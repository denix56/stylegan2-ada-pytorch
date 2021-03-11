# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
import pytorch_lightning as pl


#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, trainer: pl.Trainer = None, cache=True):
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.trainer = trainer
        #self.device         = device if device is not None else torch.device('cuda', rank)
        #self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, trainer: pl.Trainer, device, verbose=False):
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = trainer.is_global_zero
        if not is_leader and trainer.data_parallel:
            trainer.training_type_plugin.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and trainer.data_parallel:
            trainer.training_type_plugin.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------


class AllItemsMetric(pl.metrics.Metric):
    def __init__(self):
        super(AllItemsMetric, self).__init__(dist_sync_on_step=True)
        self.add_state('all_features', default=[], dist_reduce_fx=None)
        self.all_features_sync = []

    def update(self, x: torch.Tensor):
        self.all_features.append(x)

    def compute(self):
        self.all_features_sync += [feats.cpu() for feats in self.all_features]
        self.reset()
        return self.all_features_sync


class NumItemsMetric(pl.metrics.Metric):
    def __init__(self):
        super(NumItemsMetric, self).__init__(dist_sync_on_step=True)

        self.add_state('num_items', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, x: torch.Tensor):
        self.num_items += x.shape[0]

    def compute(self):
        return self.num_items


class FeatureStats(pl.metrics.Metric):
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None, num_features=None, **kwargs):
        super(FeatureStats, self).__init__(compute_on_step=False)
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_features = num_features
        self.num_items_metric = NumItemsMetric()

        if self.capture_all:
            self.all_items_metric = AllItemsMetric()

        if self.capture_mean_cov:
            self.add_state('raw_mean', default=torch.zeros([num_features], dtype=torch.float64), dist_reduce_fx='sum')
            self.add_state('raw_cov', default=torch.zeros([num_features, num_features], dtype=torch.float64),
                           dist_reduce_fx='sum')

    def reset(self):
        super(FeatureStats, self).reset()
        self.num_items_metric.reset()
        self.all_items_metric.reset()

    def update(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        num_items = self.num_items_metric.compute().item()
        if (self.max_items is not None) and (num_items + x.shape[0] > self.max_items):
            if num_items >= self.max_items:
                return
            x = x[:self.max_items - num_items]

        x = x.to(torch.float32)
        self.num_items_metric(x)
        if self.capture_all:
            self.all_items_metric(x)
        if self.capture_mean_cov:
            x64 = x.to(torch.float64)
            self.raw_mean += x64.sum(dim=0)
            self.raw_cov += x64.T @ x64

    def compute(self):
        assert self.capture_mean_cov
        num_items = self.num_items_metric.compute()
        mean = self.raw_mean / num_items
        cov = self.raw_cov / num_items
        cov = cov - torch.outer(mean, mean)
        return mean, cov

    def get_mean_cov(self):
        return self.compute()

    def is_full(self):
        return (self.max_items is not None) and (self.num_items_metric.compute() >= self.max_items)

    def get_all(self):
        assert self.capture_all
        return self.get_all_torch().numpy()

    def get_all_torch(self):
        all_features = self.all_items_metric.compute()
        return torch.cat(all_features, dim=0)

    def save(self, pkl_file):
        state = self.__dict__
        state.pop('num_items_metric', None)
        state.pop('all_items_metric', None)
        self['num_items'] = self.num_items_metric.compute().item()
        if self.capture_all:
            state['all_features'] = self.get_all()
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, capture_mean_cov=s.capture_mean_cov,
                           max_items=s.max_items, num_features=s.num_features)
        obj.__dict__.update(s)
        obj.num_items_metric.__dict__.update(s)
        if s.capture_all:
            obj.all_items_metric.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------


class FeatStatsDataset(pl.metrics.Metric):
    def __init__(self, detector_url: str, detector_kwargs: dict = None, verbose=False):
        super(FeatStatsDataset, self).__init__(compute_on_step=False)
        self.detector_url = detector_url
        self.detector_kwargs = detector_kwargs
        self.stats = None
        self.verbose = verbose
        self.detector = None
        self.cache_file = None
        self.loaded = False

    def reset(self):
        super(FeatStatsDataset, self).reset()
        self.stats = None
        self.detector = None
        self.loaded = False

    def prepare_dataset_features(self, trainer: pl.Trainer, dataset_name: str, device: torch.device, cache: str,
                                 **stats_kwargs):
        self.reset()

        self.cache_file = None
        if cache:
            # Choose cache file name.
            args = dict(dataset_name=dataset_name, detector_url=self.detector_url,
                        detector_kwargs=self.detector_kwargs, capture_mean_cov=True)
            md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
            cache_tag = f'{dataset_name}-{get_feature_detector_name(self.detector_url)}-{md5.hexdigest()}'
            self.cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

            # Check if the file exists (all processes must agree).
            flag = os.path.isfile(self.cache_file) if trainer.is_global_zero else False
            if trainer.data_parallel:
                flag = torch.as_tensor(flag, dtype=torch.float32, device=device)
                trainer.training_type_plugin.broadcast(flag, src=0)
                flag = (float(flag.cpu()) != 0)

            # Load.
            if flag:
                self.stats = FeatureStats.load(self.cache_file)
                self.loaded = True
                return

        # Initialize.
        self.stats = FeatureStats(**stats_kwargs)
        self.detector = get_feature_detector(url=self.detector_url,
                                             trainer=trainer,
                                             device=device,
                                             verbose=self.verbose)

    def update(self, images: torch.Tensor):
        # if data_loader_kwargs is None:
        #     data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

        # item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
        # loader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size,
        #                                      **data_loader_kwargs)
        # for images, _labels in loader:
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = self.detector(images, **self.detector_kwargs)
        self.stats(features)

    def compute(self):

        return self.stats

    def save(self, trainer: pl.Trainer):
        # Save to cache.
        if self.cache_file is not None and trainer.is_global_zero:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            temp_file = self.cache_file + '.' + uuid.uuid4().hex
            self.stats.save(temp_file)
            os.replace(temp_file, self.cache_file)  # atomic

    def is_full(self):
        return (self.stats is not None) and (self.loaded or self.stats.is_full())


#----------------------------------------------------------------------------

class FeatStatsGenerator(pl.metrics.Metric):
    def __init__(self, trainer: pl.Trainer, detector_url: str, detector_kwargs: dict = None, stats_kwargs=None, verbose=False):
        super(FeatStatsGenerator, self).__init__()

        self.trainer = trainer
        self.detector_url = detector_url
        self.detector_kwargs = detector_kwargs
        self.stats = None
        self.stats_kwargs = stats_kwargs
        self.verbose = verbose
        self.detector = None
        self.run_generator = None
        self.jit = False
        self.traced = False
        self.dims = None

    def reset(self):
        super(FeatStatsGenerator, self).reset()
        self.jit = False
        self.traced = False
        self.dims = None
        self.run_generator = None
        self.stats = None
        self.detector = None

    def prepare_generator_features(self, trainer: pl.Trainer, G, G_kwargs={}, device: torch.device=None, jit=False, **stats_kwargs):
        self.reset()
        self.jit = jit

        # Image generation func.
        def run_generator(z, c):
            img = G(z=z, c=c, **G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img

        self.run_generator = run_generator
        self.dims = (G.z_dim, G.c_dim)
        # Initialize.
        self.stats = FeatureStats(**stats_kwargs)
        assert self.stats.max_items is not None
        # progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
        self.detector = get_feature_detector(url=self.detector_url, trainer=trainer,
                                                               device=device, verbose=self.verbose)

    def _trace(self, data: torch.Tensor):
        batch_size = data.shape[0]
        if self.jit and not self.traced:
            z = torch.zeros([batch_size, self.dims[0]], device=data.device)
            c = torch.zeros([batch_size, self.dims[1]], device=data.device)
            self.run_generator = torch.jit.trace(self.run_generator, (z, c), check_trace=False)

    def update(self, z: torch.Tensor, c: torch.Tensor):
        batch_size = z.shape[0]
        self._trace(z)

        images = self.run_generator(z, c)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = self.detector(images, **self.detector_kwargs)
        self.stats(features)
        # progress.update(stats.num_items)
        return self.stats

    def compute(self):
        return self.stats

    def is_full(self):
        return (self.stats is not None) and self.stats.is_full()


#----------------------------------------------------------------------------
#
# class ProgressMonitor:
#     def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
#         self.tag = tag
#         self.num_items = num_items
#         self.verbose = verbose
#         self.flush_interval = flush_interval
#         self.progress_fn = progress_fn
#         self.pfn_lo = pfn_lo
#         self.pfn_hi = pfn_hi
#         self.pfn_total = pfn_total
#         self.start_time = time.time()
#         self.batch_time = self.start_time
#         self.batch_items = 0
#         if self.progress_fn is not None:
#             self.progress_fn(self.pfn_lo, self.pfn_total)
#
#     def update(self, cur_items):
#         assert (self.num_items is None) or (cur_items <= self.num_items)
#         if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
#             return
#         cur_time = time.time()
#         total_time = cur_time - self.start_time
#         time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
#         if (self.verbose) and (self.tag is not None):
#             print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
#         self.batch_time = cur_time
#         self.batch_items = cur_items
#
#         if (self.progress_fn is not None) and (self.num_items is not None):
#             self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)
#
#     def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
#         return ProgressMonitor(
#             tag             = tag,
#             num_items       = num_items,
#             flush_interval  = flush_interval,
#             verbose         = self.verbose,
#             progress_fn     = self.progress_fn,
#             pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
#             pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
#             pfn_total       = self.pfn_total,
#         )
#
# #----------------------------------------------------------------------------
