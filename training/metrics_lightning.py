from abc import ABC, abstractmethod

import torch

from . import metric_utils_lightning
import scipy
import numpy as np
import torchmetrics


class StyleGANMetric(torchmetrics.Metric, ABC):
    def __init__(self, detector_url: str, detector_kwargs: dict = None,
                 max_real=None, num_gen=None,):
        super(StyleGANMetric, self).__init__(compute_on_step=False)

        self.max_real = max_real
        self.num_gen = num_gen

        self.ds_feats = metric_utils_lightning.FeatStatsDataset(detector_url=detector_url,
                                                                detector_kwargs=detector_kwargs)

        self.gen_feats = metric_utils_lightning.FeatStatsGenerator(detector_url=detector_url,
                                                                   detector_kwargs=detector_kwargs)

    @property
    @abstractmethod
    def name(self):
        pass

    def prepare(self, opts):
        opts.update(num_items=self.max_real)
        self.ds_feats.prepare_dataset_features(**opts)

        opts.update(num_items=self.num_gen)
        self.gen_feats.prepare_generator_features(**opts)

    def reset(self):
        self.ds_feats.reset()
        self.gen_feats.reset()

    def update(self, images: torch.Tensor, z: torch.Tensor, c: torch.Tensor):
        if not self.ds_feats.is_full():
            self.ds_feats(images)

        if not self.gen_feats.is_full():
            self.gen_feats(z, c)

    def is_full(self):
        return self.ds_feats.is_full() and self.gen_feats.is_full()


class FID(StyleGANMetric):
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True)

    def __init__(self, max_real=None, num_gen=None):
        super(FID, self).__init__(detector_url=self.detector_url,
                                  detector_kwargs=self.detector_kwargs,
                                  max_real=max_real,
                                  num_gen=num_gen)

    def prepare(self, opts):
        opts.update(capture_mean_cov=True)
        super().prepare(opts)

    def compute(self):
        mu_real, sigma_real = self.ds_feats.compute().get_mean_cov()
        mu_gen, sigma_gen = self.gen_feats.compute().get_mean_cov()

        m = torch.square(mu_gen - mu_real).sum()
        m = m.cpu().numpy()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    def name(self):
        return 'FID'



