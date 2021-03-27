import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import dnnlib
import copy
import numpy as np
from torch_utils.ops import conv2d_gradfix
from functools import partial
from torch_utils import misc
from .metrics_lightning import StyleGANMetric
from .dataloader_lightning import StyleGANDataModule
from .metric_utils_lightning import LogitSign
import time
import psutil
import os
from .utils import setup_snapshot_image_grid, save_image_grid
from torchvision.utils import make_grid, save_image

class StyleGAN2(pl.LightningModule):
    def __init__(self, G, D, G_opt_kwargs, D_opt_kwargs, augment_pipe, datamodule: StyleGANDataModule,
                 style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 G_reg_interval=4, D_reg_interval=16, ema_kimg=10, ema_rampup=None, ada_target=None, ada_interval=4,
                 ada_kimg = 500, metrics = None, kimg_per_tick = 4, image_snapshot_ticks = 50, random_seed=0):
        super().__init__()
        self.G = G
        self.D = D
        self.G_ema = copy.deepcopy(self.G).eval().requires_grad_(False)
        self._G_opt_kwargs = G_opt_kwargs
        self._D_opt_kwargs = D_opt_kwargs
        self.augment_pipe = augment_pipe

        self.datamodule = datamodule

        self.G_reg_interval = G_reg_interval
        self.D_reg_interval = D_reg_interval
        self.phases = []

        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([])

        self.ema_kimg = ema_kimg
        self.ema_rampup = ema_rampup

        self.ada_target = ada_target
        self.ada_interval = ada_interval
        self.ada_kimg = ada_kimg

        if ada_target is not None:
            self.logit_sign = LogitSign()

        if metrics is None:
            metrics = []
        if isinstance(metrics, StyleGANMetric):
            metrics = [metrics]
        assert all([isinstance(metric, StyleGANMetric) for metric in metrics])
        self.metrics = nn.ModuleList(metrics)

        self.start_epoch = 0
        self.tick_start_nimg = 0
        self.kimg_per_tick = kimg_per_tick

        self.tick_start_time = 0
        self.start_time = 0
        self.tick_end_time = 0
        self.cur_nimg = 0

        self.image_snapshot_ticks = image_snapshot_ticks
        self.random_seed = random_seed

    def on_train_start(self):
        if self.trainer.is_global_zero:
            tensorboard = self.logger.experiment
            self.grid_size, images, labels = setup_snapshot_image_grid(self.datamodule.training_set, self.random_seed)

            samples = make_grid(torch.tensor(images, dtype=torch.float), nrow=self.grid_size[0], normalize=True, value_range=(0, 255))
            save_image(samples, 'reals.jpg')
            tensorboard.add_image('Original', samples, global_step=self.global_step)
            self.grid_z = torch.randn([labels.shape[0], self.G.z_dim], device=self.device)
            self.grid_c = torch.from_numpy(labels).to(self.device)
            images = torch.cat([self.G_ema(z=z[None, ...], c=c[None, ...]).cpu() for z, c in zip(self.grid_z, self.grid_c)])
            images = make_grid(images, nrow=self.grid_size[0], normalize=True, value_range=(-1,1))
            tensorboard.add_image('Generated', images, global_step=self.global_step)

    def setup(self, stage):
        self.start_epoch = self.current_epoch
        self.cur_nimg = self.global_step
        self.pl_mean = torch.zeros_like(self.pl_mean)
        for metric in self.metrics:
            metric.reset()

        self.start_time = time.time()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.cur_nimg >= self.tick_start_nimg + self.kimg_per_tick * 1000:
            return -1

    def training_step(self, batch, batch_idx, optimizer_idx):
        phase = self.phases[optimizer_idx]
        if self.global_step % phase.interval == 0:
            imgs, labels, all_gen_z, all_gen_c = batch
            loss = phase.loss(imgs, labels, all_gen_z[optimizer_idx], all_gen_c[optimizer_idx])
            self.print(loss, loss.requires_grad, phase.module.__class__.__name__)
            # for param in phase.module.parameters():
            #     print(param.requires_grad)
            return loss

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)
        phase = self.phases[optimizer_idx]
        for param in phase.module.parameters():
            if param.grad is not None:
                misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # Update G_ema.
        ema_nimg = self.ema_kimg * 1000
        batch_size = batch[0].shape[0]
        self.cur_nimg = (self.global_step + 1) * batch_size

        if self.ema_rampup is not None:
            ema_nimg = min(ema_nimg, self.cur_nimg * self.ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
            b_ema.copy_(b)

        # Execute ADA heuristic.
        if self.augment_pipe is not None and self.ada_target is not None and (self.global_step % self.ada_interval == 0):
            logit_sign = self.logit_sign.compute()
            adjust = np.sign(logit_sign - self.ada_target) * (batch_size * self.ada_interval) / (self.ada_kimg * 1000)
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=self.device)))

    def on_train_epoch_start(self):
        self.tick_start_nimg = self.cur_nimg
        self.tick_start_time = time.time()
        if self.tick_end_time != 0:
            self.log('Timing/maintenance_sec', self.tick_start_time - self.tick_end_time)

    def on_train_epoch_end(self, outputs):
        self.tick_end_time = time.time()
        mean_values = {
            'total_sec': self.tick_end_time - self.start_time,
            'Timing/sec_per_tick': self.tick_end_time - self.tick_start_time,
            'Timing/sec_per_kimg': (self.tick_end_time - self.tick_start_time) / (self.cur_nimg - self.tick_start_nimg) * 1e3,
            'Timing/total_hours': (self.tick_end_time - self.start_time) / (60 * 60),
            'Timing/total_days': (self.tick_end_time - self.start_time) / (24 * 60 * 60),
            'Progress/augment': float(self.augment_pipe.p.cpu()) if self.augment_pipe is not None else 0,
            'Resources/peak_gpu_mem_gb': torch.cuda.max_memory_allocated(self.device) / 2 ** 30
        }

        sum_values = {
            'Resources/cpu_mem_gb': psutil.Process(os.getpid()).memory_info().rss / 2 ** 30,
        }

        self.log_dict(mean_values)
        self.log_dict(sum_values, reduce_fx=torch.sum)

        if self.current_epoch % self.image_snapshot_ticks == 0:
            images = torch.cat([self.G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(self.grid_z, self.grid_c)])
            images = make_grid(images, nrow=self.grid_size[0], normalize=True, range=(-1,1))

            tensorboard = self.logger.experiment
            tensorboard.add_image('Generated', images, global_step=self.global_step)

    def configure_optimizers(self):
        self.phases = []
        opts = []

        for i, (name, module, opt_kwargs,
                reg_interval, loss_) in enumerate([('G', self.G, self._G_opt_kwargs, self.G_reg_interval, self._gen_loss),
                                                   ('D', self.D, self._D_opt_kwargs, self.D_reg_interval, self._disc_loss)
                                                   ]):
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
                                                          **opt_kwargs)  # subclass of torch.optim.Optimizer
                loss = partial(loss_, do_main=True, do_reg=True, gain=1)
                self.phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=i, interval=1, loss=loss)]
                opts.append(opt)
            else:  # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(module.parameters(),
                                                          **opt_kwargs)  # subclass of torch.optim.Optimizer
                loss = partial(loss_, do_main=True, do_reg=False, gain=1)
                self.phases += [dnnlib.EasyDict(name=name + 'main', module=module, interval=1, loss=loss)]
                opts.append(opt)
                loss = partial(loss_, do_main=False, do_reg=True, gain=reg_interval)
                self.phases += [dnnlib.EasyDict(name=name + 'reg', module=module, interval=reg_interval, loss=loss)]
                opts.append(opt)

        self.datamodule.setup_noise_params(len(self.phases), self.G.z_dim)

        return opts, []

    def on_validation_start(self):
        if self.metrics:
            opts = dnnlib.EasyDict(trainer=self.trainer, dataset_name=self.self.datamodule.name,
                                   device=self.device, G=self.G_ema)
            for metric in self.metrics:
                metric.prepare(opts)

    def validation_step(self, batch, batch_idx):
        if self.metrics:
            batch_size = batch[0].shape[0]
            z = torch.randn((batch_size, self.G.z_dim), device=self.device)
            indices = np.random.randint(self.training_set.get_len(), size=batch_size)
            c = torch.tensor([self.training_set.get_label(indices[i]) for i in range(batch_size)], device=self.device)
            for metric in self.metrics:
                metric(batch, z, c)

    def on_validation_end(self):
        if self.metrics:
            values = {}
            for metric in self.metrics:
                values[metric.name] = metric.compute()
                metric.reset()
            self.log_dict(values)

    def _gen_run(self, z: torch.Tensor, c: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        ws = self.G.mapping(z, c)
        if self.style_mixing_prob > 0:
            ws = self._style_mixing(z, c, ws)
        img = self.G.synthesis(ws)
        return img, ws

    def _disc_run(self, img: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c)
        return logits

    def _gen_main_loss(self, gen_z: torch.Tensor, gen_c: torch.Tensor, gain: int) -> torch.Tensor:
            gen_img, _gen_ws = self._gen_run(gen_z, gen_c)
            self.print(gen_img.requires_grad)
            gen_logits = self._disc_run(gen_img, gen_c)
            self.print(gen_logits.requires_grad)
            loss_Gmain = F.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain = loss_Gmain.mean()
            values = {'Loss/scores/fake': gen_logits.mean(),
                      'Loss/signs/fake': gen_logits.sign().mean(),
                      'Loss/G/loss': loss_Gmain}
            self.log_dict(values)
            return loss_Gmain.mean().mul(gain)

    def _gen_pl_loss(self, gen_z: torch.Tensor, gen_c: torch.Tensor, gain: int) -> torch.Tensor:
        batch_size = gen_z.shape[0] // self.pl_batch_shrink
        gen_img, gen_ws = self._gen_run(gen_z[:batch_size], gen_c[:batch_size])
        pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        with conv2d_gradfix.no_weight_gradients():
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                           only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        self.pl_mean = self.pl_mean.to(pl_lengths.device)
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        # training_stats.report('Loss/pl_penalty', pl_penalty)
        loss_Gpl = pl_penalty * self.pl_weight
        # training_stats.report('Loss/G/reg', loss_Gpl)

        values = {'Loss/pl_penalty': pl_penalty,
                  'Loss/G/reg': loss_Gpl}
        self.log_dict(values)

        return (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain)

    def _disc_main_loss(self, gen_z: torch.Tensor, gen_c: torch.Tensor, gain: int, log_values:dict) -> torch.Tensor:
        gen_img, _gen_ws = self._gen_run(gen_z, gen_c)
        gen_logits = self._disc_run(gen_img, gen_c)
        # training_stats.report('Loss/scores/fake', gen_logits)
        # training_stats.report('Loss/signs/fake', gen_logits.sign())
        loss_Dgen = F.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))

        log_values['Loss/scores/fake'] = gen_logits.mean()
        log_values['Loss/signs/fake'] = gen_logits.sign().mean()
        log_values['Loss/D/loss'] += loss_Dgen.mean()

        return loss_Dgen.mean().mul(gain)

    # Dmain: Maximize logits for real images.
    # Dr1: Apply R1 regularization.
    def _disc_max_logits_r1_loss(self, real_img: torch.Tensor, real_c: torch.Tensor, gain: int,
                         do_main: bool, do_reg: bool, log_values:dict) -> torch.Tensor:
        real_img_tmp = real_img.detach().requires_grad_(do_reg)
        real_logits = self._disc_run(real_img_tmp, real_c)
        self.logit_sign(real_logits)
        # training_stats.report('Loss/scores/real', real_logits)
        # training_stats.report('Loss/signs/real', real_logits.sign())
        loss_Dreal = 0
        if do_main:
            loss_Dreal = F.softplus(-real_logits)  # -log(sigmoid(real_logits))
            log_values['Loss/D/loss'] += loss_Dreal.mean()
            # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

        loss_Dr1 = 0
        if do_reg:
            with conv2d_gradfix.no_weight_gradients():
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                               only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1, 2, 3])
            loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

            log_values['Loss/r1_penalty'] = r1_penalty
            log_values['Loss/D/reg'] = loss_Dr1

        return (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain)

    def _gen_loss(self, real_img: torch.Tensor,
                  real_c: torch.Tensor, gen_z: torch.Tensor, gen_c: torch.Tensor,
                  gain: int, do_main: bool, do_reg: bool) -> torch.Tensor:
        loss = None
        do_reg = do_reg and self.pl_weight != 0
        if do_main:
            loss = self._gen_main_loss(gen_z, gen_c, gain)
            self.print('main')
            if do_reg:
                loss += self._gen_pl_loss(gen_z, gen_c, gain)
                self.print('reg')
        elif do_reg:
            loss = self._gen_pl_loss(gen_z, gen_c, gain)
            self.print('reg only')
        if loss is None:
            self.print('none')
            loss = torch.zeros(1, device=self.device)
        return loss

    def _disc_loss(self, real_img: torch.Tensor, real_c: torch.Tensor, gen_z: torch.Tensor, gen_c: torch.Tensor,
                   gain: int, do_main: bool, do_reg: bool) -> torch.Tensor:
        do_reg = do_reg and self.r1_gamma != 0
        values = {}
        if do_main:
            values['Loss/D/loss'] = torch.zeros([], device=self.device)
        loss = self._disc_max_logits_r1_loss(real_img, real_c, gain, do_main=do_main, do_reg=do_reg, log_values=values)
        if do_main:
           loss += self._disc_main_loss(gen_z, gen_c, gain, log_values=values)
        self.log_dict(values)
        return loss

    def _style_mixing(self, z: torch.Tensor, c: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                             torch.full_like(cutoff, ws.shape[1]))
        ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        return ws






