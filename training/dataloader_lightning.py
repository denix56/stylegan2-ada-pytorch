import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import dnnlib
import numpy as np


class StyleGANDataModule(pl.LightningDataModule):
    def __init__(self, batch_size_per_unit, training_set_kwargs, data_loader_kwargs):
        super().__init__()
        self._batch_size_per_unit = batch_size_per_unit
        self._data_loader_kwargs = data_loader_kwargs
        self.training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
        self.n_phases = None
        self.z_dim = None

    def setup_noise_params(self, n_phases, z_dim):
        self.n_phases = n_phases
        self.z_dim = z_dim

    @property
    def name(self):
        return self.training_set.name

    def _get_noise(self, batch: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert self.n_phases
        assert self.z_dim

        imgs, _ = batch
        batch_size = imgs.shape[0]
        all_gen_z = torch.randn([self.n_phases, batch_size, self.z_dim], device=imgs.device)
        all_gen_c = self.training_set.get_label(np.random.randint(self.training_set.get_len(),
                                                size=self.n_phases*batch_size))
        if len(all_gen_c.shape) == 1:
            all_gen_c = all_gen_c.reshape((self.n_phases, batch_size))
        else:
            print(all_gen_c.shape)
            all_gen_c = all_gen_c.reshape((self.n_phases, batch_size) + all_gen_c[1:])
        all_gen_c = torch.tensor(all_gen_c, device=imgs.device)
        return all_gen_z, all_gen_c

    def train_dataloader(self):
        return DataLoader(dataset=self.training_set, batch_size=self._batch_size_per_unit, **self._data_loader_kwargs)

    def val_dataloader(self):
        return DataLoader(dataset=self.training_set, batch_size=self._batch_size_per_unit,
                          **self._data_loader_kwargs)

    def test_dataloader(self):
        return DataLoader(dataset=self.training_set, batch_size=self._batch_size_per_unit,
                          **self._data_loader_kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch[0] = batch[0].to(torch.float32) / 127.5 - 1
        all_gen_z, all_gen_c = self._get_noise(batch)
        batch = tuple(batch) + (all_gen_z, all_gen_c)
        return batch