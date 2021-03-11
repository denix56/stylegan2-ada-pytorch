from torch.utils.data import DataLoader
import pytorch_lightning as pl
import dnnlib


class StyleGANDataModule(pl.LightningDataModule):
    def __init__(self, batch_size_per_unit, training_set_kwargs, data_loader_kwargs):
        super().__init__()
        self._batch_size_per_unit = batch_size_per_unit
        self._data_loader_kwargs = data_loader_kwargs
        self.training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)

    @property
    def label_dim(self):
        return self.training_set.label_dim

    def train_dataloader(self):
        return DataLoader(dataset=self.training_set, batch_size=self._batch_size_per_unit, **self._data_loader_kwargs)

    def val_dataloader(self):
        return DataLoader(dataset=self.training_set, batch_size=self._batch_size_per_unit,
                          **self._data_loader_kwargs)

    def test_dataloader(self):
        return DataLoader(dataset=self.training_set, batch_size=self._batch_size_per_unit,
                          **self._data_loader_kwargs)