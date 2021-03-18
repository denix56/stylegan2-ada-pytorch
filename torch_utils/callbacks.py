import pytorch_lightning as pl
from training.dataset import Dataset

class EpochCB(pl.Callback):
    def __init__(self, kimg_per_tick: int = 4):
        super(EpochCB, self).__init__()
        self.tick_start_nimg = 0
        self.kimg_per_tick = kimg_per_tick

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.tick_start_nimg = trainer.current_epoch

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx):
        cur_nimg = trainer.global_step * batch.shape[0]
        if trainer.current_epoch == 0 or cur_nimg >= self.tick_start_nimg + self.kimg_per_tick * 1000:
            trainer.
            self.dataset.stop()