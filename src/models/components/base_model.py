from typing import Any, Dict
from omegaconf import DictConfig
import torch
from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def step(self, train: bool, batch: Any):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(True, batch)

        self.log_dict(
            {"metatrain/inner_loss": inner_loss,
             "metatrain/inner_accuracy": inner_acc},
            on_epoch=False,
            on_step=True,
            prog_bar=False
        )
        self.log_dict(
            {"metatrain/outer_loss": outer_loss,
             "metatrain/outer_accuracy": outer_acc},
            on_epoch=False,
            on_step=True,
            prog_bar=True
        )

    @torch.enable_grad()
    def validation_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.model.train()
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(False, batch)
        self.log_dict(
            {"metaval/inner_loss": inner_loss,
             "metaval/inner_accuracy": inner_acc},
            prog_bar=False
        )
        self.log_dict(
            {"metaval/outer_loss": outer_loss,
             "metaval/outer_accuracy": outer_acc},
            prog_bar=True
        )

    @torch.enable_grad()
    def test_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.model.train()
        outer_loss, inner_loss, outer_acc, inner_acc = self.step(False, batch)
        self.log_dict(
            {"metatest/outer_loss": outer_loss,
             "metatest/inner_loss": inner_loss,
             "metatest/inner_accuracy": inner_acc,
             "metatest/outer_accuracy": outer_acc},
        )