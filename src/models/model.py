from typing import Any, Dict, Sequence, Tuple, Union

import higher
import hydra
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import Accuracy, F1Score, Precision, Recall
# from pytorch_lightning.metric import Accuracy
from pytorch_lightning import LightningModule

from src.models.components import BaseModel

class MAMLModel(BaseModel):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(cfg=cfg, *args, **kwargs)

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.model : nn.Module = cfg.net.to(device=self.device)
        self.inner_optimizer: Optimizer = cfg.inner_optimizer(params=self.model.parameters())

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, train: bool, batch: Any):
        self.model.zero_grad()

        # Unpack batch fom support and query sets
        support_x, support_y = batch['support']
        support_x, support_y = support_x.to(self.device), support_y.to(self.device)
        query_x, query_y = batch['query']
        query_x, query_y = query_x.to(self.device), query_y.to(self.device)

        # Initialize support and query set loss and accuracy
        support_loss = torch.tensor(0.0, device=self.device)
        support_acc = Accuracy(task="multiclass", num_classes=self.hparams.cfg.nway)
        query_loss = torch.tensor(0.0, device=self.device)
        query_acc = Accuracy(task="multiclass", num_classes=self.hparams.cfg.nway)

        # Initialize optimizer
        outer_optimizer = self.optimizers()
        inner_optimizer = self.inner_optimizer

        # Inner Loop over support & query set
        for batch_idx, (train_x, train_y, test_x, test_y) in enumerate(zip(support_x, support_y, query_x, query_y)):
            with higher.innerloop_ctx(self.model, inner_optimizer, copy_initial_weights=False, track_higher_grads=train) as (fmodel, diffopt):
                # Inner training steps for support set
                for _ in range(self.hparams.cfg.num_inner_steps):
                    train_logit = fmodel(train_x)
                    train_loss = F.cross_entropy(train_logit, train_y)
                    diffopt.step(train_loss)

                # Compute loss & acc for support set
                with torch.no_grad():
                    train_logit = fmodel(train_x)
                    train_preds = torch.softmax(train_logit, dim=-1)
                    support_loss += F.cross_entropy(train_logit, train_y).cpu()
                    support_acc.update(train_preds.cpu(), train_y.cpu())

                # Compute loss, gradient & acc for query set
                test_logit = fmodel(test_x)
                query_loss += F.cross_entropy(test_logit, test_y)
                with torch.no_grad():
                    test_preds = torch.softmax(train_logit, dim=-1)
                    query_acc.update(test_preds.cpu(), test_y.cpu())

        # Backpropagate loss for query set and update weights for learner
        if train:
            outer_optimizer.zero_grad()
            self.manual_backward(query_loss)
            outer_optimizer.step()

        # Average support and query set loss
        support_loss.div_(len(support_y))
        query_loss.div_(len(query_y))

        return support_loss.item(), query_loss.item(), support_acc.compute(), query_acc.compute()

    def configure_optimizers(self) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Lr and weight_decay are partially initialized in hydra.utils.instantiate(cfg.model)
        outer_optimizer : Optimizer = self.hparams.cfg.outer_optimizer(params=self.parameters())

        if self.hparams.cfg.use_lr_scheduler:
            scheduler = self.hparams.cfg.scheduler(optimizer=outer_optimizer)
            return [outer_optimizer], [scheduler]

        return outer_optimizer



if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "default.yaml")
    _ = hydra.utils.instantiate(cfg)
