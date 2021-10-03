import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from typing import Any
import torch.nn as nn

from src.models.modules.tcn import MS_TCN2


class MSTCNLitModel(LightningModule):

    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super().__init__()

        # model parameters
        self.num_layers_PG = num_layers_PG
        self.num_layers_R = num_layers_R
        self.num_R = num_R
        self.num_f_maps = num_f_maps
        self.dim = dim
        self.num_classes = num_classes

        # model
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)

        # loss

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x: torch.Tensor):
        pass

    def step(self, batch: Any):
        x, y, mask = batch
        loss = 0
        logits = self.forward(x)
        for p in logits:
            loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), y.view(-1))
            loss += 0.15 * torch.mean(
                torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask[:, :, 1:])
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self.step(batch)
        acc = self.train_accuracy(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)



    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
