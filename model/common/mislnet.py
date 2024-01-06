from typing import *

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torchmetrics import AUROC, Accuracy


class ConstrainedConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(ConstrainedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def constrain_weight(self) -> None:
        shape = self.weight.shape  # [0]: filter, [1]: channel, [2]: width, [3]: height
        tmp_w = self.weight.data
        center_pos = (shape[-2] * shape[-1]) // 2
        for filtr in range(shape[0]):
            for channel in range(shape[1]):
                krnl = tmp_w[filtr, channel, :, :]
                krnl = krnl.view(-1)

                krnl[center_pos] = 0
                krnl = krnl * 10000
                krnl = krnl / torch.sum(krnl)
                krnl[center_pos] = -1
                krnl = torch.reshape(krnl, (shape[-2], shape[-1]))
                tmp_w[filtr, channel, :, :] = krnl

        self.weight = nn.Parameter(tmp_w)

    def forward(self, input: Tensor) -> Tensor:
        self.constrain_weight()
        return self._conv_forward(input, self.weight, self.bias)


class MISLnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            # ConstrainedConv2d(3, 3, kernel_size=5, stride=1, padding="valid"),
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding="valid"),
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=(5, 5)),
            nn.BatchNorm2d(96),
            nn.Tanh(),
            nn.MaxPool2d(3, stride=2, padding=(1, 1)),
            nn.Conv2d(96, 64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(3, stride=2, padding=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.AvgPool2d(3, stride=2, padding=(1, 1)),
            nn.Flatten(),
            nn.LazyLinear(200),
            nn.Tanh(),
            nn.LazyLinear(200),
            nn.Tanh(),
            nn.Linear(200, num_classes),
        )
        self.init_weights()

    def init_weights(self):
        for i in [0, 1, 5, 9, 13]:
            nn.init.xavier_uniform_(self.model[i].weight)
            nn.init.zeros_(self.model[i].bias)

    def forward(self, x):
        return self.model(x)


class MISLnetPLWrapper(LightningModule):
    def __init__(
        self,
        patch_size=128,
        num_classes=70,
        lr=1e-4,
        momentum=0.95,
        decay_rate=0.75,
        decay_step=3,
    ):
        super().__init__()
        self.model = MISLnet(num_classes)
        self.lr = lr
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.decay_step = decay_step

        self.example_input_array = torch.randn(2, 3, patch_size, patch_size)

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.logger.experiment.add_graph(self, self.example_input_array.to(self.device))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=True, prog_bar=True)

    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.decay_rate)
        return [optimizer], [steplr]
