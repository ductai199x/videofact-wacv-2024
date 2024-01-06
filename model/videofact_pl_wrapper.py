from typing import *

import torch
import torchvision
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy
from torchmetrics.functional.classification.f_beta import binary_f1_score
from torchmetrics.functional.classification.matthews_corrcoef import (
    binary_matthews_corrcoef,
)

from .common.patch_predictions import PatchPredictions


class VideoFACTPLWrapper(LightningModule):
    def __init__(self, model, **config):
        super().__init__()

        self.model = model(**config)
        self.train_class_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_class_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_class_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_class_auc = AUROC(task="multiclass", num_classes=2)
        self.test_loc_f1 = []
        self.test_loc_mcc = []

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.loss_alpha = config["loss_alpha"]
        self.lr = config["lr"] or 1e-5
        self.decay_step = config["decay_step"] or 2
        self.decay_rate = config["decay_rate"] or 0.85
        self.save_hyperparameters(config)
        self.example_input_array = torch.randn(2, 3, 1080, 1920)

        self.patch_to_pixel_pred = PatchPredictions

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.logger is not None:
            self.logger.experiment.add_graph(self, self.example_input_array.to(self.device))

    def get_patch_pred(self, m):
        batch_size = m.shape[0]
        kernel_size, stride = self.patch_size, self.patch_size
        p = m.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        p = p.contiguous().view(-1, kernel_size, kernel_size)
        p = torch.flatten(p, start_dim=1, end_dim=2)
        p = torch.sum(p, dim=1) / (kernel_size * kernel_size)
        p = p.view(batch_size, -1)
        return p

    def get_pixel_pred(self, p):
        batch_size = p.shape[0]
        kernel_size, stride = self.patch_size, self.patch_size
        # TODO: change this to reflect stride < kernel_size
        m_p_size0, m_p_size1 = (
            kernel_size * (self.img_size[0] // kernel_size),
            kernel_size * (self.img_size[1] // kernel_size),
        )
        m_p = p.unsqueeze(-1).repeat(1, 1, kernel_size * kernel_size).permute(0, 2, 1)
        m_p = F.fold(
            m_p,
            output_size=(m_p_size0, m_p_size1),
            kernel_size=kernel_size,
            stride=stride,
        ).squeeze()
        return m_p, m_p_size0, m_p_size1

    def training_step(self, batch, batch_idx):
        x, y, m = batch
        B, C, H, W = x.shape
        p = self.get_patch_pred(m)

        class_logits, patch_logits = self(x.float())
        class_loss = F.cross_entropy(class_logits, y)
        patch_loss = F.binary_cross_entropy(patch_logits, p)

        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * patch_loss

        with torch.no_grad():
            if self.logger is not None and self.global_step % 1000 == 0:
                y_hat = torch.argmax(class_logits, dim=1)
                m_p, m_p_size0, m_p_size1 = self.get_pixel_pred(patch_logits)
                m = m[:, 0:m_p_size0, 0:m_p_size1]
                sample_imgs = torch.cat(
                    [
                        x,
                        m_p.unsqueeze(1).repeat(1, 3, 1, 1) * 255,
                        m.unsqueeze(1).repeat(1, 3, 1, 1) * 255,
                    ],
                    dim=-2,
                )
                grid = torchvision.utils.make_grid(
                    sample_imgs, padding=10, pad_value=255, value_range=(0, 255)
                )

                fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
                ax.imshow(grid.permute(1, 2, 0).to(torch.uint8).cpu())
                for i in range(len(y)):
                    label = f"y={y[i].item()},y^={y_hat[i].item()}"
                    ax.text(
                        300 + i * W,
                        270,
                        label,
                        color="red",
                        fontsize=13,
                        horizontalalignment="center",
                        verticalalignment="center",
                        bbox=dict(facecolor="white", alpha=0.8),
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")

                self.logger.experiment.add_figure("generated_masks", fig, self.global_step)

            self.log("train_loss", loss)
            self.log("train_class_loss", class_loss)
            self.log("train_loc_loss", patch_loss)
            self.train_class_acc(class_logits, y)
            self.log(
                "train_class_acc",
                self.train_class_acc,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m = batch
        p = self.get_patch_pred(m)

        class_logits, patch_logits = self(x.float())
        class_loss = F.cross_entropy(class_logits, y)
        patch_loss = F.binary_cross_entropy(patch_logits, p)

        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * patch_loss

        with torch.no_grad():
            self.log("val_loss", loss)
            self.log("val_class_loss", class_loss)
            self.log("val_loc_loss", patch_loss)
            self.val_class_acc(class_logits, y)
            self.log(
                "val_class_acc",
                self.val_class_acc,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )

    def test_step(self, batch, batch_idx):
        x, y, m = batch
        p = self.get_patch_pred(m)

        class_logits, patch_logits = self(x.float())
        class_loss = F.cross_entropy(class_logits, y)
        patch_loss = F.binary_cross_entropy(patch_logits, p)

        loss = self.loss_alpha * class_loss + (1 - self.loss_alpha) * patch_loss

        with torch.no_grad():
            self.log("test_loss", loss, on_epoch=True, on_step=False)
            self.log("test_class_loss", class_loss, on_epoch=True, on_step=False)
            self.log("test_loc_loss", patch_loss, on_epoch=True, on_step=False)
            self.test_class_acc(class_logits, y)

            if y.sum() > 0:
                manip_img_patch_logits = patch_logits[y == 1].cpu()
                patch_preds = [
                    self.patch_to_pixel_pred(
                        pl,
                        self.patch_size,
                        self.img_size,
                        min_thresh=0.1,
                        max_num_regions=3,
                        final_thresh=0.30,
                    )
                    for pl in manip_img_patch_logits
                ]
                pixel_preds = torch.vstack([pp.get_pixel_preds().unsqueeze(0) for pp in patch_preds])
                m_h, m_w = pixel_preds.shape[1], pixel_preds.shape[2]
                true_mask = m[y == 1, :m_h, :m_w].to(torch.uint8)
                pixel_preds, true_mask = pixel_preds.to(self.device), true_mask.to(self.device)

                for pp, gt in zip(pixel_preds, true_mask):
                    pp_neg = 1 - pp
                    f1_pos = binary_f1_score(pp, gt)
                    f1_neg = binary_f1_score(pp_neg, gt)
                    if f1_neg > f1_pos:
                        self.test_loc_f1.append(float(f1_neg))
                    else:
                        self.test_loc_f1.append(float(f1_pos))

                    mcc_pos = binary_matthews_corrcoef(pp, gt)
                    mcc_neg = binary_matthews_corrcoef(pp_neg, gt)
                    if mcc_neg > mcc_pos:
                        self.test_loc_mcc.append(float(mcc_neg))
                    else:
                        self.test_loc_mcc.append(float(mcc_pos))
            # only compute auc at the end
            self.test_class_auc(class_logits, y)

    def on_test_epoch_end(self) -> None:
        self.log("test_loc_f1", torch.nan_to_num(torch.tensor(self.test_loc_f1)).mean())
        self.log("test_loc_mcc", torch.nan_to_num(torch.tensor(self.test_loc_mcc)).mean())
        self.log("test_class_auc", self.test_class_auc.compute())
        self.log("test_class_acc", self.test_class_acc.compute())

        self.test_class_probs = torch.concat(
            [torch.softmax(preds, dim=1)[:, 1] for preds in self.test_class_auc.preds]
        )
        self.test_class_preds = torch.concat(
            [torch.argmax(torch.softmax(preds, dim=1), dim=1) for preds in self.test_class_auc.preds]
        )
        self.test_class_truths = torch.concat([truths for truths in self.test_class_auc.target])

        pos_labels = self.test_class_truths == 1
        pos_preds = self.test_class_preds[pos_labels] == 1
        neg_labels = self.test_class_truths == 0
        neg_preds = self.test_class_preds[neg_labels] == 0
        self.log("test_class_tpr", pos_preds.sum() / pos_labels.sum())
        self.log("test_class_tnr", neg_preds.sum() / neg_labels.sum())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.97)
        steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.decay_rate)
        return [optimizer], [steplr]
