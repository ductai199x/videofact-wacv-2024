from typing import *

import torch
from torch import nn
from torch.nn import functional as F

from .long_dist_attn import LongDistanceAttention
from .mislnet import MISLnetPLWrapper
from .xception import Xception, XceptionPLWrapper


class VideoFACT(nn.Module):
    def __init__(
        self,
        img_size=(1080, 1920),
        patch_size=128,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_forg_template=3,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        bb1_db_depth=1,
        fe="mislnet",
        fe_config={},
        fe_ckpt="",
        fe_freeze=True,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.fe_name = fe.lower()
        self.fe_freeze = fe_freeze

        if self.fe_name == "mislnet":
            self.pretrained_fe = MISLnetPLWrapper()
        elif self.fe_name == "xception":
            self.pretrained_fe = XceptionPLWrapper()
        else:
            raise (NotImplementedError)

        if fe_ckpt is not None and len(fe_ckpt) > 0:
            self.pretrained_fe = self.pretrained_fe.load_from_checkpoint(
                fe_ckpt, **fe_config, map_location="cpu"
            )

        if self.fe_name == "mislnet":
            self.pretrained_fe = nn.Sequential(*(list(self.pretrained_fe.model.children())[0][:-2]))
            self.pretrained_fe_con = None  # connector layer
        elif self.fe_name == "xception":
            self.pretrained_fe = nn.Sequential(*(list(self.pretrained_fe.model.children())[:-1]))
            self.pretrained_fe_con = nn.Linear(2048, 200)

        self.backbone = Xception(in_chans, embed_dim, bb1_db_depth)

        self.bb_condense = nn.Linear(embed_dim, embed_dim - 200)

        self.long_dist_attn = LongDistanceAttention(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            num_forg_template,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            act_layer,
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(2),
        )

        self.localizer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 32, 1, kernel_size=(1, 1), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # split image into non-overlapping patches
        kernel_size, stride = self.patch_size, self.patch_size
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).permute(0, 2, 3, 1, 4, 5)
        # gather up all the patch into a large batch
        patches = patches.contiguous().view(-1, 3, kernel_size, kernel_size)
        # feed large batch into the pretrained feature extractor
        if self.fe_freeze:
            with torch.no_grad():
                fe = self.pretrained_fe(patches)
        else:
            fe = self.pretrained_fe(patches)

        if self.pretrained_fe_con:
            if self.fe_name == "xception":
                fe = F.adaptive_avg_pool2d(fe, (1, 1))
                fe = fe.view(fe.size(0), -1)
                fe = self.pretrained_fe_con(fe)
        # feed large batch into the backbone to produce their features
        bb = self.backbone(patches)
        bb = self.bb_condense(bb)
        # concatinate fe and bb into a single vector of features
        bb_fe = torch.cat([bb, fe], dim=1)
        # split large batch back into embedded images
        bb_fe = bb_fe.contiguous().view(B, -1, self.embed_dim)
        # get the maps from the long distance attention
        lda_maps = self.long_dist_attn(bb_fe)
        # scale the embeddings by the attention maps
        scaled_bb_fe = torch.einsum("ijk,ilj->iljk", bb_fe, lda_maps)
        # scaled_bb = scaled_bb.contiguous().view(B, -1)
        scaled_bb_fe = torch.einsum("iljk->ijk", scaled_bb_fe)  # B, P, C = scaled_bb_fe.shape

        # feed the scaled embeddings to a classifier to get the output
        class_label = self.classifier(scaled_bb_fe)
        # for localization, we need to go from BPC to BCP1 in order to feed into the nn.Conv2d 1x1 layers (reducing C -> 1)
        patch_label = self.localizer(scaled_bb_fe.permute(0, 2, 1).unsqueeze(-1))
        patch_label = patch_label.view(B, -1)
        return class_label, patch_label
