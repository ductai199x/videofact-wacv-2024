import torch
from torch import nn

from .weight_init import trunc_normal_
from .vit_modified import VisionTransformer


class LongDistanceAttention(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
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
    ):
        super().__init__()
        # Get Latent-space representation X
        self.vit = VisionTransformer(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            act_layer,
        )

        self.forgery_template = nn.Parameter(torch.zeros(num_forg_template, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.forgery_template, std=0.02)

    def forward(self, x):
        latent_repr = self.vit(x)
        attn_map = torch.einsum("bpe, te -> btp", latent_repr, self.forgery_template)
        return attn_map
