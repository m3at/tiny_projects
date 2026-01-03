import math
from typing import Annotated

import torch
from torch import nn


# Copied from vLLM
class TensorShape:
    def __init__(
        self,
        *dims: int | str,
        dynamic_dims: set[str] | None = None,
    ) -> None:
        super().__init__()

        self.dims = dims
        self.dynamic_dims = dynamic_dims if dynamic_dims else set()

    def resolve(self, **bindings: int) -> tuple[int | str, ...]:
        resolved = list[int | str]()
        for dim in self.dims:
            if isinstance(dim, str) and dim in bindings:
                resolved.append(bindings[dim])
            else:
                resolved.append(dim)
        return tuple(resolved)

    def __str__(self) -> str:
        """Return a string representation of the tensor shape."""
        dim_strs = []
        for dim in self.dims:
            if isinstance(dim, str):
                if dim in self.dynamic_dims:
                    dim_strs.append(f"{dim}*")  # Mark dynamic dimensions with *
                else:
                    dim_strs.append(dim)
            else:
                dim_strs.append(str(dim))
        return f"({', '.join(dim_strs)})"


class DinoBilinear(nn.Module):
    """DINO-v2 ViT backbone with bilinear interpolation."""

    def __init__(self, dino):
        super().__init__()
        self.cls_token = dino.cls_token
        self.pos_embed = dino.pos_embed
        self.patch_size = dino.patch_size
        self.interpolate_offset = dino.interpolate_offset

        self.N = self.pos_embed.shape[1] - 1
        self.M = int(math.sqrt(self.N))  # type:ignore[unresolved-attribute]

        self.patch_embed = dino.patch_embed
        self.blocks = nn.Sequential(*dino.blocks)
        self.norm = dino.norm

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        if npatch == self.N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        sx = float(w0 + self.interpolate_offset) / self.M
        sy = float(h0 + self.interpolate_offset) / self.M

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, self.M, self.M, dim).permute(0, 3, 1, 2),
            mode="bilinear",
            scale_factor=(sx, sy),
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, x) -> Annotated[torch.Tensor, TensorShape("b", "n", "c")]:
        """Returns normalized tokens: (B, 1 + N, C)"""
        _, _, w, h = x.shape
        x = self.patch_embed(x)  # (B, N, C)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # (B, 1+N, C)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    # def forward_pooled(self, x):
    #     """CLS pooled feature: (B, C)"""
    #     x = self.forward(x)
    #     return x[:, 0]


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv) -> torch.Tensor:
        # q: (B, Q, D), kv: (B, S, D)
        q2 = self.ln1(q)
        q_sa, _ = self.self_attn(q2, q2, q2, need_weights=False)
        q = q + q_sa

        q2 = self.ln2(q)
        q_ca, _ = self.cross_attn(q2, kv, kv, need_weights=False)
        q = q + q_ca

        q2 = self.ln3(q)
        q = q + self.ff(q2)
        return q


class ClockModel(nn.Module):
    """
    Clock reading model with a DETR/Perceiver-style query decoder head:
    - Backbone provides patch tokens.
    - Two learned queries (hour, minute) cross-attend to tokens.
    - Each query predicts (sin, cos) and we normalize to unit length.
    """

    def __init__(
        self,
        backbone: DinoBilinear,
        backbone_channels: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        depth: int = 2,
        dim_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Normalization for DINOv2
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))

        self.backbone = backbone

        self.in_proj = nn.Linear(backbone_channels, d_model)
        self.query = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)  # [hour_query, minute_query]

        dim_ff = dim_ff or (4 * d_model)
        self.decoder = nn.ModuleList(
            [_DecoderBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff, dropout=dropout) for _ in range(depth)]
        )

        # small per-query regressor -> 2 values (sin, cos)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x) -> Annotated[torch.Tensor, TensorShape("b", 4)]:
        x = (x - self.norm_mean) / self.norm_std

        tokens = self.backbone.forward(x)  # (B, 1+N, C)

        kv = self.in_proj(tokens)  # include CLS + patches
        q = self.query.expand(kv.shape[0], -1, -1)  # (B, 2, D)

        for blk in self.decoder:
            q = blk(q, kv)

        # (B, 2, 2) -> normalize each (sin, cos) to lie on unit circle
        vec = self.out(q)
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)

        # return (B, 4): sin_h, cos_h, sin_m, cos_m
        hour = vec[:, 0]  # (B, 2)
        minute = vec[:, 1]  # (B, 2)
        return torch.cat([hour, minute], dim=-1)
