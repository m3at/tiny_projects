import math
from typing import Annotated

import torch
from torch import nn

from src.models import TensorShape


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


class DinoBackbone(nn.Module):
    """Thin wrapper around a DINOv2 ViT that returns normalized tokens (CLS + regs + patches)."""

    def __init__(self, dino):
        super().__init__()

        # Keep references so weights stay tied to the original module.
        self.patch_embed = dino.patch_embed
        self.blocks = dino.blocks
        self.norm = dino.norm

        self.cls_token = dino.cls_token
        self.pos_embed = dino.pos_embed

        # Optional bits (register tokens exist for some DINOv2 variants)
        self.register_tokens = getattr(dino, "register_tokens", None)
        self.num_register_tokens = int(getattr(dino, "num_register_tokens", 0))

        # Optional mask token (not used by you, but cheap to keep faithful)
        self.mask_token = getattr(dino, "mask_token", None)

        # Interpolation knobs (DINOv2 uses bicubic + optional antialias + offset)
        ps = getattr(dino, "patch_size", 16)
        if isinstance(ps, (tuple, list)):
            self.ph = int(ps[0])
            self.pw = int(ps[1])
        else:
            self.ph = int(ps)
            self.pw = int(ps)
        self.interpolate_offset = float(getattr(dino, "interpolate_offset", 0.1))
        self.interpolate_antialias = bool(getattr(dino, "interpolate_antialias", False))

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        # Matches upstream DINOv2 implementation (mode="bicubic", antialias, offset, asserts). :contentReference[oaicite:1]{index=1}
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.ph
        h0 = h // self.pw

        # Add small offset to avoid floating point interpolation issues (see DINO issue #8).
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset  # :contentReference[oaicite:2]{index=2}

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        # Upstream asserts: sanity check that scaling landed exactly where expected. :contentReference[oaicite:3]{index=3}
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(
        self,
        x: torch.Tensor,
        # masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Matches upstream ordering: CLS, (+regs), patches, with pos-embed added before regs insertion. :contentReference[oaicite:4]{index=4}
        _, _, w, h = x.shape
        x = self.patch_embed(x)  # (B, N, C)

        # if masks is not None:
        #     if self.mask_token is None:
        #         raise RuntimeError("masks were provided but dino.mask_token is missing.")
        #     x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # (B, 1+N, C)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward(self, x: torch.Tensor) -> Annotated[torch.Tensor, TensorShape("b", "n", "c")]:
        """Returns normalized tokens: (B, 1 + R + N, C) where R=num_register_tokens."""
        x = self.prepare_tokens_with_masks(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
