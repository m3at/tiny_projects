import math
from typing import Annotated

import torch
import torch.nn.functional as F
from torch import nn


class TensorShape:
    """Tensor shape type annotation.

    Copied from vLLM codebase.
    """

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
        dim_strs = []
        for dim in self.dims:
            if isinstance(dim, str):
                if dim in self.dynamic_dims:
                    dim_strs.append(f"{dim}*")
                else:
                    dim_strs.append(dim)
            else:
                dim_strs.append(str(dim))
        return f"({', '.join(dim_strs)})"


class DinoV3Backbone(nn.Module):
    """DINOv3 backbone from huggingface transformers.

    Returns patch tokens only: (B, seq_len, C) = patch tokens.
    """

    def __init__(self, dinov3_vit):
        super().__init__()
        self.model = dinov3_vit

        self.hidden_size = int(self.model.config.hidden_size)
        self._patch_start_idx = 1 + int(getattr(self.model.config, "num_register_tokens", 0))

    def forward(self, x: torch.Tensor) -> Annotated[torch.Tensor, TensorShape("b", "n", "c")]:
        out = self.model(pixel_values=x, return_dict=True)
        return out.last_hidden_state[:, self._patch_start_idx :, :]


def _infer_hw(n: int) -> tuple[int, int]:
    h = int(math.isqrt(n))
    if h * h != n:
        raise ValueError(f"Patch token count {n} is not a perfect square; can't reshape to grid.")
    return h, h


def _make_polar_grid(
    center_xy_norm: torch.Tensor,  # (B,2) in [-1,1]
    theta_bins: int,
    r_bins: int,
    device: torch.device,
) -> Annotated[torch.Tensor, TensorShape("b", "t", "r", 2)]:
    # theta: [0, 2pi)
    theta = (torch.arange(theta_bins, device=device, dtype=torch.float32) / theta_bins) * (2.0 * math.pi)
    r = torch.linspace(0.0, 1.0, r_bins, device=device, dtype=torch.float32)

    cos_t = torch.cos(theta)[None, :, None]  # (1,T,1)
    sin_t = torch.sin(theta)[None, :, None]  # (1,T,1)
    r = r[None, None, :]  # (1,1,R)

    cx = center_xy_norm[:, 0].to(torch.float32)[:, None, None]  # (B,1,1)
    cy = center_xy_norm[:, 1].to(torch.float32)[:, None, None]  # (B,1,1)

    # radius in normalized grid coords; padding_mode='border' makes out-of-bounds safe
    x = cx + r * cos_t  # (B,T,R)
    y = cy + r * sin_t  # (B,T,R)

    return torch.stack([x, y], dim=-1)  # (B,T,R,2) in [-1,1]


def _soft_argmax_2d01(logits: torch.Tensor) -> torch.Tensor:
    B, _, H, W = logits.shape
    w = torch.softmax(logits.view(B, -1).to(torch.float32), dim=-1).view(B, H, W)

    xs = (torch.arange(W, device=logits.device, dtype=torch.float32) + 0.5) / float(W)
    ys = (torch.arange(H, device=logits.device, dtype=torch.float32) + 0.5) / float(H)
    xx = xs[None, None, :].expand(B, H, W)
    yy = ys[None, :, None].expand(B, H, W)

    cx = (w * xx).sum(dim=(1, 2))
    cy = (w * yy).sum(dim=(1, 2))
    return torch.stack([cx, cy], dim=-1)  # (B,2) in [0,1]


class PolarMix(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.dw = nn.Conv2d(d_model, d_model, kernel_size=3, padding=0, groups=d_model)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="replicate")
        x = F.pad(x, (0, 0, 1, 1), mode="circular")
        x = self.dw(x)
        x = self.act(x)
        return self.pw(x)


class DeformableCrossAttention2D(nn.Module):
    """Lightweight 2D deformable attention over multiple 2D feature maps.

    - q: (B, Q, D)
    - maps: list of (B, D, H_l, W_l)
    - base_ref: (B, L, 2) in [0,1], per-level reference point anchor (e.g., learned center)
    """

    def __init__(self, d_model: int, n_levels: int = 2, n_points: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_levels = n_levels
        self.n_points = n_points

        self.ref_delta = nn.Linear(d_model, n_levels * 2)
        self.offset = nn.Linear(d_model, n_levels * n_points * 2)
        self.attn = nn.Linear(d_model, n_levels * n_points)

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # init: keep things stable
        nn.init.zeros_(self.ref_delta.weight)
        nn.init.zeros_(self.ref_delta.bias)

        nn.init.zeros_(self.offset.weight)
        # bias offsets in a small circle (pixel-ish); helps early exploration
        with torch.no_grad():
            bias = torch.zeros(n_levels, n_points, 2)
            for level in range(n_levels):
                for p in range(n_points):
                    ang = 2.0 * math.pi * p / n_points
                    bias[level, p, 0] = math.cos(ang)
                    bias[level, p, 1] = math.sin(ang)
            self.offset.bias.copy_(bias.reshape(-1) * 1.5)  # ~1.5 "pixels" after normalization

        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, q: torch.Tensor, maps: list[torch.Tensor], base_ref: torch.Tensor) -> torch.Tensor:
        B, Qn, D = q.shape
        assert len(maps) == self.n_levels
        eps = 1e-5

        # base_ref: (B,L,2) in [0,1] -> expand to (B,Q,L,2)
        base_ref = base_ref[:, None, :, :].expand(B, Qn, self.n_levels, 2).clamp(eps, 1.0 - eps)

        # Anchor ref at base_ref but allow learned delta in logit space (nice + bounded)
        ref_delta = self.ref_delta(q).view(B, Qn, self.n_levels, 2).to(torch.float32)
        ref = torch.sigmoid(torch.logit(base_ref.to(torch.float32)) + ref_delta)  # (B,Q,L,2) in (0,1)

        offsets = self.offset(q).view(B, Qn, self.n_levels, self.n_points, 2).to(torch.float32)
        # offsets are interpreted in "pixel" units; normalize by map shape per level below

        attn = self.attn(q).view(B, Qn, self.n_levels, self.n_points).to(torch.float32)
        attn = F.softmax(attn.flatten(2), dim=-1).view(B, Qn, self.n_levels, self.n_points)

        out = torch.zeros((B, Qn, D), device=q.device, dtype=torch.float32)

        for level, feat in enumerate(maps):
            # feat: (B,D,H,W)
            _, _, H, W = feat.shape
            normalizer = torch.tensor([W, H], device=q.device, dtype=torch.float32)  # (2,)

            loc = ref[:, :, level, None, :] + (offsets[:, :, level, :, :] / normalizer)  # (B,Q,P,2) in ~[0,1]
            if level == 1:
                loc_y = torch.remainder(loc[..., 1], 1.0)
                loc = torch.stack([loc[..., 0], loc_y], dim=-1)

            loc = loc * 2.0 - 1.0  # -> [-1,1] for grid_sample

            # grid_sample wants (B, out_h, out_w, 2)
            grid = loc.reshape(B, Qn * self.n_points, 1, 2)

            # grid_sample can be picky with bf16 on some setups; float32 here is safer
            samp = F.grid_sample(
                feat.to(torch.float32),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (B,D,Q*P,1)

            samp = samp[:, :, :, 0].transpose(1, 2).reshape(B, Qn, self.n_points, D)  # (B,Q,P,D)

            w = attn[:, :, level, :].to(torch.float32)[:, :, :, None]  # (B,Q,P,1)
            out = out + (samp * w).sum(dim=2)

        out = self.out_proj(out.to(q.dtype))
        return self.drop(out)


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0, n_points: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = DeformableCrossAttention2D(d_model, n_levels=2, n_points=n_points, dropout=dropout)

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

    def forward(self, q: torch.Tensor, maps: list[torch.Tensor], base_ref: torch.Tensor) -> torch.Tensor:
        # q: (B,Q,D)
        q2 = self.ln1(q)
        q_sa, _ = self.self_attn(q2, q2, q2, need_weights=False)
        q = q + q_sa

        q2 = self.ln2(q)
        q = q + self.cross_attn(q2, maps, base_ref)

        q2 = self.ln3(q)
        q = q + self.ff(q2)
        return q


class ClockModel(nn.Module):
    """
    Clock reading model:
    - Frozen ViT backbone -> patch tokens
    - Two branches for context:
        (a) cartesian token grid map
        (b) polar-resampled map around a learned center
    - 2 learned queries decode via deformable cross-attn over both maps
    - Predict (sin, cos) per query
    """

    def __init__(
        self,
        backbone: DinoV3Backbone,
        backbone_channels: int = 768,
        d_model: int = 1024,
        nhead: int = 8,
        depth: int = 2,
        dim_ff: int | None = None,
        dropout: float = 0.0,
        polar_theta: int = 64,
        polar_r: int = 32,
        deform_points: int = 8,
    ):
        super().__init__()

        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))

        self.backbone = backbone

        self.in_proj = nn.Linear(backbone_channels, d_model)

        self.center_head = nn.Sequential(
            nn.LayerNorm(backbone_channels),
            nn.Linear(backbone_channels, backbone_channels // 2),
            nn.GELU(),
            nn.Linear(backbone_channels // 2, 2),
        )

        last = self.center_head[-1]
        nn.init.normal_(last.weight, std=1e-4)
        nn.init.zeros_(last.bias)

        self.center_map = nn.Conv2d(backbone_channels, 1, kernel_size=1)
        nn.init.zeros_(self.center_map.weight)
        nn.init.zeros_(self.center_map.bias)

        self.polar_theta = polar_theta
        self.polar_r = polar_r

        self.polar_mix = PolarMix(d_model)

        # Per-level embedding (cart vs polar)
        self.level_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)

        self.query = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)  # [hour_query, minute_query]

        dim_ff = dim_ff or (4 * d_model)
        self.decoder = nn.ModuleList(
            [
                _DecoderBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff, dropout=dropout, n_points=deform_points)
                for _ in range(depth)
            ]
        )

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x) -> Annotated[torch.Tensor, TensorShape("b", 4)]:
        x = (x - self.norm_mean) / self.norm_std

        with torch.no_grad():
            tokens = self.backbone.forward(x)  # (B, N, Cb)

        B, N, Cb = tokens.shape
        H, W = _infer_hw(N)

        # cart map in backbone channels (for center + polar sampling)
        cart_cb = tokens.transpose(1, 2).reshape(B, Cb, H, W)

        # learned center from a token-grid heatmap
        center01 = _soft_argmax_2d01(self.center_map(cart_cb))  # (B,2) in [0,1]
        center_norm = center01 * 2.0 - 1.0  # -> [-1,1] for grid_sample

        # polar resample -> (B,Cb,T,R) then tokens -> project to d_model
        grid = _make_polar_grid(center_norm, self.polar_theta, self.polar_r, device=x.device)
        polar_cb = F.grid_sample(
            cart_cb.to(torch.float32),
            grid,  # float32 grid
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(cart_cb.dtype)  # (B,Cb,T,R)

        polar_tokens = polar_cb.permute(0, 2, 3, 1).reshape(B, self.polar_theta * self.polar_r, Cb)

        # project both branches to d_model
        kv_cart = self.in_proj(tokens)  # (B,N,D)
        kv_polar = self.in_proj(polar_tokens)  # (B,T*R,D)

        D = kv_cart.shape[-1]
        cart_map = kv_cart.transpose(1, 2).reshape(B, D, H, W)
        polar_map = kv_polar.transpose(1, 2).reshape(B, D, self.polar_theta, self.polar_r)

        # add level embeddings + a little polar mixing
        cart_map = cart_map + self.level_embed[0].view(1, D, 1, 1)
        polar_map = polar_map + self.level_embed[1].view(1, D, 1, 1)
        polar_map = polar_map + self.polar_mix(polar_map)

        # base refs per level:
        # - cart: learned center (cx,cy)
        # - polar: (r=0.0, theta=0.5) anchor; offsets will explore theta
        base_ref = torch.stack(
            [
                center01,  # (B,2) (x,y)
                torch.stack(
                    [torch.zeros_like(center01[:, 0]), torch.full_like(center01[:, 1], 0.5)],
                    dim=-1,
                ),
            ],
            dim=1,
        )  # (B,2levels,2)

        q = self.query.expand(B, -1, -1)  # (B,2,D)
        maps = [cart_map, polar_map]

        for blk in self.decoder:
            q = blk(q, maps, base_ref)

        vec = self.out(q)
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)

        hour = vec[:, 0]
        minute = vec[:, 1]
        return torch.cat([hour, minute], dim=-1)
