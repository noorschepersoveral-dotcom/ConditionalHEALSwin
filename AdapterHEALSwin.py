"""
HEALSwin U-Net for Conditional Flow Matching
=============================================
Adapted from:
  https://github.com/HuCaoFighting/Swin-Unet
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

import hp_shifting
from hp_windowing import window_partition, window_reverse, get_nest_win_idcs
import DataSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    # FIX: was [cos, sin] — now correctly [sin, cos]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Adapter(nn.Module):
    """
    Bottleneck adapter: down → GELU → up → residual.
    Initialized so the up-projection is zero, meaning the adapter
    starts as an identity and training gradually activates it.
    """
    def __init__(self, dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, dim)
        self.scale = nn.Parameter(torch.ones(1))

        # Zero-init the up-projection so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(x)))

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Window Attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias."""

    def __init__(self, dim, window_size, num_heads, rel_pos_bias=None,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 use_cos_attn=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rel_pos_bias = rel_pos_bias

        if self.use_cos_attn:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        if self.rel_pos_bias == "flat":
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (int((2 * window_size ** 0.5 - 1) * (2 * window_size ** 0.5 - 1)), num_heads)
                )
            )
            coords = torch.arange(window_size ** 0.5)
            coords = torch.stack(torch.meshgrid([coords, coords]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size ** 0.5 - 1
            relative_coords[:, :, 1] += window_size ** 0.5 - 1
            relative_coords[:, :, 0] *= 2 * window_size ** 0.5 - 1
            relative_position_index = relative_coords.sum(-1).long()
            nest_idcs = get_nest_win_idcs(window_size)
            nest_idcs_inv = nest_idcs.flatten().argsort()
            relative_position_index = relative_position_index[nest_idcs_inv][:, nest_idcs_inv]
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_cos_attn:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(
                self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if self.rel_pos_bias == "flat":
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index
            ].permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Swin Transformer Block 
# ---------------------------------------------------------------------------

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block
    """

    def __init__(self, dim, input_resolution, base_pix, num_heads,
                 window_size=4, shift_size=0, shift_strategy="nest_roll",
                 rel_pos_bias=None, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_v2_norm_placement=False, use_cos_attn=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_v2_norm_placement = use_v2_norm_placement

        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            rel_pos_bias=rel_pos_bias, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, use_cos_attn=use_cos_attn,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        nside = math.sqrt(input_resolution // base_pix)
        assert nside % 1 == 0, "nside has to be an integer in every layer"
        nside = int(nside)

        shifters = {
            "nest_roll": (hp_shifting.NestRollShift, {
                "shift_size": self.shift_size,
                "input_resolution": self.input_resolution,
                "window_size": self.window_size,
            }),
            "nest_grid_shift": (hp_shifting.NestGridShift, {
                "nside": nside, "base_pix": base_pix, "window_size": self.window_size,
            }),
            "ring_shift": (hp_shifting.RingShift, {
                "nside": nside, "base_pix": base_pix,
                "window_size": self.window_size, "shift_size": self.shift_size,
            }),
        }

        if self.shift_size > 0:
            self.shifter = shifters[shift_strategy][0](**shifters[shift_strategy][1])
        else:
            self.shifter = hp_shifting.NoShift()

        self.register_buffer("attn_mask", self.shifter.get_mask())

    def forward(self, x, emb=None):
        B, N, C = x.shape
        shortcut = x

        if not self.use_v2_norm_placement:
            x = self.norm1(x)

        if emb is not None:
            x = x + emb.unsqueeze(1)

        shifted_x = self.shifter.shift(x)
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, N)
        x = self.shifter.shift_back(shifted_x)

        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            # Adapter after attention sub-layer
            if hasattr(self, 'adapter_attn'):
                x = self.adapter_attn(x)
            x = x + self.drop_path(self.norm2(self.mlp(x)))
            # Adapter after MLP sub-layer
            if hasattr(self, 'adapter_mlp'):
                x = self.adapter_mlp(x)
        else:
            x = shortcut + self.drop_path(x)
            if hasattr(self, 'adapter_attn'):
                x = self.adapter_attn(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if hasattr(self, 'adapter_mlp'):
                x = self.adapter_mlp(x)

        return x


# ---------------------------------------------------------------------------
# Patch Merging / Expand
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, N, C = x.shape
        assert N % 4 == 0
        x0, x1, x2, x3 = x[:, 0::4, :], x[:, 1::4, :], x[:, 2::4, :], x[:, 3::4, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False) if dim_scale != 1 else nn.Identity()
        self.norm = norm_layer(dim * dim_scale // 4)

    def forward(self, x):
        x = self.expand(x)
        B, N, C = x.shape
        x = rearrange(x, "b n (p c) -> b (n p) c", p=4, c=C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, patch_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.expand(x)
        B, N, C = x.shape
        x = rearrange(x, "b n (p c) -> b (n p) c", p=self.patch_size, c=C // self.patch_size)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# BasicLayer (encoder)
# ---------------------------------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 base_pix, shift_size, shift_strategy, rel_pos_bias,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 use_v2_norm_placement=False, use_cos_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size, base_pix=base_pix,
                shift_size=0 if (i % 2 == 0) else shift_size,
                shift_strategy=shift_strategy, rel_pos_bias=rel_pos_bias,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_v2_norm_placement=use_v2_norm_placement,
                use_cos_attn=use_cos_attn,
            )
            for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, emb=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                def _fwd(_blk, _x, _emb):
                    return _blk(_x, _emb)
                x = checkpoint.checkpoint(_fwd, blk, x, emb, use_reentrant=False)
            else:
                x = blk(x, emb)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# BasicLayer_up (decoder) 
# ---------------------------------------------------------------------------

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 base_pix, shift_size, shift_strategy, rel_pos_bias,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 upsample=None, use_checkpoint=False,
                 use_v2_norm_placement=False, use_cos_attn=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size, base_pix=base_pix,
                shift_size=0 if (i % 2 == 0) else shift_size,
                shift_strategy=shift_strategy, rel_pos_bias=rel_pos_bias,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_v2_norm_placement=use_v2_norm_placement,
                use_cos_attn=use_cos_attn,
            )
            for i in range(depth)
        ])

        self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x, emb=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                def _fwd(_blk, _x, _emb):
                    return _blk(_x, _emb)
                x = checkpoint.checkpoint(_fwd, blk, x, emb, use_reentrant=False)
            else:
                x = blk(x, emb)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, config, data_spec):
        super().__init__()
        assert config.patch_size % 4 == 0
        self.config = config
        self.data_spec = data_spec
        self.num_patches = data_spec.dim_in // config.patch_size
        self.proj = nn.Conv1d(
            data_spec.f_in, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.norm = config.patch_embed_norm_layer if config.patch_embed_norm_layer is not None else None

    def forward(self, x):
        B, C, N = x.shape
        assert N == self.data_spec.dim_in
        x = self.proj(x).transpose(1, 2)   # (B, num_patches, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# UNet Decoder
# ---------------------------------------------------------------------------

class UnetDecoder(nn.Module):
    def __init__(self, config, data_spec, dpr):
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        num_patches = data_spec.dim_in // config.patch_size

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        # Per-decoder-layer projections from base embed_dim → layer-specific dim.
        # Decoder goes from deepest (num_layers-1) up to shallowest (0), so:
        #   i_layer=0  → down_idx = num_layers-1  → dim = embed_dim * 2^(num_layers-1)
        #   i_layer=1  → down_idx = num_layers-2  → dim = embed_dim * 2^(num_layers-2)
        #   ...
        # i_layer=0 is a bare PatchExpand (no blocks, no emb), so we only need
        # projections for i_layer >= 1.
        self.emb_projs = nn.ModuleList()

        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            concat_out = int(config.embed_dim * 2 ** down_idx)
            concat_in = 2 * concat_out
            concat_linear = nn.Linear(concat_in, concat_out) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                layer_up = PatchExpand(dim=concat_out, dim_scale=2, norm_layer=config.norm_layer)
                # No emb projection needed for bare PatchExpand — add a placeholder
                self.emb_projs.append(nn.Identity())
            else:
                layer_up = BasicLayer_up(
                    dim=concat_out,
                    input_resolution=num_patches // (4 ** down_idx),
                    depth=config.depths[down_idx],
                    num_heads=config.num_heads[down_idx],
                    window_size=config.window_size,
                    base_pix=data_spec.base_pix,
                    shift_size=config.shift_size,
                    shift_strategy=config.shift_strategy,
                    rel_pos_bias=config.rel_pos_bias,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_scale=config.qk_scale,
                    use_cos_attn=config.use_cos_attn,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[sum(config.depths[:down_idx]):sum(config.depths[:down_idx + 1])],
                    norm_layer=config.norm_layer,
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    upsample=PatchExpand if down_idx > 0 else None,
                    use_checkpoint=config.use_checkpoint,
                )
                # Project base_emb (embed_dim) → concat_out (the dim of this layer)
                self.emb_projs.append(nn.Linear(config.embed_dim, concat_out))

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = config.norm_layer(config.embed_dim)
        self.up = FinalPatchExpand_X4(patch_size=config.patch_size, dim=config.embed_dim)
        self.output = nn.Conv1d(
            in_channels=config.embed_dim, out_channels=data_spec.f_out,
            kernel_size=1, bias=False,
        )

    def forward(self, x, x_downsample, emb=None):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # Bare PatchExpand — no blocks, no emb injection
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], dim=-1)
                x = self.concat_back_dim[inx](x)
                # Project base_emb to the correct channel dim for this decoder layer
                layer_emb = self.emb_projs[inx](emb) if emb is not None else None
                x = layer_up(x, layer_emb)
        x = self.norm_up(x)
        x = self.up(x)
        x = x.permute(0, 2, 1)     # (B, C, N)
        x = self.output(x)
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SwinHPTransformerConfig:
    patch_size: int = 4
    window_size: int = 4
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[nn.LayerNorm] = None
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: type = nn.LayerNorm
    use_v2_norm_placement: bool = False
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    dev_mode: bool = False
    decoder_class: type = UnetDecoder


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SwinHPTransformerSys(nn.Module):
    """
    HEALSwin U-Net for Conditional Flow Matching.

    Conditioning:
      - t      : flow-matching time step (scalar per batch element)
      - cond1–11: up to 11 cosmological parameters (e.g. w0, wa, ...)

    All signals are concatenated and projected to embed_dim, then a
    per-layer linear maps the embedding to a dim-specific vector that
    is added to the residual at every block.
    """

    def __init__(self, config: SwinHPTransformerConfig, data_spec, **kwargs):
        super().__init__()
        self.config = config
        self.data_spec = data_spec
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # ---- Patch embedding ----
        self.patch_embed = PatchEmbed(config, data_spec=data_spec)
        num_patches = self.patch_embed.num_patches

        # ---- Conditioning network ----
        # Input dim: embed_dim (t_emb) + 11 (scalar cosmological params)
        self.num_cond_params = 11
        cond_in_dim = config.embed_dim + self.num_cond_params
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, 4 * config.embed_dim),
            nn.SiLU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )

        # Per-encoder-layer embedding projections: base embed_dim -> layer-specific dim.
        self.layer_emb_projs = nn.ModuleList([
            nn.Linear(config.embed_dim, int(config.embed_dim * 2 ** i))
            for i in range(self.num_layers)
        ])

        # ---- Absolute position embedding (optional) ----
        if config.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # ---- Stochastic depth schedule ----
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # ---- Encoder layers ----
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else None
            layer = BasicLayer(
                dim=int(config.embed_dim * 2 ** i_layer),
                input_resolution=num_patches // (4 ** i_layer),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                base_pix=data_spec.base_pix,
                shift_size=config.shift_size,
                shift_strategy=config.shift_strategy,
                rel_pos_bias=config.rel_pos_bias,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=config.norm_layer,
                use_v2_norm_placement=config.use_v2_norm_placement,
                downsample=downsample,
                use_checkpoint=config.use_checkpoint,
            )
            self.layers.append(layer)

        # ---- Decoder ----
        self.decoder = config.decoder_class(config, data_spec, dpr)

        # ---- Bottleneck norm ----
        self.norm = config.norm_layer(self.num_features)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    # ------------------------------------------------------------------
    def _build_emb(self, x, t, cond_params: torch.Tensor):
        """
        Build the shared conditioning embedding of shape (B, embed_dim).

        Args:
            x          : patch-embedded input, used only to infer B
            t          : (B,) or scalar flow-matching timestep
            cond_params: (B, num_cond_params) — all cosmological parameters
                         stacked along dim=-1 before calling this method.
        """
        B = x.shape[0]

        # --- time embedding ---
        if t.dim() == 0:
            t = t.expand(B)
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(B)
        t_emb = timestep_embedding(t, self.config.embed_dim)   # (B, embed_dim)

        # --- cosmological parameters ---
        # Expect (B, num_cond_params); guard against bare (B,) in the 1-param edge case
        if cond_params.dim() == 1:
            cond_params = cond_params.unsqueeze(1)
        assert cond_params.shape == (B, self.num_cond_params), (
            f"Expected cond_params shape ({B}, {self.num_cond_params}), "
            f"got {tuple(cond_params.shape)}"
        )

        # Concatenate all signals, then project to embed_dim
        raw = torch.cat([t_emb, cond_params], dim=-1)   # (B, embed_dim + num_cond_params)
        emb = self.cond_proj(raw)                        # (B, embed_dim)
        return emb

    # ------------------------------------------------------------------
    def forward_features(self, x, t, cond_params: torch.Tensor):
        x = self.patch_embed(x)
        if self.config.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Shared base embedding (B, embed_dim)
        base_emb = self._build_emb(x, t, cond_params)

        x_downsample = []
        for i, layer in enumerate(self.layers):
            x_downsample.append(x)
            layer_emb = self.layer_emb_projs[i](base_emb)   # (B, embed_dim * 2^i)
            x = layer(x, layer_emb)

        x = self.norm(x)
        return x, x_downsample, base_emb

    # ------------------------------------------------------------------
    def forward(self, x, t, cond_params: torch.Tensor):
        """
        Args:
            x          : (B, f_in, N_pix) input field
            t          : (B,) or scalar flow-matching timestep
            cond_params: (B, 11) cosmological parameters,
                         e.g. torch.stack([w0, wa, ...], dim=-1)
        """
        x, x_downsample, base_emb = self.forward_features(x, t, cond_params)
        x = self.decoder(x, x_downsample, base_emb)
        return x
