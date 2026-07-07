# ---------------------
# Imports
# ---------------------

import wandb
import numpy as np
import healpy as hp
import torch
import torchdiffeq
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import List, Optional
import CondHEALSwin as HS
import torch.nn as nn
from torchdyn.core import NeuralODE
from FM_multcond import ExactOptimalTransportConditionalFlowMatcher
import matplotlib.pyplot as plt
import os
import argparse
import h5py

# ---------------------
# Argument parsing
# ---------------------

runspecifics = 'FileName'
cond_names = ['Om', 's8', 'w0', '0cdm', '0nu', 'As', 'H0', 'm_nu', 'ns', 'Ob', 'Ol']
NUM_CONDS = len(cond_names)   # 11

patch_size      = 4
window_size     = 4
shift_size      = 2
embed_dim       = 96
depths          = (2, 2, 6, 2)
num_heads       = (3, 6, 12, 24)
drop_path_rate  = 0.1
learning_rate   = 5e-4
batch_size      = 32
n_epochs        = 100

# ---------------------
# Load Data
# ---------------------

INI_MAP = (
    "Directory name"
)

x1_maps = np.load(f"{INI_MAP}/Target.npy").astype(np.float32)[:, np.newaxis, :]   # (N, 1, P)
x0_maps = np.load(f"{INI_MAP}/Initial.npy").astype(np.float32)[:, np.newaxis, :]  # (N, 1, P)

# Load all 11 cosmological parameter arrays
cond_vals = [
    np.load(f"{INI_MAP}/{name}_standard.npy") for name in cond_names
]   # list of 11 arrays, each (N,)

assert x1_maps.shape == x0_maps.shape, (
    f"x0 and x1 shape mismatch: {x0_maps.shape} vs {x1_maps.shape}"
)

# ---------------------
# HEALPix rotation helper
# ---------------------

NSIDE       = hp.npix2nside(x1_maps.shape[-1])
N_ROTATIONS = 4
ROT_ANGLES  = [0, 90, 180, 270]


def rotate_healpix_map(healpix_map: np.ndarray, phi_deg: float, nside: int) -> np.ndarray:
    if phi_deg == 0:
        return healpix_map.copy()
    npix               = hp.nside2npix(nside)
    theta, phi         = hp.pix2ang(nside, np.arange(npix), nest=False)
    phi_src            = (phi - np.deg2rad(phi_deg)) % (2 * np.pi)
    src_pix            = hp.ang2pix(nside, theta, phi_src, nest=False)
    return healpix_map[src_pix]


# ---------------------
# Dataset
# ---------------------

class CosmologyDataset(Dataset):
    """
    Wraps paired (x0, x1) cosmology maps and quadruples the dataset by
    rotating both maps by 0°, 90°, 180°, 270°.

    Labels tensor shape: (N * n_rot * S, NUM_CONDS)
    """

    def __init__(
        self,
        x0_maps:   np.ndarray,        # (N, S, P)
        x1_maps:   np.ndarray,        # (N, S, P)
        cond_vals: list,              # list of NUM_CONDS arrays, each (N,)
    ):
        N, S, P = x1_maps.shape
        nside   = hp.npix2nside(P)

        self.N, self.S, self.P = N, S, P
        self.n_rot             = N_ROTATIONS

        rot_x0 = np.empty((self.n_rot, N, S, P), dtype=np.float32)
        rot_x1 = np.empty((self.n_rot, N, S, P), dtype=np.float32)

        for i in range(N):
            for j in range(S):
                for r, angle in enumerate(ROT_ANGLES):
                    rot_x0[r, i, j] = rotate_healpix_map(x0_maps[i, j], angle, nside)
                    rot_x1[r, i, j] = rotate_healpix_map(x1_maps[i, j], angle, nside)

        rot_x0 = rot_x0.transpose(1, 0, 2, 3).reshape(N * self.n_rot, S, P)
        rot_x1 = rot_x1.transpose(1, 0, 2, 3).reshape(N * self.n_rot, S, P)

        self.x0_maps = torch.from_numpy(rot_x0)
        self.x1_maps = torch.from_numpy(rot_x1)

        # Stack all 11 parameters into a (N, NUM_CONDS) array, then tile for
        # rotations and shells to produce (N * n_rot * S, NUM_CONDS)
        cond_matrix = np.stack(cond_vals, axis=-1)   # (N, NUM_CONDS)
        rows = []
        for i in range(N):
            for _ in range(self.n_rot):
                for j in range(S):
                    rows.append(cond_matrix[i])

        self.labels = torch.tensor(np.array(rows), dtype=torch.float32)  # (N*n_rot*S, NUM_CONDS)
        self.N_aug  = N * self.n_rot

    def __len__(self) -> int:
        return self.N_aug * self.S

    def __getitem__(self, idx: int):
        aug_idx   = idx // self.S
        shell_idx = idx %  self.S

        x0    = self.x0_maps[aug_idx, shell_idx]   # (P,)
        x1    = self.x1_maps[aug_idx, shell_idx]   # (P,)
        label = self.labels[idx]                   # (NUM_CONDS,)

        return x0, x1, label


dataset = CosmologyDataset(x0_maps, x1_maps, cond_vals)
loader  = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=7,
    pin_memory=True,
)

# ---------------------
# Weights & Biases
# ---------------------

run = wandb.init(
    entity="",
    project="",
    config={
        "learning_rate":  learning_rate,
        "batch_size":     batch_size,
        "patch_size":     patch_size,
        "window_size":    window_size,
        "shift_size":     shift_size,
        "embed_dim":      embed_dim,
        "depths":         depths,
        "num_heads":      num_heads,
        "drop_path_rate": drop_path_rate,
        "number_epochs":  n_epochs,
        "cond_params":    cond_names,
    },
)

# ---------------------
# Model setup
# ---------------------

@dataclass
class DataSpec:
    dim_in:      int
    f_in:        int
    f_out:       int
    base_pix:    Optional[int]        = 0
    class_names: Optional[List[str]] = None


nside = 64
npix  = 12 * nside**2

dataspec = DataSpec(dim_in=npix, f_in=1, f_out=1, base_pix=npix, class_names=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = HS.SwinHPTransformerConfig(
    patch_size            = patch_size,
    window_size           = window_size,
    shift_size            = shift_size,
    shift_strategy        = "nest_roll",
    rel_pos_bias          = None,
    embed_dim             = embed_dim,
    patch_embed_norm_layer= None,
    depths                = depths,
    num_heads             = num_heads,
    mlp_ratio             = 4.0,
    qkv_bias              = True,
    qk_scale              = None,
    use_cos_attn          = False,
    drop_rate             = 0.0,
    attn_drop_rate        = 0.0,
    drop_path_rate        = drop_path_rate,
    norm_layer            = nn.LayerNorm,
    use_v2_norm_placement = False,
    ape                   = False,
    patch_norm            = True,
    use_checkpoint        = False,
    dev_mode              = False,
    decoder_class         = HS.UnetDecoder,
)

model     = HS.SwinHPTransformerSys(config, dataspec).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
node      = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
wandb.watch(model, log="all", log_freq=100)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr          = 3e-4,
    steps_per_epoch = len(loader),
    epochs          = n_epochs,
    pct_start       = 0.05,
)

# ---------------------
# Training loop
# ---------------------

for epoch in range(n_epochs):
    epoch_loss = 0.0
    n_batches  = 0

    for x0_batch, x1_batch, cond in loader:
        optimizer.zero_grad()

        x0   = x0_batch.unsqueeze(1).to(device)   # (B, 1, P)
        x1   = x1_batch.unsqueeze(1).to(device)   # (B, 1, P)
        cond = cond.to(device)                     # (B, NUM_CONDS)

        t, xt, ut, _, condition = FM.guided_sample_location_and_conditional_flow(
            x0, x1, y1=cond
        )

        vt   = model(xt, t, cond)
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    wandb.log({"loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch:03d}/{n_epochs} — loss: {avg_loss:.6f}")

# ---------------------
# Image generation
# ---------------------

@torch.no_grad()
def generate_from_x0(
    model:       nn.Module,
    x0_map:      np.ndarray,    # (P,)
    cond_params: np.ndarray,    # (NUM_CONDS,)
    n_pix:       int,
    device:      torch.device,
    num_samples: int   = 1,
    rtol:        float = 1e-5,
    atol:        float = 1e-5,
    n_steps:     int   = 50,
) -> np.ndarray:
    """
    Integrate the learned vector field from a given initial map x0_map.

    Args:
        model:       Trained SwinHPTransformerSys.
        x0_map:      Initial HEALPix map, shape (P,).
        cond_params: All 11 cosmological parameters, shape (NUM_CONDS,).
        n_pix:       Number of HEALPix pixels (12 * nside**2).
        device:      Torch device.
        num_samples: How many realisations to draw from the same x0.
        rtol / atol: ODE solver tolerances.
        n_steps:     Number of time steps for the ODE integrator.

    Returns:
        Generated maps as a numpy array of shape (num_samples, 1, P),
        clipped to [0, 1].
    """
    x0_tensor = (
        torch.from_numpy(x0_map)
        .float()
        .to(device)
        .unsqueeze(0).unsqueeze(0)             # (1, 1, P)
        .expand(num_samples, 1, n_pix)
        .clone()
    )
    y0 = x0_tensor.permute(0, 2, 1).contiguous().view(num_samples, -1)

    # Pre-build the (B, NUM_CONDS) conditioning tensor — same for all ODE steps
    cond_tensor = (
        torch.from_numpy(cond_params)
        .float()
        .to(device)
        .unsqueeze(0)                           # (1, NUM_CONDS)
        .expand(num_samples, -1)               # (num_samples, NUM_CONDS)
        .clone()
    )

    def ode_func(t, y_flat):
        B = y_flat.shape[0]
        y = y_flat.view(B, n_pix, 1).permute(0, 2, 1).contiguous()   # (B, 1, P)
        t_batch = torch.full((B,), float(t), device=device)
        v = model(y, t_batch, cond_tensor)                             # (B, 1, P)
        return v.permute(0, 2, 1).contiguous().view(B, -1)

    traj = torchdiffeq.odeint(
        func   = ode_func,
        y0     = y0,
        t      = torch.linspace(0, 1, n_steps, device=device),
        rtol   = rtol,
        atol   = atol,
        method = "dopri5",
    )

    y_final = traj[-1].view(num_samples, n_pix, 1).permute(0, 2, 1)
    return y_final.clamp(0, 1).cpu().numpy()


def healpix_to_wandb_image(
    hp_map:         np.ndarray,
    nside:          int,
    xsize:          int  = 800,
    project_to_rgb: bool = True,
) -> wandb.Image:
    n_chan = hp_map.shape[0]
    assert n_chan in (1, 3)
    assert hp_map.shape[1] == 12 * nside**2

    proj   = hp.projector.MollweideProj(xsize=xsize)
    pix_fn = lambda x, y, z: hp.vec2pix(nside, x, y, z)

    if n_chan == 1:
        img        = proj.projmap(hp_map[0], pix_fn)
        img_stacked = np.stack([img] * 3, axis=-1) if project_to_rgb else img[..., None]
    else:
        img_stacked = np.stack(
            [proj.projmap(hp_map[c], pix_fn) for c in range(3)], axis=-1
        )

    mask               = np.isfinite(img_stacked[..., 0])
    img_stacked[~mask] = 0.5
    lo                 = np.percentile(img_stacked[mask], 1)
    hi                 = np.percentile(img_stacked[mask], 99)
    img_stretched      = np.clip((img_stacked - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    img_uint8          = (255 * img_stretched ** 0.7).astype(np.uint8)

    return wandb.Image(img_uint8)

# ---------------------
# Generation
# ---------------------

x0_for_gen_cosmo1 = np.load(f"{INI_MAP}/ComparisonMaps0.npy").astype(np.float32)  # (500, P)
x0_for_gen_cosmo2 = np.load(f"{INI_MAP}/ComparisonMaps1.npy").astype(np.float32)  # (500, P)
x0_for_gen_cosmo3 = np.load(f"{INI_MAP}/ComparisonMaps2.npy").astype(np.float32)  # (500, P)

cond_for_gen = np.stack([
    np.load(f"{INI_MAP}/{name}_standard.npy") for name in cond_names
], axis=-1)   # (N, NUM_CONDS)

cosmo_sim_indices = [0, 1, 2]

save_dir = ""
os.makedirs(save_dir, exist_ok=True)

n_initial_maps = x0_for_gen_cosmo1.shape[0]

for cosmo_idx, (x0_for_gen, sim_idx) in enumerate(
    zip([x0_for_gen_cosmo1, x0_for_gen_cosmo2, x0_for_gen_cosmo3], cosmo_sim_indices),
    start=1
):
    # Fixed conditioning params for this cosmology — same for all 10 maps
    cond_single = cond_for_gen[sim_idx]   # (NUM_CONDS,)
    cond_label  = "_".join(
        f"{name}{val:.3f}" for name, val in zip(cond_names, cond_single)
    )

    all_generated = []  # collect all 10 for W&B logging

    for map_idx in range(n_initial_maps):
        x0_single = x0_for_gen[map_idx]   # (P,)

        generated = generate_from_x0(
            model       = model,
            x0_map      = x0_single,
            cond_params = cond_single,
            n_pix       = npix,
            device      = device,
            num_samples = 1,
        )   # (1, 1, P)

        generated_map = generated[0, 0, :]   # (P,)
        all_generated.append(generated_map)

    all_generated = np.stack(all_generated, axis=0)   # (10, P)

    # ── Save each map individually ─────────────────────────────────
    save_path = os.path.join(
        save_dir,
        f"generated_cosmo{cosmo_idx}_{cond_label}_shell14_{runspecifics}.npy",
    )
    np.save(save_path, all_generated)   # (10, P)
    print(f"Saved {save_path}")

    # ── Log all 10 maps for this cosmology to W&B ─────────────────────
    all_generated = np.stack(all_generated, axis=0)   # (10, P)
    wandb_images = [
        healpix_to_wandb_image(all_generated[i][np.newaxis, :], nside=nside, xsize=800, project_to_rgb=False)
        for i in range(n_initial_maps)
    ]
    wandb.log({f"gen_cosmo{cosmo_idx}_{cond_label}_shell14": wandb_images})

# Log one dataset sample for visual reference
data_x0, data_x1, data_cond = next(iter(loader))
x1_sample = data_x1[0].cpu().numpy()[np.newaxis, :]
wandb.log({
    "dataset_x1_mollweide": healpix_to_wandb_image(x1_sample, nside=nside, xsize=800, project_to_rgb=False)
})

print("All generated maps saved to", save_dir)
print("Finished.")
wandb.finish()
