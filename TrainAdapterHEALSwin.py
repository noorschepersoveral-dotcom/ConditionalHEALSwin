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
import torch.nn as nn
from FM_multcond import ExactOptimalTransportConditionalFlowMatcher
import os
import HEALSwin_adapter_1706 as HS
from HEALSwin_adapter_1706 import Adapter, SwinTransformerBlock

# ---------------------
# Config
# ---------------------

runspecifics =  ''  

# 12 parameters — wa is the new one added for fine-tuning
cond_names = ['Om', 's8', 'w0', '0cdm', '0nu', 'As', 'H0', 'm_nu', 'ns', 'Ob', 'Ol', 'wa']
NUM_CONDS  = len(cond_names)   # 12

patch_size     = 4
window_size    = 4
shift_size     = 2
embed_dim      = 96
depths         = (2, 2, 6, 2)
num_heads      = (3, 6, 12, 24)
drop_path_rate = 0.1
batch_size     = 32
n_epochs       = 100
learning_rate  = 1e-5

# ---------------------
# Load Data
# ---------------------

INI_MAP = (
    " "
)

x1_maps = np.load(f"{INI_MAP}/Target_DE_norm.npy").astype(np.float32)[:, np.newaxis, :]
x0_maps = np.load(f"{INI_MAP}/initial_DE_norm_corr.npy").astype(np.float32)[:, np.newaxis, :]

# All 12 parameter arrays — including wa
cond_vals = [
    np.load(f"{INI_MAP}/labels_{name}_de.npy") for name in cond_names
]

assert x1_maps.shape == x0_maps.shape, (
    f"x0 and x1 shape mismatch: {x0_maps.shape} vs {x1_maps.shape}"
)

# ---------------------
# HEALPix rotation helper
# ---------------------

NSIDE      = hp.npix2nside(x1_maps.shape[-1])
N_ROTATIONS = 1
ROT_ANGLES  = [0]


def rotate_healpix_map(healpix_map: np.ndarray, phi_deg: float, nside: int) -> np.ndarray:
    if phi_deg == 0:
        return healpix_map.copy()
    npix    = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    phi_src = (phi - np.deg2rad(phi_deg)) % (2 * np.pi)
    src_pix = hp.ang2pix(nside, theta, phi_src, nest=False)
    return healpix_map[src_pix]


# ---------------------
# Dataset
# ---------------------

class CosmologyDataset(Dataset):
    """
    Wraps paired (x0, x1) cosmology maps.
    Labels tensor shape: (N * n_rot * S, NUM_CONDS)
    """

    def __init__(
        self,
        x0_maps:   np.ndarray,   # (N, S, P)
        x1_maps:   np.ndarray,   # (N, S, P)
        cond_vals: list,         # list of NUM_CONDS arrays, each (N,)
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

        cond_matrix = np.stack(cond_vals, axis=-1)   # (N, NUM_CONDS)
        rows = []
        for i in range(N):
            for _ in range(self.n_rot):
                for j in range(S):
                    rows.append(cond_matrix[i])

        self.labels = torch.tensor(np.array(rows), dtype=torch.float32)
        self.N_aug  = N * self.n_rot

    def __len__(self) -> int:
        return self.N_aug * self.S

    def __getitem__(self, idx: int):
        aug_idx   = idx // self.S
        shell_idx = idx %  self.S
        return (
            self.x0_maps[aug_idx, shell_idx],   # (P,)
            self.x1_maps[aug_idx, shell_idx],   # (P,)
            self.labels[idx],                   # (NUM_CONDS,)
        )


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
    entity=" ",
    project=" ",
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
        "bottleneck_dim": 64,
    },
)

# ---------------------
# DataSpec
# ---------------------

@dataclass
class DataSpec:
    dim_in:      int
    f_in:        int
    f_out:       int
    base_pix:    Optional[int]        = 0
    class_names: Optional[List[str]] = None

nside    = 64
npix     = 12 * nside**2
dataspec = DataSpec(dim_in=npix, f_in=1, f_out=1, base_pix=npix, class_names=None)
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Model config
# ---------------------

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

# ---------------------
# Adapter helper
# ---------------------

BOTTLENECK_DIM = 64

def inject_adapters(model, bottleneck_dim):
    for module in model.modules():
        if isinstance(module, SwinTransformerBlock):
            dim = module.dim
            module.adapter_attn = Adapter(dim, bottleneck_dim).to(device)
            module.adapter_mlp  = Adapter(dim, bottleneck_dim).to(device)

# ---------------------
# Checkpoint path
# ---------------------

checkpoint_path = f"checkpoints_adapted/{runspecifics}/model_latest.pt"
os.makedirs(f"checkpoints_adapted/{runspecifics}", exist_ok=True)
resuming = os.path.exists(checkpoint_path)

# Step 1 — build base model and load 11-param pretrained weights
model = HS.SwinHPTransformerSys(config, dataspec).to(device)
model.load_state_dict(
    torch.load(
        "checkpoints_1706_4/3006_300epoch_standard_fixed_weights/model_final_weights.pt",
        map_location=device,
        weights_only=True,
    )
)

# Step 2 — patch cond_proj from 11 → 12 params, preserving pretrained weights
old_cond_proj   = model.cond_proj
new_cond_in_dim = config.embed_dim + 12
model.cond_proj = nn.Sequential(
    nn.Linear(new_cond_in_dim, 4 * config.embed_dim),
    nn.SiLU(),
    nn.Linear(4 * config.embed_dim, config.embed_dim),
).to(device)
with torch.no_grad():
    model.cond_proj[0].weight[:, :-1] = old_cond_proj[0].weight  
    model.cond_proj[0].weight[:, -1]  = 0.0                      
    model.cond_proj[0].bias.copy_(old_cond_proj[0].bias)
    model.cond_proj[2].weight.copy_(old_cond_proj[2].weight)
    model.cond_proj[2].bias.copy_(old_cond_proj[2].bias)
model.num_cond_params = 12

# Step 3 — freeze all pretrained weights
for param in model.parameters():
    param.requires_grad = False

# Step 4 — inject adapter modules into every SwinTransformerBlock
inject_adapters(model, BOTTLENECK_DIM)

# Step 5 — unfreeze adapters + cond_proj + layer_emb_projs
trainable_params = []
for name, param in model.named_parameters():
    if any(key in name for key in ['adapter', 'cond_proj', 'layer_emb_projs']):
        param.requires_grad = True
        trainable_params.append(param)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")

# Step 6 — optimizer and scheduler (full n_epochs budget; state_dict restores position)
FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr          = 3e-4,
    steps_per_epoch = len(loader),
    epochs          = n_epochs,
    pct_start       = 0.05,
)
wandb.watch(model, log="all", log_freq=100)

# Step 7 — load checkpoint if resuming
start_epoch = 0
if resuming:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Do NOT restore scheduler state — the saved state is corrupted from
    # an earlier broken run. Fast-forward the scheduler manually instead.
    start_epoch = checkpoint["epoch"] + 1
    
    # Fast-forward scheduler to the correct step position
    for _ in range(start_epoch * len(loader)):
        scheduler.step()
    
    print(f"Resumed from epoch {start_epoch}, scheduler fast-forwarded to step {start_epoch * len(loader)}")
else:
    print("No checkpoint found — starting fine-tuning from pretrained weights.")

# ---------------------
# Training loop
# ---------------------

for epoch in range(start_epoch, n_epochs):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0

    for x0_batch, x1_batch, cond in loader:
        optimizer.zero_grad()

        x0   = x0_batch.unsqueeze(1).to(device)   # (B, 1, P)
        x1   = x1_batch.unsqueeze(1).to(device)   # (B, 1, P)
        cond = cond.to(device)                     # (B, 12)

        t, xt, ut, _, condition = FM.guided_sample_location_and_conditional_flow(
            x0, x1, y1=cond
        )

        vt   = model(xt, t, cond)
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    wandb.log({"loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch:03d}/{n_epochs} — loss: {avg_loss:.6f}")

    # Save latest checkpoint every epoch
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss":                 avg_loss,
        "cond_names":           cond_names,
    }, checkpoint_path)

    # Permanent snapshot every 50 epochs
    if (epoch + 1) % 50 == 0:
        snapshot_path = f"checkpoints_adapted/{runspecifics}/model_epoch{epoch+1:03d}.pt"
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(), "loss": avg_loss,
        }, snapshot_path)
        print(f"Snapshot saved → {snapshot_path}")

# Save clean weights-only file for future use
final_path = f"checkpoints_adapted/{runspecifics}/model_final_weights.pt"
torch.save(model.state_dict(), final_path)
print(f"Final weights saved → {final_path}")

# ---------------------
# Image generation
# ---------------------

@torch.no_grad()
def generate_from_x0(
    model:       nn.Module,
    x0_map:      np.ndarray,   # (P,)
    cond_params: np.ndarray,   # (NUM_CONDS,)
    n_pix:       int,
    device:      torch.device,
    num_samples: int   = 1,
    rtol:        float = 1e-5,
    atol:        float = 1e-5,
    n_steps:     int   = 50,
) -> np.ndarray:
    """
    Integrate the learned vector field from x0_map to produce samples.

    Returns:
        (num_samples, 1, P) array clipped to [0, 1].
    """
    x0_tensor = (
        torch.from_numpy(x0_map)
        .float().to(device)
        .unsqueeze(0).unsqueeze(0)        # (1, 1, P)
        .expand(num_samples, 1, n_pix)
        .clone()
    )
    y0 = x0_tensor.permute(0, 2, 1).contiguous().view(num_samples, -1)

    cond_tensor = (
        torch.from_numpy(cond_params)
        .float().to(device)
        .unsqueeze(0)                     # (1, NUM_CONDS)
        .expand(num_samples, -1)          # (num_samples, NUM_CONDS)
        .clone()
    )

    def ode_func(t, y_flat):
        B = y_flat.shape[0]
        y = y_flat.view(B, n_pix, 1).permute(0, 2, 1).contiguous()
        t_batch = torch.full((B,), float(t), device=device)
        v = model(y, t_batch, cond_tensor)
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
        img         = proj.projmap(hp_map[0], pix_fn)
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

model.eval()

x0_for_gen_cosmo1 = np.load(f"{INI_MAP}/initial_DE_norm_corr_run_cosmo0.npy").astype(np.float32)  # (500, P)
x0_for_gen_cosmo2 = np.load(f"{INI_MAP}/initial_DE_norm_corr_run_cosmo1.npy").astype(np.float32)  # (500, P)
x0_for_gen_cosmo3 = np.load(f"{INI_MAP}/initial_DE_norm_corr_run_cosmo2.npy").astype(np.float32)  # (500, P)

cond_for_gen_full = np.stack([
    np.load(f"{INI_MAP}/labels_{name}_de.npy") for name in cond_names
], axis=-1)   # (N, NUM_CONDS)

cond_for_gen = cond_for_gen_full[[2, 0, 1]]

cosmo_sim_indices = [0, 1, 2]
save_dir    = " "
os.makedirs(save_dir, exist_ok=True)

n_initial_maps = x0_for_gen_cosmo1.shape[0] 

for cosmo_idx, (x0_for_gen, sim_idx) in enumerate(
    zip([x0_for_gen_cosmo1, x0_for_gen_cosmo2, x0_for_gen_cosmo3], cosmo_sim_indices),
    start=1
):
    # Fixed conditioning params for this cosmology — same for all maps
    cond_single = cond_for_gen[sim_idx]   # (NUM_CONDS,)
    cond_label  = "_".join(
        f"{name}{val:.3f}" for name, val in zip(cond_names, cond_single)
    )

    all_generated = []  # collect all maps for this cosmology

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

    # ── Save all maps for this cosmology in a single file ─────────────
    all_generated = np.stack(all_generated, axis=0)   # (n_initial_maps, P)

    save_path = os.path.join(
        save_dir,
        f"generated_cosmo{cosmo_idx}_{cond_label}_shell14_{runspecifics}.npy",
    )
    np.save(save_path, all_generated)
    print(f"Saved {save_path}  shape={all_generated.shape}")

    # ── Log all maps for this cosmology to W&B ─────────────────────
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
