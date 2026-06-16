"""
train_inverse_2D.py
===================
Runs all 2D inverse PINN experiments and saves everything postprocessing
needs so that postprocessing_2D.py is pure plotting with zero retraining.

Experiment structure
--------------------
A. Full 4x4 grid  : sparsity_sweep x noise_sweep  (n_seeds each)
   - Sparsity figure  <- slice grid[:, noise==0.0]
   - Noise figure     <- all rows of grid overlaid
   - Heatmap          <- full grid (mean over seeds)

B. Representative run (forward verification + diagnostics)
   Fixed at (n_obs=50, noise=0, seed=0).
   Saves full-field predictions on all wet points, loss history, n-trajectory.

C. n-trajectory across seeds
   Fixed at (n_obs=20, noise=0), n_seeds runs.
   Saves per-seed loss histories for plotting.

Saved files
-----------
pred/results_2D.npz     — all sweep statistics + raw n_recs
pred/forward_2D.npz     — representative run (field predictions + hist + grid info)
pred/trajectory_2D.npz  — n-trajectories across seeds
"""

import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

from swe_configs import Config2D as cfg
from PINN_SWE_2D import InversePINN_2D
from error_swe   import l2norm_err, n_recovery_error

os.makedirs("pred",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load reference data ───────────────────────────────────────────────────────
print("Loading 2D Thacker data ...")
d      = np.load("data/thacker_2d.npz")
x_g    = d["x"]
y_g    = d["y"]
h_g    = d["h"]
u_g    = d["u"]
v_g    = d["v"]
z_g    = d["z"]
mask   = d["mask"].astype(bool)
n_true = float(d["n_true"].flat[0])

# Bed gradients via central differences on the structured grid
x_1d     = x_g[:, 0]
y_1d     = y_g[0, :]
dzb_dx_g = np.gradient(z_g, x_1d, axis=0)
dzb_dy_g = np.gradient(z_g, y_1d, axis=1)

# Flatten to wet points
x_wet      = x_g[mask]
y_wet      = y_g[mask]
h_wet      = h_g[mask]
u_wet      = u_g[mask]
v_wet      = v_g[mask]
dzb_dx_wet = dzb_dx_g[mask]
dzb_dy_wet = dzb_dy_g[mask]
N_wet      = len(x_wet)

print(f"  Wet points: {N_wet}")
print(f"  Bed slope: dzb/dx in [{dzb_dx_wet.min():+.4f}, {dzb_dx_wet.max():+.4f}]"
      f"  dzb/dy in [{dzb_dy_wet.min():+.4f}, {dzb_dy_wet.max():+.4f}]")

# Fixed collocation points (seeded for reproducibility)
rng_col = np.random.default_rng(0)
col_idx = rng_col.choice(N_wet, size=min(cfg.n_col, N_wet), replace=False)
cp = np.column_stack([
    x_wet[col_idx], y_wet[col_idx],
    dzb_dx_wet[col_idx], dzb_dy_wet[col_idx],
])  # shape (N_col, 4)

xy_all = np.column_stack([x_wet, y_wet])   # for prediction on all wet points

# ── Build fresh PINN ──────────────────────────────────────────────────────────
def build_pinn():
    inp = layers.Input(shape=(2,))
    hl  = inp
    for _ in range(cfg.n_layer):
        hl = layers.Dense(cfg.n_neural, activation=cfg.act)(hl)
    out = layers.Dense(3)(hl)
    km  = models.Model(inp, out)
    opt = optimizers.Adam(cfg.lr)
    return InversePINN_2D(km, opt, cfg.n_adam,
                          g=cfg.g, n_init=cfg.n_init,
                          lam_obs=cfg.lam_obs, lam_eq=cfg.lam_eq)

# ── Sample observations ───────────────────────────────────────────────────────
def sample_obs(n_obs, noise_frac, seed):
    rng = np.random.default_rng(seed + 2000)
    idx = rng.choice(N_wet, size=n_obs, replace=False)
    ho  = h_wet[idx].copy()
    uo  = u_wet[idx].copy()
    vo  = v_wet[idx].copy()
    if noise_frac > 0:
        ho += rng.normal(0, noise_frac * np.std(h_wet), n_obs)
        uo += rng.normal(0, noise_frac * np.std(u_wet), n_obs)
        vo += rng.normal(0, noise_frac * np.std(v_wet), n_obs)
    return np.column_stack([x_wet[idx], y_wet[idx], ho, uo, vo]), idx

# ── Single training run ───────────────────────────────────────────────────────
def one_run(n_obs, noise_frac, seed, return_pred=False):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    obs, obs_idx = sample_obs(n_obs, noise_frac, seed)
    pinn = build_pinn()

    t0   = time.time()
    hist = pinn.fit(obs, cp)
    dt   = time.time() - t0

    n_rec = pinn.get_n()
    n_err = n_recovery_error(n_true, n_rec)

    pred  = pinn.predict(xy_all)
    l2_h  = float(l2norm_err(h_wet, pred[:, 0]))
    l2_u  = float(l2norm_err(u_wet, pred[:, 1]))
    l2_v  = float(l2norm_err(v_wet, pred[:, 2]))

    print(f"  seed={seed} n_obs={n_obs} noise={noise_frac:.0%}  "
          f"n_rec={n_rec:.5f}  n_err={n_err:.2f}%  "
          f"l2_h={l2_h:.3f}%  t={dt:.0f}s")

    result = dict(n_rec=n_rec, n_err=n_err, l2_h=l2_h, l2_u=l2_u,
                  l2_v=l2_v, hist=hist, dt=dt)
    if return_pred:
        result["pred"]    = pred
        result["obs_idx"] = obs_idx
    return result

# ── Summarise a list of runs ──────────────────────────────────────────────────
def summarise(runs):
    return dict(
        n_err_mean = np.mean([r["n_err"] for r in runs]),
        n_err_std  = np.std( [r["n_err"] for r in runs]),
        l2_h_mean  = np.mean([r["l2_h"]  for r in runs]),
        l2_h_std   = np.std( [r["l2_h"]  for r in runs]),
        n_recs     = [r["n_rec"] for r in runs],
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Full 4x4 sparsity × noise grid
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT A: 4x4 sparsity × noise grid")
print(f"  sparsity: {cfg.sparsity_sweep}")
print(f"  noise:    {cfg.noise_sweep}")
print(f"  seeds:    {cfg.n_seeds}")
print("="*60)

grid_results = [[None]*len(cfg.noise_sweep) for _ in range(len(cfg.sparsity_sweep))]

for i, n_obs in enumerate(cfg.sparsity_sweep):
    for j, noise in enumerate(cfg.noise_sweep):
        print(f"\n  n_obs={n_obs}  noise={noise:.0%}")
        runs = [one_run(n_obs=n_obs, noise_frac=noise, seed=s)
                for s in range(cfg.n_seeds)]
        grid_results[i][j] = runs

# Compute summary statistics
grid_n_err_mean = np.zeros((len(cfg.sparsity_sweep), len(cfg.noise_sweep)))
grid_n_err_std  = np.zeros_like(grid_n_err_mean)
grid_l2_h_mean  = np.zeros_like(grid_n_err_mean)
grid_l2_h_std   = np.zeros_like(grid_n_err_mean)
grid_n_recs     = np.zeros((len(cfg.sparsity_sweep), len(cfg.noise_sweep), cfg.n_seeds))

for i in range(len(cfg.sparsity_sweep)):
    for j in range(len(cfg.noise_sweep)):
        s = summarise(grid_results[i][j])
        grid_n_err_mean[i, j] = s["n_err_mean"]
        grid_n_err_std[i, j]  = s["n_err_std"]
        grid_l2_h_mean[i, j]  = s["l2_h_mean"]
        grid_l2_h_std[i, j]   = s["l2_h_std"]
        grid_n_recs[i, j, :]  = s["n_recs"]

# ── Print summary tables ──────────────────────────────────────────────────────
noise_0_col = cfg.noise_sweep.index(0.0)

print("\n\nSUMMARY — Sparsity sweep (noise=0%)")
print(f"{'n_obs':>8} {'n_err_mean':>12} {'n_err_std':>10} {'l2_h_mean':>10}")
for i, n_obs in enumerate(cfg.sparsity_sweep):
    print(f"{n_obs:>8}  {grid_n_err_mean[i, noise_0_col]:>10.3f}%  "
          f"{grid_n_err_std[i, noise_0_col]:>8.3f}%  "
          f"{grid_l2_h_mean[i, noise_0_col]:>8.3f}%")

print("\nSUMMARY — Noise sweep (per n_obs row)")
header = f"{'noise':>8}" + "".join(f"  n_obs={k:>3}" for k in cfg.sparsity_sweep)
print(header)
for j, noise in enumerate(cfg.noise_sweep):
    row = f"{noise:>7.0%}"
    for i in range(len(cfg.sparsity_sweep)):
        row += f"  {grid_n_err_mean[i, j]:>9.3f}%"
    print(row)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Representative run (forward verification)
# ═══════════════════════════════════════════════════════════════════════════════
FWD_N_OBS = 50
FWD_SEED  = 0

print("\n" + "="*60)
print(f"EXPERIMENT B: Representative run  (n_obs={FWD_N_OBS}, seed={FWD_SEED})")
print("="*60)

fwd_run = one_run(n_obs=FWD_N_OBS, noise_frac=0.0,
                  seed=FWD_SEED, return_pred=True)

print(f"  n_rec = {fwd_run['n_rec']:.5f}  "
      f"n_err = {fwd_run['n_err']:.2f}%  "
      f"l2_h = {fwd_run['l2_h']:.3f}%")

np.savez_compressed(
    "pred/forward_2D.npz",
    # Grid geometry (for reconstructing contour plots)
    x_g       = x_g,
    y_g       = y_g,
    z_g       = z_g,
    mask      = mask,
    # Reference wet-point arrays
    x_wet     = x_wet,
    y_wet     = y_wet,
    h_wet     = h_wet,
    u_wet     = u_wet,
    v_wet     = v_wet,
    # Predictions on all wet points
    h_pred    = fwd_run["pred"][:, 0],
    u_pred    = fwd_run["pred"][:, 1],
    v_pred    = fwd_run["pred"][:, 2],
    obs_idx   = fwd_run["obs_idx"],
    # Training diagnostics
    hist      = fwd_run["hist"],
    n_rec     = np.array([fwd_run["n_rec"]]),
    n_err     = np.array([fwd_run["n_err"]]),
    n_true    = np.array([n_true]),
    n_init    = np.array([cfg.n_init]),
    n_adam    = np.array([cfg.n_adam]),
)
print("Saved: pred/forward_2D.npz")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — n-trajectory across seeds
# ═══════════════════════════════════════════════════════════════════════════════
TRAJ_N_OBS = 20
TRAJ_SEEDS = cfg.n_seeds

print("\n" + "="*60)
print(f"EXPERIMENT C: n-trajectory across {TRAJ_SEEDS} seeds  "
      f"(n_obs={TRAJ_N_OBS}, noise=0%)")
print("="*60)

traj_hists  = []
traj_n_recs = []

for seed in range(TRAJ_SEEDS):
    r = one_run(n_obs=TRAJ_N_OBS, noise_frac=0.0, seed=seed + 700)
    traj_hists.append(r["hist"])
    traj_n_recs.append(r["n_rec"])

max_len = max(h.shape[0] for h in traj_hists)
traj_hists_arr = np.full((TRAJ_SEEDS, max_len, traj_hists[0].shape[1]), np.nan)
for k, h in enumerate(traj_hists):
    traj_hists_arr[k, :h.shape[0], :] = h

np.savez_compressed(
    "pred/trajectory_2D.npz",
    hists   = traj_hists_arr,
    n_recs  = np.array(traj_n_recs),
    n_obs   = np.array([TRAJ_N_OBS]),
    n_true  = np.array([n_true]),
    n_init  = np.array([cfg.n_init]),
    n_adam  = np.array([cfg.n_adam]),
)
print("Saved: pred/trajectory_2D.npz")

# ═══════════════════════════════════════════════════════════════════════════════
# Save main results
# ═══════════════════════════════════════════════════════════════════════════════
np.savez_compressed(
    "pred/results_2D.npz",
    sparsity_sweep  = np.array(cfg.sparsity_sweep),
    noise_sweep     = np.array(cfg.noise_sweep),
    grid_n_err_mean = grid_n_err_mean,
    grid_n_err_std  = grid_n_err_std,
    grid_l2_h_mean  = grid_l2_h_mean,
    grid_l2_h_std   = grid_l2_h_std,
    grid_n_recs     = grid_n_recs,
    n_true          = np.array([n_true]),
    n_init          = np.array([cfg.n_init]),
)
print("Saved: pred/results_2D.npz")
print("\nDone. Run: python postprocessing_2D.py")
