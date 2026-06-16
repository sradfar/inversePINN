"""
train_inverse_1D.py
===================
Runs all 1D inverse PINN experiments and saves everything postprocessing
needs so that postprocessing_1D.py is pure plotting with zero retraining.

Experiment structure
--------------------
A. Full 4x4 grid  : sparsity_sweep x noise_sweep  (n_seeds each)
   - Sparsity figure  <- slice grid[:, noise==0.0]
   - Noise figure     <- all rows of grid overlaid
   - Heatmap          <- full grid (mean over seeds)

B. Obs-type ablation : both / h_only / u_only
   Fixed at (n_obs=ablation_n_obs, noise=ablation_noise), n_seeds runs each.

C. Representative run (forward verification + diagnostics)
   Fixed at (n_obs=50, noise=0, seed=0, obs_type="both").
   Saves full-field predictions, loss history, n-trajectory.

D. n-trajectory across seeds
   Fixed at (n_obs=20, noise=0, obs_type="both"), n_seeds runs.
   Saves per-seed loss histories for plotting.

Saved files
-----------
pred/results_1D.npz     — all sweep statistics + raw n_recs
pred/forward_1D.npz     — representative run (field predictions + hist)
pred/trajectory_1D.npz  — n-trajectories across seeds
"""

import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

from swe_configs import Config1D as cfg
from PINN_SWE_1D import InversePINN_1D
from error_swe   import l2norm_err, n_recovery_error

os.makedirs("pred",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load reference data ───────────────────────────────────────────────────────
print("Loading 1D MacDonald data ...")
d      = np.load("data/macdonald_subcritical.npz")
x      = d["x"]
h_ref  = d["h"]
u_ref  = d["u"]
n_true = float(d["n_true"].flat[0])
N      = len(x)

cp = x[::cfg.cp_step].reshape(-1, 1)

# ── Obs-mask lookup ───────────────────────────────────────────────────────────
OBS_MASKS = {
    "both"   : [1.0, 1.0],
    "h_only" : [1.0, 0.0],
    "u_only" : [0.0, 1.0],
}

# ── Build fresh PINN ──────────────────────────────────────────────────────────
def build_pinn(obs_mask=None):
    inp = layers.Input(shape=(1,))
    hl  = inp
    for _ in range(cfg.n_layer):
        hl = layers.Dense(cfg.n_neural, activation=cfg.act)(hl)
    out = layers.Dense(2)(hl)
    km  = models.Model(inp, out)
    opt = optimizers.Adam(cfg.lr)
    return InversePINN_1D(km, opt, cfg.n_adam,
                          g=cfg.g, n_init=cfg.n_init,
                          lam_obs=cfg.lam_obs, lam_eq=cfg.lam_eq,
                          obs_mask=obs_mask)

# ── Sample sparse observations ────────────────────────────────────────────────
def sample_obs(n_obs, noise_frac, seed, obs_type="both"):
    rng = np.random.default_rng(seed + 1000)
    idx = np.sort(rng.choice(N, size=n_obs, replace=False))
    h_s = h_ref[idx].copy()
    u_s = u_ref[idx].copy()
    if noise_frac > 0:
        h_s += rng.normal(0, noise_frac * np.std(h_ref), n_obs)
        u_s += rng.normal(0, noise_frac * np.std(u_ref), n_obs)
    obs = np.column_stack([x[idx], h_s, u_s])
    # Zero unused column so scaling is stable for partial obs types
    if obs_type == "h_only":
        obs[:, 2] = 0.0
    elif obs_type == "u_only":
        obs[:, 1] = 0.0
    return obs

# ── Single training run ───────────────────────────────────────────────────────
def one_run(n_obs, noise_frac, seed, obs_type="both", return_pred=False):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    obs  = sample_obs(n_obs, noise_frac, seed, obs_type)
    pinn = build_pinn(obs_mask=OBS_MASKS[obs_type])

    t0   = time.time()
    hist = pinn.fit(obs, cp)
    dt   = time.time() - t0

    n_rec = pinn.get_n()
    n_err = n_recovery_error(n_true, n_rec)

    pred  = pinn.predict(x.reshape(-1, 1))
    l2_h  = float(l2norm_err(h_ref, pred[:, 0]))
    l2_u  = float(l2norm_err(u_ref, pred[:, 1]))

    print(f"  seed={seed} n_obs={n_obs} noise={noise_frac:.0%} "
          f"type={obs_type}  n_rec={n_rec:.5f}  "
          f"n_err={n_err:.2f}%  l2_h={l2_h:.3f}%  t={dt:.0f}s")

    result = dict(n_rec=n_rec, n_err=n_err, l2_h=l2_h, l2_u=l2_u,
                  hist=hist, dt=dt)
    if return_pred:
        result["pred"] = pred
    return result

# ── Summarise a list of runs ──────────────────────────────────────────────────
def summarise(runs):
    return dict(
        n_err_mean = np.mean([r["n_err"] for r in runs]),
        n_err_std  = np.std( [r["n_err"] for r in runs]),
        l2_h_mean  = np.mean([r["l2_h"]  for r in runs]),
        l2_h_std   = np.std( [r["l2_h"]  for r in runs]),
        l2_u_mean  = np.mean([r["l2_u"]  for r in runs]),
        l2_u_std   = np.std( [r["l2_u"]  for r in runs]),
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

# grid_results[i][j] = list of runs for (sparsity_sweep[i], noise_sweep[j])
grid_results = [[None]*len(cfg.noise_sweep) for _ in range(len(cfg.sparsity_sweep))]

for i, n_obs in enumerate(cfg.sparsity_sweep):
    for j, noise in enumerate(cfg.noise_sweep):
        print(f"\n  n_obs={n_obs}  noise={noise:.0%}")
        runs = [one_run(n_obs=n_obs, noise_frac=noise, seed=s)
                for s in range(cfg.n_seeds)]
        grid_results[i][j] = runs

# Compute summary statistics for every cell
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
# EXPERIMENT B — Obs-type ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"EXPERIMENT B: Obs-type ablation  "
      f"(n_obs={cfg.ablation_n_obs}, noise={cfg.ablation_noise:.0%}, "
      f"{cfg.n_seeds} seeds)")
print("="*60)

obs_types  = ["both", "h_only", "u_only"]
ob_results = {}
for ot in obs_types:
    print(f"\n  obs_type = {ot}")
    runs = [one_run(n_obs=cfg.ablation_n_obs,
                    noise_frac=cfg.ablation_noise,
                    seed=s, obs_type=ot)
            for s in range(cfg.n_seeds)]
    ob_results[ot] = runs

ob_summary = {k: summarise(v) for k, v in ob_results.items()}

print("\nSUMMARY — Observation type")
print(f"{'type':>10} {'n_err_mean':>12} {'n_err_std':>10}")
for k in obs_types:
    s = ob_summary[k]
    print(f"{k:>10}  {s['n_err_mean']:>10.3f}%  {s['n_err_std']:>8.3f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Representative run (forward verification)
# ═══════════════════════════════════════════════════════════════════════════════
FWD_N_OBS = 50
FWD_SEED  = 0

print("\n" + "="*60)
print(f"EXPERIMENT C: Representative run  (n_obs={FWD_N_OBS}, seed={FWD_SEED})")
print("="*60)

fwd_run = one_run(n_obs=FWD_N_OBS, noise_frac=0.0,
                  seed=FWD_SEED, obs_type="both", return_pred=True)

# Also save the observation indices so postprocessing can mark them on the plot
rng_fwd  = np.random.default_rng(FWD_SEED + 1000)
fwd_idx  = np.sort(rng_fwd.choice(N, size=FWD_N_OBS, replace=False))

print(f"  n_rec = {fwd_run['n_rec']:.5f}  "
      f"n_err = {fwd_run['n_err']:.2f}%  "
      f"l2_h = {fwd_run['l2_h']:.3f}%")

np.savez_compressed(
    "pred/forward_1D.npz",
    x         = x,
    h_ref     = h_ref,
    u_ref     = u_ref,
    h_pred    = fwd_run["pred"][:, 0],
    u_pred    = fwd_run["pred"][:, 1],
    obs_idx   = fwd_idx,
    hist      = fwd_run["hist"],
    n_rec     = np.array([fwd_run["n_rec"]]),
    n_err     = np.array([fwd_run["n_err"]]),
    n_true    = np.array([n_true]),
    n_init    = np.array([cfg.n_init]),
    n_adam    = np.array([cfg.n_adam]),
)
print("Saved: pred/forward_1D.npz")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — n-trajectory across seeds
# ═══════════════════════════════════════════════════════════════════════════════
TRAJ_N_OBS = 20
TRAJ_SEEDS = cfg.n_seeds

print("\n" + "="*60)
print(f"EXPERIMENT D: n-trajectory across {TRAJ_SEEDS} seeds  "
      f"(n_obs={TRAJ_N_OBS}, noise=0%)")
print("="*60)

traj_hists = []
traj_n_recs = []

for seed in range(TRAJ_SEEDS):
    r = one_run(n_obs=TRAJ_N_OBS, noise_frac=0.0, seed=seed + 500,
                obs_type="both")
    traj_hists.append(r["hist"])
    traj_n_recs.append(r["n_rec"])

# Pad histories to same length in case of any length mismatch
max_len = max(h.shape[0] for h in traj_hists)
traj_hists_arr = np.full((TRAJ_SEEDS, max_len, traj_hists[0].shape[1]), np.nan)
for k, h in enumerate(traj_hists):
    traj_hists_arr[k, :h.shape[0], :] = h

np.savez_compressed(
    "pred/trajectory_1D.npz",
    hists   = traj_hists_arr,
    n_recs  = np.array(traj_n_recs),
    n_obs   = np.array([TRAJ_N_OBS]),
    n_true  = np.array([n_true]),
    n_init  = np.array([cfg.n_init]),
    n_adam  = np.array([cfg.n_adam]),
)
print("Saved: pred/trajectory_1D.npz")

# ═══════════════════════════════════════════════════════════════════════════════
# Save main results
# ═══════════════════════════════════════════════════════════════════════════════
np.savez_compressed(
    "pred/results_1D.npz",
    # Grid
    sparsity_sweep  = np.array(cfg.sparsity_sweep),
    noise_sweep     = np.array(cfg.noise_sweep),
    grid_n_err_mean = grid_n_err_mean,
    grid_n_err_std  = grid_n_err_std,
    grid_l2_h_mean  = grid_l2_h_mean,
    grid_l2_h_std   = grid_l2_h_std,
    grid_n_recs     = grid_n_recs,
    # Obs-type ablation
    ob_types        = np.array(obs_types),
    ob_n_err_mean   = np.array([ob_summary[k]["n_err_mean"] for k in obs_types]),
    ob_n_err_std    = np.array([ob_summary[k]["n_err_std"]  for k in obs_types]),
    ob_n_recs       = np.array([ob_summary[k]["n_recs"]     for k in obs_types]),
    # Scalars
    n_true          = np.array([n_true]),
    n_init          = np.array([cfg.n_init]),
    ablation_n_obs  = np.array([cfg.ablation_n_obs]),
    ablation_noise  = np.array([cfg.ablation_noise]),
)
print("Saved: pred/results_1D.npz")
print("\nDone. Run: python postprocessing_1D.py")
