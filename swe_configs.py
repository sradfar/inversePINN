"""
swe_configs.py
==============
Configuration for 1D and 2D inverse PINN experiments.

Design
------
- A single 4x4 sparsity x noise grid is the only sweep.
  Sparsity curve  = slice at noise = 0.0
  Noise curves    = all n_obs rows overlaid on one figure
  Heatmap         = full 4x4 grid
- n_obs_default is removed; the grid covers everything.
- Obs-type ablation (1D only) runs at n_obs=20, noise=0.0
  (hardcoded in train_inverse_1D.py, not a config param).
"""


class Config1D:
    # ── Network architecture ──────────────────────────────────────────────────
    act      = "tanh"
    n_neural = 20
    n_layer  = 8

    # ── Training ──────────────────────────────────────────────────────────────
    n_adam   = 1000          # total epochs (Phase A = n_adam//2, Phase B = n_adam//2)
    lr       = 1e-3

    # ── Physics ───────────────────────────────────────────────────────────────
    g        = 9.81
    n_init   = 0.04            # initial guess for Manning n

    # ── Loss weights ─────────────────────────────────────────────────────────
    lam_obs  = 1.0
    lam_eq   = 1.0

    # ── Collocation points ────────────────────────────────────────────────────
    cp_step  = 5               # every cp_step-th point on x grid

    # ── Sweep grid ────────────────────────────────────────────────────────────
    sparsity_sweep = [5, 10, 20, 50]    # n_obs values
    noise_sweep    = [0.0, 0.05, 0.10, 0.20]

    # ── Experiment replication ────────────────────────────────────────────────
    n_seeds  = 5

    # ── Obs-type ablation (fixed point on the grid) ───────────────────────────
    ablation_n_obs   = 20
    ablation_noise   = 0.0


class Config2D:
    # ── Network architecture ──────────────────────────────────────────────────
    act      = "tanh"
    n_neural = 20
    n_layer  = 8

    # ── Training ──────────────────────────────────────────────────────────────
    n_adam   = 3000
    lr       = 1e-3

    # ── Physics ───────────────────────────────────────────────────────────────
    g        = 9.81
    n_init   = 0.04

    # ── Loss weights ─────────────────────────────────────────────────────────
    lam_obs  = 1.0
    lam_eq   = 1.0

    # ── Collocation points ────────────────────────────────────────────────────
    n_col    = 500            # number of wet-point collocation samples

    # ── Sweep grid ────────────────────────────────────────────────────────────
    sparsity_sweep = [5, 10, 20, 50]
    noise_sweep    = [0.0, 0.05, 0.10, 0.20]

    # ── Experiment replication ────────────────────────────────────────────────
    n_seeds  = 5
