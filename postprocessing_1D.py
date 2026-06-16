"""
postprocessing_1D.py
====================
Pure plotting — no training. All data loaded from pred/*.npz.

Figures
-------
1. forward_1D.png        — field predictions + loss + n-convergence (rep. run)
2. sparsity_1D.png       — n_err vs n_obs slice at noise=0  (mean ± std + dots)
3. noise_1D.png          — n_err vs noise, 4 curves (one per n_obs)
4. heatmap_1D.png        — n_err heatmap: 4×4 sparsity × noise
5. obs_type_1D.png       — bar chart: both / h-only / u-only
6. n_trajectory_1D.png   — n_manning vs epoch across seeds
7. comparison_1D_2D.png  — sparsity curves 1D vs 2D (noise=0)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs("figs", exist_ok=True)

# ── Load all saved data ───────────────────────────────────────────────────────
res  = np.load("pred/results_1D.npz",    allow_pickle=True)
fwd  = np.load("pred/forward_1D.npz",    allow_pickle=True)
traj = np.load("pred/trajectory_1D.npz", allow_pickle=True)

# Grid sweep
sparsity_sweep  = res["sparsity_sweep"]
noise_sweep     = res["noise_sweep"]
grid_n_err_mean = res["grid_n_err_mean"]   # (n_sp, n_ns)
grid_n_err_std  = res["grid_n_err_std"]
grid_l2_h_mean  = res["grid_l2_h_mean"]
grid_l2_h_std   = res["grid_l2_h_std"]
grid_n_recs     = res["grid_n_recs"]       # (n_sp, n_ns, n_seeds)

# Obs-type ablation
ob_types      = list(res["ob_types"])
ob_n_err_mean = res["ob_n_err_mean"]
ob_n_err_std  = res["ob_n_err_std"]
ob_n_recs     = res["ob_n_recs"]

# Scalars
n_true         = float(res["n_true"].flat[0])
n_init         = float(res["n_init"].flat[0])
ablation_n_obs = int(res["ablation_n_obs"].flat[0])
ablation_noise = float(res["ablation_noise"].flat[0])

# Forward run
x       = fwd["x"]
h_ref   = fwd["h_ref"]
u_ref   = fwd["u_ref"]
h_pred  = fwd["h_pred"]
u_pred  = fwd["u_pred"]
obs_idx = fwd["obs_idx"]
hist    = fwd["hist"]
n_rec   = float(fwd["n_rec"].flat[0])
n_adam  = int(fwd["n_adam"].flat[0])
n_phase_a = n_adam // 2

# Trajectory
traj_hists  = traj["hists"]    # (n_seeds, epochs, hist_cols)
traj_n_recs = traj["n_recs"]
traj_n_obs  = int(traj["n_obs"].flat[0])

noise_0_col = list(noise_sweep).index(0.0)

# ── Figure 1: Forward verification ───────────────────────────────────────────
print("Generating Fig 1: forward verification ...")
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

for col, (name, p, r) in enumerate([("h [m]",   h_pred, h_ref),
                                     ("u [m/s]", u_pred, u_ref)]):
    axs[0, col].plot(x, p, color="#1f77b4", lw=1.5, label="PINN")
    axs[0, col].plot(x, r, color="#d62728", lw=1.5, ls="--", label="Analytical")
    axs[0, col].plot(x[obs_idx], r[obs_idx], "ko", ms=4, label="Observations")
    axs[0, col].set_ylabel(name); axs[0, col].legend(fontsize=9)
    axs[0, col].set_title(f"Prediction vs Reference — {name.split()[0]}")
    axs[0, col].grid(True, ls=":", alpha=0.5)

    axs[1, col].plot(x, np.abs(r - p), color="#2ca02c", lw=1.5)
    axs[1, col].set_ylabel(f"|error| {name}")
    axs[1, col].set_title("Absolute Error")
    axs[1, col].grid(True, ls=":", alpha=0.5)
    axs[1, col].yaxis.set_major_formatter(
        ticker.ScalarFormatter(useMathText=True))
    axs[1, col].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

axs[2, 0].semilogy(hist[:, 1], label="Obs loss",  lw=1.5, color="#1f77b4")
axs[2, 0].semilogy(hist[:, 2], label="Eq loss",   lw=1.5, color="#d62728", ls="--")
axs[2, 0].axvline(n_phase_a, color="grey", ls=":", lw=1.2, alpha=0.7)
axs[2, 0].set_xlabel("Epoch"); axs[2, 0].set_ylabel("Loss")
axs[2, 0].set_title("Training Loss"); axs[2, 0].legend(fontsize=9)
axs[2, 0].grid(True, which="both", ls=":", alpha=0.5)

axs[2, 1].plot(hist[:, 3], color="purple", lw=1.5)
axs[2, 1].axhline(n_true,  color="red",  ls="--", lw=1.5, label=f"True n={n_true}")
axs[2, 1].axhline(n_init,  color="grey", ls=":",  lw=1.2, label=f"n_init={n_init}")
axs[2, 1].axvline(n_phase_a, color="grey", ls=":", lw=1.2, alpha=0.7)
axs[2, 1].set_xlabel("Epoch"); axs[2, 1].set_ylabel("n Manning")
axs[2, 1].set_title(f"n recovery: {n_rec:.5f} (true: {n_true})")
axs[2, 1].legend(fontsize=9); axs[2, 1].grid(True, ls=":", alpha=0.5)

plt.suptitle(f"1D Inverse PINN — Forward Verification (n_obs={len(obs_idx)}, seed=0)\n"
             f"Recovered n = {n_rec:.5f}  |  True n = {n_true}",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figs/forward_1D.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: figs/forward_1D.png")

# ── Figure 2: Sparsity sweep (noise=0 slice) ──────────────────────────────────
print("Generating Fig 2: sparsity sweep ...")
sp_n_err_mean = grid_n_err_mean[:, noise_0_col]
sp_n_err_std  = grid_n_err_std[:,  noise_0_col]
sp_l2_h_mean  = grid_l2_h_mean[:, noise_0_col]
sp_l2_h_std   = grid_l2_h_std[:,  noise_0_col]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.errorbar(sparsity_sweep, sp_n_err_mean, yerr=sp_n_err_std,
            fmt="o-", color="#1f77b4", capsize=5, lw=1.8, ms=7,
            label="n error (mean ± std)")
for i, n_obs in enumerate(sparsity_sweep):
    dots = np.abs(grid_n_recs[i, noise_0_col, :] - n_true) / n_true * 100
    ax.scatter([n_obs] * len(dots), dots,
               color="grey", alpha=0.4, s=20, zorder=5)
ax.set_xlabel("Number of observations", fontsize=12)
ax.set_ylabel("Relative error in recovered n [%]", fontsize=12)
ax.set_title("Manning n Recovery vs Observation Count", fontsize=13)
ax.set_xscale("log"); ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=10)

ax2 = axes[1]
ax2.errorbar(sparsity_sweep, sp_l2_h_mean, yerr=sp_l2_h_std,
             fmt="s-", color="#d62728", capsize=5, lw=1.8, ms=7,
             label="L2 h error (mean ± std)")
ax2.set_xlabel("Number of observations", fontsize=12)
ax2.set_ylabel("L2-norm error in h [%]", fontsize=12)
ax2.set_title("h Field Accuracy vs Observation Count", fontsize=13)
ax2.set_xscale("log"); ax2.grid(True, which="both", ls=":", alpha=0.5)
ax2.legend(fontsize=10)

plt.suptitle("1D Inverse PINN — Sparsity Sweep  (noise=0%)", fontsize=14)
plt.tight_layout()
plt.savefig("figs/sparsity_1D.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: figs/sparsity_1D.png")

# ── Figure 3: Noise sweep (4 curves, one per n_obs) ───────────────────────────
print("Generating Fig 3: noise sweep ...")
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sparsity_sweep)))
noise_pct = noise_sweep * 100

for i, (n_obs, c) in enumerate(zip(sparsity_sweep, colors)):
    ax.errorbar(noise_pct, grid_n_err_mean[i, :], yerr=grid_n_err_std[i, :],
                fmt="o-", color=c, capsize=4, lw=1.8, ms=6,
                label=f"n_obs={n_obs}")

ax.set_xlabel("Observation noise level [%]", fontsize=12)
ax.set_ylabel("Relative error in recovered n [%]", fontsize=12)
ax.set_title("1D Inverse PINN — Manning n Recovery vs Noise Level", fontsize=12)
ax.legend(fontsize=10, title="Observation count")
ax.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("figs/noise_1D.png", dpi=200)
plt.close()
print("  Saved: figs/noise_1D.png")

# ── Figure 4: Heatmap ─────────────────────────────────────────────────────────
print("Generating Fig 4: heatmap ...")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid_n_err_mean, aspect="auto", origin="lower",
               cmap="RdYlGn_r", vmin=0)
ax.set_xticks(range(len(noise_sweep)))
ax.set_xticklabels([f"{n*100:.0f}%" for n in noise_sweep], fontsize=11)
ax.set_yticks(range(len(sparsity_sweep)))
ax.set_yticklabels([str(n) for n in sparsity_sweep], fontsize=11)
ax.set_xlabel("Observation noise level", fontsize=12)
ax.set_ylabel("Number of observations", fontsize=12)
ax.set_title("1D Inverse PINN — Manning n Recovery Error [%]\n"
             "Sparsity × Noise Heatmap", fontsize=13)
thresh = grid_n_err_mean.max() * 0.6
for i in range(len(sparsity_sweep)):
    for j in range(len(noise_sweep)):
        ax.text(j, i, f"{grid_n_err_mean[i, j]:.1f}",
                ha="center", va="center", fontsize=10,
                color="black" if grid_n_err_mean[i, j] < thresh else "white")
plt.colorbar(im, label="Relative n error [%]")
plt.tight_layout()
plt.savefig("figs/heatmap_1D.png", dpi=200)
plt.close()
print("  Saved: figs/heatmap_1D.png")

# ── Figure 5: Observation-type ablation ───────────────────────────────────────
print("Generating Fig 5: observation-type ablation ...")
fig, ax = plt.subplots(figsize=(6, 5))
colors_ob = ["#1f77b4", "#ff7f0e", "#2ca02c"]
x_pos     = np.arange(len(ob_types))
bars      = ax.bar(x_pos, ob_n_err_mean, yerr=ob_n_err_std,
                   color=colors_ob, capsize=6, edgecolor="k",
                   lw=0.8, alpha=0.85, error_kw={"lw": 2})
ax.set_xticks(x_pos)
ax.set_xticklabels(["Both h & u", "h only", "u only"], fontsize=12)
ax.set_ylabel("Relative error in recovered n [%]", fontsize=12)
ax.set_title(f"1D Inverse PINN — Observation Type Ablation\n"
             f"(n_obs={ablation_n_obs}, noise={ablation_noise:.0%})", fontsize=12)
for bar, mean, std in zip(bars, ob_n_err_mean, ob_n_err_std):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.05,
            f"{mean:.2f}%", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(ob_n_err_mean + ob_n_err_std) * 1.4)
ax.grid(True, axis="y", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("figs/obs_type_1D.png", dpi=200)
plt.close()
print("  Saved: figs/obs_type_1D.png")

# ── Figure 6: n-trajectory across seeds ───────────────────────────────────────
print("Generating Fig 6: n trajectories ...")
n_seeds_traj = traj_hists.shape[0]
fig, ax = plt.subplots(figsize=(8, 5))
cmap_t = plt.cm.viridis(np.linspace(0.15, 0.85, n_seeds_traj))

for seed, (h_t, c) in enumerate(zip(traj_hists, cmap_t)):
    # Drop NaN-padded tail
    valid = ~np.isnan(h_t[:, 3])
    ax.plot(np.where(valid)[0], h_t[valid, 3], lw=1.5, color=c,
            label=f"seed {seed} → n={traj_n_recs[seed]:.5f}")

ax.axhline(n_true, color="red",  ls="--", lw=1.5, label=f"True n={n_true}")
ax.axhline(n_init, color="grey", ls=":",  lw=1.2, label=f"n_init={n_init}")
ax.axvline(n_phase_a, color="grey", ls=":", lw=1.3, alpha=0.8)
ylim = ax.get_ylim()
ax.text(n_phase_a, ylim[1] * 0.95, "  Phase A → B",
        fontsize=10, color="grey", va="top")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Manning n", fontsize=12)
ax.set_title(f"1D Inverse PINN — n Convergence Across Seeds\n"
             f"(n_obs={traj_n_obs}, noise=0%, {n_seeds_traj} seeds)", fontsize=12)
ax.legend(fontsize=9, loc="best")
ax.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("figs/n_trajectory_1D.png", dpi=200)
plt.close()
print("  Saved: figs/n_trajectory_1D.png")

# ── Figure 7: 1D vs 2D comparison ─────────────────────────────────────────────
print("Generating Fig 7: 1D vs 2D comparison ...")
try:
    res2 = np.load("pred/results_2D.npz", allow_pickle=True)
    sp2_n_err_mean = res2["grid_n_err_mean"][:, list(res2["noise_sweep"]).index(0.0)]
    sp2_n_err_std  = res2["grid_n_err_std"][:,  list(res2["noise_sweep"]).index(0.0)]
    sp2_n_obs      = res2["sparsity_sweep"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sparsity_sweep, sp_n_err_mean, yerr=sp_n_err_std,
                fmt="o-",  color="#1f77b4", capsize=4, lw=1.8, ms=7,
                label="1D MacDonald")
    ax.errorbar(sp2_n_obs, sp2_n_err_mean, yerr=sp2_n_err_std,
                fmt="s--", color="#d62728", capsize=4, lw=1.8, ms=7,
                label="2D Thacker")
    ax.set_xlabel("Number of observations", fontsize=12)
    ax.set_ylabel("Relative error in recovered n [%]", fontsize=12)
    ax.set_title("Inverse PINN — Manning n Recovery: 1D vs 2D\n"
                 "(noise=0%)", fontsize=12)
    ax.set_xscale("log"); ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("figs/comparison_1D_2D.png", dpi=200)
    plt.close()
    print("  Saved: figs/comparison_1D_2D.png")
except FileNotFoundError:
    print("  Skipped comparison_1D_2D.png (pred/results_2D.npz not found yet)")

print("\nAll 1D figures saved to figs/")
