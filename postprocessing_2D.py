"""
postprocessing_2D.py
====================
Pure plotting — no training. All data loaded from pred/*.npz.

Figures
-------
1. geometry_2D.png       — Thacker basin: bed elevation, water depth, wet mask
2. forward_2D.png        — contour maps: prediction / reference / error (h, u, v)
3. sparsity_2D.png       — n_err vs n_obs slice at noise=0  (mean ± std + dots)
4. noise_2D.png          — n_err vs noise, 4 curves (one per n_obs)
5. heatmap_2D.png        — n_err heatmap: 4×4 sparsity × noise
6. n_convergence_2D.png  — loss + n convergence from representative run
7. n_trajectory_2D.png   — n_manning vs epoch across seeds
8. comparison_1D_2D.png  — sparsity curves 1D vs 2D (noise=0)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("figs", exist_ok=True)

# ── Load all saved data ───────────────────────────────────────────────────────
res  = np.load("pred/results_2D.npz",    allow_pickle=True)
fwd  = np.load("pred/forward_2D.npz",    allow_pickle=True)
traj = np.load("pred/trajectory_2D.npz", allow_pickle=True)

# Grid sweep
sparsity_sweep  = res["sparsity_sweep"]
noise_sweep     = res["noise_sweep"]
grid_n_err_mean = res["grid_n_err_mean"]
grid_n_err_std  = res["grid_n_err_std"]
grid_l2_h_mean  = res["grid_l2_h_mean"]
grid_l2_h_std   = res["grid_l2_h_std"]
grid_n_recs     = res["grid_n_recs"]

n_true = float(res["n_true"].flat[0])
n_init = float(res["n_init"].flat[0])

# Forward run
x_g     = fwd["x_g"];    y_g  = fwd["y_g"]
z_g     = fwd["z_g"];    mask = fwd["mask"].astype(bool)
x_wet   = fwd["x_wet"];  y_wet = fwd["y_wet"]
h_wet   = fwd["h_wet"];  u_wet = fwd["u_wet"];  v_wet = fwd["v_wet"]
h_pred  = fwd["h_pred"]; u_pred = fwd["u_pred"]; v_pred = fwd["v_pred"]
obs_idx = fwd["obs_idx"]
hist    = fwd["hist"]
n_rec   = float(fwd["n_rec"].flat[0])
n_adam  = int(fwd["n_adam"].flat[0])
n_phase_a = n_adam // 2

# Trajectory
traj_hists  = traj["hists"]
traj_n_recs = traj["n_recs"]
traj_n_obs  = int(traj["n_obs"].flat[0])

noise_0_col = list(noise_sweep).index(0.0)
N_wet       = len(x_wet)

# ── Helper: scatter wet-point array back onto grid ────────────────────────────
def to_grid(arr_wet):
    g = np.full(x_g.shape, np.nan)
    g[mask] = arr_wet
    return g

# ── Figure 1: Thacker geometry overview ──────────────────────────────────────
print("Generating Fig 1: Thacker geometry overview ...")
h_g_plot  = to_grid(h_wet)
ws_g_plot = to_grid(h_wet + z_g[mask])   # water-surface elevation

fig, axs = plt.subplots(1, 3, figsize=(16, 5))

im0 = axs[0].contourf(x_g, y_g, z_g, levels=20, cmap="terrain")
axs[0].contour(x_g, y_g, mask.astype(float), levels=[0.5],
               colors="k", linewidths=1.2)
axs[0].set_title("Bed Elevation z_b [m]\n(black contour: wet/dry boundary)",
                 fontsize=11)
plt.colorbar(im0, ax=axs[0], orientation="horizontal", pad=0.15, shrink=0.9)

im1 = axs[1].contourf(x_g, y_g, h_g_plot, levels=20, cmap="Blues")
axs[1].set_title("Water Depth h [m]  (wet region only)", fontsize=11)
plt.colorbar(im1, ax=axs[1], orientation="horizontal", pad=0.15, shrink=0.9)

im2 = axs[2].contourf(x_g, y_g, ws_g_plot, levels=20, cmap="viridis")
axs[2].set_title("Water-Surface Elevation z_b + h [m]", fontsize=11)
plt.colorbar(im2, ax=axs[2], orientation="horizontal", pad=0.15, shrink=0.9)

for ax in axs:
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

plt.suptitle(f"2D Sloped Channel with Parabolic Cross-Section — Test Case Geometry\n"
             f"N_wet = {N_wet}  |  True Manning n = {n_true}",
             fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig("figs/geometry_2D.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: figs/geometry_2D.png")

# ── Figure 2: Forward verification (contour maps) ────────────────────────────
print("Generating Fig 2: 2D forward verification ...")

h_pred_g = to_grid(h_pred)
u_pred_g = to_grid(u_pred)
v_pred_g = to_grid(v_pred)
h_ref_g  = to_grid(h_wet)
u_ref_g  = to_grid(u_wet)
v_ref_g  = to_grid(v_wet)

fig, axs = plt.subplots(3, 3, figsize=(15, 12))
pairs = [("h [m]",   h_pred_g, h_ref_g),
         ("u [m/s]", u_pred_g, u_ref_g),
         ("v [m/s]", v_pred_g, v_ref_g)]

for col, (label, pg, rg) in enumerate(pairs):
    vmin = np.nanmin(rg); vmax = np.nanmax(rg)
    im0 = axs[0, col].contourf(x_g, y_g, pg, levels=20,
                                vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axs[0, col].set_title(f"Prediction — {label}")
    plt.colorbar(im0, ax=axs[0, col], orientation="horizontal", pad=0.18, shrink=0.9)
    axs[0, col].plot(x_wet[obs_idx], y_wet[obs_idx], "k.", ms=3)

    im1 = axs[1, col].contourf(x_g, y_g, rg, levels=20,
                                vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axs[1, col].set_title(f"Reference — {label}")
    plt.colorbar(im1, ax=axs[1, col], orientation="horizontal", pad=0.18, shrink=0.9)

    err_g = np.abs(rg - pg)
    im2 = axs[2, col].contourf(x_g, y_g, err_g, levels=20, cmap="YlOrRd")
    axs[2, col].set_title(f"|Error| — {label}")
    plt.colorbar(im2, ax=axs[2, col], orientation="horizontal", pad=0.18, shrink=0.9)

for ax_row in axs:
    for ax in ax_row:
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

plt.suptitle(f"2D Inverse PINN — 2D Sloped Channel Forward Verification\n"
             f"Recovered n = {n_rec:.5f}  |  True n = {n_true}",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("figs/forward_2D.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figs/forward_2D.png")

# ── Figure 3: Sparsity sweep (noise=0 slice) ──────────────────────────────────
print("Generating Fig 3: sparsity sweep ...")
sp_n_err_mean = grid_n_err_mean[:, noise_0_col]
sp_n_err_std  = grid_n_err_std[:,  noise_0_col]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(sparsity_sweep, sp_n_err_mean, yerr=sp_n_err_std,
            fmt="o-", color="#1f77b4", capsize=5, lw=1.8, ms=7,
            label="2D Sloped Channel (mean ± std)")
for i, n_obs in enumerate(sparsity_sweep):
    dots = np.abs(grid_n_recs[i, noise_0_col, :] - n_true) / n_true * 100
    ax.scatter([n_obs] * len(dots), dots,
               color="grey", alpha=0.4, s=20, zorder=5)
ax.set_xlabel("Number of observations", fontsize=12)
ax.set_ylabel("Relative error in recovered n [%]", fontsize=12)
ax.set_title("2D Inverse PINN — Manning n Recovery vs Observation Count",
             fontsize=12)
ax.set_xscale("log"); ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("figs/sparsity_2D.png", dpi=200)
plt.close()
print("  Saved: figs/sparsity_2D.png")

# ── Figure 4: Noise sweep (4 curves, one per n_obs) ───────────────────────────
print("Generating Fig 4: noise sweep ...")
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sparsity_sweep)))
noise_pct = noise_sweep * 100

for i, (n_obs, c) in enumerate(zip(sparsity_sweep, colors)):
    ax.errorbar(noise_pct, grid_n_err_mean[i, :], yerr=grid_n_err_std[i, :],
                fmt="o-", color=c, capsize=4, lw=1.8, ms=6,
                label=f"n_obs={n_obs}")

ax.set_xlabel("Observation noise level [%]", fontsize=12)
ax.set_ylabel("Relative error in recovered n [%]", fontsize=12)
ax.set_title(f"2D Inverse PINN — Manning n Recovery vs Noise Level\n"
             f"(2D Sloped Channel)", fontsize=12)
ax.legend(fontsize=10, title="Observation count")
ax.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("figs/noise_2D.png", dpi=200)
plt.close()
print("  Saved: figs/noise_2D.png")

# ── Figure 5: Heatmap ─────────────────────────────────────────────────────────
print("Generating Fig 5: heatmap ...")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

bounds = [0, 6, 25, 45]
cmap_3 = mcolors.ListedColormap(['#1D9E75', '#EF9F27', '#E24B4A'])
norm_3 = mcolors.BoundaryNorm(bounds, cmap_3.N)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.imshow(grid_n_err_mean, aspect='auto', origin='lower',
          cmap=cmap_3, norm=norm_3)

# Cell gap effect via white grid lines
ax.set_xticks(np.arange(-0.5, len(noise_sweep), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(sparsity_sweep), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=2.5)
ax.tick_params(which='minor', bottom=False, left=False)

# Axis labels and ticks
ax.set_xticks(range(len(noise_sweep)))
ax.set_xticklabels([f"{n*100:.0f}%" for n in noise_sweep], fontsize=12)
ax.set_yticks(range(len(sparsity_sweep)))
ax.set_yticklabels([str(n) for n in sparsity_sweep], fontsize=12)
ax.set_xlabel("Observation noise level", fontsize=13)
ax.set_ylabel("Number of observations", fontsize=13)

# Cell annotations
for i in range(len(sparsity_sweep)):
    for j in range(len(noise_sweep)):
        val = grid_n_err_mean[i, j]
        txt_color = '#E1F5EE' if val < 6 else ('#412402' if val < 25 else '#FCEBEB')
        ax.text(j, i, f"{val:.1f}%",
                ha='center', va='center',
                fontsize=13, fontweight='bold', color=txt_color)

# Categorical legend
patches = [
    mpatches.Patch(color='#1D9E75', label='Robust  (<6%)'),
    mpatches.Patch(color='#EF9F27', label='Marginal  (6–25%)'),
    mpatches.Patch(color='#E24B4A', label='Non-identifiable  (>25%)'),
]
ax.legend(handles=patches, loc='upper center',
          bbox_to_anchor=(0.5, -0.14), ncol=3,
          fontsize=10.5, frameon=False)

plt.tight_layout()
plt.savefig("figs/heatmap_2D.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: figs/heatmap_2D.png")

# ── Figure 6: Loss + n convergence from representative run ───────────────────
print("Generating Fig 6: n convergence ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].semilogy(hist[:, 1], label="Obs loss", lw=1.5, color="#1f77b4")
axes[0].semilogy(hist[:, 2], label="Eq loss",  lw=1.5, color="#d62728", ls="--")
axes[0].axvline(n_phase_a, color="grey", ls=":", lw=1.2, alpha=0.7)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title(f"Training Loss (2D, n_obs={len(obs_idx)})")
axes[0].legend(fontsize=10); axes[0].grid(True, which="both", ls=":", alpha=0.5)

axes[1].plot(hist[:, 3], color="purple", lw=1.5)
axes[1].axhline(n_true, color="red",  ls="--", lw=1.5, label=f"True n={n_true}")
axes[1].axhline(n_init, color="grey", ls=":",  lw=1.2, label=f"n_init={n_init}")
axes[1].axvline(n_phase_a, color="grey", ls=":", lw=1.2, alpha=0.7)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("n Manning")
axes[1].set_title(f"n Convergence → {n_rec:.5f}")
axes[1].legend(fontsize=10); axes[1].grid(True, ls=":", alpha=0.5)

plt.suptitle("2D Inverse PINN — Training Diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig("figs/n_convergence_2D.png", dpi=200)
plt.close()
print("  Saved: figs/n_convergence_2D.png")

# ── Figure 7: n-trajectory across seeds ───────────────────────────────────────
print("Generating Fig 7: n trajectories ...")
n_seeds_traj = traj_hists.shape[0]
fig, ax = plt.subplots(figsize=(8, 5))
cmap_t = plt.cm.viridis(np.linspace(0.15, 0.85, n_seeds_traj))

for seed, (h_t, c) in enumerate(zip(traj_hists, cmap_t)):
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
ax.set_title(f"2D Inverse PINN — n Convergence Across Seeds\n"
             f"(n_obs={traj_n_obs}, noise=0%, {n_seeds_traj} seeds)", fontsize=12)
ax.legend(fontsize=9, loc="best")
ax.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("figs/n_trajectory_2D.png", dpi=200)
plt.close()
print("  Saved: figs/n_trajectory_2D.png")

# ── Figure 8: 1D vs 2D comparison ─────────────────────────────────────────────
print("Generating Fig 8: 1D vs 2D comparison ...")
try:
    res1 = np.load("pred/results_1D.npz", allow_pickle=True)
    sp1_n_err_mean = res1["grid_n_err_mean"][:, list(res1["noise_sweep"]).index(0.0)]
    sp1_n_err_std  = res1["grid_n_err_std"][:,  list(res1["noise_sweep"]).index(0.0)]
    sp1_n_obs      = res1["sparsity_sweep"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sp1_n_obs, sp1_n_err_mean, yerr=sp1_n_err_std,
                fmt="o-",  color="#1f77b4", capsize=4, lw=1.8, ms=7,
                label="1D MacDonald")
    ax.errorbar(sparsity_sweep, sp_n_err_mean, yerr=sp_n_err_std,
                fmt="s--", color="#d62728", capsize=4, lw=1.8, ms=7,
                label="2D Sloped Channel")
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
    print("  Skipped comparison_1D_2D.png (pred/results_1D.npz not found yet)")

print("\nAll 2D figures saved to figs/")
