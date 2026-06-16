"""
test_e2e_pinn_recovery.py
=========================
End-to-end smoke test: train the patched PINN_SWE_2D on the newly generated
data and verify that n_manning moves meaningfully from n_init = 0.035 toward
n_true = 0.020. Quick run (small Adam budget) — not the full paper sweep.
"""

import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import sys
sys.path.insert(0, ".")
from PINN_SWE_2D import InversePINN_2D

np.random.seed(0)
tf.random.set_seed(0)

# ── Load new data ────────────────────────────────────────────────────────────
d = np.load("data/thacker_2d.npz")
x_g  = d["x"]; y_g = d["y"]
h_g  = d["h"]; u_g = d["u"]; v_g = d["v"]
z_g  = d["z"]
mask = d["mask"].astype(bool)
n_true = float(d["n_true"].flat[0])

x_1d = x_g[:, 0]; y_1d = y_g[0, :]
dzb_dx_g = np.gradient(z_g, x_1d, axis=0)
dzb_dy_g = np.gradient(z_g, y_1d, axis=1)

x_wet = x_g[mask]; y_wet = y_g[mask]
h_wet = h_g[mask]; u_wet = u_g[mask]; v_wet = v_g[mask]
dzb_dx_wet = dzb_dx_g[mask]; dzb_dy_wet = dzb_dy_g[mask]
N_wet = len(x_wet)
print(f"N_wet = {N_wet},  n_true = {n_true}")

# Pick obs and cp
rng = np.random.default_rng(42)
n_obs = 50
obs_idx = rng.choice(N_wet, n_obs, replace=False)
obs = np.column_stack([
    x_wet[obs_idx], y_wet[obs_idx],
    h_wet[obs_idx], u_wet[obs_idx], v_wet[obs_idx],
])

n_col = 1500
col_idx = rng.choice(N_wet, min(n_col, N_wet), replace=False)
cp = np.column_stack([
    x_wet[col_idx], y_wet[col_idx],
    dzb_dx_wet[col_idx], dzb_dy_wet[col_idx],
])

# Build model
inp = layers.Input(shape=(2,))
h = inp
for _ in range(5):
    h = layers.Dense(40, activation='tanh')(h)
out = layers.Dense(3)(h)
model = models.Model(inp, out)
opt = optimizers.Adam(1e-3)

n_init = 0.035
pinn = InversePINN_2D(model, opt, epochs=4000,
                      g=9.81, n_init=n_init,
                      lam_obs=1.0, lam_eq=1.0)

t0 = time.time()
pinn.fit(obs, cp)
dt = time.time() - t0

n_rec = pinn.get_n()
n_err = abs(n_rec - n_true) / n_true * 100
print()
print(f"=== RESULT ===")
print(f"  n_true  = {n_true:.5f}")
print(f"  n_init  = {n_init:.5f}")
print(f"  n_rec   = {n_rec:.5f}")
print(f"  n_err   = {n_err:.2f}%   (was ~71.6% in the broken pipeline)")
print(f"  runtime = {dt:.1f}s")
print()
moved = abs(n_rec - n_init) > 0.1 * abs(n_true - n_init)
print(f"n_raw MOVED meaningfully from n_init: {'YES ✓' if moved else 'NO ✗'}")
print(f"n_rec WITHIN 25% of n_true:           {'YES ✓' if n_err < 25 else 'NO ✗'}")
