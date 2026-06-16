"""
test_normal_depth.py
====================
Verification 2: friction-balanced uniform flow.

A long sloped channel with Manning friction reaches "normal depth" h_n where
the bed slope exactly balances the friction slope:
    S_0 = S_f = n²·U²/h^(4/3),  with  U = q/h
    →  h_n = (q·n / sqrt(S_0))^(3/5)

Setup: rectangular channel, sloped bed z_b = -S_0·x, inflow q_x = q on west,
free outflow on east, walls N/S. Run to steady state, check h matches h_n
in the channel interior (away from boundary transients).

Pass criterion: relative error on mean interior depth < 2%.
"""

import numpy as np
import sys
sys.path.insert(0, ".")
from swe_solver_2d import SWE2D

# ── Setup ────────────────────────────────────────────────────────────────────
g     = 9.81
n     = 0.02
S_0   = 0.001          # bed slope (downhill in +x)
q_in  = 0.5            # m²/s, unit discharge

# Normal depth analytical
h_n = (q_in * n / np.sqrt(S_0)) ** (3.0/5.0)
U_n = q_in / h_n
Fr  = U_n / np.sqrt(g * h_n)
print(f"Normal-depth verification")
print(f"  q_in = {q_in:.3f} m²/s,  n = {n:.3f},  S_0 = {S_0:.4f}")
print(f"  h_n  = {h_n:.4f} m,   U_n = {U_n:.4f} m/s,   Fr = {Fr:.3f}")

# ── Grid ─────────────────────────────────────────────────────────────────────
Lx, Ly = 2000.0, 100.0
Nx, Ny = 201, 11
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
z_b = -S_0 * X     # downhill in +x

# Initial: uniform h_n, U_n in x (start near the expected answer)
h0  = np.full((Nx, Ny), h_n)
hu0 = np.full((Nx, Ny), q_in)
hv0 = np.zeros((Nx, Ny))
U0 = np.stack([h0, hu0, hv0], axis=0)

# ── BCs ──────────────────────────────────────────────────────────────────────
def bc(h_p, hu_p, hv_p, z_p):
    # West inflow: fix h, hu, hv at ghost (Dirichlet)
    h_p[0, :]  = h_n
    hu_p[0, :] = q_in
    hv_p[0, :] = 0.0
    z_p[0, :]  = -S_0 * (x[0] - (x[1]-x[0]))   # extend bed
    # East: zero-gradient outflow
    h_p[-1, :]  = h_p[-2, :]
    hu_p[-1, :] = hu_p[-2, :]
    hv_p[-1, :] = hv_p[-2, :]
    z_p[-1, :]  = z_p[-2, :]
    # N/S walls: free-slip on u, no normal v
    h_p[:, 0]  = h_p[:, 1]
    hu_p[:, 0] =  hu_p[:, 1]
    hv_p[:, 0] = -hv_p[:, 1]
    z_p[:, 0]  = z_p[:, 1]
    h_p[:, -1]  = h_p[:, -2]
    hu_p[:, -1] =  hu_p[:, -2]
    hv_p[:, -1] = -hv_p[:, -2]
    z_p[:, -1]  = z_p[:, -2]

solver = SWE2D(x, y, z_b, n_manning=n, g=g, h_min=1e-3, cfl=0.4)
U, info = solver.run(U0, bc, max_steps=20000, tol=1e-7, report_every=2000)

# ── Check interior region (away from inflow/outflow transients) ──────────────
i_int = slice(Nx//4, 3*Nx//4)   # middle half of channel
h_mid = U[0, i_int, Ny//2]
u_mid = U[1, i_int, Ny//2] / np.maximum(U[0, i_int, Ny//2], 1e-6)

h_mean = float(np.mean(h_mid))
h_std  = float(np.std(h_mid))
err_h  = abs(h_mean - h_n) / h_n * 100

print()
print(f"Steady state reached at step {info['steps']}, t = {info['t_final']:.1f} s")
print(f"Interior depth (mid-channel, middle half of domain):")
print(f"  mean h     = {h_mean:.4f} m   (analytical h_n = {h_n:.4f})")
print(f"  std  h     = {h_std:.4e} m")
print(f"  relative error vs h_n = {err_h:.3f}%")

ok = err_h < 2.0
print()
print(f"NORMAL-DEPTH: {'PASS ✓' if ok else 'FAIL ✗'}  (target < 2.0%)")
