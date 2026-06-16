"""
test_lake_at_rest.py
====================
Verification 1: well-balancedness.

Setup: parabolic bowl (Thacker geometry, h0=10, R=1000). Initial condition is
a flat free surface (eta=0, u=v=0). With a well-balanced scheme, no spurious
currents should develop — h, u, v stay at their initial values to machine
precision regardless of how many steps we take.

Pass criterion: after 5000 steps,
    max|u|, max|v|   < 1e-10 m/s
    max|h - h_init|  < 1e-10 m
"""

import numpy as np
import sys
sys.path.insert(0, ".")
from swe_solver_2d import SWE2D

# ── Geometry (Thacker basin interior) ─────────────────────────────────────────
R, h0 = 1000.0, 10.0
L = 500.0                      # half-width of computational box (stays wet)
N = 51                          # smaller grid → faster verification
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y, indexing='ij')
z_b = h0 * (X**2 + Y**2) / R**2 - h0   # parabolic bowl, z_b(0) = -h0

# ── Initial condition: flat free surface eta = 0 ─────────────────────────────
eta0 = 0.0
h0_init = eta0 - z_b
hu0 = np.zeros_like(h0_init)
hv0 = np.zeros_like(h0_init)
U0 = np.stack([h0_init, hu0, hv0], axis=0)

print(f"Lake-at-rest test")
print(f"  Grid:       {N} x {N}")
print(f"  Initial h range: [{h0_init.min():.4f}, {h0_init.max():.4f}] m")
print(f"  Bed slope max:   {np.max(np.gradient(z_b, x[1]-x[0], axis=0)):.4f}")

# ── BC: closed walls (zero normal velocity, free-slip tangential) ────────────
def bc_walls(h_p, hu_p, hv_p, z_p):
    # Reflective at all four boundaries: copy interior, reverse normal momentum
    # Padding scheme: index 0 and -1 are ghost cells.
    # West wall (x = -L): ghost at i=0, mirror i=1
    h_p[0, :]  = h_p[1, :]
    hu_p[0, :] = -hu_p[1, :]
    hv_p[0, :] =  hv_p[1, :]
    z_p[0, :]  = z_p[1, :]
    # East
    h_p[-1, :]  = h_p[-2, :]
    hu_p[-1, :] = -hu_p[-2, :]
    hv_p[-1, :] =  hv_p[-2, :]
    z_p[-1, :]  = z_p[-2, :]
    # South
    h_p[:, 0]  = h_p[:, 1]
    hu_p[:, 0] =  hu_p[:, 1]
    hv_p[:, 0] = -hv_p[:, 1]
    z_p[:, 0]  = z_p[:, 1]
    # North
    h_p[:, -1]  = h_p[:, -2]
    hu_p[:, -1] =  hu_p[:, -2]
    hv_p[:, -1] = -hv_p[:, -2]
    z_p[:, -1]  = z_p[:, -2]

solver = SWE2D(x, y, z_b, n_manning=0.0, g=9.81, h_min=1e-3, cfl=0.4)

U, info = solver.run(U0, bc_walls, max_steps=5000, report_every=1000)

# ── Pass criterion ───────────────────────────────────────────────────────────
u_final = U[1] / np.maximum(U[0], 1e-10)
v_final = U[2] / np.maximum(U[0], 1e-10)
err_h = np.max(np.abs(U[0] - h0_init))
err_u = np.max(np.abs(u_final))
err_v = np.max(np.abs(v_final))

print()
print(f"After {info['steps']} steps, t = {info['t_final']:.3f} s:")
print(f"  max|h - h_init| = {err_h:.3e}  m       (target < 1e-10)")
print(f"  max|u|          = {err_u:.3e}  m/s    (target < 1e-10)")
print(f"  max|v|          = {err_v:.3e}  m/s    (target < 1e-10)")

ok = (err_h < 1e-10 and err_u < 1e-10 and err_v < 1e-10)
print()
print(f"LAKE-AT-REST: {'PASS ✓' if ok else 'FAIL ✗'}")
