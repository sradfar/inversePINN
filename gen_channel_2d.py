"""
gen_channel_2d.py
=================
2D inverse-PINN benchmark dataset (replaces the broken gen_thacker.py).

Geometry: sloped channel with parabolic transverse bed
    z_b(x, y) = -S_0·x  +  h_y · (2·y / W)²

This is a standard SWE benchmark configuration:
  - Streamwise: gentle downhill slope → friction-balanced normal-depth flow
  - Transverse: parabolic U-shape → deeper in centre, shallower at sides
  - 2D effects: depth-varying friction induces transverse v component

Driven by Dirichlet inflow on west, free outflow on east, walls N/S.
At steady state, Manning friction is the SAME ORDER as the bed-slope source
(unlike the original gen_thacker, where friction was 6 OOM smaller).

Saves data/thacker_2d.npz   (same filename so train_inverse_2D.py runs unchanged)
"""

import numpy as np
import os
import sys
sys.path.insert(0, ".")
from swe_solver_2d import SWE2D

# ── Parameters ────────────────────────────────────────────────────────────────
g       = 9.81
n_true  = 0.02

# Channel geometry
Lx      = 2000.0    # length [m]
W       = 400.0     # width [m]
S_0     = 0.002     # streamwise bed slope
h_y     = 0.3       # transverse bed amplitude [m]
q_in    = 1.0       # inflow unit discharge [m²/s]

# Grid (matched to deliver ~6000 usable points like original)
Nx, Ny  = 121, 41

# Analytical normal depth at centreline (deepest section)
h_n = (q_in * n_true / np.sqrt(S_0)) ** (3.0/5.0)
U_n = q_in / h_n
print(f"2D channel generator (replacement for Thacker)")
print(f"  Geometry  : {Lx} x {W} m,   {Nx} x {Ny} grid")
print(f"  S_0       : {S_0}   h_y : {h_y}   q_in : {q_in}")
print(f"  n_true    : {n_true}")
print(f"  Centreline normal depth : h_n = {h_n:.4f} m,  U_n = {U_n:.4f} m/s,  "
      f"Fr = {U_n/np.sqrt(g*h_n):.3f}")

# ── Grid ──────────────────────────────────────────────────────────────────────
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(-W/2, W/2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
z_b = -S_0 * X + h_y * (2.0 * Y / W) ** 2

# ── Initial condition: uniform normal depth ──────────────────────────────────
h_init = np.full((Nx, Ny), h_n)
u_init = np.full((Nx, Ny), U_n)
v_init = np.zeros((Nx, Ny))
U0 = np.stack([h_init, h_init * u_init, h_init * v_init], axis=0)

# ── Boundary conditions ──────────────────────────────────────────────────────
def bc(h_p, hu_p, hv_p, z_p):
    # West inflow: uniform discharge, normal depth
    h_p[0, :]  = h_n
    hu_p[0, :] = q_in
    hv_p[0, :] = 0.0
    z_p[0, :]  = z_b[0, 0]    # bed at x=0
    # East outflow: zero-gradient
    h_p[-1, :]  = h_p[-2, :]
    hu_p[-1, :] = hu_p[-2, :]
    hv_p[-1, :] = hv_p[-2, :]
    z_p[-1, :]  = z_p[-2, :]
    # N/S walls (y=±W/2): free-slip on u, no normal v
    h_p[:, 0]  = h_p[:, 1]
    hu_p[:, 0] =  hu_p[:, 1]
    hv_p[:, 0] = -hv_p[:, 1]
    z_p[:, 0]  = z_p[:, 1]
    h_p[:, -1]  = h_p[:, -2]
    hu_p[:, -1] =  hu_p[:, -2]
    hv_p[:, -1] = -hv_p[:, -2]
    z_p[:, -1]  = z_p[:, -2]

# ── Solve to steady state ────────────────────────────────────────────────────
solver = SWE2D(x, y, z_b, n_manning=n_true, g=g, h_min=1e-3, cfl=0.4)
U, info = solver.run(U0, bc, max_steps=40000, tol=1e-8, report_every=4000)

# ── Extract fields ───────────────────────────────────────────────────────────
h = U[0]
u = U[1] / np.maximum(U[0], 1e-6)
v = U[2] / np.maximum(U[0], 1e-6)
eta = h + z_b
U_mag = np.sqrt(u**2 + v**2)
Sf = (n_true * U_mag / np.maximum(h, 1e-6)**(2.0/3.0))**2
Fr = U_mag / np.sqrt(g * np.maximum(h, 1e-6))

# Wet mask: exclude boundary cells (Dirichlet inflow / extrapolated outflow / wall ghost)
mask = np.ones_like(h, dtype=bool)
mask[0, :] = False; mask[-1, :] = False
mask[:, 0] = False; mask[:, -1] = False

print()
print(f"Steady state at step {info['steps']}, t = {info['t_final']:.1f} s")
print(f"  h     range : [{h.min():.4f}, {h.max():.4f}] m")
print(f"  u     range : [{u.min():.4f}, {u.max():.4f}] m/s")
print(f"  v     range : [{v.min():.4f}, {v.max():.4f}] m/s")
print(f"  Fr    max   : {Fr.max():.3f}")
print(f"  |v|/|u| max : {np.max(np.abs(v)/np.maximum(np.abs(u),1e-6)):.3f}")

# Identifiability check: back-solve n² pointwise from x-momentum
dx, dy = x[1]-x[0], y[1]-y[0]
def ddx(f):
    out = np.zeros_like(f); out[1:-1,:] = (f[2:,:] - f[:-2,:])/(2*dx); return out
def ddy(f):
    out = np.zeros_like(f); out[:,1:-1] = (f[:,2:] - f[:,:-2])/(2*dy); return out

F2x = u*u*h + 0.5*g*h*h
F2y = u*v*h
dFx_dx = ddx(F2x)
dFy_dy = ddy(F2y)
dzb_dx = ddx(z_b)
denom_x = g * h * u * U_mag / np.maximum(h, 1e-6)**(4.0/3.0)
denom_x_safe = np.where(np.abs(denom_x) > 1e-12, denom_x, 1.0)
n2_pt = -(dFx_dx + dFy_dy + g * h * dzb_dx) / denom_x_safe

# Use central interior (well away from boundaries) for FD reliability
interior = mask.copy()
interior[:3, :] = False; interior[-3:, :] = False
interior[:, :3] = False; interior[:, -3:] = False
vals = n2_pt[interior]
print()
print(f"Back-solved n² from x-momentum (central interior):")
print(f"  true n²    = {n_true**2:.3e}")
print(f"  median n²  = {np.median(vals):.3e}")
print(f"  iqr        = [{np.percentile(vals,25):.3e}, {np.percentile(vals,75):.3e}]")
print(f"  fraction negative = {(vals<0).mean()*100:.1f}%   "
      f"(was 100% in old gen_thacker)")
err_pct = abs(np.median(vals) - n_true**2)/n_true**2 * 100
print(f"  median error vs truth = {err_pct:.1f}%")

# ── Save (same field names as before) ────────────────────────────────────────
h_save = np.where(mask, h, np.nan)
u_save = np.where(mask, u, np.nan)
v_save = np.where(mask, v, np.nan)
Sf_save = np.where(mask, Sf, np.nan)
Fr_save = np.where(mask, Fr, np.nan)

os.makedirs("data", exist_ok=True)
np.savez_compressed(
    "data/thacker_2d.npz",
    x      = X, y      = Y,
    h      = h_save, u  = u_save, v  = v_save,
    z      = z_b,    eta = eta,
    Sf     = Sf_save, Fr = Fr_save,
    mask   = mask,
    q      = np.array([q_in]),
    n_true = np.array([n_true]),
    g      = np.array([g]),
    R      = np.array([1.0]),     # not used; kept for format compatibility
    h0     = np.array([h_n]),     # reference depth instead of bowl depth
)
print(f"\nSaved: data/thacker_2d.npz  ({mask.sum()} usable points)")
