"""
gen_macdonald.py  (patched)
============================
MacDonald (1996) 1D subcritical benchmark for inverse PINN training.

Previous version: prescribed h(x), u=q/h, flat bed (z=0), reported Sf as a
diagnostic. The data did NOT satisfy steady SWE momentum with a constant
Manning n: pointwise n² back-solve gave 50% negative values, the symmetric
sin profile cannot be a steady flat-bed friction-balanced flow.

Patched: proper MacDonald construction. Prescribe h(x), q, n_true and
BACK-SOLVE z(x) such that the steady 1D SWE momentum equation is exactly
satisfied:

    d/dx(u²h + g h²/2) + g h S_f + g h dz/dx = 0
    →  dz/dx = -(dF/dx)/(g h) - S_f         with S_f = (n·q/h^(5/3))²
    →  z(x) = ∫₀ˣ dz/dx' dx'                 (z(0) := 0 reference)

The h(x) profile and all other fields are preserved; only z is now non-trivial
(and eta, since eta = h + z). File format unchanged so train_inverse_1D.py
runs without modification.
"""

import numpy as np
import os

# ── Parameters (unchanged) ────────────────────────────────────────────────────
g       = 9.81
L       = 1000.0
N       = 501
n_true  = 0.02
q       = 0.5

# ── Prescribed water depth profile (unchanged) ────────────────────────────────
x  = np.linspace(0.0, L, N)
h  = 0.5 + 0.1 * np.sin(np.pi * x / L)
u  = q / h
Fr = u / np.sqrt(g * h)
Sf = (n_true * q / h**(5.0/3.0))**2          # friction slope from Manning

# ── Analytical d/dx(u²h + g h²/2) for the prescribed h ────────────────────────
# F = q²/h + g h²/2     →     dF/dx = dh/dx · (g h - q²/h²)
dh_dx = 0.1 * (np.pi / L) * np.cos(np.pi * x / L)
dF_dx = dh_dx * (g * h - q**2 / h**2)

# ── Back-solve bed slope and bed elevation ────────────────────────────────────
dz_dx = -dF_dx / (g * h) - Sf                # bed slope required by momentum
z     = np.concatenate([[0.0], np.cumsum(0.5*(dz_dx[1:] + dz_dx[:-1])) * (x[1]-x[0])])
z     = z - z.max()                          # shift so highest bed point = 0
eta   = h + z                                # free surface elevation

h_upstream   = float(h[0])
h_downstream = float(h[-1])

# ── Save (same field names) ───────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.savez_compressed(
    "data/macdonald_subcritical.npz",
    x            = x,
    h            = h,
    u            = u,
    z            = z,
    eta          = eta,
    q            = np.array([q]),
    n_true       = np.array([n_true]),
    g            = np.array([g]),
    L            = np.array([L]),
    h_upstream   = np.array([h_upstream]),
    h_downstream = np.array([h_downstream]),
    Sf           = Sf,
    Fr           = Fr,
)

# ── Diagnostics ───────────────────────────────────────────────────────────────
print("MacDonald subcritical benchmark (PATCHED: bed back-solved)")
print("=" * 60)
print(f"  N points     : {N}")
print(f"  h range      : [{h.min():.4f}, {h.max():.4f}] m")
print(f"  u range      : [{u.min():.4f}, {u.max():.4f}] m/s")
print(f"  Fr max       : {Fr.max():.4f}   ({'subcritical ✓' if Fr.max()<1 else 'NOT SUBCRITICAL'})")
print(f"  z range      : [{z.min():.4f}, {z.max():.4f}] m   "
      f"(was 0 in broken version)")
print(f"  eta range    : [{eta.min():.4f}, {eta.max():.4f}] m")
print(f"  dz/dx range  : [{dz_dx.min():+.4e}, {dz_dx.max():+.4e}]")

# Verify the data satisfies steady SWE with constant n_true.
# Recompute dF/dx and dz/dx via finite differences, then back-solve n².
dF_fd   = np.gradient(u*u*h + 0.5*g*h*h, x)
dz_fd   = np.gradient(z, x)
Sf_req  = -(dF_fd + g*h*dz_fd) / (g * h)
n2_req  = Sf_req * h**(4/3) / u**2

print()
print("Consistency check — pointwise n² back-solved from FD on saved data:")
print(f"  true n²                  : {n_true**2:.4e}")
print(f"  median back-solved n²    : {np.median(n2_req):.4e}")
print(f"  fraction negative        : {(n2_req<0).mean()*100:.1f}%  "
      f"(was 49.9% before)")
print(f"  rel. error vs truth      : "
      f"{abs(np.median(n2_req) - n_true**2)/n_true**2*100:.2f}%")
print()
print("Saved: data/macdonald_subcritical.npz")
