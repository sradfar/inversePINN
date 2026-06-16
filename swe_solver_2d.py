"""
swe_solver_2d.py
================
2D shallow-water finite-volume solver.

Scheme:
  - HLL Riemann solver at each interface
  - Audusse hydrostatic reconstruction → well-balanced bed-slope treatment
  - Semi-implicit Manning friction (point-implicit)
  - Forward-Euler time stepping with CFL-bounded dt
  - Ghost-cell boundary conditions

Conservative variables U = (h, hu, hv) at cell centres on a uniform grid.

Governing equations:
  ∂h/∂t  + ∂(uh)/∂x + ∂(vh)/∂y = 0
  ∂(uh)/∂t + ∂(u²h + gh²/2)/∂x + ∂(uvh)/∂y = -g·h·∂z_b/∂x - g·h·Sf_x
  ∂(vh)/∂t + ∂(uvh)/∂x + ∂(v²h + gh²/2)/∂y = -g·h·∂z_b/∂y - g·h·Sf_y
  Sf_x = n²·u·|U|/h^(4/3),   Sf_y = n²·v·|U|/h^(4/3)
"""

import numpy as np


class SWE2D:

    def __init__(self, x, y, z_b, n_manning=0.02, g=9.81,
                 h_min=1e-3, cfl=0.4):
        """
        x       : (Nx,) cell centres in x, uniform spacing
        y       : (Ny,) cell centres in y, uniform spacing
        z_b     : (Nx, Ny) bed elevation at cell centres
        n_manning : Manning roughness coefficient (scalar)
        """
        assert z_b.shape == (len(x), len(y))
        self.x = x.copy()
        self.y = y.copy()
        self.z_b = z_b.copy()
        self.n = float(n_manning)
        self.g = g
        self.h_min = h_min
        self.cfl = cfl
        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        self.Nx = len(x)
        self.Ny = len(y)

    # ── Flux at all x-interfaces ─────────────────────────────────────────────
    def x_interface_fluxes(self, h, hu, hv):
        """
        Compute HLL fluxes at all x-interfaces (i+1/2, j), i = 0..Nx-2.
        Returns (F1, F2_L, F2_R, F3) each of shape (Nx-1, Ny).
        F2_L: momentum flux to apply on the LEFT side of the interface
              (i.e., what cell i sees on its right face)
        F2_R: momentum flux to apply on the RIGHT side of the interface
              (i.e., what cell i+1 sees on its left face)
        """
        h_L = h[:-1, :]; h_R = h[1:, :]
        hu_L = hu[:-1, :]; hu_R = hu[1:, :]
        hv_L = hv[:-1, :]; hv_R = hv[1:, :]
        z_L = self.z_b[:-1, :]; z_R = self.z_b[1:, :]

        # Safe velocities at full states (used only for wave speeds)
        h_L_safe = np.maximum(h_L, self.h_min)
        h_R_safe = np.maximum(h_R, self.h_min)
        u_L = hu_L / h_L_safe; v_L = hv_L / h_L_safe
        u_R = hu_R / h_R_safe; v_R = hv_R / h_R_safe

        # Hydrostatic reconstruction
        z_star = np.maximum(z_L, z_R)
        h_L_star = np.maximum(0.0, h_L + z_L - z_star)
        h_R_star = np.maximum(0.0, h_R + z_R - z_star)

        # Wave speeds (Davis estimate)
        c_L = np.sqrt(self.g * h_L_star)
        c_R = np.sqrt(self.g * h_R_star)
        S_L = np.minimum(u_L - c_L, u_R - c_R)
        S_R = np.maximum(u_L + c_L, u_R + c_R)

        # Fluxes from reconstructed (star) states
        F1_L = h_L_star * u_L
        F2_L_raw = h_L_star * u_L * u_L + 0.5 * self.g * h_L_star ** 2
        F3_L = h_L_star * u_L * v_L
        F1_R = h_R_star * u_R
        F2_R_raw = h_R_star * u_R * u_R + 0.5 * self.g * h_R_star ** 2
        F3_R = h_R_star * u_R * v_R

        # Conservative star states
        U1_L = h_L_star
        U1_R = h_R_star
        U2_L = h_L_star * u_L
        U2_R = h_R_star * u_R
        U3_L = h_L_star * v_L
        U3_R = h_R_star * v_R

        # HLL combination (mask-safe — guard division)
        dS = S_R - S_L
        dS_safe = np.where(np.abs(dS) > 1e-12, dS, 1e-12)
        F1_hll = (S_R * F1_L - S_L * F1_R + S_L * S_R * (U1_R - U1_L)) / dS_safe
        F2_hll = (S_R * F2_L_raw - S_L * F2_R_raw +
                  S_L * S_R * (U2_R - U2_L)) / dS_safe
        F3_hll = (S_R * F3_L - S_L * F3_R + S_L * S_R * (U3_R - U3_L)) / dS_safe

        F1 = np.where(S_L >= 0, F1_L,
                      np.where(S_R <= 0, F1_R, F1_hll))
        F2 = np.where(S_L >= 0, F2_L_raw,
                      np.where(S_R <= 0, F2_R_raw, F2_hll))
        F3 = np.where(S_L >= 0, F3_L,
                      np.where(S_R <= 0, F3_R, F3_hll))

        # Audusse momentum corrections (different on each side of interface)
        F2_L_side = F2 + 0.5 * self.g * (h_L ** 2 - h_L_star ** 2)
        F2_R_side = F2 + 0.5 * self.g * (h_R ** 2 - h_R_star ** 2)

        # Max wave speed for CFL
        smax = np.max(np.maximum(np.abs(S_L), np.abs(S_R)))
        return F1, F2_L_side, F2_R_side, F3, smax

    # ── Flux at all y-interfaces (analogous to x) ───────────────────────────
    def y_interface_fluxes(self, h, hu, hv):
        h_L = h[:, :-1]; h_R = h[:, 1:]
        hu_L = hu[:, :-1]; hu_R = hu[:, 1:]
        hv_L = hv[:, :-1]; hv_R = hv[:, 1:]
        z_L = self.z_b[:, :-1]; z_R = self.z_b[:, 1:]

        h_L_safe = np.maximum(h_L, self.h_min)
        h_R_safe = np.maximum(h_R, self.h_min)
        u_L = hu_L / h_L_safe; v_L = hv_L / h_L_safe
        u_R = hu_R / h_R_safe; v_R = hv_R / h_R_safe

        z_star = np.maximum(z_L, z_R)
        h_L_star = np.maximum(0.0, h_L + z_L - z_star)
        h_R_star = np.maximum(0.0, h_R + z_R - z_star)

        c_L = np.sqrt(self.g * h_L_star)
        c_R = np.sqrt(self.g * h_R_star)
        S_L = np.minimum(v_L - c_L, v_R - c_R)
        S_R = np.maximum(v_L + c_L, v_R + c_R)

        # G(U) = (h v, u v h, v² h + g h²/2)
        G1_L = h_L_star * v_L
        G2_L = h_L_star * u_L * v_L
        G3_L_raw = h_L_star * v_L * v_L + 0.5 * self.g * h_L_star ** 2
        G1_R = h_R_star * v_R
        G2_R = h_R_star * u_R * v_R
        G3_R_raw = h_R_star * v_R * v_R + 0.5 * self.g * h_R_star ** 2

        U1_L = h_L_star;            U1_R = h_R_star
        U2_L = h_L_star * u_L;      U2_R = h_R_star * u_R
        U3_L = h_L_star * v_L;      U3_R = h_R_star * v_R

        dS = S_R - S_L
        dS_safe = np.where(np.abs(dS) > 1e-12, dS, 1e-12)
        G1_hll = (S_R * G1_L - S_L * G1_R + S_L * S_R * (U1_R - U1_L)) / dS_safe
        G2_hll = (S_R * G2_L - S_L * G2_R + S_L * S_R * (U2_R - U2_L)) / dS_safe
        G3_hll = (S_R * G3_L_raw - S_L * G3_R_raw +
                  S_L * S_R * (U3_R - U3_L)) / dS_safe

        G1 = np.where(S_L >= 0, G1_L, np.where(S_R <= 0, G1_R, G1_hll))
        G2 = np.where(S_L >= 0, G2_L, np.where(S_R <= 0, G2_R, G2_hll))
        G3 = np.where(S_L >= 0, G3_L_raw, np.where(S_R <= 0, G3_R_raw, G3_hll))

        # Audusse corrections on y-momentum flux (component G3)
        G3_L_side = G3 + 0.5 * self.g * (h_L ** 2 - h_L_star ** 2)
        G3_R_side = G3 + 0.5 * self.g * (h_R ** 2 - h_R_star ** 2)

        smax = np.max(np.maximum(np.abs(S_L), np.abs(S_R)))
        return G1, G2, G3_L_side, G3_R_side, smax

    # ── Single explicit step ─────────────────────────────────────────────────
    def step(self, U, dt, bc_func):
        """
        U: (3, Nx, Ny) -> (h, hu, hv)
        bc_func(U_padded): mutates U_padded with one layer of ghost cells.
        """
        h, hu, hv = U[0], U[1], U[2]

        # Pad with ghost cells (1 layer)
        h_p  = np.pad(h,  1, mode='edge')
        hu_p = np.pad(hu, 1, mode='edge')
        hv_p = np.pad(hv, 1, mode='edge')
        z_p  = np.pad(self.z_b, 1, mode='edge')

        # Apply BCs (overwrites ghost layers)
        bc_func(h_p, hu_p, hv_p, z_p)

        # Build solver instance on padded grid temporarily? No — just compute
        # fluxes on padded arrays with bed extended too.
        # Use the same algorithm inline (we need the padded z_b for reconstruction).
        F1, F2_L, F2_R, F3, smax_x = _x_fluxes(h_p, hu_p, hv_p, z_p,
                                               self.g, self.h_min)
        G1, G2, G3_L, G3_R, smax_y = _y_fluxes(h_p, hu_p, hv_p, z_p,
                                               self.g, self.h_min)

        # Updates for interior cells (which sit at index [1:-1, 1:-1] in padded)
        # x-direction: cell (i,j) interior; padded index (i+1, j+1)
        #   right face = interface (i+1/2, j) → padded x-interface index i+1
        #   left  face = interface (i-1/2, j) → padded x-interface index i
        # F1 has shape (Nx_p - 1, Ny_p)
        dFh  = (F1   [1:, 1:-1] - F1   [:-1, 1:-1]) / self.dx
        dFhu = (F2_L [1:, 1:-1] - F2_R [:-1, 1:-1]) / self.dx
        dFhv = (F3   [1:, 1:-1] - F3   [:-1, 1:-1]) / self.dx

        dGh  = (G1   [1:-1, 1:] - G1   [1:-1, :-1]) / self.dy
        dGhu = (G2   [1:-1, 1:] - G2   [1:-1, :-1]) / self.dy
        dGhv = (G3_L [1:-1, 1:] - G3_R [1:-1, :-1]) / self.dy

        h_new  = h  - dt * (dFh  + dGh)
        hu_new = hu - dt * (dFhu + dGhu)
        hv_new = hv - dt * (dFhv + dGhv)

        # Semi-implicit Manning friction (point-implicit)
        #   (hu)^* = hu_new,  then solve (1 + dt·K)·(hu)^{n+1} = (hu)^*
        #   K = g·n² · |U|/h^(4/3) ,  |U| computed from (hu_new, hv_new)
        if self.n > 0:
            h_safe = np.maximum(h_new, self.h_min)
            u_tmp  = hu_new / h_safe
            v_tmp  = hv_new / h_safe
            U_mag  = np.sqrt(u_tmp ** 2 + v_tmp ** 2)
            K = self.g * self.n ** 2 * U_mag / h_safe ** (4.0 / 3.0)
            denom = 1.0 + dt * K
            hu_new = hu_new / denom
            hv_new = hv_new / denom

        # Wet/dry: zero out momentum where h is near-zero
        dry = h_new < self.h_min
        hu_new = np.where(dry, 0.0, hu_new)
        hv_new = np.where(dry, 0.0, hv_new)
        h_new  = np.where(dry, 0.0, h_new)

        U_new = np.stack([h_new, hu_new, hv_new], axis=0)
        smax = max(smax_x, smax_y, 1e-6)
        return U_new, smax

    def compute_dt(self, smax):
        return self.cfl * min(self.dx, self.dy) / smax

    def run(self, U0, bc_func, t_end=None, max_steps=200000, tol=None,
            report_every=2000, save_history=False):
        """
        Integrate. Stops on:
          - t >= t_end (if t_end given)
          - residual ||dU/dt||_inf < tol (steady state) if tol given
          - max_steps reached
        Returns (U_final, info)
        """
        U = U0.copy()
        t = 0.0
        history = []
        info = {'steps': 0, 't_final': 0.0, 'res_h': [], 'res_hu': [], 'res_hv': [],
                'mass': [], 'momentum_x': [], 'momentum_y': []}

        for step in range(max_steps):
            # Compute dt from current state
            _, _, _, _, smax_x = self.x_interface_fluxes(U[0], U[1], U[2])
            _, _, _, _, smax_y = self.y_interface_fluxes(U[0], U[1], U[2])
            smax = max(smax_x, smax_y, 1e-6)
            dt = self.compute_dt(smax)
            if t_end is not None:
                dt = min(dt, t_end - t)
            if dt <= 0:
                break

            U_old = U.copy()
            U, _ = self.step(U, dt, bc_func)
            t += dt
            info['steps'] = step + 1
            info['t_final'] = t

            # Diagnostics
            if (step + 1) % report_every == 0 or step == 0:
                res_h  = np.max(np.abs(U[0] - U_old[0])) / dt
                res_hu = np.max(np.abs(U[1] - U_old[1])) / dt
                res_hv = np.max(np.abs(U[2] - U_old[2])) / dt
                mass   = np.sum(U[0]) * self.dx * self.dy
                mx     = np.sum(U[1]) * self.dx * self.dy
                my     = np.sum(U[2]) * self.dx * self.dy
                info['res_h'].append(res_h)
                info['res_hu'].append(res_hu)
                info['res_hv'].append(res_hv)
                info['mass'].append(mass)
                info['momentum_x'].append(mx)
                info['momentum_y'].append(my)
                print(f"  step {step+1:6d}  t={t:9.3f}  dt={dt:.4f}  "
                      f"res_h={res_h:.3e}  res_hu={res_hu:.3e}  "
                      f"res_hv={res_hv:.3e}  mass={mass:.4e}")

                if tol is not None and res_h < tol and res_hu < tol and res_hv < tol:
                    print(f"  ── steady-state reached at step {step+1}, t={t:.3f}")
                    break

                if save_history:
                    history.append(U.copy())

            if t_end is not None and t >= t_end - 1e-12:
                break

        info['history'] = history
        return U, info


# ── Module-level vectorised flux routines (used by step on padded arrays) ──

def _x_fluxes(h, hu, hv, z_b, g, h_min):
    h_L = h[:-1, :]; h_R = h[1:, :]
    hu_L = hu[:-1, :]; hu_R = hu[1:, :]
    hv_L = hv[:-1, :]; hv_R = hv[1:, :]
    z_L = z_b[:-1, :]; z_R = z_b[1:, :]
    h_L_safe = np.maximum(h_L, h_min); h_R_safe = np.maximum(h_R, h_min)
    u_L = hu_L / h_L_safe; v_L = hv_L / h_L_safe
    u_R = hu_R / h_R_safe; v_R = hv_R / h_R_safe
    z_star = np.maximum(z_L, z_R)
    h_L_star = np.maximum(0.0, h_L + z_L - z_star)
    h_R_star = np.maximum(0.0, h_R + z_R - z_star)
    c_L = np.sqrt(g * h_L_star); c_R = np.sqrt(g * h_R_star)
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)
    F1_L = h_L_star * u_L
    F2_L_raw = h_L_star * u_L * u_L + 0.5 * g * h_L_star ** 2
    F3_L = h_L_star * u_L * v_L
    F1_R = h_R_star * u_R
    F2_R_raw = h_R_star * u_R * u_R + 0.5 * g * h_R_star ** 2
    F3_R = h_R_star * u_R * v_R
    U1_L, U1_R = h_L_star, h_R_star
    U2_L, U2_R = h_L_star * u_L, h_R_star * u_R
    U3_L, U3_R = h_L_star * v_L, h_R_star * v_R
    dS = S_R - S_L
    dS_safe = np.where(np.abs(dS) > 1e-12, dS, 1e-12)
    F1_hll = (S_R*F1_L - S_L*F1_R + S_L*S_R*(U1_R-U1_L))/dS_safe
    F2_hll = (S_R*F2_L_raw - S_L*F2_R_raw + S_L*S_R*(U2_R-U2_L))/dS_safe
    F3_hll = (S_R*F3_L - S_L*F3_R + S_L*S_R*(U3_R-U3_L))/dS_safe
    F1 = np.where(S_L >= 0, F1_L, np.where(S_R <= 0, F1_R, F1_hll))
    F2 = np.where(S_L >= 0, F2_L_raw, np.where(S_R <= 0, F2_R_raw, F2_hll))
    F3 = np.where(S_L >= 0, F3_L, np.where(S_R <= 0, F3_R, F3_hll))
    F2_L_side = F2 + 0.5 * g * (h_L ** 2 - h_L_star ** 2)
    F2_R_side = F2 + 0.5 * g * (h_R ** 2 - h_R_star ** 2)
    smax = np.max(np.maximum(np.abs(S_L), np.abs(S_R))) if F1.size else 0.0
    return F1, F2_L_side, F2_R_side, F3, smax


def _y_fluxes(h, hu, hv, z_b, g, h_min):
    h_L = h[:, :-1]; h_R = h[:, 1:]
    hu_L = hu[:, :-1]; hu_R = hu[:, 1:]
    hv_L = hv[:, :-1]; hv_R = hv[:, 1:]
    z_L = z_b[:, :-1]; z_R = z_b[:, 1:]
    h_L_safe = np.maximum(h_L, h_min); h_R_safe = np.maximum(h_R, h_min)
    u_L = hu_L / h_L_safe; v_L = hv_L / h_L_safe
    u_R = hu_R / h_R_safe; v_R = hv_R / h_R_safe
    z_star = np.maximum(z_L, z_R)
    h_L_star = np.maximum(0.0, h_L + z_L - z_star)
    h_R_star = np.maximum(0.0, h_R + z_R - z_star)
    c_L = np.sqrt(g * h_L_star); c_R = np.sqrt(g * h_R_star)
    S_L = np.minimum(v_L - c_L, v_R - c_R)
    S_R = np.maximum(v_L + c_L, v_R + c_R)
    G1_L = h_L_star * v_L
    G2_L = h_L_star * u_L * v_L
    G3_L_raw = h_L_star * v_L * v_L + 0.5 * g * h_L_star ** 2
    G1_R = h_R_star * v_R
    G2_R = h_R_star * u_R * v_R
    G3_R_raw = h_R_star * v_R * v_R + 0.5 * g * h_R_star ** 2
    U1_L, U1_R = h_L_star, h_R_star
    U2_L, U2_R = h_L_star * u_L, h_R_star * u_R
    U3_L, U3_R = h_L_star * v_L, h_R_star * v_R
    dS = S_R - S_L
    dS_safe = np.where(np.abs(dS) > 1e-12, dS, 1e-12)
    G1_hll = (S_R*G1_L - S_L*G1_R + S_L*S_R*(U1_R-U1_L))/dS_safe
    G2_hll = (S_R*G2_L - S_L*G2_R + S_L*S_R*(U2_R-U2_L))/dS_safe
    G3_hll = (S_R*G3_L_raw - S_L*G3_R_raw + S_L*S_R*(U3_R-U3_L))/dS_safe
    G1 = np.where(S_L >= 0, G1_L, np.where(S_R <= 0, G1_R, G1_hll))
    G2 = np.where(S_L >= 0, G2_L, np.where(S_R <= 0, G2_R, G2_hll))
    G3 = np.where(S_L >= 0, G3_L_raw, np.where(S_R <= 0, G3_R_raw, G3_hll))
    G3_L_side = G3 + 0.5 * g * (h_L ** 2 - h_L_star ** 2)
    G3_R_side = G3 + 0.5 * g * (h_R ** 2 - h_R_star ** 2)
    smax = np.max(np.maximum(np.abs(S_L), np.abs(S_R))) if G1.size else 0.0
    return G1, G2, G3_L_side, G3_R_side, smax
