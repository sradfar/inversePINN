"""
PINN_SWE_2D.py  (patched)
==========================
Inverse PINN for the 2D steady shallow-water equations.

Patches vs previous version:
  * Bed-slope source term g·h·∇z_b is now included in momentum residuals.
    Previously absent, which is part of why every recovery converged to the
    same n ≈ 0.0343 regardless of data conditions.
  * cp is now (N_col, 4) = [x, y, dz_b/dx, dz_b/dy]. The extra columns are
    physical-units bed-slope components evaluated at each collocation point.
    Caller must pre-compute these from the bathymetry on the data grid.

All other behaviour is unchanged:
  * lam_obs / lam_eq as tf.Variable for Phase A → Phase B switching.
  * n_raw as softplus-parameterised tf.Variable.
  * No normalisation on the loss (preserves gradient flow to n_raw).
  * Two-phase Adam training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models


class InversePINN_2D(models.Model):

    def __init__(self, model, optimizer, epochs,
                 g=9.81, n_init=0.035,
                 lam_obs=1.0, lam_eq=1.0, **kwargs):
        super(InversePINN_2D, self).__init__(**kwargs)
        self.model   = model
        self.opt     = optimizer
        self.epochs  = epochs
        self.g       = tf.constant(g, dtype=tf.float32)

        self.lam_obs = tf.Variable(lam_obs, dtype=tf.float32,
                                   trainable=False, name="lam_obs")
        self.lam_eq  = tf.Variable(lam_eq,  dtype=tf.float32,
                                   trainable=False, name="lam_eq")

        self.hist  = []
        self.epoch = 0
        self.n_init_value = float(n_init)

        n_raw_init = float(np.log(np.exp(n_init) - 1.0))
        self.n_raw = tf.Variable(n_raw_init, trainable=True,
                                 dtype=tf.float32, name="n_manning_raw")

    @property
    def n_manning(self):
        return tf.nn.softplus(self.n_raw)

    @property
    def trainable_variables(self):
        return list(self.model.trainable_variables) + [self.n_raw]

    # ── Scaling ───────────────────────────────────────────────────────────────
    def fit_scale(self, xy_all, y_obs):
        self.xy_min = tf.reduce_min(xy_all, axis=0)
        self.xy_max = tf.reduce_max(xy_all, axis=0)
        y_max_raw   = tf.reduce_max(tf.abs(y_obs), axis=0)
        self.y_max = tf.where(y_max_raw > 1e-10,
                              y_max_raw, tf.ones_like(y_max_raw))

    @tf.function
    def _sxy(self, xy):
        return (xy - self.xy_min) / (self.xy_max - self.xy_min)

    @tf.function
    def _sy_r(self, ys):
        return ys * self.y_max

    # ── Physics residual ─────────────────────────────────────────────────────
    def net_f(self, xy_col_s, dzb_col):
        """
        xy_col_s : (N, 2) scaled collocation coords
        dzb_col  : (N, 2) PHYSICAL bed-slope components [dz_b/dx, dz_b/dy]
        """
        x = xy_col_s[:, 0]
        y = xy_col_s[:, 1]
        Lx = self.xy_max[0] - self.xy_min[0]
        Ly = self.xy_max[1] - self.xy_min[1]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x); tape.watch(y)
            xy_s   = tf.stack([x, y], axis=-1)
            pred_s = self.model(xy_s)
            pred   = self._sy_r(pred_s)
            h = pred[:, 0]; u = pred[:, 1]; v = pred[:, 2]

            uh    = u * h
            vh    = v * h
            uuh_p = u * u * h + 0.5 * self.g * h * h
            uvh   = u * v * h
            vvh_p = v * v * h + 0.5 * self.g * h * h

        duh_dx  = tape.gradient(uh,    x) / Lx
        dvh_dy  = tape.gradient(vh,    y) / Ly
        duuh_dx = tape.gradient(uuh_p, x) / Lx
        duvh_dy = tape.gradient(uvh,   y) / Ly
        duvh_dx = tape.gradient(uvh,   x) / Lx
        dvvh_dy = tape.gradient(vvh_p, y) / Ly
        del tape

        # Manning friction
        h_safe = tf.maximum(h, 1e-6)
        U_mag  = tf.sqrt(u*u + v*v + 1e-10)
        n2     = tf.square(self.n_manning)
        coeff  = n2 * U_mag / tf.pow(h_safe, 4.0/3.0)
        Sf_x   = coeff * u
        Sf_y   = coeff * v

        # Bed-slope components (pre-computed, physical units)
        dzb_dx = dzb_col[:, 0]
        dzb_dy = dzb_col[:, 1]

        f1 = duh_dx  + dvh_dy
        f2 = duuh_dx + duvh_dy + self.g * h * dzb_dx + self.g * h * Sf_x
        f3 = duvh_dx + dvvh_dy + self.g * h * dzb_dy + self.g * h * Sf_y
        return tf.stack([f1, f2, f3], axis=-1)

    # ── Training step ────────────────────────────────────────────────────────
    @tf.function
    def train_step(self, obs_data, xy_col_s, dzb_col):
        xy_obs = obs_data[:, :2]
        y_obs  = obs_data[:, 2:]

        with tf.GradientTape() as tape:
            pred_obs     = self.model(xy_obs)
            raw_loss_obs = tf.reduce_mean(tf.square(y_obs - pred_obs))
            f            = self.net_f(xy_col_s, dzb_col)
            raw_loss_eq  = tf.reduce_mean(tf.square(f))
            loss = self.lam_obs * raw_loss_obs + self.lam_eq * raw_loss_eq

        grads = tape.gradient(loss, self.trainable_variables)
        g_n_raw = grads[-1]
        return loss, grads, tf.stack([raw_loss_obs + raw_loss_eq,
                                      raw_loss_obs, raw_loss_eq,
                                      self.n_manning]), g_n_raw

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, obs, cp):
        """
        obs: (N_obs, 5) = [x, y, h, u, v]
        cp : (N_col, 4) = [x, y, dz_b/dx, dz_b/dy]   ← physical units
        """
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        cp  = tf.convert_to_tensor(cp,  dtype=tf.float32)

        if cp.shape[1] != 4:
            raise ValueError(
                f"cp must have 4 columns [x, y, dzb_dx, dzb_dy]; got shape {cp.shape}. "
                f"Compute bed-slope at collocation points from the data grid."
            )

        xy_obs_phys = obs[:, :2]
        y_obs_phys  = obs[:, 2:]
        xy_col_phys = cp[:, :2]
        dzb_col     = cp[:, 2:]                 # physical bed slopes — NOT scaled

        xy_all = tf.concat([xy_obs_phys, xy_col_phys], axis=0)
        self.fit_scale(xy_all, y_obs_phys)

        xy_obs_s = self._sxy(xy_obs_phys)
        xy_col_s = self._sxy(xy_col_phys)
        y_obs_s  = y_obs_phys / self.y_max
        obs_data = tf.concat([xy_obs_s, y_obs_s], axis=1)

        lam_eq_orig = float(self.lam_eq.numpy())

        print("\n" + "=" * 72)
        print(f"DIAG  n_init        = {self.n_init_value:.6f}")
        print(f"DIAG  lam_obs       = {float(self.lam_obs.numpy()):.3e}")
        print(f"DIAG  lam_eq_orig   = {lam_eq_orig:.3e}")
        print(f"DIAG  n_obs_points  = {int(obs_data.shape[0])}")
        print(f"DIAG  n_col_points  = {int(xy_col_s.shape[0])}")
        print(f"DIAG  total epochs  = {self.epochs}  (Adam only)")
        print(f"DIAG  bed slope mean = ({tf.reduce_mean(dzb_col[:,0]).numpy():+.4e}, "
              f"{tf.reduce_mean(dzb_col[:,1]).numpy():+.4e})")
        print("=" * 72)

        # Phase A: obs-only
        self.lam_eq.assign(0.0)
        n_phase_a = self.epochs // 2
        n_phase_b = self.epochs - n_phase_a

        print(f"\n--- PHASE A: obs-only, {n_phase_a} steps ---")
        a_log = {0, 1, 9, 49, max(n_phase_a // 2, 1), n_phase_a - 1}
        for k in range(n_phase_a):
            loss, grads, h_vec, g_n = self.train_step(obs_data, xy_col_s, dzb_col)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(h_vec.numpy())
            if k in a_log:
                print(f"  step {k:5d}  loss_obs={float(h_vec[1]):.3e}  "
                      f"loss_eq={float(h_vec[2]):.3e}  "
                      f"n={float(h_vec[3]):.6f}  g_n_raw={float(g_n):.3e}")

        n_end_phase_a = float(self.n_manning.numpy())
        print(f"--- end PHASE A: n = {n_end_phase_a:.6f} ---")

        # Phase B: physics on
        self.lam_eq.assign(lam_eq_orig)
        print(f"\n--- PHASE B: physics on, {n_phase_b} steps ---")
        b_log = {0, 1, 9, 49, max(n_phase_b // 2, 1), n_phase_b - 1}
        for k in range(n_phase_b):
            loss, grads, h_vec, g_n = self.train_step(obs_data, xy_col_s, dzb_col)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(h_vec.numpy())
            if k in b_log:
                print(f"  step {k:5d}  loss_obs={float(h_vec[1]):.3e}  "
                      f"loss_eq={float(h_vec[2]):.3e}  "
                      f"n={float(h_vec[3]):.6f}  g_n_raw={float(g_n):.3e}")

        n_end_phase_b = float(self.n_manning.numpy())
        print(f"--- end PHASE B: n = {n_end_phase_b:.6f} ---")
        print(f"\nDIAG SUMMARY:  n_init={self.n_init_value:.4f}, "
              f"Phase A end n={n_end_phase_a:.4f}, Phase B end n={n_end_phase_b:.4f}\n")

        return np.array(self.hist)

    def predict(self, xy_phys):
        xy_phys = tf.convert_to_tensor(xy_phys, dtype=tf.float32)
        xy_s    = self._sxy(xy_phys)
        pred_s  = self.model(xy_s)
        return self._sy_r(pred_s).numpy()

    def get_n(self):
        return float(self.n_manning.numpy())
