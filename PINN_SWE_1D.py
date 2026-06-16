"""
PINN_SWE_1D.py
==============
Inverse PINN for 1D steady shallow-water equations.

Solution A: L-BFGS removed.

Rationale (from diagnostic run, see chat log):
  Phase B Adam consistently lands n within 0.5–3.5% of truth across all
  sparsity/noise/obs-type combinations. L-BFGS then slides along an
  identifiability null direction — both losses keep decreasing while n
  drifts toward 0.001–0.008. The polish step was actively harmful, not
  beneficial, because the network has enough capacity (~3000 params) to
  compensate for wrong n while still reducing the joint loss.

  Pure Adam (Phase A obs-only + Phase B obs+physics) is the recovery
  procedure. n_adam stays at 1000 (500 + 500) as configured.

Critical design notes preserved from earlier iterations:
  1. net_f is NOT @tf.function — the outer GradientTape in train_step
     must see n_raw through Sf; @tf.function + inner persistent tape
     breaks that path.
  2. lam_obs / lam_eq are tf.Variable (not tf.constant) so that
     @tf.function on train_step reads their values at execution time,
     not trace time. This is what lets Phase A → Phase B switching
     actually take effect.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models


class InversePINN_1D(models.Model):

    def __init__(self, model, optimizer, epochs,
                 g=9.81, n_init=0.035,
                 lam_obs=1.0, lam_eq=1.0,
                 obs_mask=None,
                 **kwargs):
        super(InversePINN_1D, self).__init__(**kwargs)
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

        if obs_mask is None:
            obs_mask = [1.0, 1.0]
        self.obs_mask = tf.constant(obs_mask, dtype=tf.float32)

        n_raw_init = float(np.log(np.exp(n_init) - 1.0))
        self.n_raw = tf.Variable(n_raw_init, trainable=True,
                                 dtype=tf.float32, name="n_manning_raw")

        self._tracked_extra = [self.n_raw]

    @property
    def n_manning(self):
        return tf.nn.softplus(self.n_raw)

    @property
    def trainable_variables(self):
        base = self.model.trainable_variables
        return list(base) + [self.n_raw]

    # ── Scaling ───────────────────────────────────────────────────────────────

    def fit_scale(self, x_all, y_obs_phys):
        self.x_min = tf.reduce_min(x_all)
        self.x_max = tf.reduce_max(x_all)
        y_max_raw  = tf.reduce_max(tf.abs(y_obs_phys), axis=0)
        self.y_max = tf.where(y_max_raw > 1e-10,
                              y_max_raw, tf.ones_like(y_max_raw))

    @tf.function
    def _sx(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

    @tf.function
    def _sy_r(self, ys):
        return ys * self.y_max

    # ── Physics residual ─────────────────────────────────────────────────────

    def net_f(self, x_col_s):
        x = x_col_s[:, 0]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            pred_s   = self.model(tf.stack([x], axis=-1))
            pred     = self._sy_r(pred_s)
            h        = pred[:, 0]
            u        = pred[:, 1]
            q_flux   = u * h
            mom_flux = u * u * h + 0.5 * self.g * h * h

        dq_dx   = tape.gradient(q_flux,   x)
        dmom_dx = tape.gradient(mom_flux, x)
        del tape

        inv_L   = 1.0 / (self.x_max - self.x_min)
        dq_dx   = dq_dx   * inv_L
        dmom_dx = dmom_dx * inv_L

        h_safe = tf.maximum(h, 1e-6)
        Sf     = tf.square(self.n_manning * u / tf.pow(h_safe, 2.0/3.0))

        f1 = dq_dx
        f2 = dmom_dx + self.g * h * Sf
        return tf.stack([f1, f2], axis=-1)

    # ── Training step ────────────────────────────────────────────────────────

    @tf.function
    def train_step(self, obs_data, x_col_s):
        x_obs = obs_data[:, :1]
        y_obs = obs_data[:, 1:]

        with tf.GradientTape() as tape:
            pred_obs     = self.model(x_obs)
            diff         = tf.square(y_obs - pred_obs) * self.obs_mask
            raw_loss_obs = tf.reduce_mean(diff)

            f            = self.net_f(x_col_s)
            raw_loss_eq  = tf.reduce_mean(tf.square(f))

            loss = self.lam_obs * raw_loss_obs + self.lam_eq * raw_loss_eq

        grads = tape.gradient(loss, self.trainable_variables)
        g_n_raw = grads[-1]

        return loss, grads, tf.stack([raw_loss_obs + raw_loss_eq,
                                      raw_loss_obs, raw_loss_eq,
                                      self.n_manning]), g_n_raw

    # ── fit (Adam-only, two-phase) ───────────────────────────────────────────

    def fit(self, obs, cp):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        cp  = tf.convert_to_tensor(cp,  dtype=tf.float32)

        x_obs_phys = obs[:, :1]
        y_obs_phys = obs[:, 1:]
        x_all      = tf.concat([x_obs_phys, cp], axis=0)
        self.fit_scale(x_all, y_obs_phys)

        x_obs_s  = self._sx(x_obs_phys)
        x_col_s  = self._sx(cp)
        y_obs_s  = y_obs_phys / self.y_max
        obs_data = tf.concat([x_obs_s, y_obs_s], axis=1)

        lam_eq_orig = float(self.lam_eq.numpy())

        print("\n" + "=" * 72)
        print(f"DIAG  n_init        = {self.n_init_value:.6f}")
        print(f"DIAG  lam_obs       = {float(self.lam_obs.numpy()):.3e}")
        print(f"DIAG  lam_eq_orig   = {lam_eq_orig:.3e}")
        print(f"DIAG  n_obs_points  = {int(obs_data.shape[0])}")
        print(f"DIAG  n_col_points  = {int(x_col_s.shape[0])}")
        print(f"DIAG  total epochs  = {self.epochs}  "
              f"(L-BFGS disabled — Adam only)")
        print("=" * 72)

        # ── Phase A: observation fit (lam_eq = 0) ────────────────────────────
        self.lam_eq.assign(0.0)
        n_phase_a = self.epochs // 2
        n_phase_b = self.epochs - n_phase_a

        print(f"\n--- PHASE A: obs-only, {n_phase_a} steps (lam_eq=0) ---")
        a_log = {0, 1, 9, 49, max(n_phase_a // 2, 1), n_phase_a - 1}
        for k in range(n_phase_a):
            loss, grads, h_vec, g_n = self.train_step(obs_data, x_col_s)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(h_vec.numpy())
            if k in a_log:
                print(f"  step {k:5d}  loss_obs={float(h_vec[1]):.3e}  "
                      f"loss_eq={float(h_vec[2]):.3e}  "
                      f"n={float(h_vec[3]):.6f}  g_n_raw={float(g_n):.3e}")

        n_end_phase_a = float(self.n_manning.numpy())
        print(f"--- end PHASE A: n = {n_end_phase_a:.6f} ---")

        # ── Phase B: joint obs + physics ─────────────────────────────────────
        self.lam_eq.assign(lam_eq_orig)
        print(f"\n--- PHASE B: physics on, {n_phase_b} steps "
              f"(lam_eq={lam_eq_orig:.3e}) ---")

        b_log = {0, 1, 9, 49, max(n_phase_b // 2, 1), n_phase_b - 1}
        for k in range(n_phase_b):
            loss, grads, h_vec, g_n = self.train_step(obs_data, x_col_s)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(h_vec.numpy())
            if k in b_log:
                print(f"  step {k:5d}  loss_obs={float(h_vec[1]):.3e}  "
                      f"loss_eq={float(h_vec[2]):.3e}  "
                      f"weighted_eq={lam_eq_orig * float(h_vec[2]):.3e}  "
                      f"n={float(h_vec[3]):.6f}  g_n_raw={float(g_n):.3e}")

        n_end_phase_b = float(self.n_manning.numpy())
        print(f"--- end PHASE B: n = {n_end_phase_b:.6f} ---")

        # ── Summary ──────────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print(f"DIAG SUMMARY")
        print(f"  n_init        = {self.n_init_value:.6f}")
        print(f"  n end Phase A = {n_end_phase_a:.6f}")
        print(f"  n end Phase B = {n_end_phase_b:.6f}   <- final recovered n")
        print(f"  Phase B moved n by: {n_end_phase_b - n_end_phase_a:+.6f}")
        print("=" * 72 + "\n")

        return np.array(self.hist)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, x_phys):
        x_phys = tf.convert_to_tensor(x_phys, dtype=tf.float32)
        xs     = self._sx(x_phys)
        pred_s = self.model(xs)
        return self._sy_r(pred_s).numpy()

    def get_n(self):
        return float(self.n_manning.numpy())
