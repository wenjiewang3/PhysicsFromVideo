import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_omega_zeta(gamma0: float, gamma1: float):
    zeta = gamma1 / 2.0
    disc = gamma0 - zeta**2
    if disc > 1e-12:
        omega = float(np.sqrt(disc))
        regime = "Underdamped"
    elif disc < -1e-12:
        omega = 0.0
        regime = "Overdamped"
    else:
        omega = 0.0
        regime = "Critical"
    return zeta, omega, regime

def _to_1d_numpy(z):
    if isinstance(z, torch.Tensor):
        if z.ndim == 3: z = z.squeeze(0).squeeze(-1)
        elif z.ndim == 2: z = z.squeeze(-1)
        z = z.detach().cpu().numpy()
    else:
        z = np.asarray(z)
        if z.ndim == 3: z = z.squeeze(0).squeeze(-1)
        elif z.ndim == 2: z = z.squeeze(-1)
    assert z.ndim == 1
    return z

def true_z_series(g0, g1, z0, z1, T_len, dt):
    t = np.arange(T_len, dtype=np.float64) * float(dt)
    D = g1**2 - 4.0*g0
    eps = 1e-12
    if D < -eps:
        zeta = g1/2.0
        w2 = max(g0 - zeta**2, 0.0)
        omega = np.sqrt(w2)
        C1 = z0
        C2 = (z1 + zeta*z0)/(omega + 1e-30)
        z = np.exp(-zeta*t) * (C1*np.cos(omega*t) + C2*np.sin(omega*t))
    elif D > eps:
        s = np.sqrt(D); r1 = (-g1 + s)/2.0; r2 = (-g1 - s)/2.0
        denom = (r1 - r2) if abs(r1 - r2) > 0 else 1e-30
        A = (z1 - r2*z0)/denom; B = (r1*z0 - z1)/denom
        z = A*np.exp(r1*t) + B*np.exp(r2*t)
    else:
        zeta = g1/2.0
        C1 = z0; C2 = z1 + zeta*z0
        z = (C1 + C2*t)*np.exp(-zeta*t)
    return z.astype(np.float32)

def plot_est_vs_true(
    z_est,
    dt: float,
    gamma0: float, gamma1: float, z0: float, z1: float,
    save_path: str = "z_compare.png",
    title_top: str = "Estimated z(t)",
    title_bottom: str = "True z(t)",
    match_ylims: bool = False,
    gamma0_hat  = None,
    gamma1_hat  = None,
    z0_hat  = None,
    z1_hat = None,
):
    z_est_np = _to_1d_numpy(z_est)
    T_len = z_est_np.shape[0]
    t = np.arange(T_len, dtype=np.float32) * float(dt)
    z_true_np = true_z_series(gamma0, gamma1, z0, z1, T_len=T_len, dt=dt)

    if z0_hat is None: z0_hat = float(z_est_np[0])
    if z1_hat is None and T_len >= 2:
        z1_hat = float((z_est_np[1] - z_est_np[0]) / float(dt))

    zeta_true, omega_true, regime_true = compute_omega_zeta(gamma0, gamma1)
    zeta_hat = omega_hat = None; regime_hat = "N/A"
    if (gamma0_hat is not None) and (gamma1_hat is not None):
        zeta_hat, omega_hat, regime_hat = compute_omega_zeta(float(gamma0_hat), float(gamma1_hat))

    def _fmt(v):
        return "N/A" if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{v:.6g}"

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)

    axes[0].plot(t, z_est_np, linewidth=1.5)
    axes[0].set_title(title_top); axes[0].set_ylabel("z (rad)")
    text_est = (
        f"Estimated params ({regime_hat}):\n"
        f"γ0={_fmt(gamma0_hat)}  γ1={_fmt(gamma1_hat)}\n"
        f"ζ={_fmt(zeta_hat)}  ω={_fmt(omega_hat)} \n"
        f"z(0)={_fmt(z0_hat)}  z'(0)={_fmt(z1_hat)}"
    )
    axes[0].text(0.98, 0.98, text_est, ha="right", va="top",
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.35"))

    # --- True subplot ---
    axes[1].plot(t, z_true_np, linewidth=1.5)
    axes[1].set_title(title_bottom); axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("z (rad)")
    text_true = (
        f"True params ({regime_true}):\n"
        f"γ0={gamma0:.6g}  γ1={gamma1:.6g}\n"
        f"ζ={zeta_true:.6g}  ω={omega_true:.6g}\n"
        f"z(0)={z0:.6g}  z'(0)={z1:.6g}"
    )
    axes[1].text(0.98, 0.98, text_true, ha="right", va="top",
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, boxstyle="round,pad=0.35"))

    if match_ylims:
        ymin = min(np.nanmin(z_est_np), np.nanmin(z_true_np))
        ymax = max(np.nanmax(z_est_np), np.nanmax(z_true_np))
        pad = 0.05 * (ymax - ymin + 1e-9)
        y0, y1 = ymin - pad, ymax + pad
        axes[0].set_ylim(y0, y1); axes[1].set_ylim(y0, y1)

    plt.savefig(save_path, dpi=220); plt.close(fig)
    print(f"[OK] saved -> {save_path}")
