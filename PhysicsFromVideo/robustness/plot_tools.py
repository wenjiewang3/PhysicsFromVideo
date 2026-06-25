import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from textwrap import fill

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


def save_experiment_summary_card(
    first_frame,
    save_path: str,
    gamma0_hat: float,
    gamma1_hat: float,
    fps: int,
    duration_pi: float,
    duration_seconds: float,
    theta0_deg: float,
    dz: float,
    video_path: str,
    setting_case: str | None = None,
    seed: int | None = None,
    noise_level: int | None = None,
    setting_param_lines: list[str] | None = None,
):
    if isinstance(first_frame, torch.Tensor):
        frame_np = first_frame.detach().cpu().numpy()
    else:
        frame_np = np.asarray(first_frame)

    if frame_np.ndim == 3:
        frame_np = np.squeeze(frame_np)
    if frame_np.ndim != 2:
        raise ValueError(f"Expected first_frame to be 2D after squeeze, got shape {frame_np.shape}")

    def _fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "N/A"
        return f"{v:.6g}"

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(15.2, 7.6), constrained_layout=True, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.58])
    ax_img = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[0.88, 1.12], hspace=0.18)
    ax_summary = fig.add_subplot(right_gs[0, 0])
    bottom_gs = right_gs[1, 0].subgridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.18)
    ax_params = fig.add_subplot(bottom_gs[0, 0])
    ax_path = fig.add_subplot(bottom_gs[0, 1])

    ax_img.imshow(frame_np, cmap="gray", vmin=0.0, vmax=1.0)
    ax_img.set_title("First Frame", fontsize=17, pad=12)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_edgecolor("black")

    ax_summary.axis("off")
    ax_params.axis("off")
    ax_path.axis("off")
    header = "Ablation Result Summary"
    if setting_case is not None:
        header += f"\n{setting_case}"

    summary_lines = [
        f"gamma0_hat   {_fmt(gamma0_hat)}",
        f"gamma1_hat   {_fmt(gamma1_hat)}",
        "",
        f"fps          {int(fps)}",
        f"duration     {duration_pi:.2f}pi ({duration_seconds:.4f}s)",
        f"degree       {theta0_deg}",
        f"dz           {dz}",
    ]
    if noise_level is not None:
        summary_lines.append(f"level        {noise_level}")
    if seed is not None:
        summary_lines.append(f"seed         {seed}")

    params_lines = []
    if setting_param_lines:
        params_lines = ["Video Params", *setting_param_lines]

    path_lines = [
        "Video File",
        os.path.basename(video_path),
        "",
        "Directory",
        fill(os.path.dirname(video_path), width=34),
    ]

    ax_summary.text(
        0.02,
        0.98,
        header + "\n\n" + "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=17,
        family="monospace",
        linespacing=1.28,
        transform=ax_summary.transAxes,
    )

    if params_lines:
        ax_params.text(
            0.02,
            0.98,
            "\n".join(params_lines),
            ha="left",
            va="top",
            fontsize=13.5,
            family="monospace",
            linespacing=1.22,
            bbox=dict(facecolor="#f7f7f7", edgecolor="black", alpha=0.95, boxstyle="round,pad=0.6"),
            transform=ax_params.transAxes,
        )

    ax_path.text(
        0.02,
        0.98,
        "\n".join(path_lines),
        ha="left",
        va="top",
        fontsize=13.5,
        family="monospace",
        linespacing=1.22,
        bbox=dict(facecolor="#fbfbfb", edgecolor="black", alpha=0.95, boxstyle="round,pad=0.6"),
        transform=ax_path.transAxes,
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.06, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] saved summary card -> {save_path}")
