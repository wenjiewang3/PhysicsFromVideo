

import argparse
import numpy as np
import imageio
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def theta_series(gamma0: float,
                 gamma1: float,
                 duration: float = 10.0,
                 fps: int = 60,
                 theta0: float = 1.0,
                 dtheta0: float = 0.0):

    n = int(round(duration * fps))
    t = np.linspace(0.0, duration, n, endpoint=False)

    D = gamma1**2 - 4.0 * gamma0
    eps = 1e-12

    if D < -eps:
        zeta = gamma1 / 2.0
        omega = np.sqrt(gamma0 - zeta**2)
        A = theta0
        B = (dtheta0 + zeta * theta0) / (omega if omega != 0 else 1.0)
        theta = np.exp(-zeta * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))

    elif D > eps:
        s = np.sqrt(D)
        r1 = (-gamma1 + s) / 2.0
        r2 = (-gamma1 - s) / 2.0
        denom = (r1 - r2) if (r1 - r2) != 0 else 1.0
        C1 = (dtheta0 - r2 * theta0) / denom
        C2 = (r1 * theta0 - dtheta0) / denom
        theta = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

    else:
        zeta = gamma1 / 2.0
        A = theta0
        B = dtheta0 + zeta * theta0
        theta = (A + B * t) * np.exp(-zeta * t)

    return t, theta


def render_bw(T, TH, L=1.0, out="pendulum.mp4", fps=60,
              size=512, rod_px=6.0, bob_ratio=0.06, trail=0,
              pivot_y_frac=0.5, zoom=1.0):

    xs = L * np.sin(TH)
    ys = -L * np.cos(TH)


    margin = 0.15 * L
    span = 2.0 * (L + margin) * zoom
    half = span * 0.5
    

    p = float(np.clip(pivot_y_frac, 0.0, 1.0))
    xmin, xmax = -half, half
    ymin = -p * span
    ymax = ymin + span

    bob_r = bob_ratio * L
    color = "black"

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for i in range(len(T)):
            base_res = 512
            fig = plt.figure(figsize=(base_res/100, base_res/100), dpi=size/base_res*100, facecolor="white")
            # 全屏绘制，无坐标轴
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)

            
            ax.set_aspect('equal', adjustable='box')
            ax.set_autoscale_on(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            x, y = xs[i], ys[i]
            ax.plot([0, x], [0, y], color=color, linewidth=rod_px, solid_capstyle="round")
            ax.add_patch(Circle((x, y), radius=bob_r, facecolor=color, edgecolor=color))

            if trail > 0:
                j0 = max(0, i - trail)
                ax.plot(xs[j0:i+1], ys[j0:i+1], color=color, alpha=0.6,
                        linewidth=max(1.0, rod_px * 0.25))

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frame = rgba[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()
    print(f"[OK] saved -> {out}")



def theta_series_with_derivs(gamma0: float, gamma1: float,
                             duration: float = 10.0, fps: int = 60,
                             theta0: float = 1.0, dtheta0: float = 0.0):
    n = int(round(duration * fps))
    t = np.linspace(0.0, duration, n, endpoint=False)

    D = gamma1**2 - 4.0*gamma0
    eps = 1e-12

    if D < -eps:
        zeta = gamma1/2.0
        omega = np.sqrt(max(gamma0 - zeta**2, 0.0))
        A = theta0
        B = (dtheta0 + zeta*theta0) / (omega if omega != 0 else 1.0)
        exp = np.exp(-zeta*t)
        cos = np.cos(omega*t); sin = np.sin(omega*t)
        th  = exp * (A*cos + B*sin)
        thd = exp * ( -zeta*(A*cos + B*sin) + (-A*omega*sin + B*omega*cos) )
    elif D > eps:
        s = np.sqrt(D)
        r1 = (-gamma1 + s)/2.0; r2 = (-gamma1 - s)/2.0
        denom = (r1 - r2) if (r1 - r2) != 0 else 1.0
        C1 = (dtheta0 - r2*theta0)/denom
        C2 = (r1*theta0 - dtheta0)/denom
        e1 = np.exp(r1*t); e2 = np.exp(r2*t)
        th  = C1*e1 + C2*e2
        thd = C1*r1*e1 + C2*r2*e2
    else:
        zeta = gamma1/2.0
        A = theta0; B = dtheta0 + zeta*theta0
        exp = np.exp(-zeta*t)
        th  = (A + B*t) * exp
        thd = (B - zeta*(A + B*t)) * exp

    thdd = -gamma1*thd - gamma0*th
    return t, th, thd, thdd


def main(args):
    theta0 = np.deg2rad(args.theta0_deg)
    dtheta0 = np.deg2rad(args.dtheta0_deg_s)

    T, TH = theta_series(args.gamma0, args.gamma1,
                         duration=args.duration, fps=args.fps,
                         theta0=theta0, dtheta0=dtheta0)

    render_bw(T, TH, L=args.L, out=args.out, fps=args.fps, size=args.size,
              rod_px=args.rod_px, bob_ratio=args.bob_ratio, trail=args.trail,
              pivot_y_frac=args.pivot_y_frac, zoom=args.zoom)

    T, TH, THD, THDD = theta_series_with_derivs(args.gamma0, args.gamma1,
                                                duration=args.duration, fps=args.fps,
                                                theta0=theta0, dtheta0=dtheta0)
    # print("z'(0) =", THD[0], " z''(0) =", THDD[0])


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="B/W pendulum: θ'' + γ1 θ' + γ0 θ = 0 (custom θ0 in degrees)")

    ap.add_argument("--gamma0", type=float, default=4.0, help="γ0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=5.0, help="γ1 (damping)")



    ap.add_argument("--theta0_deg", type=float, default=90,
                    help="initial angle in degrees (θ(0))")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0.0,
                    help="initial angular velocity in degrees per second (θ'(0))")

    ap.add_argument("--duration", type=float, default=10.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=60, help="frames per second")
    ap.add_argument("--L", type=float, default=1.0, help="visual rod length")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--rod_px", type=float, default=8.0, help="rod line width (px)")
    ap.add_argument("--bob_ratio", type=float, default=0.1, help="bob radius = ratio * L")
    ap.add_argument("--trail", type=int, default=0, help="trail length in frames (0 off)")
    ap.add_argument("--pivot_y_frac", type=float, default=0.5,
                    help="pivot vertical position in frame: 0.0=bottom, 0.5=center, 1.0=top")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="camera zoom (1.0 gives span = 2*(L+margin))")
    
    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_10s.mp4",
                    help="output MP4 path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_folder = "pendulum_dataset/undamped"
    os.makedirs(save_folder, exist_ok=True)
    for args.theta0_deg in [90]:
        for args.dtheta0_deg_s in [0, -100, 100]:
            for  args.duration in [1/2]:
                for args.fps in [6]:
                    args.gamma0 = 4.0
                    args.gamma1 = 0.0
                    args.out = f"{save_folder}/{args.duration:.1f}pi_{int(args.fps)}fps_degree{args.theta0_deg}_degs{args.dtheta0_deg_s}.mp4"
                    args.duration = args.duration * np.pi
                    main(args)

    # save_folder = "pendulum/underdamped"
    # os.makedirs(save_folder, exist_ok=True)
    # for args.theta0_deg in [90]:
    #     for args.dtheta0_deg_s in [0]:
    #         for  args.duration in [1/2]:
    #             for args.fps in [6]:
    #                 args.gamma0 = 4.0016
    #                 args.gamma1 = 0.08
    #                 args.out = f"{save_folder}/{args.duration:.1f}pi_{int(args.fps)}fps_degree{args.theta0_deg}_degs{args.dtheta0_deg_s}.mp4"
    #                 args.duration = args.duration * np.pi
    #                 main(args)
    #
    # save_folder = "pendulum/critical"
    # os.makedirs(save_folder, exist_ok=True)
    # for args.theta0_deg in [90]:
    #     for args.dtheta0_deg_s in [0]:
    #         for  args.duration in [1/2]:
    #             for args.fps in [6]:
    #                 args.gamma0 = 4.0
    #                 args.gamma1 = 4.0
    #                 args.out = f"pendulum/critical/{args.duration:.1f}pi_{int(args.fps)}fps_degree{args.theta0_deg}_degs{args.dtheta0_deg_s}.mp4"
    #                 args.duration = args.duration * np.pi
    #                 main(args)
    #
    # save_folder = "pendulum/overdamped"
    # os.makedirs(save_folder, exist_ok=True)
    # for args.theta0_deg in [90]:
    #     for args.dtheta0_deg_s in [0]:
    #         for  args.duration in [1/2]:
    #             for args.fps in [6]:
    #                 args.gamma0 = 4.0
    #                 args.gamma1 = 5.0
    #                 args.out = f"{save_folder}/{args.duration:.1f}pi_{int(args.fps)}fps_degree{args.theta0_deg}_degs{args.dtheta0_deg_s}.mp4"
    #                 args.duration = args.duration * np.pi
    #                 main(args)