import argparse
import os

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from noise_background_gen import theta_series, theta_series_with_derivs


CAMERA_JITTER_LEVELS = {
    1: {
        "jitter_profile": "linear_sine",
        "jitter_amp_x_px": 1,
        "jitter_amp_y_px": 1,
        "jitter_cycles": 1.0,
        "jitter_direction_deg": 18.0,
    },
    2: {
        "jitter_profile": "linear_sine",
        "jitter_amp_x_px": 2,
        "jitter_amp_y_px": 1,
        "jitter_cycles": 1.8,
        "jitter_direction_deg": 18.0,
    },
    3: {
        "jitter_profile": "random_walk",
        "jitter_max_x_px": 1,
        "jitter_max_y_px": 1,
        "jitter_step_choices": [0, 1],
        "jitter_step_prob": 0.38,
        "jitter_momentum": 0.15,
    },
    4: {
        "jitter_profile": "random_walk",
        "jitter_max_x_px": 2,
        "jitter_max_y_px": 2,
        "jitter_step_choices": [0, 1],
        "jitter_step_prob": 0.55,
        "jitter_momentum": 0.35,
    },
    5: {
        "jitter_profile": "random_walk",
        "jitter_max_x_px": 3,
        "jitter_max_y_px": 2,
        "jitter_step_choices": [0, 1, 2],
        "jitter_step_prob": 0.72,
        "jitter_momentum": 0.45,
    },
    6: {
        "jitter_profile": "staged_horizontal_sweep",
        "jitter_stage_targets_px": [14, 30],
        "jitter_sweep_frac": 0.10,
    },
    7: {
        "jitter_profile": "vertical_shake",
        "jitter_amp_y_px": 8,
        "jitter_cycles": 7.0,
    },
}


def compute_view_bounds(L, pivot_y_frac, zoom):
    margin = 0.15 * L
    span = 2.0 * (L + margin) * zoom
    half = span * 0.5
    p = float(np.clip(pivot_y_frac, 0.0, 1.0))
    xmin, xmax = -half, half
    ymin = -p * span
    ymax = ymin + span
    return xmin, xmax, ymin, ymax


def render_clean_frame(
    theta,
    L=1.0,
    size=64,
    rod_px=8.0,
    bob_ratio=0.1,
    trail_xs=None,
    trail_ys=None,
    pivot_y_frac=0.5,
    zoom=1.0,
):
    xmin, xmax, ymin, ymax = compute_view_bounds(L, pivot_y_frac, zoom)
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    bob_r = bob_ratio * L

    base_res = 512
    fig = plt.figure(
        figsize=(base_res / 100, base_res / 100),
        dpi=size / base_res * 100,
        facecolor="white",
    )
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_autoscale_on(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if trail_xs is not None and trail_ys is not None and len(trail_xs) > 1:
        ax.plot(
            trail_xs,
            trail_ys,
            color="black",
            alpha=0.6,
            linewidth=max(1.0, rod_px * 0.25),
            zorder=1,
        )

    ax.plot([0, x], [0, y], color="black", linewidth=rod_px, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((x, y), radius=bob_r, facecolor="black", edgecolor="black", zorder=3))

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


def translate_frame(frame, shift_x_px, shift_y_px, pad_value=255):
    h, w = frame.shape[:2]
    out = np.full_like(frame, pad_value)

    dst_x0 = max(0, int(shift_x_px))
    dst_x1 = min(w, w + int(shift_x_px))
    dst_y0 = max(0, int(shift_y_px))
    dst_y1 = min(h, h + int(shift_y_px))

    src_x0 = max(0, -int(shift_x_px))
    src_x1 = src_x0 + max(0, dst_x1 - dst_x0)
    src_y0 = max(0, -int(shift_y_px))
    src_y1 = src_y0 + max(0, dst_y1 - dst_y0)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def build_linear_jitter_offsets(num_frames, amp_x_px, amp_y_px, cycles, direction_deg, jitter_seed):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    rng = np.random.default_rng(jitter_seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.deg2rad(direction_deg)
    axis_x = np.cos(direction)
    axis_y = np.sin(direction)

    idx = np.arange(num_frames, dtype=np.float64)
    denom = max(num_frames - 1, 1)
    wave = np.sin(2.0 * np.pi * float(cycles) * idx / denom + phase)

    shift_x = np.rint(float(amp_x_px) * axis_x * wave).astype(np.int32)
    shift_y = np.rint(float(amp_y_px) * axis_y * wave).astype(np.int32)
    return shift_x, shift_y


def sample_random_step(rng, step_choices, step_prob):
    if rng.random() > step_prob:
        return 0, 0

    step_size = int(rng.choice(step_choices))
    if step_size <= 0:
        return 0, 0

    direction = int(rng.integers(0, 4))
    if direction == 0:
        return step_size, 0
    if direction == 1:
        return -step_size, 0
    if direction == 2:
        return 0, step_size
    return 0, -step_size


def build_random_walk_jitter_offsets(
    num_frames,
    max_x_px,
    max_y_px,
    step_choices,
    step_prob,
    momentum,
    jitter_seed,
):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    rng = np.random.default_rng(jitter_seed)
    shift_x = np.zeros(num_frames, dtype=np.int32)
    shift_y = np.zeros(num_frames, dtype=np.int32)
    last_dx, last_dy = 0, 0

    for t in range(1, num_frames):
        if rng.random() < momentum and (last_dx != 0 or last_dy != 0):
            dx, dy = last_dx, last_dy
        else:
            dx, dy = sample_random_step(rng, step_choices, step_prob)

        next_x = int(np.clip(shift_x[t - 1] + dx, -int(max_x_px), int(max_x_px)))
        next_y = int(np.clip(shift_y[t - 1] + dy, -int(max_y_px), int(max_y_px)))

        last_dx = next_x - shift_x[t - 1]
        last_dy = next_y - shift_y[t - 1]
        shift_x[t] = next_x
        shift_y[t] = next_y

    return shift_x, shift_y


def build_one_way_horizontal_offsets(num_frames, max_x_px):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    shift_x = np.rint(np.linspace(0.0, float(max_x_px), num_frames)).astype(np.int32)
    shift_y = np.zeros(num_frames, dtype=np.int32)
    return shift_x, shift_y


def build_staged_horizontal_sweep_offsets(num_frames, stage_targets_px, sweep_frac):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    shift_x = np.zeros(num_frames, dtype=np.int32)
    shift_y = np.zeros(num_frames, dtype=np.int32)
    if not stage_targets_px:
        return shift_x, shift_y

    stage_targets = [int(v) for v in stage_targets_px]
    sweep_len = max(2, int(round(num_frames * float(sweep_frac))))
    anchor_points = np.linspace(0.28, 0.72, len(stage_targets))
    prev_value = 0

    for stage_idx, target in enumerate(stage_targets):
        center = int(round(anchor_points[stage_idx] * max(num_frames - 1, 1)))
        start = max(0, center - sweep_len // 2)
        end = min(num_frames, start + sweep_len)
        start = max(0, end - sweep_len)

        if start > 0:
            shift_x[:start] = prev_value

        ramp = np.rint(np.linspace(prev_value, target, max(end - start, 2))).astype(np.int32)
        shift_x[start:end] = ramp[: end - start]
        if end < num_frames:
            shift_x[end:] = target
        prev_value = target

    return shift_x, shift_y


def build_vertical_shake_offsets(num_frames, amp_y_px, cycles, jitter_seed):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    rng = np.random.default_rng(jitter_seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    idx = np.arange(num_frames, dtype=np.float64)
    denom = max(num_frames - 1, 1)
    wave = np.sin(2.0 * np.pi * float(cycles) * idx / denom + phase)

    shift_x = np.zeros(num_frames, dtype=np.int32)
    shift_y = np.rint(float(amp_y_px) * wave).astype(np.int32)
    return shift_x, shift_y


def build_camera_jitter_offsets(num_frames, level_cfg, jitter_seed):
    profile = level_cfg["jitter_profile"]
    if profile == "linear_sine":
        return build_linear_jitter_offsets(
            num_frames=num_frames,
            amp_x_px=level_cfg["jitter_amp_x_px"],
            amp_y_px=level_cfg["jitter_amp_y_px"],
            cycles=level_cfg["jitter_cycles"],
            direction_deg=level_cfg["jitter_direction_deg"],
            jitter_seed=jitter_seed,
        )
    if profile == "random_walk":
        return build_random_walk_jitter_offsets(
            num_frames=num_frames,
            max_x_px=level_cfg["jitter_max_x_px"],
            max_y_px=level_cfg["jitter_max_y_px"],
            step_choices=level_cfg["jitter_step_choices"],
            step_prob=level_cfg["jitter_step_prob"],
            momentum=level_cfg["jitter_momentum"],
            jitter_seed=jitter_seed,
        )
    if profile == "one_way_horizontal":
        return build_one_way_horizontal_offsets(
            num_frames=num_frames,
            max_x_px=level_cfg["jitter_max_x_px"],
        )
    if profile == "staged_horizontal_sweep":
        return build_staged_horizontal_sweep_offsets(
            num_frames=num_frames,
            stage_targets_px=level_cfg["jitter_stage_targets_px"],
            sweep_frac=level_cfg.get("jitter_sweep_frac", 0.10),
        )
    if profile == "vertical_shake":
        return build_vertical_shake_offsets(
            num_frames=num_frames,
            amp_y_px=level_cfg["jitter_amp_y_px"],
            cycles=level_cfg["jitter_cycles"],
            jitter_seed=jitter_seed,
        )
    raise ValueError(f"Unsupported jitter profile: {profile}")


def render_bw_with_camera_jitter(
    T,
    TH,
    L=1.0,
    out="pendulum.mp4",
    fps=60,
    size=64,
    rod_px=8.0,
    bob_ratio=0.1,
    trail=0,
    pivot_y_frac=0.5,
    zoom=1.0,
    jitter_level=1,
    jitter_seed=42,
):
    if jitter_level not in CAMERA_JITTER_LEVELS:
        raise ValueError(f"Unsupported jitter level: {jitter_level}")

    level_cfg = CAMERA_JITTER_LEVELS[jitter_level]
    shift_x, shift_y = build_camera_jitter_offsets(
        num_frames=len(T),
        level_cfg=level_cfg,
        jitter_seed=jitter_seed,
    )

    xs = L * np.sin(TH)
    ys = -L * np.cos(TH)

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for i in range(len(T)):
            trail_xs = trail_ys = None
            if trail > 0:
                j0 = max(0, i - trail)
                trail_xs = xs[j0 : i + 1]
                trail_ys = ys[j0 : i + 1]

            clean_frame = render_clean_frame(
                theta=TH[i],
                L=L,
                size=size,
                rod_px=rod_px,
                bob_ratio=bob_ratio,
                trail_xs=trail_xs,
                trail_ys=trail_ys,
                pivot_y_frac=pivot_y_frac,
                zoom=zoom,
            )
            jittered_frame = translate_frame(clean_frame, shift_x[i], shift_y[i], pad_value=255)
            writer.append_data(jittered_frame)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


def main(args):
    theta0 = np.deg2rad(args.theta0_deg)
    dtheta0 = np.deg2rad(args.dtheta0_deg_s)

    T, TH = theta_series(
        args.gamma0,
        args.gamma1,
        duration=args.duration,
        fps=args.fps,
        theta0=theta0,
        dtheta0=dtheta0,
    )

    render_bw_with_camera_jitter(
        T,
        TH,
        L=args.L,
        out=args.out,
        fps=args.fps,
        size=args.size,
        rod_px=args.rod_px,
        bob_ratio=args.bob_ratio,
        trail=args.trail,
        pivot_y_frac=args.pivot_y_frac,
        zoom=args.zoom,
        jitter_level=args.jitter_level,
        jitter_seed=args.jitter_seed,
    )

    theta_series_with_derivs(
        args.gamma0,
        args.gamma1,
        duration=args.duration,
        fps=args.fps,
        theta0=theta0,
        dtheta0=dtheta0,
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="B/W pendulum with camera jitter: whole-frame translation with white padding"
    )
    ap.add_argument("--gamma0", type=float, default=4.0, help="gamma0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=5.0, help="gamma1 (damping)")
    ap.add_argument("--theta0_deg", type=float, default=90.0, help="initial angle in degrees")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0.0, help="initial angular velocity in degrees per second")
    ap.add_argument("--duration", type=float, default=10.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=60, help="frames per second")
    ap.add_argument("--L", type=float, default=1.0, help="visual rod length")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--rod_px", type=float, default=8.0, help="rod line width (px)")
    ap.add_argument("--bob_ratio", type=float, default=0.1, help="bob radius = ratio * L")
    ap.add_argument("--trail", type=int, default=0, help="trail length in frames (0 off)")
    ap.add_argument("--pivot_y_frac", type=float, default=0.5, help="pivot vertical position in frame")
    ap.add_argument("--zoom", type=float, default=1.0, help="camera zoom")
    ap.add_argument("--jitter_level", type=int, default=1, help="camera jitter level in [1, 7]")
    ap.add_argument("--jitter_seed", type=int, default=42, help="random seed for jitter generation")
    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_10s.mp4", help="output MP4 path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_folder = "ablation_data/camera_jitter_underdamped"
    os.makedirs(save_folder, exist_ok=True)
    for args.theta0_deg in [60]:
        for args.dtheta0_deg_s in [0]:
            for args.duration in [2.5]:
                for args.fps in [20]:
                    duration_pi = args.duration
                    for level in sorted(CAMERA_JITTER_LEVELS.keys()):
                        args.gamma0 = 4.0016
                        args.gamma1 = 0.08
                        args.jitter_level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
