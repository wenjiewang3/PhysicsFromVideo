import argparse
import os

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from noise_background_gen import theta_series, theta_series_with_derivs
from setting_registry import get_setting_level_params, get_setting_levels, get_setting_save_folder


def compute_view_bounds(L, pivot_y_frac, zoom):
    margin = 0.15 * L
    span = 2.0 * (L + margin) * zoom
    half = span * 0.5
    p = float(np.clip(pivot_y_frac, 0.0, 1.0))
    xmin, xmax = -half, half
    ymin = -p * span
    ymax = ymin + span
    return xmin, xmax, ymin, ymax, span


def rects_overlap(rect_a, rect_b):
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def subtract_rect(region, blocker):
    rx0, ry0, rx1, ry1 = region
    bx0, by0, bx1, by1 = blocker

    ix0 = max(rx0, bx0)
    iy0 = max(ry0, by0)
    ix1 = min(rx1, bx1)
    iy1 = min(ry1, by1)

    if ix0 >= ix1 or iy0 >= iy1:
        return [region]

    pieces = []
    if ix0 > rx0:
        pieces.append((rx0, ry0, ix0, ry1))
    if ix1 < rx1:
        pieces.append((ix1, ry0, rx1, ry1))
    if iy0 > ry0:
        pieces.append((ix0, ry0, ix1, iy0))
    if iy1 < ry1:
        pieces.append((ix0, iy1, ix1, ry1))
    return pieces


def get_allowed_regions(bounds_rect, exclusion_rects):
    regions = [bounds_rect]
    for blocker in exclusion_rects:
        next_regions = []
        for region in regions:
            next_regions.extend(subtract_rect(region, blocker))
        regions = next_regions

    filtered = []
    for x0, y0, x1, y1 in regions:
        if x1 > x0 and y1 > y0:
            filtered.append((x0, y0, x1, y1))
    return filtered


def intersect_rects(rect_a, rect_b):
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return None
    return (ix0, iy0, ix1, iy1)


def build_quadrant_cells(bounds_rect):
    x0, y0, x1, y1 = bounds_rect
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    return [
        (x0, ym, xm, y1),
        (xm, ym, x1, y1),
        (x0, y0, xm, ym),
        (xm, y0, x1, ym),
    ]


def build_motion_exclusion_rect(
    xs,
    ys,
    span,
    size_px,
    bob_r,
    rod_px,
    max_square_px,
    motion_amp_x_px,
    motion_amp_y_px,
    motion_guard_px,
):
    px_to_world = span / float(size_px)
    pad_world = (
        bob_r
        + 0.5 * max_square_px * px_to_world
        + max(rod_px, 1.0) * px_to_world
        + motion_guard_px * px_to_world
    )
    amp_x_world = motion_amp_x_px * px_to_world
    amp_y_world = motion_amp_y_px * px_to_world
    x_min = min(0.0, float(np.min(xs))) - pad_world - amp_x_world
    x_max = max(0.0, float(np.max(xs))) + pad_world + amp_x_world
    y_min = min(0.0, float(np.min(ys))) - pad_world - amp_y_world
    y_max = max(0.0, float(np.max(ys))) + pad_world + amp_y_world
    return (x_min, y_min, x_max, y_max)


def sample_square_noise_with_motion(
    rng,
    count,
    xmin,
    xmax,
    ymin,
    ymax,
    size_px,
    min_size_px,
    max_size_px,
    gray_min,
    gray_max,
    motion_amp_x_px,
    motion_amp_y_px,
    exclusion_rects,
    motion_guard_px,
    placement_mode="quadrants",
    allow_wall_collision=False,
):
    if count <= 0:
        return []

    min_size_px = max(1, int(min_size_px))
    max_size_px = max(min_size_px, int(max_size_px))
    gray_min = int(np.clip(gray_min, 0, 255))
    gray_max = int(np.clip(gray_max, gray_min, 255))

    span = min(xmax - xmin, ymax - ymin)
    px_to_world = span / float(size_px)
    amp_x_world = motion_amp_x_px * px_to_world
    amp_y_world = motion_amp_y_px * px_to_world
    guard_world = motion_guard_px * px_to_world

    if allow_wall_collision:
        bounds_rect = (xmin, ymin, xmax, ymax)
    else:
        bounds_rect = (
            xmin + amp_x_world,
            ymin + amp_y_world,
            xmax - amp_x_world,
            ymax - amp_y_world,
        )
    if bounds_rect[2] <= bounds_rect[0] or bounds_rect[3] <= bounds_rect[1]:
        raise ValueError("Motion amplitude is too large for the current frame bounds.")

    allowed_regions = get_allowed_regions(bounds_rect, exclusion_rects)
    if not allowed_regions:
        raise RuntimeError("No valid clutter region remains after excluding the pendulum motion corridor.")

    region_usage = np.zeros(len(allowed_regions), dtype=np.int32)
    quadrant_order = list(rng.permutation(len(build_quadrant_cells(bounds_rect))))

    def get_regions_for_patch(patch_idx):
        if placement_mode != "quadrants":
            return allowed_regions, list(range(len(allowed_regions)))

        cells = build_quadrant_cells(bounds_rect)
        target_cell = cells[quadrant_order[patch_idx % len(cells)]]
        cell_regions = []
        cell_indices = []
        for region_idx, region in enumerate(allowed_regions):
            inter = intersect_rects(region, target_cell)
            if inter is not None:
                cell_regions.append(inter)
                cell_indices.append(region_idx)

        if cell_regions:
            return cell_regions, cell_indices
        return allowed_regions, list(range(len(allowed_regions)))

    def largest_square_px_that_fits():
        max_side_world = 0.0
        for x0, y0, x1, y1 in allowed_regions:
            max_side_world = max(max_side_world, min(x1 - x0, y1 - y0))
        return max(0, int(np.floor(max_side_world / px_to_world)))

    squares = []
    for patch_idx in range(count):
        target_side_px = int(rng.integers(min_size_px, max_size_px + 1))
        gray = int(rng.integers(gray_min, gray_max + 1))
        candidate_base_regions, candidate_base_indices = get_regions_for_patch(patch_idx)

        side_px = None
        valid_regions = []
        valid_weights = []
        valid_region_indices = []
        for candidate_side_px in range(target_side_px, min_size_px - 1, -1):
            candidate_side_world = candidate_side_px * px_to_world
            candidate_regions = []
            candidate_weights = []
            candidate_indices = []
            for local_idx, (x0, y0, x1, y1) in enumerate(candidate_base_regions):
                region_idx = candidate_base_indices[local_idx]
                free_w = (x1 - x0) - candidate_side_world
                free_h = (y1 - y0) - candidate_side_world
                if free_w > 0 and free_h > 0:
                    candidate_regions.append((x0, y0, x1, y1))
                    usage_penalty = 1.0 / (1.0 + float(region_usage[region_idx])) ** 2
                    candidate_weights.append(free_w * free_h * usage_penalty)
                    candidate_indices.append(region_idx)

            if candidate_regions:
                side_px = candidate_side_px
                valid_regions = candidate_regions
                valid_weights = candidate_weights
                valid_region_indices = candidate_indices
                break

        if side_px is None:
            fallback_max_px = largest_square_px_that_fits()
            if fallback_max_px <= 0:
                print("[WARN] no valid region remains for additional moving clutter patches; using fewer patches.")
                break

            fallback_side_px = min(target_side_px, fallback_max_px)
            candidate_side_world = fallback_side_px * px_to_world
            candidate_regions = []
            candidate_weights = []
            candidate_indices = []
            for local_idx, (x0, y0, x1, y1) in enumerate(candidate_base_regions):
                region_idx = candidate_base_indices[local_idx]
                free_w = (x1 - x0) - candidate_side_world
                free_h = (y1 - y0) - candidate_side_world
                if free_w > 0 and free_h > 0:
                    candidate_regions.append((x0, y0, x1, y1))
                    usage_penalty = 1.0 / (1.0 + float(region_usage[region_idx])) ** 2
                    candidate_weights.append(free_w * free_h * usage_penalty)
                    candidate_indices.append(region_idx)

            if not candidate_regions:
                print("[WARN] fallback placement failed for moving clutter; using fewer patches.")
                break

            side_px = fallback_side_px
            valid_regions = candidate_regions
            valid_weights = candidate_weights
            valid_region_indices = candidate_indices

        probs = np.asarray(valid_weights, dtype=np.float64)
        probs = probs / probs.sum()
        region_idx = int(rng.choice(len(valid_regions), p=probs))
        rx0, ry0, rx1, ry1 = valid_regions[region_idx]
        chosen_region_idx = valid_region_indices[region_idx]
        side_world = side_px * px_to_world
        x0 = float(rng.uniform(rx0, rx1 - side_world))
        y0 = float(rng.uniform(ry0, ry1 - side_world))
        squares.append((x0, y0, side_world, gray))
        region_usage[chosen_region_idx] += 1

    if len(squares) < count:
        print(f"[WARN] placed {len(squares)} / {count} moving clutter patches under current motion constraints.")

    return squares


def build_global_motion_offsets(
    num_frames,
    motion_amp_x_px,
    motion_amp_y_px,
    motion_cycles_x,
    motion_cycles_y,
    motion_harmonic_ratio,
    motion_seed,
):
    if num_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    rng = np.random.default_rng(motion_seed)
    phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_y = float(rng.uniform(0.0, 2.0 * np.pi))

    idx = np.arange(num_frames, dtype=np.float64)
    denom = max(num_frames - 1, 1)

    theta_x = 2.0 * np.pi * float(motion_cycles_x) * idx / denom + phase_x
    theta_y = 2.0 * np.pi * float(motion_cycles_y) * idx / denom + phase_y

    x_wave = np.sin(theta_x)
    y_wave = np.cos(theta_y)

    harmonic_ratio = float(motion_harmonic_ratio)
    if harmonic_ratio > 0.0:
        x_wave = x_wave + harmonic_ratio * np.sin(2.0 * theta_x + 0.5 * phase_x)
        y_wave = y_wave + harmonic_ratio * np.cos(2.0 * theta_y + 0.5 * phase_y)
        x_wave = x_wave / (1.0 + harmonic_ratio)
        y_wave = y_wave / (1.0 + harmonic_ratio)

    shift_x = np.rint(float(motion_amp_x_px) * x_wave).astype(np.int32)
    shift_y = np.rint(float(motion_amp_y_px) * y_wave).astype(np.int32)
    return shift_x, shift_y


def build_independent_motion_offsets(
    squares,
    num_frames,
    xmin,
    xmax,
    ymin,
    ymax,
    px_to_world,
    motion_speed_px,
    motion_seed,
):
    num_squares = len(squares)
    offsets_x = np.zeros((num_frames, num_squares), dtype=np.float32)
    offsets_y = np.zeros((num_frames, num_squares), dtype=np.float32)
    if num_frames <= 0 or num_squares == 0:
        return offsets_x, offsets_y

    rng = np.random.default_rng(motion_seed)
    base_speed_world = float(motion_speed_px) * px_to_world

    current_x = np.asarray([sq[0] for sq in squares], dtype=np.float32)
    current_y = np.asarray([sq[1] for sq in squares], dtype=np.float32)
    side_world = np.asarray([sq[2] for sq in squares], dtype=np.float32)

    min_x = np.full(num_squares, xmin, dtype=np.float32)
    max_x = np.asarray([xmax - s for s in side_world], dtype=np.float32)
    min_y = np.full(num_squares, ymin, dtype=np.float32)
    max_y = np.asarray([ymax - s for s in side_world], dtype=np.float32)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=num_squares)
    speed_scale = rng.uniform(0.85, 1.15, size=num_squares)
    velocity_x = np.cos(angles) * base_speed_world * speed_scale
    velocity_y = np.sin(angles) * base_speed_world * speed_scale

    for t in range(num_frames):
        offsets_x[t] = current_x - np.asarray([sq[0] for sq in squares], dtype=np.float32)
        offsets_y[t] = current_y - np.asarray([sq[1] for sq in squares], dtype=np.float32)
        if t == num_frames - 1:
            break

        next_x = current_x + velocity_x
        next_y = current_y + velocity_y

        hit_left = next_x < min_x
        hit_right = next_x > max_x
        velocity_x[hit_left | hit_right] *= -1.0
        next_x = np.clip(next_x, min_x, max_x)

        hit_bottom = next_y < min_y
        hit_top = next_y > max_y
        velocity_y[hit_bottom | hit_top] *= -1.0
        next_y = np.clip(next_y, min_y, max_y)

        current_x = next_x
        current_y = next_y

    return offsets_x, offsets_y


def draw_square_noise(ax, squares, offset_x_world=0.0, offset_y_world=0.0):
    if np.isscalar(offset_x_world):
        offset_x_values = [float(offset_x_world)] * len(squares)
    else:
        offset_x_values = offset_x_world

    if np.isscalar(offset_y_world):
        offset_y_values = [float(offset_y_world)] * len(squares)
    else:
        offset_y_values = offset_y_world

    for (x0, y0, side, gray), dx, dy in zip(squares, offset_x_values, offset_y_values):
        ax.add_patch(
            Rectangle(
                (x0 + float(dx), y0 + float(dy)),
                side,
                side,
                facecolor=str(gray / 255.0),
                edgecolor="none",
                zorder=0,
            )
        )


def render_bw_with_moving_clutter(
    T,
    TH,
    L=1.0,
    out="pendulum.mp4",
    fps=60,
    size=512,
    rod_px=6.0,
    bob_ratio=0.06,
    trail=0,
    pivot_y_frac=0.5,
    zoom=1.0,
    noise_count=8,
    noise_min_size_px=4,
    noise_max_size_px=10,
    noise_gray_min=0,
    noise_gray_max=0,
    noise_seed=42,
    motion_amp_x_px=2,
    motion_amp_y_px=1,
    motion_cycles_x=1.0,
    motion_cycles_y=1.0,
    motion_harmonic_ratio=0.0,
    motion_speed_px=0.0,
    motion_profile="global_sinusoid",
    motion_guard_px=4,
    allow_wall_collision=False,
    allow_pendulum_overlap=False,
    placement_mode="quadrants",
):
    xs = L * np.sin(TH)
    ys = -L * np.cos(TH)
    xmin, xmax, ymin, ymax, span = compute_view_bounds(L, pivot_y_frac, zoom)

    bob_r = bob_ratio * L
    color = "black"
    rng = np.random.default_rng(noise_seed)

    exclusion_rects = []
    if not allow_pendulum_overlap:
        exclusion_rects.append(
            build_motion_exclusion_rect(
                xs=xs,
                ys=ys,
                span=span,
                size_px=size,
                bob_r=bob_r,
                rod_px=rod_px,
                max_square_px=noise_max_size_px,
                motion_amp_x_px=motion_amp_x_px,
                motion_amp_y_px=motion_amp_y_px,
                motion_guard_px=motion_guard_px,
            )
        )

    squares = sample_square_noise_with_motion(
        rng=rng,
        count=noise_count,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        size_px=size,
        min_size_px=noise_min_size_px,
        max_size_px=noise_max_size_px,
        gray_min=noise_gray_min,
        gray_max=noise_gray_max,
        motion_amp_x_px=motion_amp_x_px,
        motion_amp_y_px=motion_amp_y_px,
        exclusion_rects=exclusion_rects,
        motion_guard_px=motion_guard_px,
        placement_mode=placement_mode,
        allow_wall_collision=allow_wall_collision,
    )

    px_to_world = span / float(size)
    if motion_profile == "independent_bounce":
        offset_x_world, offset_y_world = build_independent_motion_offsets(
            squares=squares,
            num_frames=len(T),
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            px_to_world=px_to_world,
            motion_speed_px=motion_speed_px,
            motion_seed=noise_seed + 17,
        )
    else:
        offset_x_px, offset_y_px = build_global_motion_offsets(
            num_frames=len(T),
            motion_amp_x_px=motion_amp_x_px,
            motion_amp_y_px=motion_amp_y_px,
            motion_cycles_x=motion_cycles_x,
            motion_cycles_y=motion_cycles_y,
            motion_harmonic_ratio=motion_harmonic_ratio,
            motion_seed=noise_seed + 17,
        )
        offset_x_world = offset_x_px.astype(np.float32) * px_to_world
        offset_y_world = offset_y_px.astype(np.float32) * px_to_world

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8)
    try:
        for i in range(len(T)):
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

            draw_square_noise(
                ax,
                squares,
                offset_x_world=offset_x_world[i],
                offset_y_world=offset_y_world[i],
            )

            x, y = xs[i], ys[i]
            ax.plot([0, x], [0, y], color=color, linewidth=rod_px, solid_capstyle="round", zorder=2)
            ax.add_patch(Circle((x, y), radius=bob_r, facecolor=color, edgecolor=color, zorder=3))

            if trail > 0:
                j0 = max(0, i - trail)
                ax.plot(
                    xs[j0 : i + 1],
                    ys[j0 : i + 1],
                    color=color,
                    alpha=0.6,
                    linewidth=max(1.0, rod_px * 0.25),
                    zorder=1,
                )

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            frame = rgba[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()

    print(f"[OK] saved -> {out}")


def apply_moving_level(args, level):
    cfg = get_setting_level_params(args.setting_case, level)
    args.noise_count = cfg["noise_count"]
    args.noise_min_size_px = cfg["noise_min_size_px"]
    args.noise_max_size_px = cfg["noise_max_size_px"]
    args.noise_gray_min = cfg["noise_gray_min"]
    args.noise_gray_max = cfg["noise_gray_max"]
    args.motion_amp_x_px = cfg.get("motion_amp_x_px", 0.0)
    args.motion_amp_y_px = cfg.get("motion_amp_y_px", 0.0)
    args.motion_cycles_x = cfg.get("motion_cycles_x", 0.0)
    args.motion_cycles_y = cfg.get("motion_cycles_y", 0.0)
    args.motion_harmonic_ratio = cfg.get("motion_harmonic_ratio", 0.0)
    args.motion_speed_px = cfg.get("motion_speed_px", 0.0)
    args.motion_profile = cfg.get("motion_profile", "global_sinusoid")
    args.motion_guard_px = cfg["motion_guard_px"]
    args.placement_mode = cfg.get("placement_mode", "quadrants")
    args.allow_wall_collision = cfg.get("allow_wall_collision", False)
    args.allow_pendulum_overlap = cfg.get("allow_pendulum_overlap", False)
    args.noise_level = level
    return args


def main(args):
    if not hasattr(args, "setting_case"):
        args.setting_case = "moving_clutter_underdamped"

    if getattr(args, "noise_level", None) is not None:
        apply_moving_level(args, args.noise_level)

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

    render_bw_with_moving_clutter(
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
        noise_count=args.noise_count,
        noise_min_size_px=args.noise_min_size_px,
        noise_max_size_px=args.noise_max_size_px,
        noise_gray_min=args.noise_gray_min,
        noise_gray_max=args.noise_gray_max,
        noise_seed=args.noise_seed,
        motion_amp_x_px=args.motion_amp_x_px,
        motion_amp_y_px=args.motion_amp_y_px,
        motion_cycles_x=args.motion_cycles_x,
        motion_cycles_y=args.motion_cycles_y,
        motion_harmonic_ratio=args.motion_harmonic_ratio,
        motion_speed_px=args.motion_speed_px,
        motion_profile=args.motion_profile,
        motion_guard_px=args.motion_guard_px,
        allow_wall_collision=args.allow_wall_collision,
        allow_pendulum_overlap=args.allow_pendulum_overlap,
        placement_mode=args.placement_mode,
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
        description="B/W pendulum with moving clutter: low-frequency global clutter translation"
    )

    ap.add_argument("--gamma0", type=float, default=4.0, help="γ0 (stiffness-like)")
    ap.add_argument("--gamma1", type=float, default=5.0, help="γ1 (damping)")
    ap.add_argument("--theta0_deg", type=float, default=90.0, help="initial angle in degrees")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0.0, help="initial angular velocity in degrees per second")
    ap.add_argument("--duration", type=float, default=10.0, help="video seconds")
    ap.add_argument("--fps", type=int, default=60, help="frames per second")
    ap.add_argument("--L", type=float, default=1.0, help="visual rod length")
    ap.add_argument("--size", type=int, default=64, help="square resolution (px)")
    ap.add_argument("--rod_px", type=float, default=8.0, help="rod line width (px)")
    ap.add_argument("--bob_ratio", type=float, default=0.1, help="bob radius = ratio * L")
    ap.add_argument("--trail", type=int, default=0, help="trail length in frames (0 off)")
    ap.add_argument("--pivot_y_frac", type=float, default=0.5, help="pivot vertical position")
    ap.add_argument("--zoom", type=float, default=1.0, help="camera zoom")

    ap.add_argument("--noise_count", type=int, default=8, help="number of square clutter patches")
    ap.add_argument("--noise_min_size_px", type=int, default=4, help="min square side length in pixels")
    ap.add_argument("--noise_max_size_px", type=int, default=10, help="max square side length in pixels")
    ap.add_argument("--noise_gray_min", type=int, default=0, help="min grayscale value for clutter blocks")
    ap.add_argument("--noise_gray_max", type=int, default=0, help="max grayscale value for clutter blocks")
    ap.add_argument("--noise_level", type=int, default=None, help="motion level in [1, 5]")
    ap.add_argument("--noise_seed", type=int, default=42, help="random seed for reproducible clutter")

    ap.add_argument("--motion_amp_x_px", type=float, default=2.0, help="x amplitude of global clutter motion in pixels")
    ap.add_argument("--motion_amp_y_px", type=float, default=1.0, help="y amplitude of global clutter motion in pixels")
    ap.add_argument("--motion_cycles_x", type=float, default=1.0, help="x motion cycles over the full clip")
    ap.add_argument("--motion_cycles_y", type=float, default=1.0, help="y motion cycles over the full clip")
    ap.add_argument("--motion_harmonic_ratio", type=float, default=0.0, help="extra low-frequency harmonic weight")
    ap.add_argument("--motion_speed_px", type=float, default=0.0, help="independent clutter speed in pixels per frame")
    ap.add_argument("--motion_profile", type=str, default="global_sinusoid", help="motion profile: global_sinusoid or independent_bounce")
    ap.add_argument("--motion_guard_px", type=int, default=4, help="extra guard margin around the pendulum corridor")
    ap.add_argument("--placement_mode", type=str, default="quadrants", help="initial clutter placement mode")
    ap.add_argument("--allow_wall_collision", action="store_true", help="allow clutter motion to clip at image boundaries")
    ap.add_argument("--allow_pendulum_overlap", action="store_true", help="allow clutter to overlap the pendulum corridor")

    ap.add_argument("-o", "--out", type=str, default="generated_video/pendulum_10s.mp4", help="output MP4 path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.setting_case = "moving_clutter_underdamped"
    save_folder = get_setting_save_folder(args.setting_case)
    os.makedirs(save_folder, exist_ok=True)
    for args.theta0_deg in [60]:
        for args.dtheta0_deg_s in [0]:
            for args.duration in [2.5]:
                for args.fps in [20]:
                    duration_pi = args.duration
                    for level in get_setting_levels(args.setting_case):
                        setting_params = get_setting_level_params(args.setting_case, level)
                        args.gamma0 = setting_params["gamma0"]
                        args.gamma1 = setting_params["gamma1"]
                        args.noise_level = level
                        args.out = (
                            f"{save_folder}/{duration_pi:.1f}pi_{int(args.fps)}fps_"
                            f"degree{args.theta0_deg}_degs{args.dtheta0_deg_s}_level{level}.mp4"
                        )
                        args.duration = duration_pi * np.pi
                        main(args)
                        args.duration = duration_pi
