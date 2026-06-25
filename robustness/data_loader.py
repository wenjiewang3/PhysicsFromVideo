import numpy as np
import imageio.v3 as iio
import torch

def load_bw_video_to_tensor(
    path: str,
    binarize = True,
    threshold = 0.5,
    device = None,
):
    frames = iio.imread(path, index=None)  # requires imageio>=2.29
    if frames.ndim == 3:
        # Already grayscale (T, H, W)
        gray = frames.astype(np.float32)
    else:
        # (T, H, W, C) -> Grayscale; drop alpha
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        # Y = 0.299 R + 0.587 G + 0.114 B
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = np.tensordot(frames.astype(np.float32), weights, axes=([-1], [0]))
        # Result: (T, H, W)

    # Normalize to [0,1]
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    if gray.max() > 1.0:  # Handle 0-255 range
        gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)

    # Optional binarization
    if binarize:
        gray = (gray > threshold).astype(np.float32)

    # Add channel dim -> (T, 1, H, W)
    gray = gray[:, None, :, :].copy()  # contiguous


    t = torch.from_numpy(gray)
    if device is not None:
        t = t.to(device)
    return t

def load_bw_videos_same_shape(
    paths: list[str],
    binarize: bool = True,
    threshold: float = 0.5,
    device=None,
):
    """
    Load multiple videos that all share the same shape (T,1,H,W).
    Returns a single tensor stacked on batch dim: (B,T,1,H,W).
    """
    vids = []
    for p in paths:
        v = load_bw_video_to_tensor(p, binarize=binarize, threshold=threshold, device="cpu")
        vids.append(v)

    # Verify same shape for all
    T, C, H, W = vids[0].shape
    for i, v in enumerate(vids[1:], start=1):
        if v.shape != (T, C, H, W):
            raise ValueError(
                f"Video shape mismatch at {paths[i]}: got {v.shape}, expected {(T, C, H, W)}. "
                "Make sure all videos have identical length/FPS/resolution."
            )

    x = torch.stack(vids, dim=0)  # (B,T,1,H,W)
    if device is not None:
        x = x.to(device)
    return x