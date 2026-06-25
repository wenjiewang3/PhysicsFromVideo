import torch
import torch.nn.functional as F
import sys

@torch.no_grad()
def estimate_z0_and_derivs(z: torch.Tensor, dt: float):
    """
    Input z: (B,T,1) or (T,1) or (T,)
    Returns dict:
      {
        'z0':  z(0),
        'z0p': z'(0) approx (z1 - z0) / dt,
        'z0pp': z''(0) approx (z2 - 2*z1 + z0) / dt^2
      }
    """
    # Flatten to (T,)
    if z.ndim == 3:
        # If (B,T,1), default to first batch
        z = z[0, :, 0]
    elif z.ndim == 2:
        z = z[:, 0]
    assert z.ndim == 1, f"z must be 1D over time, got shape {tuple(z.shape)}"

    T = z.shape[0]
    if T < 3:
        raise ValueError(f"Need at least 3 frames to estimate derivatives, got T={T}")

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)
    z0, z1, z2 = z[0], z[1], z[2]
    z0p  = (z1 - z0) / h
    z0pp = (z2 - 2*z1 + z0) / (h * h)
    return {"z0": z0.item(), "z0p": z0p.item(), "z0pp": z0pp.item()}


@torch.no_grad()
def smooth_time_1d(z: torch.Tensor):
    """5-point lightweight smoothing (1,4,6,4,1), along time dim only"""
    k = torch.tensor([1.,4.,6.,4.,1.], device=z.device, dtype=z.dtype)
    k = (k / k.sum()).view(1,1,-1)                 # (1,1,5)
    return F.conv1d(z.unsqueeze(1), k, padding=2).squeeze(1)

@torch.no_grad()
def estimate_z0_and_derivs_batch(
    z: torch.Tensor, dt: float, use_smoothing: bool = True
):
    """
    Input: z shape (B,T), (B,T,1), (T,) or (T,1)
    Output: dict where values are tensors:
      'z0' (B,), 'z0p' (B,), 'z0pp' (B,)
    """
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)                 # (B,T)
    elif z.ndim == 2:
        # Could be (B,T) or (T,1). If (T,1), squeeze last dim:
        if z.size(1) == 1: z = z.squeeze(1)  # -> (T,)
    if z.ndim == 1:
        z = z.unsqueeze(0)                 # (1,T)

    B, T = z.shape
    if T < 4:
        raise ValueError(f"Need at least 4 frames, got T={T}")

    z_proc = z - z.mean(dim=1, keepdim=True)  # Remove mean for stability
    if use_smoothing:
        z_proc = smooth_time_1d(z_proc)

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    # z(0)
    z0 = z_proc[:, 0]                      # (B,)

    # 1st deriv: Forward 3-point O(h^2): (-3 f0 + 4 f1 - f2)/(2h)
    z0p = (-3*z_proc[:,0] + 4*z_proc[:,1] - z_proc[:,2]) / (2*h)

    # 2nd deriv: Forward 4-point O(h^2): (2 f0 - 5 f1 + 4 f2 - f3)/h^2
    z0pp = (2*z_proc[:,0] - 5*z_proc[:,1] + 4*z_proc[:,2] - z_proc[:,3]) / (h*h)

    return {"z0": z0, "z0p": z0p, "z0pp": z0pp}


class TeeLogger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.file = open(filename, 'w')
        self.stream = stream
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.file.flush()
        self.stream.flush()
    def close(self):
        self.file.close()