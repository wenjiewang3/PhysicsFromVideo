import torch

def ode_residual_loss(z: torch.Tensor, dt: float, gamma0: torch.Tensor, gamma1: torch.Tensor) -> torch.Tensor:
    """
    z: (B,T,1) or (B,T)
    dt: scalar (seconds)
    gamma0, gamma1: learnable scalars (broadcasted)
    """
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)           # (B,T)
    B, T = z.shape
    assert T >= 3, "Need at least 3 frames to calculate 2nd order difference"

    z_prev = z[:, :-2]              # (B, T-2)  k-1
    z_k    = z[:, 1:-1]             # (B, T-2)  k
    z_next = z[:, 2: ]              # (B, T-2)  k+1

    dt_k = dt                       # scalar
    d1 = (z_k - z_prev) / dt_k                     # 1st order diff ~ z'(t_k)
    d2 = (z_next - 2.0*z_k + z_prev) / (dt_k**2)   # 2nd order diff ~ z''(t_k)

    # broadcast to (B, T-2)
    g0 = gamma0.view(1, 1).expand_as(z_k)
    g1 = gamma1.view(1, 1).expand_as(z_k)

    resid = d2 + g1 * d1 + g0 * z_k              # (B, T-2)
    return (resid**2).mean()


def ode_residual_loss_centered(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
    tau: float = 0.7,        # Minimum standard deviation threshold (e.g., sqrt(10))
    var_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)  # (B,T)
    B, T = z.shape
    assert T >= 3, "Need at least 3 frames to calculate 2nd order difference"

    z_prev = z[:, :-2]
    z_k    = z[:, 1:-1]
    z_next = z[:, 2: ]

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    # Central difference
    d1 = (z_next - z_prev) / (2.0 * h)
    d2 = (z_next - 2.0 * z_k + z_prev) / (h * h)



    g0 = gamma0.view(1, 1).expand_as(z_k)
    g1 = gamma1.view(1, 1).expand_as(z_k)

    resid = d2 + g1 * d1 + g0 * z_k
    resid_mse = (resid ** 2).mean()

    # --- Variance lower bound regularization (std threshold) ---
    var_per_seq = z.var(dim=1, unbiased=False)           # (B,)
    sigma_per_seq = torch.sqrt(var_per_seq + eps)        # (B,)
    std_min = torch.as_tensor(tau, dtype=z.dtype, device=z.device)
    slack = torch.relu(std_min - sigma_per_seq)          # Penalty if below threshold
    L_var = (slack ** 2).mean()

    loss = resid_mse + var_weight * L_var
    return loss

# ablation study: centered loss without regularizer
def ode_residual_loss_no_reg(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
) -> torch.Tensor:
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)  # (B, T)

    B, T = z.shape
    assert T >= 3, "Need at least 3 frames"

    z_prev = z[:, :-2]
    z_k    = z[:, 1:-1]
    z_next = z[:, 2:]

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    d1 = (z_next - z_prev) / (2.0 * h)
    d2 = (z_next - 2.0 * z_k + z_prev) / (h * h)

    g0 = gamma0.view(1, 1).expand_as(z_k)
    g1 = gamma1.view(1, 1).expand_as(z_k)

    resid = d2 + g1 * d1 + g0 * z_k
    resid_mse = (resid ** 2).mean()

    return resid_mse


# ablation study: centered loss with KL (N(0,1)) regularizer
def ode_residual_loss_with_kl(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
    kl_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)  # (B, T)

    B, T = z.shape
    assert T >= 3, "Need at least 3 frames"

    z_prev = z[:, :-2]
    z_k    = z[:, 1:-1]
    z_next = z[:, 2:]

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    d1 = (z_next - z_prev) / (2.0 * h)
    d2 = (z_next - 2.0 * z_k + z_prev) / (h * h)

    g0 = gamma0.view(1, 1).expand_as(z_k)
    g1 = gamma1.view(1, 1).expand_as(z_k)

    resid = d2 + g1 * d1 + g0 * z_k
    resid_mse = (resid ** 2).mean()

    # --- KL regularizer: q(z) vs N(0,1) ---
    mu = z.mean(dim=1)                        # (B,)
    var = z.var(dim=1, unbiased=False) + eps  # (B,)
    logvar = torch.log(var)

    kl_per_seq = 0.5 * (mu**2 + var - logvar - 1.0)
    kl_loss = kl_per_seq.mean()

    loss = resid_mse + kl_weight * kl_loss
    return loss


def ode_residual_loss_backward(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
    tau: float = 0.7,        # Minimum std threshold
    var_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Unify shape (B, T)
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)
    B, T = z.shape
    assert T >= 3, "Need >= 3 frames for 3-point central diff of 2nd derivative"

    # Align to k = 1..T-2
    z_prev = z[:, :-2]   # z_{k-1}
    z_k    = z[:, 1:-1]  # z_k
    z_next = z[:, 2: ]   # z_{k+1}

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    # 1st derivative (Backward Diff): z'(t_k) ≈ (z_k - z_{k-1}) / Δt
    d1 = (z_k - z_prev) / h

    # 2nd derivative (3-point Central Diff): z''(t_k) ≈ (z_{k+1} - 2 z_k + z_{k-1}) / Δt^2
    d2 = (z_next - 2.0 * z_k + z_prev) / (h * h)

    # Residual: z'' + γ1 z' + γ0 z
    g0 = gamma0.view(1, 1).expand_as(z_k)
    g1 = gamma1.view(1, 1).expand_as(z_k)
    resid = d2 + g1 * d1 + g0 * z_k
    resid_mse = (resid ** 2).mean()

    # Variance lower bound regularization
    var_per_seq = z.var(dim=1, unbiased=False)           # (B,)
    sigma_per_seq = torch.sqrt(var_per_seq + eps)        # (B,)
    std_min = torch.as_tensor(tau, dtype=z.dtype, device=z.device)
    slack = torch.relu(std_min - sigma_per_seq)
    L_var = (slack ** 2).mean()

    loss = resid_mse + var_weight * L_var
    return loss


def ode_residual_loss_with_intercept(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
    tau: float = 0.7,          # Minimum std threshold (prevent collapse)
    var_weight: float = 1.0,
    eps: float = 1e-8,
):
    """
    Model: z'' + gamma1*z' + gamma0*z = c_i   (constant intercept per trajectory)
    Training: Estimate c_i per trajectory via closed-form solution: c_i = mean_t(z'' + gamma1*z' + gamma0*z),
    then substitute back into the residual.

    z: (B,T,1) or (B,T) - Same length/sample rate within batch
    dt: scalar (seconds)
    """
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)  # (B,T)
    B, T = z.shape
    assert T >= 3, "Need at least 3 frames"

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    # 3-point central difference (robust to noise), align to k=1..T-2
    z_prev = z[:, :-2]
    z_mid  = z[:, 1:-1]
    z_next = z[:, 2: ]

    z_p  = (z_next - z_prev) / (2.0 * h)         # z'(t_k)
    z_pp = (z_next - 2.0*z_mid + z_prev) / (h*h) # z''(t_k)

    g0 = gamma0.view(1, 1).expand_as(z_mid)
    g1 = gamma1.view(1, 1).expand_as(z_mid)

    # First compute a = z'' + gamma1*z' + gamma0*z
    a = z_pp + g1 * z_p + g0 * z_mid             # (B, T-2)

    # Optimal intercept per trajectory (closed-form): time mean
    c = a.mean(dim=1, keepdim=True)              # (B,1)

    # Final residual after removing constant
    resid = a - c                                 # (B, T-2)
    resid_mse = (resid ** 2).mean()

    # --- Variance lower bound reg to prevent z collapsing to constant ---
    var_per_seq = z.var(dim=1, unbiased=False)          # (B,)
    sigma_per_seq = torch.sqrt(var_per_seq + eps)       # (B,)
    std_min = torch.as_tensor(tau, dtype=z.dtype, device=z.device)
    slack = torch.relu(std_min - sigma_per_seq)
    L_var = (slack ** 2).mean()

    loss = resid_mse + var_weight * L_var
    # Return mean(c) for logging
    return loss, {"loss_ode": resid_mse.detach(), "loss_var": L_var.detach(), "c_mean": c.mean().detach()}


# ablation study: centred loss with intercept no regularizer
def ode_residual_loss_with_intercept_no_reg(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
):
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)  # (B,T)

    B, T = z.shape
    assert T >= 3, "Need at least 3 frames"

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    z_prev = z[:, :-2]
    z_mid  = z[:, 1:-1]
    z_next = z[:, 2:]

    z_p  = (z_next - z_prev) / (2.0 * h)
    z_pp = (z_next - 2.0*z_mid + z_prev) / (h*h)

    g0 = gamma0.view(1, 1).expand_as(z_mid)
    g1 = gamma1.view(1, 1).expand_as(z_mid)

    a = z_pp + g1 * z_p + g0 * z_mid        # (B, T-2)
    c = a.mean(dim=1, keepdim=True)         # per-trajectory intercept
    resid = a - c

    resid_mse = (resid ** 2).mean()

    return resid_mse, {
        "loss_ode": resid_mse.detach(),
        "c_mean": c.mean().detach(),
    }

# ablation study: centred loss with intercept with KL regularizer
def ode_residual_loss_with_intercept_kl(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
    kl_weight: float = 1.0,
    eps: float = 1e-8,
):
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)  # (B,T)

    B, T = z.shape
    assert T >= 3, "Need at least 3 frames"

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    z_prev = z[:, :-2]
    z_mid  = z[:, 1:-1]
    z_next = z[:, 2:]

    z_p  = (z_next - z_prev) / (2.0 * h)
    z_pp = (z_next - 2.0*z_mid + z_prev) / (h*h)

    g0 = gamma0.view(1, 1).expand_as(z_mid)
    g1 = gamma1.view(1, 1).expand_as(z_mid)

    a = z_pp + g1 * z_p + g0 * z_mid
    c = a.mean(dim=1, keepdim=True)
    resid = a - c
    resid_mse = (resid ** 2).mean()

    # --- KL regularizer on latent z ---
    mu = z.mean(dim=1)                              # (B,)
    var = z.var(dim=1, unbiased=False) + eps        # (B,)
    logvar = torch.log(var)

    kl_per_seq = 0.5 * (mu**2 + var - logvar - 1.0)
    kl_loss = kl_per_seq.mean()

    loss = resid_mse + kl_weight * kl_loss

    return loss, {
        "loss_ode": resid_mse.detach(),
        "loss_kl": kl_loss.detach(),
        "c_mean": c.mean().detach(),
    }


def ode_residual_loss_with_intercept_backward(
    z: torch.Tensor,
    dt: float,
    gamma0: torch.Tensor,
    gamma1: torch.Tensor,
    tau: float = 0.7,
    var_weight: float = 1.0,
    eps: float = 1e-8,
):
    if z.ndim == 3 and z.size(-1) == 1:
        z = z.squeeze(-1)
    B, T = z.shape
    assert T >= 3, "Need at least 3 frames"

    h = torch.as_tensor(dt, dtype=z.dtype, device=z.device)

    z_prev = z[:, :-2]      # k-1
    z_mid  = z[:, 1:-1]     # k
    z_next = z[:, 2: ]      # k+1

    z_p  = (z_mid - z_prev) / h                   # Backward diff
    z_pp = (z_next - 2.0*z_mid + z_prev) / (h*h)  # 3-point 2nd order

    g0 = gamma0.view(1, 1).expand_as(z_mid)
    g1 = gamma1.view(1, 1).expand_as(z_mid)

    a = z_pp + g1 * z_p + g0 * z_mid
    c = a.mean(dim=1, keepdim=True)
    resid = a - c
    resid_mse = (resid ** 2).mean()

    var_per_seq = z.var(dim=1, unbiased=False)
    sigma_per_seq = torch.sqrt(var_per_seq + eps)
    std_min = torch.as_tensor(tau, dtype=z.dtype, device=z.device)
    slack = torch.relu(std_min - sigma_per_seq)
    L_var = (slack ** 2).mean()

    loss = resid_mse + var_weight * L_var
    return loss, {"loss_ode": resid_mse.detach(), "loss_var": L_var.detach(), "c_mean": c.mean().detach()}