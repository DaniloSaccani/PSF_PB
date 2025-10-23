import torch
import torch.nn.functional as F


def loss_state_tracking(state, target_positions, Q):
    """
    Computes a quadratic loss penalizing the deviation of the system's state from the target positions.

    Args:
        state (torch.Tensor): The current state of the system.
        target_positions (torch.Tensor): The target positions or desired state.
        Q (torch.Tensor): State weighting matrix for penalizing the deviation.

    Returns:
        torch.Tensor: The total loss based on the state deviation.
    """
    dx = state - target_positions
    return (F.linear(dx, Q) * dx).sum()

# ---- Absolute control effort  ϕ(u) = uᵀ R u  ----
def loss_control_effort_abs(u, R):
    """
    Penalizes absolute effort. Supports R as (m,m) or (m,) diagonal.
    u: shape (m,) or (1,m) or (m,1)
    """
    u = u.reshape(-1, 1)                             # (m,1)
    if R.ndim == 1:
        Rm = torch.diag(R)                           # (m,m)
    else:
        Rm = R
    return (u.t() @ Rm @ u).squeeze()                # scalar

def loss_control_effort(u, R, uL=None):
    """
    Computes the loss associated with minimizing control effort by penalizing large control inputs.

    Args:
        u (torch.Tensor): Control input vector.
        R (torch.Tensor): Weighting matrix for control effort penalty.

    Returns:
        torch.Tensor: The total control effort loss.
    """
    if uL is None:
        return (F.linear(u, R) * u).sum()
    else:
        return (F.linear((uL-u), R) * (uL-u) ).sum() #+ 1 * (F.linear(uL, R) * uL).sum()

def loss_control_effort_regularized(u, u_prev, R):
    """
    Computes the loss that penalizes the change in control inputs (delta_u) to ensure smooth control actions.

    Args:
        u (torch.Tensor): Current control input vector.
        u_prev (torch.Tensor): Previous control input vector.
        R (torch.Tensor): Weighting matrix for control effort penalty.

    Returns:
        torch.Tensor: The total loss for the change in control inputs.
    """
    delta_u = u - u_prev
    return (F.linear(delta_u, R) * delta_u).sum()


# loss_function.py
def loss_obstacle_avoidance_pdf(state, obstacle_position, obstacle_cov):
    mu = obstacle_position
    cov = torch.tensor([[obstacle_cov, obstacle_cov]], dtype=state.dtype, device=state.device)
    Q, min_dis = normpdf(state, mu=mu, cov=cov)
    return Q.sum(), min_dis

def normpdf(q, mu, cov):
    d = 2
    mu  = mu.view(1, d)
    cov = cov.view(1, d)
    qs = torch.split(q, d)
    out = torch.zeros((), dtype=q.dtype, device=q.device)
    dists = []
    for qi in qs:
        den = (2 * torch.pi) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
        num = torch.exp((-0.5 * (qi - mu) ** 2 / cov).sum())
        out = out + num / den
        dists.append(torch.norm(qi - mu, dim=1))
    all_dists = torch.cat(dists)
    return out, all_dists.min()


# --- PDF clipped at 95% Mahalanobis contour (χ2_2(0.95) = 5.991464547) ---
def loss_obstacle_avoidance_pdf95clip(state, obstacle_position, obstacle_cov,
                                      chi2_2_95: float = 5.991464547, smooth: float = 0.0):
    """
    ℓ_obs = ReLU(pdf(x) - pdf_95), where pdf_95 is the Gaussian PDF value
    on the 95% ellipse (Mahalanobis^2 = χ^2_2(0.95)).
    Outside that ellipse the loss is 0; inside it grows with the PDF.
    If 'smooth' > 0, Softplus smoothing replaces the hard ReLU.
    """
    d = 2
    x  = state.reshape(-1, d)
    mu = obstacle_position.reshape(1, d).to(dtype=state.dtype, device=state.device)

    # σ² (scalar tensor on the right device/dtype)
    var = torch.as_tensor(obstacle_cov, dtype=state.dtype, device=state.device)

    # Mahalanobis^2 for isotropic covariance
    dx = x - mu
    r2 = (dx * dx).sum(dim=1) / var  # shape (N,)

    # Normalizer: (2π)^(d/2) * sqrt(|Σ|); with Σ=diag(σ²,σ²), sqrt(|Σ|)=σ²
    den = (2 * torch.pi) ** (d * 0.5) * var  # scalar tensor

    pdf = torch.exp(-0.5 * r2) / den  # (N,)

    # Ensure chi2 is a tensor (fixes your error)
    chi2 = torch.as_tensor(chi2_2_95, dtype=state.dtype, device=state.device)
    pdf_95 = torch.exp(-0.5 * chi2) / den  # scalar tensor

    raw = pdf - pdf_95
    if smooth > 0.0:
        loss = torch.nn.functional.softplus(raw / smooth, beta=1.0).sum() * smooth
    else:
        loss = torch.clamp(raw, min=0.0).sum()

    min_dis = torch.norm(dx, dim=1).min()
    return loss, min_dis

# 97.5% ellipse: χ²₂(0.975) = 7.377758908
def loss_obstacle_avoidance_pdf975clip(state, obstacle_position, obstacle_cov, smooth: float = 0.05):
    return loss_obstacle_avoidance_pdf95clip(
        state, obstacle_position, obstacle_cov,
        chi2_2_95=7.377758908, smooth=smooth
    )

# 99% ellipse: χ²₂(0.99) = 9.210340372
def loss_obstacle_avoidance_pdf99clip(state, obstacle_position, obstacle_cov, smooth: float = 0.05):
    return loss_obstacle_avoidance_pdf95clip(
        state, obstacle_position, obstacle_cov,
        chi2_2_95=9.210340372, smooth=smooth
    )