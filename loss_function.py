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
