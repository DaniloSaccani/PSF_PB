import torch

class MovingObstacle:
    """
    A class to represent moving obstacles with fixed covariances (cov).

    Args:
        positions (torch.Tensor or array-like):
            Tensor of shape (num_obs, T, 2) giving the (x, y) position
            of each obstacle at each time step t = 0, …, T-1.
        cov (torch.Tensor or array-like):
            Tensor of shape (num_obs,) giving the fixed covariance  for each obstacle.
    """

    def __init__(self, positions, cov):
        positions = torch.as_tensor(positions)
        cov = torch.as_tensor(cov)
        assert positions.ndim == 3 and positions.size(2) == 2, \
            "positions must have shape (num_obs, T, 2)"
        assert positions.size(0) == cov.numel(), \
            "One cov required per obstacle"
        self.positions = positions      # shape (num_obs, T, 2)
        self.cov = cov                  # shape (num_obs,)

    def get_obstacles(self, t):
        """
        Get the obstacle positions at time index t, and their cov.

        Args:
            t (int): Time index, 0 <= t < T.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                positions_t: shape (num_obs, 2)
                cov:        shape (num_obs,)
        """
        T = self.positions.size(1)
        if not (0 <= t < T):
            raise IndexError(f"time index t must be in [0, {T-1}], got {t}")
        positions_t = self.positions[:, t, :]  # (num_obs, 2)
        return positions_t, self.cov

    def get_obstacle_avoidance_loss(self, state, loss_function, t):
        positions_t, cov = self.get_obstacles(t)
        total_loss = torch.tensor(0.0, device=state.device)
        min_dis_total = torch.tensor(float('inf'), device=state.device)
        for i in range(positions_t.size(0)):
            loss, min_dis = loss_function(state, positions_t[i], cov[i])
            total_loss += loss
            min_dis_total = torch.minimum(min_dis_total, min_dis)
        return total_loss, min_dis_total


import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian_pdf(cov, r_max=0.6, num_points=200):
    """
    Plot the radial profile of a 2D isotropic Gaussian PDF (for publication).

    Args:
        cov (float): Variance along each axis (assumes isotropic Gaussian).
        r_max (float): Maximum radial distance to display.
        num_points (int): Number of radial samples.
    """
    r = np.linspace(0, r_max, num_points)
    normalizer = 1 / (2 * np.pi * cov)
    p = normalizer * np.exp(-r**2 / (2 * cov))

    fig, ax = plt.subplots(figsize=(3.2, 2.4))  # approx 8cm x 6cm
    ax.plot(r, p, linewidth=1)

    # Clean axis formatting
    ax.set_xlabel(r'Distance from center $r$', fontsize=10)
    ax.set_ylabel(r'PDF value', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

    # Tight layout and clean borders
    plt.tight_layout(pad=0.4)
    plt.savefig("gaussian_pdf_plot.eps", dpi=300, bbox_inches='tight')  # for paper
    plt.show()
# Example usage
if __name__ == "__main__":
    plot_gaussian_pdf(cov=0.005, r_max=0.6)
