import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.stats import multivariate_normal

try:
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif',
        'font.serif': ['Arial'],
        'mathtext.fontset': 'cm',
        'font.size': 10,            # 10 pt for axis labels, titles
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,       # 8 pt for tick labels
        'ytick.labelsize': 8,
        'legend.fontsize': 8,       # 8 pt for legend text
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,      # 0.5 pt axes spine
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf'     # save as PDF (vector)
    })

except Exception:
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Arial'],
        'mathtext.fontset': 'cm',
        'font.size': 10,            # 10 pt for axis labels, titles
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,       # 8 pt for tick labels
        'ytick.labelsize': 8,
        'legend.fontsize': 8,       # 8 pt for legend text
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,      # 0.5 pt axes spine
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf'     # save as PDF (vector)
    })
def plot_results(x_log,
                 u_log,
                 u_L_Log=None,
                 dt=0.01,
                 length=1.0,
                 plot_trj=True,
                 x_bar=np.array([np.pi, 0]),
                 file_path: str = None,
                 obstacle_centers = None,
                 obstacle_covs = None,
                 state_lower_bound = np.array([[0.5, -np.inf]]).T,
                 state_upper_bound = np.array([[2 * np.pi - 0.5, np.inf]]).T,
                 control_lower_bound = np.array([-3.0]),
                 control_upper_bound = np.array([3.0])

):
    """
    Plots the pendulum states (theta, omega) and control signals.
    If u_L_Log is provided, shows both u and u_L; otherwise only u.
    Also optionally plots the tip‐trajectory in a second figure.

    If `file_path` is given (as a string), saves two PNGs:
      1) {file_path}_timeseries.png
      2) {file_path}_trajectory.png   (only if plot_trj=True)
    Otherwise, just does plt.show().

    Parameters
    ----------
    x_log : np.ndarray
        shape (1, T+1, 2), containing [theta, omega] over time.
    u_log : np.ndarray
        shape (1, T, 1), nominal control input over time.
    u_L_Log : np.ndarray or None
        shape (1, T, 1), filtered/secondary control input over time. If None,
        only plots u_log.
    dt : float
        Time step between samples.
    length : float
        Length of the pendulum stick (for tip‐trajectory).
    plot_trj : bool
        If True, generate a second figure showing the tip trajectory.
    x_bar : np.ndarray of shape (2,)
        Reference [theta_ref, omega_ref] (drawn as dashed lines).
    file_path : str or None
        If a string, used as the prefix for saving:
            {file_path}_timeseries.png
            {file_path}_trajectory.png   (if plot_trj=True)
        If None, no files are saved and plt.show() is called.

    Returns
    -------
    None
    """
    # --- 1) Extract theta, omega, time vector t ---
    theta = x_log[0, :, 0]
    omega = x_log[0, :, 1]
    T = theta.shape[0]
    t = np.arange(T) * dt

    # --- 2) Compute tip coordinates (for trajectory) ---
    x_tip = length * np.sin(theta)
    y_tip = -length * np.cos(theta)

    theta_lo, theta_hi = state_lower_bound[0, 0], state_upper_bound[0, 0]      # NEW
    u_lo, u_hi = float(control_lower_bound), float(control_upper_bound) # NEW

    # === Figure #1: time‐series of (theta, omega) and control signals ===
    if u_L_Log is None:
        fig1, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Subplot 0: theta & omega vs. time
        axs[0].plot(t, theta, label=r'$\theta$')
        axs[0].axhline(y=x_bar[0], color='r', linestyle='--', label=r'$\theta_{ref}$')
        axs[0].plot(t, omega, label=r'$\omega$')
        axs[0].axhline(y=x_bar[1], color='r', linestyle='--', label=r'$\omega_{ref}$')
        # axs[0].set_title(r'State')
        axs[0].legend()

        # Subplot 1: only u vs. time
        axs[1].plot(t[:-1], u_log[0, :, 0], label='u')
        axs[1].axhline(u_lo, color='k', ls='--', label='u bounds')      # NEW
        axs[1].axhline(u_hi, color='k', ls='--', label='_nolegend_')    # NEW
        # axs[1].set_title('Control Input u')
        axs[1].legend()

    else:
        fig1, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Subplot 0: theta & omega vs. time
        axs[0].plot(t, theta, color='b', label=r'$\theta$')
        axs[0].axhline(y=x_bar[0], color='b', linestyle='--', label=r'$\theta_{ref}$')
        axs[0].plot(t, omega, color='tab:orange', label=r'$\omega$')
        axs[0].axhline(y=x_bar[1], color='tab:orange', linestyle='--', label=r'$\omega_{ref}$')
        # axs[0].set_title(r'State')
        axs[0].legend()

        # Subplot 1: u and u_L vs. time
        axs[1].plot(t[:-1], u_L_Log[0, :, 0], label=r'$u_L$', color='b')
        axs[1].plot(t[:-1], u_log[0, :, 0], label=r'$u$', color='r')
        axs[1].axhline(u_lo, color='g', ls='--', label='u bounds')      # NEW
        axs[1].axhline(u_hi, color='g', ls='--', label='_nolegend_')    # NEW
        # axs[1].set_title('Control Inputs')
        axs[1].legend()

        # Subplot 2: only u_L vs. time
        axs[2].plot(t[:-1], u_L_Log[0, :, 0], label=r'$u_L$', color='b')
        axs[2].axhline(u_lo, color='g', ls='--', label='u bounds')      # NEW
        axs[2].axhline(u_hi, color='g', ls='--', label='_nolegend_')    # NEW
        # axs[2].set_title(r'Control Input $u_L$')
        axs[2].legend()

    # === Figure #2: tip trajectory (if requested) ===
    fig2 = None
    if plot_trj:
        fig2, axs2 = plt.subplots(figsize=(5, 5))
        axs2.plot(x_tip, y_tip,'--', label='Tip Trajectory')
        axs2.plot(x_tip[0], y_tip[0], 'go', label='Start')
        axs2.plot([0, x_tip[-1]], [0, y_tip[-1]], color='k', lw=2, label='_nolegend_')
        axs2.plot(x_tip[-1], y_tip[-1], 'ro', label='End')
        if np.isfinite(theta_lo):
            axs2.plot([0, 1 * np.sin(theta_lo)],
                    [0, -1 * np.cos(theta_lo)],
                    color='r', ls='--', label=r'$\theta$ bound')  # NEW
        if np.isfinite(theta_hi):
            axs2.plot([0, 1 * np.sin(theta_hi)],
                    [0, -1 * np.cos(theta_hi)],
                    color='r', ls='--', label='_nolegend_')  # NEW


        # axs2.set_title('Tip Trajectory')
        if obstacle_centers is not None:
            yy, xx = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
            zz = xx * 0
            for center, cov in zip(obstacle_centers, obstacle_covs):
                distr = multivariate_normal(
                    cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                    mean=center.detach().clone().cpu().numpy().flatten()
                )
                for i in range(xx.shape[0]):
                    for j in range(xx.shape[1]):
                        zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])

            # --- Add the two grey lines ---
            cx, cy = center.detach().cpu().numpy().flatten()
            axs2.plot([cx, 1], [cy, 0.5], color='grey', linewidth=1, alpha=0.6)  # solid line to (1, 0.5)

            z_min, z_max = np.abs(zz).min(), np.abs(zz).max()

            axs2.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')
        axs2.legend()
        axs2.set_xlim([-1, 1])
        axs2.set_ylim([-1, 1])

    # === 3) Save or Show ===
    if file_path is not None:
        # Ensure the directory for file_path exists
        folder = os.path.dirname(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # Save figure #1 (timeseries)
        ts_path = f"{file_path}_timeseries.png"
        fig1.savefig(ts_path, dpi=300, bbox_inches='tight')

        # Save figure #2 (trajectory), if it was created
        if fig2 is not None:
            trj_path = f"{file_path}_trajectory.png"
            fig2.savefig(trj_path, dpi=300, bbox_inches='tight')

        # Close both figures to free memory
        plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)

    else:
        # No file_path provided → just display
        plt.show()
        # After showing, close the figures
        plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)
