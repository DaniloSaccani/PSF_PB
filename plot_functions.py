# plot_functions.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.stats import multivariate_normal

# ---------------- Styling (kept from your file) ----------------
try:
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif',
        'font.serif': ['Arial'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf'
    })
except Exception:
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Arial'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf'
    })

# ---------------- Helpers for overlay geometry ----------------
CHI2_2_95 = 5.991464547107979  # chi^2(df=2, 0.95)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def _wrap_0_2pi(x):
    return np.mod(x, 2*np.pi)

def _theta_arcs_hit_obstacle(center_xy, L, cov_var):
    """
    Intersect the pendulum tip circle (radius L, centered at origin)
    with the obstacle's 95% ellipse. For isotropic cov_var (σ²), the
    95% contour is a circle of radius r = sqrt(χ²₂(0.95) * σ²).
    Returns arcs in ABSOLUTE θ ∈ [0,2π): list[(θ_lo, θ_hi)].
    """
    r = np.sqrt(CHI2_2_95 * float(cov_var))
    c = np.asarray(center_xy, dtype=float).reshape(2,)
    nc = np.linalg.norm(c)
    if nc < 1e-12:
        return [(0.0, 2*np.pi)]  # obstacle covers origin → whole ring
    zeta = (L**2 + nc**2 - r**2) / (2.0 * L * nc)
    if zeta <= -1.0:
        return [(0.0, 2*np.pi)]  # full ring
    if zeta >=  1.0:
        return []                # no intersection
    delta = np.arccos(zeta)
    phi = np.arctan2(c[1], c[0])
    # tip direction ψ = θ - π/2  ⇒  θ = ψ + π/2
    th_lo = _wrap_0_2pi(phi - delta + np.pi/2)
    th_hi = _wrap_0_2pi(phi + delta + np.pi/2)
    return [(th_lo, th_hi)] if th_lo <= th_hi else [(0.0, th_hi), (th_lo, 2*np.pi)]

def _arcs_absolute_to_error(arcs_abs, theta_ref=np.pi):
    """
    Map absolute-θ arcs [θ_lo, θ_hi] (in [0,2π)) to error frame (-π,π]
    by subtracting theta_ref and wrapping.
    """
    out = []
    for th_lo, th_hi in arcs_abs:
        lo_e = _wrap_pi(th_lo - theta_ref)
        hi_e = _wrap_pi(th_hi - theta_ref)
        if lo_e <= hi_e:
            out.append((lo_e, hi_e))
        else:
            out.append((-np.pi, hi_e))
            out.append((lo_e, np.pi))
    return out


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
                 control_upper_bound = np.array([3.0]),
                 # -------- NEW: overlay on θ̃–time subplot --------
                 overlay_ts_obstacle: bool = False,
                 obstacle_band_centers_t = None,    # [T,2] torch or np
                 obstacle_band_cov_var: float = None,
                 theta_ref: float = np.pi
                 ):
    """
    Plots time series and (optionally) trajectory.

    NEW: if `overlay_ts_obstacle=True` and both `obstacle_band_centers_t` and
    `obstacle_band_cov_var` are provided, the first subplot overlays the
    95% obstacle footprint in the θ̃ frame (stacked time slabs).
    """
    # --- 1) Extract signals & time vector ---
    theta = x_log[0, :, 0]
    omega = x_log[0, :, 1]
    T = theta.shape[0]
    t = np.arange(T) * dt

    # Error frame for theta
    theta_err = _wrap_pi(theta - theta_ref)

    # tip (for trajectory)
    x_tip = length * np.sin(theta)
    y_tip = -length * np.cos(theta)

    theta_lo, theta_hi = state_lower_bound[0, 0], state_upper_bound[0, 0]
    u_lo, u_hi = float(control_lower_bound), float(control_upper_bound)

    # === Figure #1: time‐series ===
    if u_L_Log is None:
        fig1, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Subplot 0: θ̃ & ω vs. time (θ̃ main)
        axs[0].plot(t, theta_err, label=r'$\tilde{\theta}$')
        axs[0].axhline(0.0, color='k', linestyle='--', alpha=0.6, label=r'$\tilde{\theta}=0$')
        axs[0].axhline(-np.pi, color='k', linestyle=':', alpha=0.5, label='_nolegend_')
        axs[0].plot(t, omega, label=r'$\omega$', alpha=0.65)

        # Optional overlay of obstacle footprint in θ̃
        if overlay_ts_obstacle and (obstacle_band_centers_t is not None) and (obstacle_band_cov_var is not None):
            centers_np = _to_numpy(obstacle_band_centers_t)
            Tts = centers_np.shape[0]
            t_band = np.arange(Tts) * float(dt)
            for k in range(Tts - 1):
                arcs_abs = _theta_arcs_hit_obstacle(centers_np[k], float(length), float(obstacle_band_cov_var))
                if not arcs_abs:
                    continue
                arcs_err = _arcs_absolute_to_error(arcs_abs, float(theta_ref))
                x_span = [t_band[k], t_band[k+1]]
                for (lo_e, hi_e) in arcs_err:
                    axs[0].fill_between(x_span, [lo_e, lo_e], [hi_e, hi_e],
                                        facecolor=(0.95, 0.55, 0.55, 0.35),
                                        edgecolor='none', linewidth=0.0, zorder=0)

        axs[0].set_ylabel(r'$\tilde{\theta}$ [rad]')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Subplot 1: u
        axs[1].plot(t[:-1], u_log[0, :, 0], label='u')
        axs[1].axhline(u_lo, color='k', ls='--', label='u bounds')
        axs[1].axhline(u_hi, color='k', ls='--', label='_nolegend_')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

    else:
        fig1, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Subplot 0: θ̃ & ω vs. time
        axs[0].plot(t, theta_err, color='b', label=r'$\tilde{\theta}$')
        axs[0].axhline(0.0, color='k', linestyle='--', alpha=0.6, label=r'$\tilde{\theta}=0$')
        axs[0].axhline(-np.pi, color='k', linestyle=':', alpha=0.5, label='_nolegend_')
        axs[0].plot(t, omega, color='tab:orange', label=r'$\omega$', alpha=0.65)

        if overlay_ts_obstacle and (obstacle_band_centers_t is not None) and (obstacle_band_cov_var is not None):
            centers_np = _to_numpy(obstacle_band_centers_t)
            Tts = centers_np.shape[0]
            t_band = np.arange(Tts) * float(dt)
            for k in range(Tts - 1):
                arcs_abs = _theta_arcs_hit_obstacle(centers_np[k], float(length), float(obstacle_band_cov_var))
                if not arcs_abs:
                    continue
                arcs_err = _arcs_absolute_to_error(arcs_abs, float(theta_ref))
                x_span = [t_band[k], t_band[k+1]]
                for (lo_e, hi_e) in arcs_err:
                    axs[0].fill_between(x_span, [lo_e, lo_e], [hi_e, hi_e],
                                        facecolor=(0.95, 0.55, 0.55, 0.35),
                                        edgecolor='none', linewidth=0.0, zorder=0)

        axs[0].legend()
        axs[0].set_ylabel(r'$\tilde{\theta}$ [rad]')
        axs[0].grid(True, alpha=0.3)

        # Subplot 1: u and u_L
        axs[1].plot(t[:-1], u_L_Log[0, :, 0], label=r'$u_L$', color='b')
        axs[1].plot(t[:-1], u_log[0, :, 0], label=r'$u$', color='r')
        axs[1].axhline(u_lo, color='g', ls='--', label='u bounds')
        axs[1].axhline(u_hi, color='g', ls='--', label='_nolegend_')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # Subplot 2: only u_L
        axs[2].plot(t[:-1], u_L_Log[0, :, 0], label=r'$u_L$', color='b')
        axs[2].axhline(u_lo, color='g', ls='--', label='u bounds')
        axs[2].axhline(u_hi, color='g', ls='--', label='_nolegend_')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

    # === Figure #2: tip trajectory (unchanged from your version) ===
    fig2 = None
    if plot_trj:
        fig2, axs2 = plt.subplots(figsize=(5, 5))
        axs2.plot(x_tip, y_tip, '--', label='Tip Trajectory')
        axs2.plot(x_tip[0], y_tip[0], 'go', label='Start')
        axs2.plot([0, x_tip[-1]], [0, y_tip[-1]], color='k', lw=2, label='_nolegend_')
        axs2.plot(x_tip[-1], y_tip[-1], 'ro', label='End')
        if np.isfinite(theta_lo):
            axs2.plot([0, 1 * np.sin(theta_lo)],
                      [0, -1 * np.cos(theta_lo)],
                      color='r', ls='--', label=r'$\theta$ bound')
        if np.isfinite(theta_hi):
            axs2.plot([0, 1 * np.sin(theta_hi)],
                      [0, -1 * np.cos(theta_hi)],
                      color='r', ls='--', label='_nolegend_')

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

            # draw the same grey helper line you had
            cx, cy = center.detach().cpu().numpy().flatten()
            axs2.plot([cx, 1], [cy, 0.5], color='grey', linewidth=1, alpha=0.6)

            z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
            axs2.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

        axs2.legend()
        axs2.set_xlim([-1, 1]); axs2.set_ylim([-1, 1])

    # === Save or Show ===
    if file_path is not None:
        folder = os.path.dirname(file_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        ts_path = f"{file_path}_timeseries.png"
        fig1.savefig(ts_path, dpi=300, bbox_inches='tight')

        if fig2 is not None:
            trj_path = f"{file_path}_trajectory.png"
            fig2.savefig(trj_path, dpi=300, bbox_inches='tight')

        plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)
    else:
        plt.show()
        plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)
