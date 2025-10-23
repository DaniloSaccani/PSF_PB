# run_prestabilized_open_loop.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

# -----------------------------
# Configuration
# -----------------------------
DT = 0.05           # must match the model's step (your models use h = 0.05)
T_STEPS = 150       # simulation length (T_STEPS * DT seconds)
HORIZON = 20        # PSF horizon
EPSILON = 0.05      # scheduler "knee" (||u_L|| scale)
RHO_MAX = 1.0       # max scheduler value (1 = strongest decrease constraint)
RHO_FIXED = None    # leave None to use the scheduler with rho_bar
SAFE_TH_LO = 0.5
SAFE_TH_HI = 2*np.pi - 0.5
U_MIN, U_MAX = -5.0, 5.0
q_theta = 50
q_theta_dot = 10
r_u = 0.1
Q = np.diag([q_theta, q_theta_dot]).astype(float)   # shape (2,2)
R = np.array([[r_u]], dtype=float)

# Initial condition near horizontal; target is upright (pi, 0)
x0 = np.array([np.pi/2, 0.0], dtype=float)
xbar = np.array([np.pi, 0.0], dtype=float)

# We'll compare multiple rho_bar values
RHOBAR_LIST = [10.00, 0.25, 0.50, 0.75]

# -----------------------------
# Obstacle (Gaussian, moving)
#   We draw the 95% iso-probability contour.
# -----------------------------
OBS_CENTERS = np.array([[1, 0.5]], dtype=float)     # shape (n_obs, 2)
OBS_COVS    = np.array([0.005], dtype=float)          # variance (isotropic)
OBS_VEL     = np.array([[-0.2, 0.0]], dtype=float)    # shape (n_obs, 2)
CHI2_2_95   = 5.991464547107979                       # chi2 df=2 at 95%

# Uniform shading color (RGBA) for obstacle area
OBSTACLE_SHADE_COLOR = (0.85, 0.2, 0.2, 0.18)  # soft red with alpha

def obstacle_centers_at_step(t_idx: int) -> np.ndarray:
    """Return obstacle centers at discrete step t_idx."""
    return OBS_CENTERS + t_idx * DT * OBS_VEL  # (n_obs, 2)

def wrap_0_2pi(a):
    """Wrap angle to [0, 2π)."""
    return (a + 2*np.pi) % (2*np.pi)

def theta_arcs_hit_obstacle(center_xy, link_length, cov_var, chi2_val=CHI2_2_95):
    """
    For a circular obstacle centered at 'center_xy' with radius r = sqrt(chi2*cov),
    return the interval(s) of θ (in [0,2π)) such that the tip p(θ) = [l*sinθ, -l*cosθ]
    lies inside the circle. Returns a list of (θ_lo, θ_hi); may be empty or split into 2.
    """
    r = np.sqrt(chi2_val * cov_var)
    c = np.asarray(center_xy, dtype=float)
    L = float(link_length)
    norm_c = np.linalg.norm(c)
    # Handle degenerate center
    if norm_c < 1e-12:
        if r >= L:  # circle around origin contains the whole unit circle of radius L
            return [(0.0, 2*np.pi)]
        else:
            return []

    # On the unit circle parametrized by α (standard), our tip is at angle α = θ - π/2
    # Distance^2 between tip and center: d^2 = L^2 + ||c||^2 - 2 L ||c|| cos(α - φ)
    # Condition d <= r  ->  cos(α - φ) >= (L^2 + ||c||^2 - r^2) / (2 L ||c||) = ζ
    zeta = (L**2 + norm_c**2 - r**2) / (2.0 * L * norm_c)

    if zeta <= -1.0:  # always true -> full circle
        return [(0.0, 2*np.pi)]
    if zeta >=  1.0:  # never true
        return []

    delta = np.arccos(zeta)              # half-width in α
    phi   = np.arctan2(c[1], c[0])       # center direction in α-space
    alpha_lo, alpha_hi = phi - delta, phi + delta

    # Convert back to θ = α + π/2
    th_lo = wrap_0_2pi(alpha_lo + np.pi/2)
    th_hi = wrap_0_2pi(alpha_hi + np.pi/2)

    # If wrapped, split into two intervals
    if th_lo <= th_hi:
        return [(th_lo, th_hi)]
    else:
        return [(0.0, th_hi), (th_lo, 2*np.pi)]

# u_L(t): "open-loop" learning action sent to the PSF.
# Setting it to zero means the PSF alone prestabilizes the system.
def desired_action_uL(t_step: int, x: np.ndarray) -> np.ndarray:
    return np.array([0.00], dtype=float)



# -----------------------------
# One closed-loop run (PSF + plant) for a given rho_bar
# -----------------------------
def simulate_one_rhobar(rho_bar: float):
    # Plant (torch) + CasADi model for PSF
    import torch
    model = SinglePendulumCasadi(xbar=xbar.copy())
    L_link = float(model.l)  # use the same arm length as the PSF model
    plant = SinglePendulum(
        xbar=torch.tensor(xbar, dtype=torch.double),
        x_init=torch.tensor(x0, dtype=torch.double).view(1, -1),
        u_init=torch.zeros(1, 1, dtype=torch.double),
    )

    # PSF with scheduled rho (rho=None) and selectable rho_bar
    psf = MPCPredictSafetyFilter(
        model,
        horizon=HORIZON,
        state_lower_bound=np.array([SAFE_TH_LO, -20], dtype=float),
        state_upper_bound=np.array([SAFE_TH_HI,  20], dtype=float),
        control_lower_bound=np.array([U_MIN], dtype=float),
        control_upper_bound=np.array([U_MAX], dtype=float),
        Q=Q, R=R, solver_opts=None,
        set_lyacon=True,
        epsilon=EPSILON,
        rho=RHO_FIXED,        # keep None to activate scheduler
        rho_bar=rho_bar,
        rho_max=RHO_MAX,
    )

    # Logs
    th_log   = [x0[0]]
    om_log   = [x0[1]]
    u_log    = []
    Jstar_log= []
    feas_log = []

    # For obstacle shading: list of per-time lists of (θ_lo, θ_hi)
    obs_arcs = []

    # Internal warm starts for PSF
    U_prev = None
    X_prev = None

    # Current plant state (torch, shape (1,1,2))
    x_t = torch.tensor(x0, dtype=torch.double).view(1,1,-1)

    for t in range(T_STEPS):
        # Compose inputs for PSF
        uL = desired_action_uL(t, x_t.detach().cpu().numpy().reshape(-1))
        x_np = x_t.detach().cpu().numpy().reshape(-1)

        # Solve one PSF OCP and extract first control
        try:
            if t == 0:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, xbar, uL)
            else:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, xbar, uL, U_prev, X_prev)
        except Exception:
            U_sol, X_sol, J_curr = None, None, None

        if U_sol is None or X_sol is None:
            # Infeasible → pass-through uL, keep warm-starts as None
            u_cmd = uL.copy().reshape(1,1,1)
            U_prev, X_prev = None, None
            feas_log.append(False)
            Jstar_log.append(np.nan)
        else:
            # Apply first move and warm start with the solution
            u_cmd = U_sol[:, 0:1].reshape(1,1,1)
            U_prev, X_prev = U_sol, X_sol
            feas_log.append(True)
            Jstar_log.append(float(J_curr))

        # Plant step (filtered input)
        u_torch = torch.tensor(u_cmd, dtype=torch.double)
        x_t = plant.rk4_integration(x_t, u_torch)

        # Log state
        th_log.append(float(x_t.view(-1)[0]))
        om_log.append(float(x_t.view(-1)[1]))
        u_log.append(float(u_cmd.reshape(-1)[0]))

        # --- obstacle arcs at time slice [t, t+DT] ---
        centers_t = obstacle_centers_at_step(t)  # (n_obs,2)
        arcs_t = []
        for i in range(centers_t.shape[0]):
            arcs_t.extend(theta_arcs_hit_obstacle(
                centers_t[i], L_link, cov_var=float(OBS_COVS[i]), chi2_val=CHI2_2_95
            ))
        obs_arcs.append(arcs_t)

    # Time axis
    t_axis = np.arange(T_STEPS+1) * DT
    t_axis_u = np.arange(T_STEPS) * DT

    return {
        "rho_bar": rho_bar,
        "t": t_axis,
        "t_u": t_axis_u,
        "theta": np.array(th_log),
        "omega": np.array(om_log),
        "u": np.array(u_log),
        "Jstar": np.array(Jstar_log),
        "feasible": np.array(feas_log, dtype=bool),
        "obstacle_arcs": obs_arcs,   # list of per-step lists of (θ_lo, θ_hi)
    }

# -----------------------------
# Run all sims and plot
# -----------------------------
if __name__ == "__main__":
    all_runs = [simulate_one_rhobar(rb) for rb in RHOBAR_LIST]

    # ---------- θ(t) with obstacle-arc shading ----------
    fig1, ax1 = plt.subplots(figsize=(10, 4.8))
    # draw curves
    for run in all_runs:
        ax1.plot(run["t"], run["theta"], label=f"rho_bar={run['rho_bar']:.2f}")
    ax1.axhline(np.pi, linestyle="--", linewidth=1, label="θ*=π")
    ax1.set_title("Pendulum angle θ(t) with obstacle arc (95% contour) shaded")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("θ [rad]")
    ax1.grid(True, alpha=0.3)

    # obstacle shading (uniform color)
    arc_runs = all_runs[0]["obstacle_arcs"]  # same arcs for all runs → use the first
    for k, arcs in enumerate(arc_runs):
        if not arcs:
            continue
        x_span = [k*DT, (k+1)*DT]
        for (th_lo, th_hi) in arcs:
            ax1.fill_between(
                x_span,
                [th_lo, th_lo],
                [th_hi, th_hi],
                facecolor=OBSTACLE_SHADE_COLOR,  # <- uniform color
                edgecolor='none',
                linewidth=0.0,
                zorder=0
            )

    # Safe θ bounds for reference
    ax1.axhline(SAFE_TH_LO, color="k", lw=0.8, ls=":")
    ax1.axhline(SAFE_TH_HI, color="k", lw=0.8, ls=":")

    # Legend (include obstacle)
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Patch(facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', label='Obstacle (95%)'))
    ax1.legend(handles=handles, loc="best")

    # ---------- J*(t) ----------
    plt.figure(figsize=(10, 4.5))
    for run in all_runs:
        label = f"rho_bar={run['rho_bar']:.2f}"
        plt.plot(run["t_u"], run["Jstar"], label=label)
    plt.title("Optimal value $J^*$ (per PSF solve) vs time")
    plt.xlabel("time [s]")
    plt.ylabel("$J^*$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ---------- u(t) ----------
    plt.figure(figsize=(10, 4.5))
    for run in all_runs:
        label = f"rho_bar={run['rho_bar']:.2f}"
        plt.plot(run["t_u"], run["u"], label=label)
    plt.axhline(U_MIN, linestyle="--", linewidth=1, color="k")
    plt.axhline(U_MAX, linestyle="--", linewidth=1, color="k")
    plt.title("Filtered input u(t) for different $\\bar{\\rho}$")
    plt.xlabel("time [s]")
    plt.ylabel("u [Nm]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
