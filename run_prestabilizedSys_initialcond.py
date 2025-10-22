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

# Nominal target is upright (pi, 0)
xbar = np.array([np.pi, 0.0], dtype=float)

# We'll compare multiple rho_bar values
RHOBAR_LIST = [10.00, 0.25, 0.50, 0.75]

# -----------------------------
# Obstacle (Gaussian, moving) for shading
# -----------------------------
OBS_CENTERS = np.array([[1.0, 0.5]], dtype=float)     # shape (n_obs, 2)
OBS_COVS    = np.array([0.005], dtype=float)          # variance (isotropic)
OBS_VEL     = np.array([[-0.2, 0.0]], dtype=float)    # shape (n_obs, 2)
CHI2_2_95   = 5.991464547107979                       # chi2 df=2 at 95%
OBSTACLE_SHADE_COLOR = (0.85, 0.2, 0.2, 0.18)         # soft red with alpha

def obstacle_centers_at_step(t_idx: int) -> np.ndarray:
    return OBS_CENTERS + t_idx * DT * OBS_VEL  # (n_obs, 2)

def wrap_0_2pi(a):
    return (a + 2*np.pi) % (2*np.pi)

def theta_arcs_hit_obstacle(center_xy, link_length, cov_var, chi2_val=CHI2_2_95):
    r = np.sqrt(chi2_val * cov_var)
    c = np.asarray(center_xy, dtype=float)
    L = float(link_length)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        if r >= L:
            return [(0.0, 2*np.pi)]
        else:
            return []
    zeta = (L**2 + norm_c**2 - r**2) / (2.0 * L * norm_c)
    if zeta <= -1.0: return [(0.0, 2*np.pi)]
    if zeta >=  1.0: return []
    delta = np.arccos(zeta)
    phi   = np.arctan2(c[1], c[0])
    alpha_lo, alpha_hi = phi - delta, phi + delta
    th_lo = wrap_0_2pi(alpha_lo + np.pi/2)
    th_hi = wrap_0_2pi(alpha_hi + np.pi/2)
    if th_lo <= th_hi:
        return [(th_lo, th_hi)]
    else:
        return [(0.0, th_hi), (th_lo, 2*np.pi)]

# u_L(t): keep zero to isolate PSF behavior & first-step feasibility.
def desired_action_uL(t_step: int, x: np.ndarray) -> np.ndarray:
    return np.array([0.00], dtype=float)


# -----------------------------
# Helpers for feasibility scan
# -----------------------------
def build_psf(rho_bar: float) -> MPCPredictSafetyFilter:
    model = SinglePendulumCasadi(xbar=xbar.copy())
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
        rho=RHO_FIXED,        # None -> scheduled PSF
        rho_bar=rho_bar,
        rho_max=RHO_MAX,
    )
    return psf

def first_step_feasible(psf: MPCPredictSafetyFilter, theta0: float, omega0: float = 0.0) -> bool:
    x0_test = np.array([theta0, omega0], dtype=float)
    uL0 = desired_action_uL(0, x0_test)
    try:
        U_sol, X_sol, _ = psf.solve_mpc(x0_test, xbar, uL0)
        return (U_sol is not None and X_sol is not None)
    except Exception:
        return False

def clamp_theta(th: float) -> float:
    return float(np.clip(th, SAFE_TH_LO, SAFE_TH_HI))

def scan_theta_range_for_rhobar(rho_bar: float,
                                center_theta: float = np.pi,
                                omega0: float = 0.0,
                                coarse_step: float = 0.02,
                                bisect_tol: float = 1e-3,
                                max_steps: int = 1000):
    """
    Expand around center_theta in both directions until the first-step becomes infeasible.
    Then refine the boundary with a simple bisection. Returns (theta_min, theta_max).
    """
    psf = build_psf(rho_bar)

    center_theta = clamp_theta(center_theta)
    if not first_step_feasible(psf, center_theta, omega0):
        # If even the center is infeasible, return empty range (or the center itself)
        return center_theta, center_theta

    # --- scan downward (lower theta) ---
    th = center_theta
    for k in range(1, max_steps+1):
        cand = clamp_theta(center_theta - k*coarse_step)
        if cand == th:  # hit bound
            break
        feas = first_step_feasible(psf, cand, omega0)
        if not feas:
            # bracket [cand, th] where th is last feasible
            lo, hi = cand, th
            while (hi - lo) > bisect_tol:
                mid = 0.5*(lo + hi)
                if first_step_feasible(psf, mid, omega0):
                    hi = mid
                else:
                    lo = mid
            theta_min = hi
            break
        th = cand
    else:
        theta_min = th
    if 'theta_min' not in locals():
        theta_min = th

    # --- scan upward (higher theta) ---
    th = center_theta
    for k in range(1, max_steps+1):
        cand = clamp_theta(center_theta + k*coarse_step)
        if cand == th:
            break
        feas = first_step_feasible(psf, cand, omega0)
        if not feas:
            lo, hi = th, cand
            while (hi - lo) > bisect_tol:
                mid = 0.5*(lo + hi)
                if first_step_feasible(psf, mid, omega0):
                    lo = mid
                else:
                    hi = mid
            theta_max = lo
            break
        th = cand
    else:
        theta_max = th
    if 'theta_max' not in locals():
        theta_max = th

    return float(theta_min), float(theta_max)


# -----------------------------
# One closed-loop run (PSF + plant) for a given rho_bar and x0
# -----------------------------
def simulate_one_rhobar(rho_bar: float, x0_override=None):
    import torch
    model = SinglePendulumCasadi(xbar=xbar.copy())
    L_link = float(model.l)
    plant = SinglePendulum(
        xbar=torch.tensor(xbar, dtype=torch.double),
        x_init=torch.tensor((x0_override if x0_override is not None else np.array([np.pi/2, 0.0], dtype=float)),
                            dtype=torch.double).view(1, -1),
        u_init=torch.zeros(1, 1, dtype=torch.double),
    )

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
    if x0_override is None:
        x0_used = np.array([np.pi/2, 0.0], dtype=float)
    else:
        x0_used = np.array(x0_override, dtype=float)

    th_log   = [x0_used[0]]
    om_log   = [x0_used[1]]
    u_log    = []
    Jstar_log= []
    feas_log = []

    # For obstacle shading
    obs_arcs = []

    # Internal warm starts
    U_prev = None
    X_prev = None

    # Current plant state
    x_t = plant.x_init.view(1,1,-1)  # already set above

    for t in range(T_STEPS):
        uL = desired_action_uL(t, x_t.detach().cpu().numpy().reshape(-1))
        x_np = x_t.detach().cpu().numpy().reshape(-1)

        try:
            if t == 0:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, xbar, uL)
            else:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, xbar, uL, U_prev, X_prev)
        except Exception:
            U_sol, X_sol, J_curr = None, None, None

        if U_sol is None or X_sol is None:
            u_cmd = uL.copy().reshape(1,1,1)
            U_prev, X_prev = None, None
            feas_log.append(False)
            Jstar_log.append(np.nan)
        else:
            u_cmd = U_sol[:, 0:1].reshape(1,1,1)
            U_prev, X_prev = U_sol, X_sol
            feas_log.append(True)
            Jstar_log.append(float(J_curr))

        # Plant step (filtered input)
        import torch as _torch
        u_torch = _torch.tensor(u_cmd, dtype=_torch.double)
        x_t = plant.rk4_integration(x_t, u_torch)

        # Log state/input
        th_log.append(float(x_t.view(-1)[0]))
        om_log.append(float(x_t.view(-1)[1]))
        u_log.append(float(u_cmd.reshape(-1)[0]))

        # obstacle arcs at time t
        centers_t = obstacle_centers_at_step(t)
        arcs_t = []
        for i in range(centers_t.shape[0]):
            arcs_t.extend(theta_arcs_hit_obstacle(
                centers_t[i], L_link, cov_var=float(OBS_COVS[i]), chi2_val=CHI2_2_95
            ))
        obs_arcs.append(arcs_t)

    t_axis = np.arange(T_STEPS+1) * DT
    t_axis_u = np.arange(T_STEPS) * DT

    return {
        "rho_bar": rho_bar,
        "x0": x0_used.copy(),
        "t": t_axis,
        "t_u": t_axis_u,
        "theta": np.array(th_log),
        "omega": np.array(om_log),
        "u": np.array(u_log),
        "Jstar": np.array(Jstar_log),
        "feasible": np.array(feas_log, dtype=bool),
        "obstacle_arcs": obs_arcs,
    }


# -----------------------------
# Run: scan ranges and plot tubes
# -----------------------------
if __name__ == "__main__":
    center_theta = np.pi
    omega0 = 0.0

    # 1) Scan feasible θ-range for each rho_bar
    ranges = {}  # rho_bar -> (theta_min, theta_max)
    print("\n=== First-step feasible θ-range around π (ω0=0) ===")
    for rb in RHOBAR_LIST:
        th_min, th_max = scan_theta_range_for_rhobar(
            rb, center_theta=center_theta, omega0=omega0,
            coarse_step=0.02, bisect_tol=1e-3
        )
        ranges[rb] = (th_min, th_max)
        deg = lambda r: 180.0*r/np.pi
        print(f"rho_bar={rb:.2f}: θ ∈ [{th_min:.4f}, {th_max:.4f}] rad  "
              f"(≈ [{deg(th_min):.2f}°, {deg(th_max):.2f}°], width {th_max-th_min:.4f} rad)")

    # 2) Simulate at the two boundary ICs for each rho_bar and plot the tube
    cmap = plt.get_cmap("tab10")
    color_map = {rb: cmap(i % 10) for i, rb in enumerate(RHOBAR_LIST)}

    # Gather runs
    tubes = {}  # rb -> dict with 'low' and 'high'
    for rb in RHOBAR_LIST:
        th_min, th_max = ranges[rb]
        x0_low  = np.array([th_min, omega0], dtype=float)
        x0_high = np.array([th_max, omega0], dtype=float)
        run_low  = simulate_one_rhobar(rb, x0_override=x0_low)
        run_high = simulate_one_rhobar(rb, x0_override=x0_high)
        tubes[rb] = {"low": run_low, "high": run_high}

    # Extra trajectory for θ(0)=2, ω(0)=0 with ρ̄ = 0.50
    rb_mid = 0.50
    x0_mid = np.array([2.0, 0.0], dtype=float)
    run_mid = simulate_one_rhobar(rb_mid, x0_override=x0_mid)

    # ---------- FIGURE 1: θ(t) tubes with obstacle-arc shading ----------
    fig1, ax1 = plt.subplots(figsize=(10, 4.8))
    # obstacle shading (same across runs)
    any_run = next(iter(tubes.values()))["low"]
    arc_runs = any_run["obstacle_arcs"]
    for k, arcs in enumerate(arc_runs):
        if not arcs: continue
        x_span = [k*DT, (k+1)*DT]
        for (th_lo, th_hi) in arcs:
            ax1.fill_between(
                x_span, [th_lo, th_lo], [th_hi, th_hi],
                facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', linewidth=0.0, zorder=0
            )

    # plot tubes per rho_bar
    for rb in RHOBAR_LIST:
        color = color_map[rb]
        runL = tubes[rb]["low"]
        runH = tubes[rb]["high"]
        t = runL["t"]
        ax1.fill_between(t, runL["theta"], runH["theta"], color=color, alpha=0.25, label=f"ρ̄={rb:.2f} tube")
        ax1.plot(t, runL["theta"], color=color, linewidth=1.5, linestyle='-')
        ax1.plot(t, runH["theta"], color=color, linewidth=1.5, linestyle='-')

    # overlay θ(t) for θ0=2, ρ̄=0.50
    mid_color = color_map.get(rb_mid, 'k')
    ax1.plot(run_mid["t"], run_mid["theta"],
             color=mid_color, linewidth=2.2, linestyle='--',
             label=r"$\theta_0=2$, $\bar{\rho}=0.50$")

    ax1.axhline(np.pi, linestyle="--", linewidth=1, color="k", label="θ*=π")
    ax1.axhline(SAFE_TH_LO, color="k", lw=0.8, ls=":")
    ax1.axhline(SAFE_TH_HI, color="k", lw=0.8, ls=":")
    ax1.set_title("θ(t) tubes from boundary ICs (per ρ̄) with obstacle arc shaded")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("θ [rad]")
    ax1.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Patch(facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', label='Obstacle (95%)'))
    ax1.legend(handles=handles, loc="best")

    # ---------- FIGURE 2: Phase plane (θ, ω) for θ0 ∈ [0.5, 2.0], ω0=0, ρ̄=0.5, u_L=0 ----------
    from matplotlib import cm, colors as mcolors
    TH0_MIN, TH0_MAX = 0.5, 2.0
    N_TH0 = 2 # number of initial theta samples
    TH0_LIST = np.linspace(TH0_MIN, TH0_MAX, N_TH0)

    fig2, ax2 = plt.subplots(figsize=(6.8, 5.6))
    cmap_phase = cm.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=TH0_MIN, vmax=TH0_MAX)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_phase)

    for th0 in TH0_LIST:
        run = simulate_one_rhobar(0.50, x0_override=np.array([th0, 0.0], dtype=float))
        col = cmap_phase(norm(th0))
        ax2.plot(run["theta"], run["omega"], color=col, linewidth=1.8)
        # mark start/end
        ax2.plot(run["theta"][0], run["omega"][0], marker='o', color=col, markersize=4)
        ax2.plot(run["theta"][-1], run["omega"][-1], marker='x', color=col, markersize=5)

    # goal and safe bounds
    ax2.plot([np.pi], [0.0], marker='*', color='k', markersize=10, label="target (π, 0)")
    ax2.axvline(SAFE_TH_LO, ls=":", c="k", lw=0.8)
    ax2.axvline(SAFE_TH_HI, ls=":", c="k", lw=0.8)

    ax2.set_title(r"Phase plane: $\theta$–$\omega$ trajectories ($u_L=0$, $\bar{\rho}=0.5$)")
    ax2.set_xlabel(r"$\theta$ [rad]")
    ax2.set_ylabel(r"$\omega$ [rad/s]")
    ax2.grid(True, alpha=0.3)

    # colorbar for θ0
    cbar = fig2.colorbar(sm, ax=ax2, pad=0.02)
    cbar.set_label(r"initial $\theta_0$ [rad]")

    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()
