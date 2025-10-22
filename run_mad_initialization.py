# run_mad_observe_rho.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Callable, Dict

from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

# =========================
# Config
# =========================
DT = 0.05
HORIZON = 20
T_STEPS = 150

# PSF scheduling
EPSILON = 0.05
RHO_MAX = 1.0
RHOBAR_SCHEDULED = 0.50   # scheduler baseline (rho=None)

# Constraints
SAFE_TH_LO = 0.5
SAFE_TH_HI = 2*np.pi - 0.5
U_MIN, U_MAX = -5.0, 5.0

# Quadratic weights
q_theta = 50.0
q_theta_dot = 10.0
r_u = 0.1
Q = np.diag([q_theta, q_theta_dot]).astype(float)
R = np.array([[r_u]], dtype=float)

# Target and initial condition
XBAR = np.array([np.pi, 0.0], dtype=float)
X0_MAIN = np.array([np.pi/2, 0.0], dtype=float)

# Sweep for modulus r  (restricted to [0.95, 0.99])
R_LIST = np.linspace(0.95, 0.99, 6)

# tiny floor to avoid u_L=0 nondifferentiability
U_L_FLOOR = 1e-6

# =========================
# Obstacle settings (Gaussian, moving) + helpers
# =========================
OBS_CENTERS = np.array([[1.0, 0.5]], dtype=float)     # (n_obs, 2)
OBS_COVS    = np.array([0.005], dtype=float)          # isotropic variance
OBS_VEL     = np.array([[-0.2, 0.0]], dtype=float)    # (n_obs, 2)
CHI2_2_95   = 5.991464547107979                       # chi2(2) 95%
OBSTACLE_SHADE_COLOR = (0.85, 0.2, 0.2, 0.18)         # soft red with alpha

def obstacle_centers_at_step(t_idx: int) -> np.ndarray:
    return OBS_CENTERS + t_idx * DT * OBS_VEL

def wrap_0_2pi(a):
    return (a + 2*np.pi) % (2*np.pi)

def theta_arcs_hit_obstacle(center_xy, link_length, cov_var, chi2_val=CHI2_2_95):
    """
    For a circular obstacle centered at 'center_xy' with radius r = sqrt(chi2*cov),
    return intervals of θ ∈ [0,2π) such that tip p(θ) = [L sin θ, -L cos θ] lies inside the circle.
    Returns a list of (θ_lo, θ_hi); may be empty or split into two.
    """
    r = np.sqrt(chi2_val * cov_var)
    c = np.asarray(center_xy, dtype=float)
    L = float(link_length)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        if r >= L: return [(0.0, 2*np.pi)]
        else:      return []
    # On the unit circle parameterized by α (tip angle), our tip angle α = θ - π/2.
    # d^2 = L^2 + ||c||^2 - 2 L ||c|| cos(α - φ) <= r^2
    # -> cos(α - φ) >= (L^2 + ||c||^2 - r^2)/(2 L ||c||) = ζ
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

# =========================
# Generators
# =========================
def A0_linear(x0: np.ndarray, k_theta: float = 2.5, k_omega: float = 0.2) -> float:
    """A0 = kθ|θ0-π| + kω|ω0|, clipped to actuator limits."""
    theta0, omega0 = float(x0[0]), float(x0[1])
    amp = k_theta*abs(theta0 - np.pi) + k_omega*abs(omega0)
    return float(np.clip(amp, 0.0, min(abs(U_MIN), abs(U_MAX))))

def make_uL_generator(r: float, A0_fn: Callable[[np.ndarray], float], x0_ref: np.ndarray):
    """Amplitude-only LRU/MAD: u_L(t)=A0(x0_ref)*r^t*sgn(π-θ_t), clipped."""
    A0_val = max(U_L_FLOOR, A0_fn(x0_ref))
    def uL(t_step: int, x: np.ndarray) -> np.ndarray:
        theta = float(x[0])
        sgn = 1.0 if (np.pi - theta) >= 0.0 else -1.0
        A_t = max(U_L_FLOOR, A0_val*(r**t_step))
        u = float(np.clip(sgn*A_t, U_MIN, U_MAX))
        return np.array([u], dtype=float)
    return uL

def uL_pure_sequence(r: float, A0: float, sign0: float, T: int) -> np.ndarray:
    """
    Pure LRU with max_phase = 0 (no oscillation):
      u_L^pure(t) = sign0 * A0 * r^t, clipped to [U_MIN, U_MAX]
    """
    t = np.arange(T, dtype=float)
    seq = sign0 * (A0 * (r ** t))
    return np.clip(seq, U_MIN, U_MAX)

def uL_complex_sequence(r: float, omega: float, A0: float, sign0: float,
                        T: int, phi0: float = 0.0) -> np.ndarray:
    """
    Pure LRU with eigenvalue angle per step = omega (radians/sample).
    u_L^pure(t) = sign0 * A0 * r^t * cos(omega*t + phi0)
    """
    t = np.arange(T, dtype=float)
    seq = sign0 * A0 * (r ** t) * np.cos(omega * t + phi0)
    return np.clip(seq, U_MIN, U_MAX)

# =========================
# PSF builder
# =========================
def build_psf_scheduled(rho_bar: float) -> MPCPredictSafetyFilter:
    model = SinglePendulumCasadi(xbar=XBAR.copy())
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
        rho=None,                # <-- scheduled
        rho_bar=rho_bar,
        rho_max=RHO_MAX,
    )
    return psf

# =========================
# Simulation + induced rho
# =========================
def induced_rho_from_uL(uL_scalar: float,
                        epsilon: float = EPSILON,
                        rho_bar: float = RHOBAR_SCHEDULED,
                        rho_max: float = RHO_MAX) -> float:
    """
    Matches (piecewise-linear) scheduler:
      ratio = (|uL|-eps)/eps, clipped to [0,1]
      rho   = rho_bar + (rho_max - rho_bar) * ratio_clipped
    """
    ratio = (abs(float(uL_scalar)) - epsilon) / max(epsilon, 1e-12)
    ratio_clipped = float(np.clip(ratio, 0.0, 1.0))
    return float(rho_bar + (rho_max - rho_bar) * ratio_clipped)

def simulate(psf: MPCPredictSafetyFilter,
             x0: np.ndarray,
             uL_fn: Callable[[int, np.ndarray], np.ndarray],
             T: int = T_STEPS,
             compute_rho_induced: bool = False) -> Dict[str, np.ndarray]:
    import torch

    plant = SinglePendulum(
        xbar=torch.tensor(XBAR, dtype=torch.double),
        x_init=torch.tensor(x0, dtype=torch.double).view(1, -1),
        u_init=torch.zeros(1, 1, dtype=torch.double),
    )

    th_log, om_log, u_log, uL_log, Jstar_log, feas_log = [], [], [], [], [], []
    rho_induced_log = []

    U_prev, X_prev = None, None
    x_t = plant.x_init.view(1, 1, -1)

    for t in range(T):
        x_np = x_t.detach().cpu().numpy().reshape(-1)
        uL = uL_fn(t, x_np)

        try:
            if t == 0:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, XBAR, uL)
            else:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, XBAR, uL, U_prev, X_prev)
        except Exception:
            U_sol, X_sol, J_curr = None, None, None

        if U_sol is None or X_sol is None:
            u_cmd = uL.copy().reshape(1, 1, 1)
            U_prev, X_prev = None, None
            feas_log.append(False)
            Jstar_log.append(np.nan)
        else:
            u_cmd = U_sol[:, 0:1].reshape(1, 1, 1)
            U_prev, X_prev = U_sol, X_sol
            feas_log.append(True)
            Jstar_log.append(float(J_curr))

        # integrate plant
        u_torch = torch.tensor(u_cmd, dtype=torch.double)
        x_t = plant.rk4_integration(x_t, u_torch)

        th_log.append(float(x_t.view(-1)[0]))
        om_log.append(float(x_t.view(-1)[1]))
        u_log.append(float(u_cmd.reshape(-1)[0]))
        uL_log.append(float(uL.reshape(-1)[0]))

        if compute_rho_induced:
            rho_induced_log.append(induced_rho_from_uL(uL_log[-1]))

    t_axis = np.arange(T + 1) * DT
    t_axis_u = np.arange(T) * DT
    out = {
        "t": t_axis,
        "t_u": t_axis_u,
        "theta": np.array([float(plant.x_init.view(-1)[0])] + th_log),
        "omega": np.array([float(plant.x_init.view(-1)[1])] + om_log),
        "u": np.array(u_log),
        "uL": np.array(uL_log),
        "Jstar": np.array(Jstar_log),
        "feasible": np.array(feas_log, dtype=bool),
    }
    if compute_rho_induced:
        out["rho_t"] = np.array(rho_induced_log, dtype=float)
    return out

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Colors per r
    cmap = plt.get_cmap("viridis")
    colors = {r: cmap(i / max(1, len(R_LIST)-1)) for i, r in enumerate(R_LIST)}

    # Build PSF (scheduled)
    psf_sched = build_psf_scheduled(RHOBAR_SCHEDULED)

    # Induced rho_t per r (scheduled PSF)
    induced_results = {}
    print("=== Induced ρ_t under scheduled PSF (per r) ===")
    for r in R_LIST:
        uL_fn = make_uL_generator(r, A0_fn=A0_linear, x0_ref=X0_MAIN)
        run = simulate(psf_sched, x0=X0_MAIN, uL_fn=uL_fn, T=T_STEPS, compute_rho_induced=True)
        induced_results[r] = run
        rho_seq = run["rho_t"]
        print(f"r={r:.4f} | ρ_t min={rho_seq.min():.3f}, max={rho_seq.max():.3f}, mean={rho_seq.mean():.3f}")

    # θ(t) overlay with u_L == 0 (scheduled PSF, same rhobar)
    def uL_zero(_t, _x): return np.array([0.0], dtype=float)
    run_theta = simulate(psf_sched, x0=X0_MAIN, uL_fn=uL_zero, T=T_STEPS, compute_rho_induced=False)

    # Precompute obstacle arcs (need link length from model)
    model_tmp = SinglePendulumCasadi(xbar=XBAR.copy())
    L_link = float(model_tmp.l)
    obs_arcs = []
    for k in range(T_STEPS):
        centers_t = obstacle_centers_at_step(k)  # (n_obs, 2)
        arcs_t = []
        for i in range(centers_t.shape[0]):
            arcs_t.extend(theta_arcs_hit_obstacle(
                centers_t[i], L_link, cov_var=float(OBS_COVS[i]), chi2_val=CHI2_2_95
            ))
        obs_arcs.append(arcs_t)

    # === FIGURE 1: Induced rho_t + theta(t) + obstacle arcs ===
    fig, ax_rho = plt.subplots(figsize=(10, 5.0))

    # left axis: induced rho_t per r
    for r, run in induced_results.items():
        ax_rho.plot(run["t_u"], run["rho_t"], label=f"r={r:.3f}", color=colors[r])
    ax_rho.set_ylim(0.0, 1.02)
    ax_rho.set_xlabel("time [s]")
    ax_rho.set_ylabel(r"induced $\rho_t$")
    ax_rho.grid(True, alpha=0.3)
    ax_rho.legend(ncol=3, loc="upper left")

    # right axis: theta(t) for u_L=0, rhobar=0.5
    ax_th = ax_rho.twinx()

    # obstacle shading on theta-axis
    for k, arcs in enumerate(obs_arcs):
        if not arcs:
            continue
        x_span = [k*DT, (k+1)*DT]
        for (th_lo, th_hi) in arcs:
            ax_th.fill_between(
                x_span,
                [th_lo, th_lo],
                [th_hi, th_hi],
                facecolor=OBSTACLE_SHADE_COLOR,
                edgecolor='none',
                linewidth=0.0,
                zorder=0
            )

    # theta(t) overlay
    ax_th.plot(run_theta["t"], run_theta["theta"], color="k", linewidth=2.0,
               label=r"$\theta(t)$ | $u_L=0$, $\bar{\rho}=0.5$")
    ax_th.axhline(np.pi, ls="--", c="k", lw=1, alpha=0.6)
    ax_th.axhline(SAFE_TH_LO, ls=":", c="k", lw=0.8)
    ax_th.axhline(SAFE_TH_HI, ls=":", c="k", lw=0.8)
    ax_th.set_ylabel(r"$\theta$ [rad] (right axis)")

    # Legends
    handles_th, labels_th = ax_th.get_legend_handles_labels()
    handles_th.append(Patch(facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', label='Obstacle (95%)'))
    ax_th.legend(handles_th, labels_th, loc="upper right")

    # Title
    ax_rho.set_title(rf"Induced $\rho_t$ (scheduled PSF, $\bar{{\rho}}={RHOBAR_SCHEDULED:.2f}$) "
                     r"with $\theta(t)$ and obstacle arcs")

    plt.tight_layout()

    # === FIGURE 2: Pure LRU with different phase (omega) ===
    # Choose a fixed radius and several per-step angles (phase increments)
    R_FIXED = 0.96
    OMEGA_LIST = [0.0, np.pi/8, np.pi/4, 3*np.pi/8]   # radians per sample
    PHI0 = 0.0

    A0_val = max(U_L_FLOOR, A0_linear(X0_MAIN))
    sign0 = 1.0 if (np.pi - X0_MAIN[0]) >= 0.0 else -1.0
    t_u = np.arange(T_STEPS) * DT

    plt.figure(figsize=(10, 3.8))
    tab = plt.get_cmap("tab10")

    def format_omega(w):
        # pretty print e.g. 0.25π
        frac = w / np.pi
        return f"{frac:.2g}π"

    for i, w in enumerate(OMEGA_LIST):
        ul = uL_complex_sequence(R_FIXED, w, A0=A0_val, sign0=sign0, T=T_STEPS, phi0=PHI0)
        plt.plot(t_u, ul, label=rf"$\omega={format_omega(w)}$", color=tab(i % 10))

    plt.axhline(U_MIN, ls="--", c="k", lw=1)
    plt.axhline(U_MAX, ls="--", c="k", lw=1)
    plt.xlabel("time [s]")
    plt.ylabel(r"$u_L^{pure}$ [Nm]")
    plt.title(rf"Pure LRU dynamics for different per-step phases $\omega$ (phase$_0$={PHI0:.1f}, $r={R_FIXED}$)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4)

    plt.show()
