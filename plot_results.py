#!/usr/bin/env python3
"""
plot_results.py (equilibrium-at-zero view)

- Loads results/<RUN_FOLDER_NAME>/model_best.pth (best actor).
- Rebuilds env+controller, generates u_L online from the best actor.
- Runs PSF forward (baseline vs learned).
- Figure 1: plots θ_error(t) = wrap_pi(θ(t) - THETA_REF) with obstacle band mapped into error frame.
  • Only LOWER θ bound (also mapped) is shown; y-lims are [lower_bound_error - 0.1, FIG1_MAXY].
  • Optional θ0 sweep overlay.
- Figure 2: ρ_t (kept with title/legend).
- Figure 3: J*(t) with shared anchor (NO title/legend), now with θ0 sweep overlay like Figure 1.
- Figure 4: three subplots for the single θ0:
    (a) θ (error frame if enabled) and ω vs time
    (b) u_L vs applied u
    (c) (u_L - u) for the learned approach

Saves Figure 1 and Figure 3 into results/plots and also shows all figures.
"""

# =========================== IMPORTS (TOP) ===========================
import os, re, ast
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Project imports (run from repo root)
from mad_controller import MADController
from pendulum_env import PendulumEnv
from obstacles import MovingObstacle
from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

# =================== HYPERPARAMETERS / USER SETTINGS =================
# --- Run / model selection ---
RUN_FOLDER_NAME = "PSF_SSM_NS_10_30_11_17_51"  # results/<RUN_FOLDER_NAME>

THETA0 = 1.57079632679  # single run initial angle [rad]; None -> π/2
RHO_FIXED = 0.5  # baseline (and optional learned) fixed ρ
LEARNED_RHO_MODE = "scheduled"  # "scheduled" or "fixed"

# --- Display: shift to equilibrium at 0 ---
SHIFT_TO_ERROR_FRAME = True  # if True, plot θ̃=wrap_pi(θ-THETA_REF) with THETA_REF as desired eq.
THETA_REF = np.pi  # desired equilibrium in absolute angle (π is upright)

# --- Visual anchor for J* plot (baseline-derived “one-step increase”) ---
PLOT_SHARED_ANCHOR = True

# --- θ0 sweep (FIRST & THIRD plot overlays) ---
ENABLE_THETA0_SWEEP = True  # set False to disable overlay
THETA0_MIN = -2.6 + 3.14  # radians conversion for 0 in pi
THETA0_MAX = 2.5 + 3.14  # radians conversion for 0 in pi
THETA0_N = 3  # number of θ0 samples in [min,max]
PLOT_BASELINE_IN_SWEEP = True  # overlay PSF-only curves (baseline)
PLOT_LEARNED_IN_SWEEP = True  # overlay Actor+PSF curves (learned)
rho_bar = 0.5
rho_max = 10.0
PSFhorizon = 20

# --- Publication styling knobs (labels & tick numbers) ---
AXIS_LABEL_FONTSIZE = 16  # fontsize for x/y labels (figs 1–3)
TICK_LABEL_FONTSIZE = 13  # fontsize for tick numbers (figs 1–3)
FIG1_SIZE = (10, 4.8)  # inches
FIG3_SIZE = (10, 4.6)  # inches
SAVE_DPI = 300

# Optional custom ticks (set to None to use Matplotlib defaults)
FIG1_XTICKS = None  # e.g., np.arange(0, 12.5, 2.5)
FIG1_YTICKS = None  # e.g., [-3, -2, -1, 0, 1, 2, 3]
FIG3_XTICKS = None
FIG3_YTICKS = None
FIG1_MAXY = max(0.86, 2.7)  # max y-limit for fig 1

# --- Colors (approaches & obstacle) ---
COLOR_BASELINE = 'tab:orange'  # baseline (fixed ρ / rhobar)
COLOR_LEARNED = 'tab:blue'  # learned (scheduled or fixed)
OBSTACLE_SHADE_COLOR = (0.95, 0.55, 0.55, 0.35)  # RGBA for obstacle band shading
# =====================================================================

# Matplotlib baseline config
mpl.rcParams['text.usetex'] = False
torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Resolve paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", RUN_FOLDER_NAME)
BEST_PATH = os.path.join(RESULTS_DIR, "model_best.pth")
LOG_PATH = os.path.join(RESULTS_DIR, "training.log")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")  # <- where figs 1 & 3 are saved
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(RESULTS_DIR):
    raise FileNotFoundError(f"Results dir not found: {RESULTS_DIR}")
if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(f"Best model not found: {BEST_PATH}")
if not os.path.exists(LOG_PATH):
    print(f"[WARN] training.log not found at {LOG_PATH}. Will use default hyperparams.")

# ---------- Tiny log parser ----------
FLOAT_RE = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'


def parse_last_float(line, default=None):
    ms = re.findall(FLOAT_RE, line);
    return float(ms[-1]) if ms else default


def parse_last_int(line, default=None):
    ms = re.findall(r'\d+', line);
    return int(ms[-1]) if ms else default


def parse_tensor_like(line, default=None):
    m = re.search(r'\[(.*)\]', line)
    if not m: return default
    inner = m.group(0)
    try:
        return np.array(ast.literal_eval(inner), dtype=float)
    except Exception:
        vals = re.findall(FLOAT_RE, inner)
        return np.array([float(v) for v in vals], dtype=float) if vals else default


def load_meta_from_log(log_path):
    meta = {}
    if not os.path.exists(log_path): return meta
    with open(log_path, 'r') as f:
        for ln in f:
            if 'Simulation horizon (T)' in ln:
                meta['sim_horizon'] = parse_last_int(ln)
            elif 'Epsilon for PSF' in ln:
                meta['epsilon'] = parse_last_float(ln)
            elif 'RENs dim_internal' in ln:
                meta['dim_internal'] = parse_last_int(ln)
            elif 'obs_centers' in ln:
                meta['obs_centers'] = parse_tensor_like(ln)
            elif 'obs_covs' in ln:
                meta['obs_covs'] = parse_tensor_like(ln)
            elif 'obs_vel' in ln:
                meta['obs_vel'] = parse_tensor_like(ln)
            elif 'state_lower_bound' in ln:
                meta['state_lower_bound'] = parse_tensor_like(ln)
            elif 'state_upper_bound' in ln:
                meta['state_upper_bound'] = parse_tensor_like(ln)
            elif 'controller_lower_bound' in ln or 'control_lower_bound' in ln:
                meta['control_lower_bound'] = parse_tensor_like(ln)
            elif 'controller_upper_bound' in ln or 'control_upper_bound' in ln:
                meta['control_upper_bound'] = parse_tensor_like(ln)
    return meta


meta = load_meta_from_log(LOG_PATH)

# ---------- Defaults (match training script) ----------
# --- MODIFIED: Renamed sim_horizon to train_horizon ---
train_horizon = int(meta.get('sim_horizon', 250))
plot_horizon = train_horizon + 50  # Define longer horizon for plotting
# --- END MODIFIED ---
epsilon = float(meta.get('epsilon', 0.05))
DT = 0.05

state_lower_bound = np.array(meta.get('state_lower_bound', [0.5, -20.0]), dtype=float)
state_upper_bound = np.array(meta.get('state_upper_bound', [2 * np.pi - 0.5, 20.0]), dtype=float)
control_lower_bound = np.array(meta.get('control_lower_bound', [-3.0]), dtype=float)
control_upper_bound = np.array(meta.get('control_upper_bound', [3.0]), dtype=float)
obs_centers = np.array(meta.get('obs_centers', [[1.0, 0.5]]), dtype=float).reshape(-1, 2)
obs_covs = np.array(meta.get('obs_covs', [0.005]), dtype=float).reshape(-1)
obs_vel = np.array(meta.get('obs_vel', [[-0.2, 0.0]]), dtype=float).reshape(-1, 2)

if THETA0 is None:
    THETA0 = np.pi / 2


# ---------- Auto-detect SSM size from checkpoint ----------
def load_actor_state_dict_from_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 1: return ckpt[0]
    if isinstance(ckpt, dict):
        for k in ('actor', 'actor_state_dict', 'actor_model', 'state_dict'):
            if k in ckpt:
                if k == 'state_dict':
                    sd = {kk.split('actor_model.', 1)[-1]: vv for kk, vv in ckpt[k].items()
                          if kk.startswith('actor_model.')}
                    if sd: return sd
                else:
                    return ckpt[k]
        return ckpt
    raise RuntimeError("Unrecognized checkpoint format for actor weights.")


def detect_dim_internal(actor_sd):
    candidates = ['m_dynamics.LRUR.B', 'm_dynamics.model.0.B',
                  'm_dynamics.LRUR.C', 'm_dynamics.model.0.C',
                  'm_dynamics.LRUR.state', 'm_dynamics.model.0.state',
                  'm_dynamics.LRUR.nu_log', 'm_dynamics.model.0.nu_log']
    for key in candidates:
        if key in actor_sd:
            t = actor_sd[key]
            shape = tuple(t.shape) if isinstance(t, torch.Tensor) else tuple(t.size())
            if key.endswith('.B'): return int(shape[0])  # (k,2)
            if key.endswith('.C'): return int(shape[1])  # (1,k)
            return int(shape[0])  # vectors
    return None


actor_sd = load_actor_state_dict_from_checkpoint(BEST_PATH)
detected_dim = detect_dim_internal(actor_sd)
log_dim = int(meta.get('dim_internal')) if 'dim_internal' in meta else None
if detected_dim is None and log_dim is None:
    num_dynamics_states = 24
    print("[WARN] Could not detect SSM size from checkpoint/log; using default 24.")
elif detected_dim is None:
    num_dynamics_states = log_dim;
    print(f"[INFO] Using SSM size from log: {num_dynamics_states}")
elif log_dim is None:
    num_dynamics_states = detected_dim;
    print(f"[INFO] Using SSM size from checkpoint: {num_dynamics_states}")
else:
    num_dynamics_states = detected_dim
    if detected_dim != log_dim:
        print(
            f"[WARN] Log dim_internal={log_dim}, checkpoint shows {detected_dim}. Using {detected_dim} to match checkpoint.")

# ---------- Build obstacle signal (same as training) ----------
# --- MODIFIED: Use plot_horizon for T_total ---
T_total = plot_horizon + 1 + 500
# --- END MODIFIED ---
t = torch.arange(T_total, dtype=torch.double)
obs_pos_raw = torch.tensor(obs_centers, dtype=torch.double) + (t[:, None] * DT) * torch.tensor(obs_vel,
                                                                                               dtype=torch.double)
obs_pos = obs_pos_raw.unsqueeze(0)  # (1, T_total, 2)
moving_obs = MovingObstacle(obs_pos, torch.tensor(obs_covs, dtype=torch.double))

# ---------- Rebuild env ----------
q_theta, q_theta_dot, r_u = 50.0, 10.0, 0.1
Qlyapunov = np.diag([q_theta, q_theta_dot]).astype(float)
Rlyapunov = np.array([[r_u]], dtype=float)
target_positions = torch.tensor([np.pi, 0.0], dtype=torch.double)  # absolute reference

env = PendulumEnv(
    initial_state_low=torch.tensor([THETA0, 0.0], dtype=torch.double),
    initial_state_high=torch.tensor([THETA0, 0.0], dtype=torch.double),
    state_limit_low=torch.tensor(state_lower_bound, dtype=torch.double),
    state_limit_high=torch.tensor(state_upper_bound, dtype=torch.double),
    control_limit_low=torch.tensor(control_lower_bound, dtype=torch.double),
    control_limit_high=torch.tensor(control_upper_bound, dtype=torch.double),
    alpha_state=0.03, alpha_control=2.0, alpha_obst=3.0,
    obstacle=moving_obs, obstacle_avoidance=True,
    alpha_cer=2.0, control_reward_regularization=True,
    epsilon=epsilon,
    rho=0.5, rho_bar=rho_bar, rho_max=rho_max,
    Qlyapunov=Qlyapunov, Rlyapunov=Rlyapunov,
    # --- MODIFIED: Pass train_horizon to env ---
    sim_horizon=train_horizon,
    # --- END MODIFIED ---
    obs_vel=torch.tensor(obs_vel, dtype=torch.double)
)

# ---------- Controller with *matching* SSM size ----------
mad = MADController(
    env=env,
    buffer_capacity=100000,
    target_state=target_positions,
    num_dynamics_states=num_dynamics_states,
    # --- MODIFIED: Pass train_horizon to controller ---
    dynamics_input_time_window_length=train_horizon + 1,
    # --- END MODIFIED ---
    batch_size=64,
    gamma=0.99, tau=0.005,
    actor_lr=5e-4, critic_lr=1e-3,
    std_dev=0.5,
    control_action_upper_bound=float(control_upper_bound[0]),
    control_action_lower_bound=float(control_lower_bound[0]),
    nn_type='mad'
)

# Load best weights
mad.load_model_weight(BEST_PATH)
print(f"DDPG Model weights loaded from {BEST_PATH} successfully.")

# ---------- PSF model ----------
model = SinglePendulumCasadi(xbar=np.array([np.pi, 0.0], dtype=float))
L_link = float(model.l)


def make_psf(rho_mode, rho_fixed_val):
    return MPCPredictSafetyFilter(
        model,
        horizon=PSFhorizon,
        state_lower_bound=state_lower_bound.astype(float),
        state_upper_bound=state_upper_bound.astype(float),
        control_lower_bound=control_lower_bound.astype(float),
        control_upper_bound=control_upper_bound.astype(float),
        Q=Qlyapunov, R=Rlyapunov, solver_opts=None,
        set_lyacon=True,
        epsilon=epsilon,
        rho=(rho_fixed_val if rho_mode == 'fixed' else None),
        rho_bar=rho_bar,
        rho_max=rho_max,
    )


# ---------- Helpers ----------
# CHI2 = -2 ln(1-%)
# safety margin
CHI2_2_93 = 5.31852007387


def wrap_pi(x):  # Wrap to (-pi, pi]
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrap_0_2pi(x): return np.mod(x, 2 * np.pi)


def theta_arcs_hit_obstacle(center_xy, L, cov_var):
    r = np.sqrt(CHI2_2_93 * float(cov_var))
    c = np.asarray(center_xy, dtype=float).reshape(2, )
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12: return [(0.0, 2 * np.pi)]
    zeta = (L ** 2 + norm_c ** 2 - r ** 2) / (2.0 * L * norm_c)
    if zeta <= -1.0: return [(0.0, 2 * np.pi)]
    if zeta >= 1.0:  return []
    delta = np.arccos(zeta);
    phi = np.arctan2(c[1], c[0])
    th_lo = wrap_0_2pi(phi - delta + np.pi / 2);
    th_hi = wrap_0_2pi(phi + delta + np.pi / 2)
    return [(th_lo, th_hi)] if th_lo <= th_hi else [(0.0, th_hi), (th_lo, 2 * np.pi)]


def rho_schedule_from_uL(uL_scalar, rho_bar, rho_max, epsilon):
    r2 = float(uL_scalar ** 2);
    sigma = r2 / (epsilon * epsilon + r2)
    return float(rho_bar + (rho_max - rho_bar) * sigma)


def arcs_absolute_to_error(arcs_abs, theta_ref):
    """Map absolute-θ arcs [θ_lo, θ_hi] in [0,2π) into error arcs over (-π,π] by subtracting theta_ref and wrapping."""
    out = []
    for th_lo, th_hi in arcs_abs:
        lo_e = wrap_pi(th_lo - theta_ref);
        hi_e = wrap_pi(th_hi - theta_ref)
        if lo_e <= hi_e:
            out.append((lo_e, hi_e))
        else:
            out.append((-np.pi, hi_e)); out.append((lo_e, np.pi))
    return out


# ---------- (1) Single-run baseline ----------
psf_base = make_psf('fixed', RHO_FIXED)
plant_single = SinglePendulum(
    xbar=torch.tensor([np.pi, 0.0], dtype=torch.double),
    x_init=torch.tensor([THETA0, 0.0], dtype=torch.double).view(1, -1),
    u_init=torch.zeros(1, 1, dtype=torch.double),
)
theta_base, omega_base, u_cmd_base, rho_base, J_base = [THETA0], [0.0], [], [], []
U_prev = X_prev = None
x_t = torch.tensor([THETA0, 0.0], dtype=torch.double).view(1, 1, -1)
# --- MODIFIED: Use plot_horizon ---
for k in range(plot_horizon):
    # --- END MODIFIED ---
    x_np = x_t.view(-1).cpu().numpy()
    uL = np.zeros((1,), dtype=float)
    try:
        if k == 0:
            U_sol, X_sol, J_curr = psf_base.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL)
        else:
            U_sol, X_sol, J_curr = psf_base.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL, U_prev, X_prev)
    except Exception:
        U_sol = X_sol = J_curr = None
    if U_sol is None:
        u_cmd = uL.reshape(1, 1, 1);
        U_prev = X_prev = None;
        J_base.append(np.nan)
    else:
        u_cmd = U_sol[:, 0:1].reshape(1, 1, 1);
        U_prev, X_prev = U_sol, X_sol;
        J_base.append(float(J_curr))
    u_cmd_base.append(float(u_cmd.reshape(-1)[0]))
    rho_base.append(RHO_FIXED)
    x_t = plant_single.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
    x_vec = x_t.view(-1).cpu().numpy()
    theta_base.append(float(x_vec[0]));
    omega_base.append(float(x_vec[1]))
theta_base = np.array(theta_base);
omega_base = np.array(omega_base)
u_cmd_base = np.array(u_cmd_base);
rho_base = np.array(rho_base);
J_base = np.array(J_base)

# ---------- (2) Single-run learned ----------
psf_learn = make_psf(LEARNED_RHO_MODE, RHO_FIXED)
plant_single_L = SinglePendulum(
    xbar=torch.tensor([np.pi, 0.0], dtype=torch.double),
    x_init=torch.tensor([THETA0, 0.0], dtype=torch.double).view(1, -1),
    u_init=torch.zeros(1, 1, dtype=torch.double),
)
theta_learn, omega_learn, uL_log, u_cmd_learn, rho_learn, J_learn = [THETA0], [0.0], [], [], [], []
U_prev = X_prev = None
x_t = torch.tensor([THETA0, 0.0], dtype=torch.double).view(1, 1, -1)

# Manually init env and controller for aug_state
env.state = x_t.clone().to(env.state_limit_low.device)
env.t = 0
aug_state_t = env._get_augmented_state().to(device)
mad.set_ep_initial_state(initial_aug_state=aug_state_t)

mad.reset_ep_timestep()
mad.update_dynamics_input_time_window()
mad.w = torch.zeros(mad.num_physical_states, dtype=torch.double).to(device)

# --- MODIFIED: Use plot_horizon ---
for k in range(plot_horizon):
    # --- END MODIFIED ---
    x_np = x_t.view(-1).cpu().numpy()  # physical state for PSF

    # Get aug_state and call policy
    aug_state_t = env._get_augmented_state().to(device)
    uL_tensor = mad.learned_policy(
        aug_state_t,
        mad.dynamics_input_time_window,
        mad.dynamics_disturbance_time_window
    )

    uL = np.asarray(uL_tensor, dtype=float).reshape(-1)
    uL_log.append(float(uL[0]))
    try:
        if k == 0:
            U_sol, X_sol, J_curr = psf_learn.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL)
        else:
            U_sol, X_sol, J_curr = psf_learn.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL, U_prev, X_prev)
    except Exception:
        U_sol = X_sol = J_curr = None
    if U_sol is None:
        u_cmd = uL.reshape(1, 1, 1);
        U_prev = X_prev = None;
        J_learn.append(np.nan)
    else:
        u_cmd = U_sol[:, 0:1].reshape(1, 1, 1);
        U_prev, X_prev = U_sol, X_sol;
        J_learn.append(float(J_curr))
    u_cmd_learn.append(float(u_cmd.reshape(-1)[0]))
    # ρ_t for plotting
    if LEARNED_RHO_MODE == 'scheduled':
        rho_learn.append(rho_schedule_from_uL(uL[0], rho_bar=rho_bar, rho_max=rho_max, epsilon=epsilon))
    else:
        rho_learn.append(RHO_FIXED)

    x_t = plant_single_L.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))

    # Update env state/time for next loop iter
    env.state = x_t.clone().to(env.state_limit_low.device)
    env.t += 1

    x_vec = x_t.view(-1).cpu().numpy()
    theta_learn.append(float(x_vec[0]));
    omega_learn.append(float(x_vec[1]))
    # update controller episode windows
    mad.update_ep_timestep()
    mad.update_dynamics_input_time_window()

theta_learn = np.array(theta_learn);
omega_learn = np.array(omega_learn)
uL_log = np.array(uL_log);
u_cmd_learn = np.array(u_cmd_learn)
rho_learn = np.array(rho_learn);
J_learn = np.array(J_learn)

# ======================= PLOTTING (SAVE + SHOW) ========================
# --- MODIFIED: Define t_states and t_inputs here using plot_horizon ---
t_states = np.arange(plot_horizon + 1) * DT
t_inputs = np.arange(plot_horizon) * DT
# --- END MODIFIED ---
SAFE_TH_LO = float(state_lower_bound[0])  # absolute lower bound (for frame shift)
SAFE_TH_HI = float(state_upper_bound[0])  # absolute upper bound (for frame shift)
RHO_T = r'$\rho_t$'


# ---------- Visual anchor for J* plot (baseline-derived one-step increase) ----------
def _angdiff(a, b):  # shortest angular difference a->b in [-pi, pi]
    return np.arctan2(np.sin(b - a), np.cos(b - a))


def _first_finite(x, default=0.0):
    for v in np.asarray(x).ravel():
        if np.isfinite(v):
            return float(v)
    return float(default)


# --- MODIFIED: Calculate separate anchors ---
if PLOT_SHARED_ANCHOR:
    e0 = np.array([_angdiff(THETA0, np.pi), 0.0], dtype=float)
    u0_base = np.array([0.0], dtype=float)  # Baseline uL is 0
    u0_learn = np.array([_first_finite(uL_log, 0.0)], dtype=float)  # Learned uL

    # Calculate penalty s(x_0, u_0) for each case
    penalty0_base = float(e0.T @ Qlyapunov @ e0 + u0_base.T @ Rlyapunov @ u0_base)
    # Note: The *learned* anchor uses the *learned* uL_0, but the *baseline* rho (RHO_FIXED)
    # This is consistent with the paper's "one-step increase" idea.
    penalty0_learn = float(e0.T @ Qlyapunov @ e0 + u0_learn.T @ Rlyapunov @ u0_learn)

    J0_base = _first_finite(J_base, default=0.0)
    J0_learn = _first_finite(J_learn, default=0.0)
    eps_margin = 1e-6

    # Anchor for baseline
    J_anchor_base = J0_base + (1.0 - float(RHO_FIXED)) * penalty0_base + eps_margin
    # Anchor for learned (uses its own J0 and penalty, but RHO_FIXED for "shared" logic)
    J_anchor_learn = J0_learn + (1.0 - float(RHO_FIXED)) * penalty0_learn + eps_margin

    # --- MODIFIED: Use t_inputs (which is already plot_horizon long) ---
    t_inputs_plot = np.concatenate(([-DT], t_inputs))
    # --- END MODIFIED ---
    J_base_plot = np.concatenate(([J_anchor_base], J_base))
    J_learn_plot = np.concatenate(([J_anchor_learn], J_learn))
else:
    # --- MODIFIED: Use t_inputs (which is already plot_horizon long) ---
    t_inputs_plot = t_inputs
    # --- END MODIFIED ---
    J_base_plot = J_base
    J_learn_plot = J_learn
# --- END MODIFIED ---

# ---------- θ0 sweep simulations (Figure 1 and Figure 3 overlays) ----------
sweep_data = []  # list of dicts: {theta0, theta_base?, theta_learn?, J_base?, J_learn?}
if ENABLE_THETA0_SWEEP:
    theta0_grid = np.linspace(THETA0_MIN, THETA0_MAX, THETA0_N)
    for th0 in theta0_grid:
        record = {'theta0': float(th0)}
        # Baseline sweep (store theta and J)
        if PLOT_BASELINE_IN_SWEEP:
            psf_b = make_psf('fixed', RHO_FIXED)
            plant_b = SinglePendulum(
                xbar=torch.tensor([np.pi, 0.0], dtype=torch.double),
                x_init=torch.tensor([th0, 0.0], dtype=torch.double).view(1, -1),
                u_init=torch.zeros(1, 1, dtype=torch.double),
            )
            x_t = torch.tensor([th0, 0.0], dtype=torch.double).view(1, 1, -1)
            tb = [th0];
            Jb = []
            U_prev = X_prev = None
            # --- MODIFIED: Use plot_horizon ---
            for k in range(plot_horizon):
                # --- END MODIFIED ---
                x_np = x_t.view(-1).cpu().numpy()
                uL = np.zeros((1,), dtype=float)
                try:
                    if k == 0:
                        U_sol, X_sol, J_curr = psf_b.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL)
                    else:
                        U_sol, X_sol, J_curr = psf_b.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL, U_prev,
                                                               X_prev)
                except Exception:
                    U_sol = X_sol = J_curr = None
                if U_sol is None:
                    u_cmd = uL.reshape(1, 1, 1);
                    U_prev = X_prev = None;
                    Jb.append(np.nan)
                else:
                    u_cmd = U_sol[:, 0:1].reshape(1, 1, 1);
                    U_prev, X_prev = U_sol, X_sol;
                    Jb.append(float(J_curr))
                x_t = plant_b.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
                tb.append(float(x_t.view(-1)[0]))
            record['theta_base'] = np.array(tb);
            record['J_base'] = np.array(Jb)

        # Learned sweep (store theta and J)
        if PLOT_LEARNED_IN_SWEEP:
            psf_l = make_psf(LEARNED_RHO_MODE, RHO_FIXED)
            plant_l = SinglePendulum(
                xbar=torch.tensor([np.pi, 0.0], dtype=torch.double),
                x_init=torch.tensor([th0, 0.0], dtype=torch.double).view(1, -1),
                u_init=torch.zeros(1, 1, dtype=torch.double),
            )
            x_t = torch.tensor([th0, 0.0], dtype=torch.double).view(1, 1, -1)

            # Reset env/actor episode state
            env.state = x_t.clone().to(env.state_limit_low.device)
            env.t = 0
            aug_state_t = env._get_augmented_state().to(device)
            mad.set_ep_initial_state(initial_aug_state=aug_state_t)
            mad.reset_ep_timestep()
            mad.update_dynamics_input_time_window()
            mad.w = torch.zeros(mad.num_physical_states, dtype=torch.double).to(device)

            tl = [th0];
            Jl = [];
            uL_log_sweep = []
            U_prev = X_prev = None
            # --- MODIFIED: Use plot_horizon ---
            for k in range(plot_horizon):
                # --- END MODIFIED ---
                x_np = x_t.view(-1).cpu().numpy()
                aug_state_t = env._get_augmented_state().to(device)
                uL_tensor = mad.learned_policy(
                    aug_state_t,
                    mad.dynamics_input_time_window,
                    mad.dynamics_disturbance_time_window
                )
                uL = np.asarray(uL_tensor, dtype=float).reshape(-1)
                uL_log_sweep.append(float(uL[0]))  # Store uL_0
                try:
                    if k == 0:
                        U_sol, X_sol, J_curr = psf_l.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL)
                    else:
                        U_sol, X_sol, J_curr = psf_l.solve_mpc(x_np, np.array([np.pi, 0.0], dtype=float), uL, U_prev,
                                                               X_prev)
                except Exception:
                    U_sol = X_sol = J_curr = None
                if U_sol is None:
                    u_cmd = uL.reshape(1, 1, 1);
                    U_prev = X_prev = None;
                    Jl.append(np.nan)
                else:
                    u_cmd = U_sol[:, 0:1].reshape(1, 1, 1);
                    U_prev, X_prev = U_sol, X_sol;
                    Jl.append(float(J_curr))

                x_t = plant_l.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
                env.state = x_t.clone().to(env.state_limit_low.device)
                env.t += 1

                tl.append(float(x_t.view(-1)[0]))
                mad.update_ep_timestep()
                mad.update_dynamics_input_time_window()
            record['theta_learn'] = np.array(tl);
            record['J_learn'] = np.array(Jl)
            record['uL0_learn'] = _first_finite(uL_log_sweep, 0.0)  # Store first learned uL

        sweep_data.append(record)


# --- Build error-frame versions for single-run traces (if enabled) ---
def to_error(arr_abs):
    return wrap_pi(arr_abs - THETA_REF) if SHIFT_TO_ERROR_FRAME else arr_abs


theta_base_err = to_error(theta_base)
theta_learn_err = to_error(theta_learn)

# error-frame lower/upper bounds
SAFE_TH_LO_ERR = to_error(np.array([SAFE_TH_LO]))[0]
SAFE_TH_HI_ERR = to_error(np.array([SAFE_TH_HI]))[0]

# 1) θ̃(t) with obstacle band + θ0 sweep overlay (NO title/legend)
fig1, ax1 = plt.subplots(figsize=FIG1_SIZE)

# Sweep overlay: same color per approach, only alpha changes (in error frame)
if ENABLE_THETA0_SWEEP and len(sweep_data) > 0:
    sweep_sorted = sorted(sweep_data, key=lambda d: d['theta0'])
    n = len(sweep_sorted)
    alpha_min, alpha_max = 0.25, 0.90
    for idx, d in enumerate(sweep_sorted):
        a = alpha_min + (alpha_max - alpha_min) * (idx / (n - 1)) if n > 1 else 0.6
        if PLOT_BASELINE_IN_SWEEP and ('theta_base' in d):
            ax1.plot(t_states[:d['theta_base'].shape[0]], to_error(d['theta_base']),
                     ls='--', lw=1.0, color=COLOR_BASELINE, alpha=a, label=None)
        if PLOT_LEARNED_IN_SWEEP and ('theta_learn' in d):
            ax1.plot(t_states[:d['theta_learn'].shape[0]], to_error(d['theta_learn']),
                     ls='-', lw=1.0, color=COLOR_LEARNED, alpha=a, label=None)

# Single-run thick lines (error frame)
ax1.plot(t_states[:theta_base_err.shape[0]], theta_base_err, lw=2.2, color=COLOR_BASELINE)
ax1.plot(t_states[:theta_learn_err.shape[0]], theta_learn_err, lw=2.2, color=COLOR_LEARNED)

# Reference line at θ̃=0 (equilibrium at zero)
ax1.axhline(0.0, linestyle='--', linewidth=1.0, color='k')

# Lower and upper bound (error frame)
ax1.axhline(SAFE_TH_LO_ERR, color='k', lw=0.8, ls=':')
ax1.axhline(SAFE_TH_HI_ERR, color='k', lw=0.8, ls=':')

# Labels and limits
ax1.set_xlabel("time [s]", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_ylabel(r"$\theta$ [rad]" if SHIFT_TO_ERROR_FRAME else r"$\theta$ [rad]",
               fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_ylim([SAFE_TH_LO_ERR - 0.1, FIG1_MAXY])  # requested y-lims

# Obstacle shading — convert arcs to error frame before filling
# --- MODIFIED: Use plot_horizon ---
for k in range(plot_horizon):
    # --- END MODIFIED ---
    centers_k = obs_centers + (k * DT) * obs_vel
    for i in range(centers_k.shape[0]):
        arcs_abs = theta_arcs_hit_obstacle(centers_k[i], L_link, cov_var=float(obs_covs[i]))
        if not arcs_abs: continue
        arcs_err = arcs_absolute_to_error(arcs_abs, THETA_REF) if SHIFT_TO_ERROR_FRAME else arcs_abs
        x_span = [k * DT, (k + 1) * DT]
        for (th_lo_e, th_hi_e) in arcs_err:
            ax1.fill_between(x_span, [th_lo_e, th_lo_e], [th_hi_e, th_hi_e],
                             facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', linewidth=0.0, zorder=0)

ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
if FIG1_XTICKS is not None: ax1.set_xticks(FIG1_XTICKS)
if FIG1_YTICKS is not None: ax1.set_yticks(FIG1_YTICKS)
fig1.tight_layout()

# Save Figure 1
fig1_path = os.path.join(PLOTS_DIR, "figure1_theta_error_overlay.png")
fig1.savefig(fig1_path, dpi=SAVE_DPI, bbox_inches='tight')
print(f"[Saved] {fig1_path}")

# 2) ρ_t (single-run) — keep title/legend (unchanged by frame)
fig2, ax2 = plt.subplots(figsize=(10, 4.6))
ax2.plot(t_inputs[:rho_base.shape[0]], rho_base, label='PSF-only ' + RHO_T + ' (baseline fixed)', lw=1.8,
         color=COLOR_BASELINE)
ax2.plot(t_inputs[:rho_learn.shape[0]], rho_learn,
         label=(
             'Actor ' + RHO_T + ' (scheduled)' if LEARNED_RHO_MODE == 'scheduled' else 'Actor ' + RHO_T + ' (fixed)'),
         lw=1.8, color=COLOR_LEARNED)
ax2.set_title("Tightening schedule " + RHO_T)
ax2.set_xlabel("time [s]", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel(RHO_T, fontsize=AXIS_LABEL_FONTSIZE)
ax2.grid(True, alpha=0.3);
ax2.legend(loc='best')
ax2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
fig2.tight_layout()

# 3) J*(t) with shared anchor — θ0 sweep overlay like Figure 1 (NO title/legend)
fig3, ax3 = plt.subplots(figsize=FIG3_SIZE)

# Sweep overlay for J*: per-theta0 shared anchor so baseline/learned comparable
if ENABLE_THETA0_SWEEP and len(sweep_data) > 0:
    sweep_sorted = sorted(sweep_data, key=lambda d: d['theta0'])
    n = len(sweep_sorted)
    alpha_min, alpha_max = 0.25, 0.90
    for idx, d in enumerate(sweep_sorted):
        a = alpha_min + (alpha_max - alpha_min) * (idx / (n - 1)) if n > 1 else 0.6

        # --- MODIFIED: Calculate separate anchors for sweep ---
        th0 = d['theta0']
        e0 = np.array([_angdiff(th0, np.pi), 0.0], dtype=float)

        # Baseline J sweep
        if PLOT_BASELINE_IN_SWEEP and ('J_base' in d):
            u0_base_local = np.array([0.0], dtype=float)
            penalty0_base_local = float(e0.T @ Qlyapunov @ e0 + u0_base_local.T @ Rlyapunov @ u0_base_local)
            J0_base_local = _first_finite(d.get('J_base', []), default=0.0)
            J_anchor_base_local = J0_base_local + (
                        1.0 - float(RHO_FIXED)) * penalty0_base_local + 1e-6 if PLOT_SHARED_ANCHOR else 0.0

            Jb = d['J_base'];
            Jb_plot = np.concatenate(([J_anchor_base_local], Jb)) if PLOT_SHARED_ANCHOR else Jb
            # --- MODIFIED: Use t_inputs (plot_horizon) for sweep plot anchor ---
            t_plot = np.concatenate(([-DT], t_inputs)) if PLOT_SHARED_ANCHOR else t_inputs
            # --- END MODIFIED ---
            ax3.plot(t_plot[:Jb_plot.shape[0]], Jb_plot, ls='--', lw=1.0, color=COLOR_BASELINE, alpha=a, label=None)

        # Learned J sweep
        if PLOT_LEARNED_IN_SWEEP and ('J_learn' in d):
            u0_learn_local = np.array([d.get('uL0_learn', 0.0)], dtype=float)
            penalty0_learn_local = float(e0.T @ Qlyapunov @ e0 + u0_learn_local.T @ Rlyapunov @ u0_learn_local)
            J0_learn_local = _first_finite(d.get('J_learn', []), default=0.0)
            J_anchor_learn_local = J0_learn_local + (
                        1.0 - float(RHO_FIXED)) * penalty0_learn_local + 1e-6 if PLOT_SHARED_ANCHOR else 0.0

            Jl = d['J_learn'];
            Jl_plot = np.concatenate(([J_anchor_learn_local], Jl)) if PLOT_SHARED_ANCHOR else Jl
            # --- MODIFIED: Use t_inputs (plot_horizon) for sweep plot anchor ---
            t_plot = np.concatenate(([-DT], t_inputs)) if PLOT_SHARED_ANCHOR else t_inputs
            # --- END MODIFIED ---
            ax3.plot(t_plot[:Jl_plot.shape[0]], Jl_plot, ls='-', lw=1.0, color=COLOR_LEARNED, alpha=a, label=None)
        # --- END MODIFIED ---

# Single-run thick lines
ax3.plot(t_inputs_plot[:J_base_plot.shape[0]], J_base_plot, lw=1.8, color=COLOR_BASELINE)
ax3.plot(t_inputs_plot[:J_learn_plot.shape[0]], J_learn_plot, lw=1.8, color=COLOR_LEARNED)
ax3.set_xlabel("time [s]", fontsize=AXIS_LABEL_FONTSIZE)
ax3.set_ylabel(r"$J^*$", fontsize=AXIS_LABEL_FONTSIZE)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
if FIG3_XTICKS is not None: ax3.set_xticks(FIG3_XTICKS)
if FIG3_YTICKS is not None: ax3.set_yticks(FIG3_YTICKS)
fig3.tight_layout()

# Save Figure 3
fig3_path = os.path.join(PLOTS_DIR, "figure3_jstar_overlay.png")
fig3.savefig(fig3_path, dpi=SAVE_DPI, bbox_inches='tight')
print(f"[Saved] {fig3_path}")

# 4) Figure 4 — 3 subplots for the single θ0
fig4, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=False)

# (a) θ & ω vs time
ax_a = axes[0]
ax_a.plot(t_states, theta_base_err, lw=1.6, color=COLOR_BASELINE, label=r'$\theta$ baseline')
ax_a.plot(t_states, theta_learn_err, lw=1.6, color=COLOR_LEARNED, label=r'$\theta$ learned')
ax_a.plot(t_states, omega_base, lw=1.2, ls='--', color=COLOR_BASELINE, alpha=0.75, label=r'$\omega$ baseline')
ax_a.plot(t_states, omega_learn, lw=1.2, ls='--', color=COLOR_LEARNED, alpha=0.75, label=r'$\omega$ learned')
ax_a.axhline(0.0, linestyle=':', linewidth=0.9, color='k', alpha=0.9)
ax_a.set_xlabel("time [s]")
ax_a.set_ylabel(r"$\theta$, $\omega$")
ax_a.grid(True, alpha=0.3)
ax_a.legend(loc='best', fontsize=10)

# (b) u_L vs applied u
ax_b = axes[1]
# learned
ax_b.plot(t_inputs, uL_log, lw=1.8, color=COLOR_LEARNED, label=r'$u_L$ learned')
ax_b.plot(t_inputs, u_cmd_learn, lw=1.8, color=COLOR_LEARNED, ls='--', label=r'$u$ learned (applied)')
# baseline (reference)
ax_b.plot(t_inputs, np.zeros_like(u_cmd_base), lw=1.2, color=COLOR_BASELINE, alpha=0.6,
          label=r'$u_L$ baseline ($\equiv 0$)')
ax_b.plot(t_inputs, u_cmd_base, lw=1.2, color=COLOR_BASELINE, ls='--', alpha=0.8, label=r'$u$ baseline (applied)')
ax_b.set_xlabel("time [s]")
ax_b.set_ylabel("control")
ax_b.grid(True, alpha=0.3)
ax_b.legend(loc='best', fontsize=10)

# (c) (u_L - u) for learned (optionally show baseline diff as dashed)
ax_c = axes[2]
diff_learn = uL_log - u_cmd_learn
ax_c.plot(t_inputs, diff_learn, lw=1.8, color=COLOR_LEARNED, label=r'$u_L - u$ (learned)')
ax_c.plot(t_inputs, -u_cmd_base, lw=1.2, color=COLOR_BASELINE, ls='--', alpha=0.8, label=r'$-u$ baseline')
ax_c.axhline(0.0, linestyle=':', linewidth=0.9, color='k', alpha=0.9)
ax_c.set_xlabel("time [s]")
ax_c.set_ylabel(r"$u_L - u$")
ax_c.grid(True, alpha=0.3)
ax_c.legend(loc='best', fontsize=10)

fig4.tight_layout()

# Show all figures
plt.show()