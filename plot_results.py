#!/usr/bin/env python3
"""
plot_results.py

Loads results/<RUN_FOLDER_NAME>/model_best.pth, auto-detects SSM size,
rebuilds env+controller, generates u_L online from the best actor, runs PSF
forward (baseline vs learned), and SHOWS figures (no saving).

Extra figure with 3 subplots:
  (i)   theta & omega vs time
  (ii)  u_L vs filtered u vs time (+ bounds)
  (iii) PSF activation indicator (1 if u != u_L else 0)
"""

# ======================= USER SETTINGS (EDIT ME) =======================
#RUN_FOLDER_NAME    = "PSF_SSM_NS_10_22_13_42_51"  # results/<RUN_FOLDER_NAME>
#RUN_FOLDER_NAME    = "PSF_SSM_NS_10_22_12_30_21"
RUN_FOLDER_NAME    = "PSF_SSM_NS_10_22_13_27_47"
THETA0             = 1.57079632679                # initial angle [rad]; set None to use π/2
RHO_FIXED          = 0.5                          # baseline (and optional learned) fixed ρ
LEARNED_RHO_MODE   = "scheduled"                  # "scheduled" or "fixed"
PLOT_SHARED_ANCHOR = True                         # prepend a shared visual J*_{-1} anchor
ANCHOR_MODE        = "common_zero"                # (kept for compatibility; not used now)
# ======================================================================

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

mpl.rcParams['text.usetex'] = False
torch.set_default_dtype(torch.double)

# ---------- Resolve paths ----------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", RUN_FOLDER_NAME)
BEST_PATH   = os.path.join(RESULTS_DIR, "model_best.pth")
LOG_PATH    = os.path.join(RESULTS_DIR, "training.log")

if not os.path.exists(RESULTS_DIR):
    raise FileNotFoundError(f"Results dir not found: {RESULTS_DIR}")
if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(f"Best model not found: {BEST_PATH}")
if not os.path.exists(LOG_PATH):
    print(f"[WARN] training.log not found at {LOG_PATH}. Will use default hyperparams.")

# ---------- Tiny log parser ----------
FLOAT_RE = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'

def parse_last_float(line, default=None):
    ms = re.findall(FLOAT_RE, line)
    return float(ms[-1]) if ms else default

def parse_last_int(line, default=None):
    ms = re.findall(r'\d+', line)
    return int(ms[-1]) if ms else default

def parse_tensor_like(line, default=None):
    m = re.search(r'\[(.*)\]', line)
    if not m:
        return default
    inner = m.group(0)
    try:
        arr = ast.literal_eval(inner)
        return np.array(arr, dtype=float)
    except Exception:
        vals = re.findall(FLOAT_RE, inner)
        if not vals:
            return default
        return np.array([float(v) for v in vals], dtype=float)

def load_meta_from_log(log_path):
    meta = {}
    if not os.path.exists(log_path): return meta
    with open(log_path, 'r') as f:
        lines = f.read().splitlines()
    for ln in lines:
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
sim_horizon = int(meta.get('sim_horizon', 250))
epsilon     = float(meta.get('epsilon', 0.05))
DT          = 0.05
rho_bar     = 0.5
rho_max     = 2.0

state_lower_bound   = np.array(meta.get('state_lower_bound', [0.5, -20.0]), dtype=float)
state_upper_bound   = np.array(meta.get('state_upper_bound', [2*np.pi - 0.5, 20.0]), dtype=float)
control_lower_bound = np.array(meta.get('control_lower_bound', [-3.0]), dtype=float)
control_upper_bound = np.array(meta.get('control_upper_bound', [ 3.0]), dtype=float)
obs_centers         = np.array(meta.get('obs_centers', [[1.0, 0.5]]), dtype=float).reshape(-1,2)
obs_centers         = np.array(meta.get('obs_centers', [[9.0, 1.5]]), dtype=float).reshape(-1,2)
obs_covs            = np.array(meta.get('obs_covs', [0.005]), dtype=float).reshape(-1)
obs_vel             = np.array(meta.get('obs_vel', [[-0.2, 0.0]]), dtype=float).reshape(-1,2)

if THETA0 is None:
    THETA0 = np.pi/2

# ---------- Auto-detect SSM size from checkpoint ----------
def load_actor_state_dict_from_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 1:
        return ckpt[0]
    if isinstance(ckpt, dict):
        for k in ('actor', 'actor_state_dict', 'actor_model', 'state_dict'):
            if k in ckpt:
                if k == 'state_dict':
                    sd = {kk.split('actor_model.',1)[-1]: vv for kk, vv in ckpt[k].items()
                          if kk.startswith('actor_model.')}
                    if sd: return sd
                else:
                    return ckpt[k]
        return ckpt
    raise RuntimeError("Unrecognized checkpoint format for actor weights.")

def detect_dim_internal(actor_sd):
    candidates = [
        'm_dynamics.LRUR.B', 'm_dynamics.model.0.B',
        'm_dynamics.LRUR.C', 'm_dynamics.model.0.C',
        'm_dynamics.LRUR.state', 'm_dynamics.model.0.state',
        'm_dynamics.LRUR.nu_log', 'm_dynamics.model.0.nu_log',
    ]
    for key in candidates:
        if key in actor_sd:
            t = actor_sd[key]
            shape = tuple(t.shape) if isinstance(t, torch.Tensor) else tuple(t.size())
            if key.endswith('.B'):  # (k, 2)
                return int(shape[0])
            if key.endswith('.C'):  # (1, k)
                return int(shape[1])
            return int(shape[0])   # vectors
    return None

actor_sd = load_actor_state_dict_from_checkpoint(BEST_PATH)
detected_dim = detect_dim_internal(actor_sd)
log_dim      = int(meta.get('dim_internal')) if 'dim_internal' in meta else None
if detected_dim is None and log_dim is None:
    num_dynamics_states = 24
    print("[WARN] Could not detect SSM size from checkpoint/log; using default 24.")
elif detected_dim is None:
    num_dynamics_states = log_dim
    print(f"[INFO] Using SSM size from log: {num_dynamics_states}")
elif log_dim is None:
    num_dynamics_states = detected_dim
    print(f"[INFO] Using SSM size from checkpoint: {num_dynamics_states}")
else:
    num_dynamics_states = detected_dim
    if detected_dim != log_dim:
        print(f"[WARN] Log dim_internal={log_dim}, checkpoint shows {detected_dim}. Using {detected_dim} to match checkpoint.")

# ---------- Build obstacle signal (same as training) ----------
T_total = sim_horizon + 1 + 500
t = torch.arange(T_total, dtype=torch.double)
obs_pos_raw = torch.tensor(obs_centers, dtype=torch.double) + (t[:, None] * DT) * torch.tensor(obs_vel, dtype=torch.double)
obs_pos = obs_pos_raw.unsqueeze(0)  # (1, T_total, 2)
moving_obs = MovingObstacle(obs_pos, torch.tensor(obs_covs, dtype=torch.double))

# ---------- Rebuild env ----------
q_theta, q_theta_dot, r_u = 50.0, 10.0, 0.1
Qlyapunov = np.diag([q_theta, q_theta_dot]).astype(float)
Rlyapunov = np.array([[r_u]], dtype=float)
target_positions = torch.tensor([np.pi, 0.0], dtype=torch.double)

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
    Qlyapunov=Qlyapunov, Rlyapunov=Rlyapunov
)

# ---------- Controller with *matching* SSM size ----------
mad = MADController(
    env=env,
    buffer_capacity=100000,
    target_state=target_positions,
    num_dynamics_states=num_dynamics_states,
    dynamics_input_time_window_length=sim_horizon+1,
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

# ---------- PSF model + torch plant ----------
model = SinglePendulumCasadi(xbar=np.array([np.pi, 0.0], dtype=float))
plant = SinglePendulum(
    xbar=torch.tensor([np.pi, 0.0], dtype=torch.double),
    x_init=torch.tensor([THETA0, 0.0], dtype=torch.double).view(1, -1),
    u_init=torch.zeros(1, 1, dtype=torch.double),
)
L_link = float(model.l)

def make_psf(rho_mode, rho_fixed_val):
    return MPCPredictSafetyFilter(
        model,
        horizon=25,
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
CHI2_2_95 = 5.991464547
def wrap_0_2pi(x): return np.mod(x, 2*np.pi)
def theta_arcs_hit_obstacle(center_xy, L, cov_var):
    r = np.sqrt(CHI2_2_95 * float(cov_var))
    c = np.asarray(center_xy, dtype=float).reshape(2,)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12: return [(0.0, 2*np.pi)]
    zeta = (L**2 + norm_c**2 - r**2) / (2.0 * L * norm_c)
    if zeta <= -1.0: return [(0.0, 2*np.pi)]
    if zeta >=  1.0: return []
    delta = np.arccos(zeta); phi = np.arctan2(c[1], c[0])
    th_lo = wrap_0_2pi(phi - delta + np.pi/2)
    th_hi = wrap_0_2pi(phi + delta + np.pi/2)
    return [(th_lo, th_hi)] if th_lo <= th_hi else [(0.0, th_hi), (th_lo, 2*np.pi)]
def rho_schedule_from_uL(uL_scalar, rho_bar, rho_max, epsilon):
    r2 = float(uL_scalar**2); sigma = r2 / (epsilon*epsilon + r2)
    return float(rho_bar + (rho_max - rho_bar) * sigma)

# ---------- (1) Baseline: PSF-only, uL=0, fixed rho ----------
psf_base = make_psf('fixed', RHO_FIXED)
theta_base, rho_base, J_base = [THETA0], [], []
U_prev = X_prev = None
x_t = torch.tensor([THETA0, 0.0], dtype=torch.double).view(1,1,-1)
for k in range(sim_horizon):
    x_np = x_t.view(-1).cpu().numpy()
    uL = np.zeros((1,), dtype=float)
    try:
        if k == 0:
            U_sol, X_sol, J_curr = psf_base.solve_mpc(x_np, np.array([np.pi,0.0],dtype=float), uL)
        else:
            U_sol, X_sol, J_curr = psf_base.solve_mpc(x_np, np.array([np.pi,0.0],dtype=float), uL, U_prev, X_prev)
    except Exception:
        U_sol = X_sol = J_curr = None
    if U_sol is None:
        u_cmd = uL.reshape(1,1,1); U_prev = X_prev = None; J_base.append(np.nan)
    else:
        u_cmd = U_sol[:,0:1].reshape(1,1,1); U_prev, X_prev = U_sol, X_sol; J_base.append(float(J_curr))
    rho_base.append(RHO_FIXED)
    x_t = plant.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
    theta_base.append(float(x_t.view(-1)[0]))
theta_base = np.array(theta_base); rho_base = np.array(rho_base); J_base = np.array(J_base)

# ---------- (2) Learned: Actor-generated uL with PSF ----------
psf_learn = make_psf(LEARNED_RHO_MODE, RHO_FIXED)
theta_learn, rho_learn, J_learn = [THETA0], [], []
omega_learn = [0.0]
uL_series, u_series, psf_active = [], [], []

U_prev = X_prev = None
x_t = torch.tensor([THETA0, 0.0], dtype=torch.double).view(1,1,-1)

# Initialize episode windows so SSM sees [x0-x*, 0, 0, ...]
mad.set_ep_initial_state(initial_state=x_t.view(1,-1))
mad.reset_ep_timestep()
mad.update_dynamics_input_time_window()
mad.w = torch.zeros_like(x_t.view(-1))

for k in range(sim_horizon):
    x_np = x_t.view(-1).cpu().numpy()

    # Actor (no OU noise)
    state_error = (x_t.view(1,-1) - mad.target_state).to(next(mad.actor_model.parameters()).device)
    uL_tensor = mad.learned_policy(
        state_error,
        mad.dynamics_input_time_window,
        mad.dynamics_disturbance_time_window
    )
    uL = np.asarray(uL_tensor, dtype=float).reshape(-1)

    try:
        if k == 0:
            U_sol, X_sol, J_curr = psf_learn.solve_mpc(x_np, np.array([np.pi,0.0],dtype=float), uL)
        else:
            U_sol, X_sol, J_curr = psf_learn.solve_mpc(x_np, np.array([np.pi,0.0],dtype=float), uL, U_prev, X_prev)
    except Exception:
        U_sol = X_sol = J_curr = None

    if U_sol is None:
        u_cmd = uL.reshape(1,1,1); U_prev = X_prev = None; J_learn.append(np.nan)
    else:
        u_cmd = U_sol[:,0:1].reshape(1,1,1); U_prev, X_prev = U_sol, X_sol; J_learn.append(float(J_curr))

    # PSF activation & logs
    u_cmd_scalar = float(u_cmd.reshape(-1)[0])
    uL_scalar    = float(uL[0])
    uL_series.append(uL_scalar)
    u_series.append(u_cmd_scalar)
    psf_active.append(1.0 if abs(u_cmd_scalar - uL_scalar) > 1e-8 else 0.0)

    # ρ_t for plotting
    if LEARNED_RHO_MODE == 'scheduled':
        rho_learn.append(rho_schedule_from_uL(uL_scalar, rho_bar=rho_bar, rho_max=rho_max, epsilon=epsilon))
    else:
        rho_learn.append(RHO_FIXED)

    # plant step
    x_t = plant.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
    x_np_next = x_t.view(-1).detach().cpu().numpy()
    theta_learn.append(float(x_np_next[0]))
    omega_learn.append(float(x_np_next[1]))

    # update controller episode windows
    mad.update_ep_timestep()
    mad.update_dynamics_input_time_window()

theta_learn = np.array(theta_learn); omega_learn = np.array(omega_learn)
rho_learn = np.array(rho_learn); J_learn = np.array(J_learn)
uL_series = np.array(uL_series); u_series = np.array(u_series); psf_active = np.array(psf_active)

# ---------- Visual anchor: use ONLY the fixed-ρ baseline (one-step increase) ----------
def _angdiff(a, b):
    return np.arctan2(np.sin(b - a), np.cos(b - a))

def _first_finite(x, default=0.0):
    for v in np.asarray(x).ravel():
        if np.isfinite(v):
            return float(v)
    return float(default)

if PLOT_SHARED_ANCHOR:
    # penalty at k=0 with u0 = 0 (neutral for both approaches)
    e0 = np.array([_angdiff(THETA0, np.pi), 0.0], dtype=float)  # [theta error, omega error]
    u0 = np.array([0.0], dtype=float)
    penalty0 = float(e0.T @ Qlyapunov @ e0 + u0.T @ Rlyapunov @ u0)

    # anchor strictly above the fixed-ρ baseline's first J*
    J0_base = _first_finite(J_base, default=0.0)
    eps_margin = 1e-6
    J_anchor = J0_base + (1.0 - float(RHO_FIXED)) * penalty0 + eps_margin

    # use the SAME anchor for both curves
    t_inputs_plot = np.concatenate(([-DT], np.arange(sim_horizon) * DT))
    J_base_plot   = np.concatenate(([J_anchor], J_base))
    J_learn_plot  = np.concatenate(([J_anchor], J_learn))
else:
    t_inputs_plot = np.arange(sim_horizon) * DT
    J_base_plot   = J_base
    J_learn_plot  = J_learn

# ---------- Plotting (SHOW, don't save) ----------
t_states = np.arange(sim_horizon + 1) * DT
t_inputs = np.arange(sim_horizon) * DT
SAFE_TH_LO = float(state_lower_bound[0]); SAFE_TH_HI = float(state_upper_bound[0])
OBSTACLE_SHADE_COLOR = (0.95, 0.55, 0.55, 0.35)
RHO_T = r'$\rho_t$'  # safe mathtext token

# 1) θ(t) with obstacle band
fig1, ax1 = plt.subplots(figsize=(10, 4.8))
ax1.plot(t_states[:theta_base.shape[0]],  theta_base,
         label=rf'PSF-only ($\bar{{\rho}}={RHO_FIXED:.2f}$, $u_L\equiv 0$)', lw=1.4)
ax1.plot(t_states[:theta_learn.shape[0]], theta_learn,
         label=('Actor + PSF (scheduled ' + RHO_T + ')' if LEARNED_RHO_MODE=='scheduled'
                else 'Actor + PSF (fixed $\\rho$)'), lw=1.4)
ax1.axhline(np.pi, linestyle='--', linewidth=1.0, color='k', label=r'$\theta^\star=\pi$')
ax1.set_title(r"Pendulum angle $\theta(t)$ with obstacle (95%)")
ax1.set_xlabel("time [s]"); ax1.set_ylabel(r"$\theta$ [rad]")
ax1.set_ylim([SAFE_TH_LO-0.1, SAFE_TH_HI+0.1])
ax1.grid(True, alpha=0.3)
for k in range(sim_horizon):
    centers_k = obs_centers + (k * DT) * obs_vel
    for i in range(centers_k.shape[0]):
        arcs = theta_arcs_hit_obstacle(centers_k[i], L_link, cov_var=float(obs_covs[i]))
        if not arcs: continue
        x_span = [k*DT, (k+1)*DT]
        for (th_lo, th_hi) in arcs:
            ax1.fill_between(x_span, [th_lo, th_lo], [th_hi, th_hi],
                             facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axhline(SAFE_TH_LO, color='k', lw=0.8, ls=':'); ax1.axhline(SAFE_TH_HI, color='k', lw=0.8, ls=':')
handles, labels = ax1.get_legend_handles_labels()
handles.append(Patch(facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', label='Obstacle (95%)'))
ax1.legend(loc='best')
fig1.tight_layout()

# 2) ρ_t
fig2, ax2 = plt.subplots(figsize=(10, 4.6))
ax2.plot(t_inputs[:rho_base.shape[0]],  rho_base,  label='PSF-only ' + RHO_T + ' (baseline fixed)', lw=1.4)
ax2.plot(t_inputs[:rho_learn.shape[0]], rho_learn,
         label=('Actor ' + RHO_T + ' (scheduled)' if LEARNED_RHO_MODE=='scheduled' else 'Actor ' + RHO_T + ' (fixed)'), lw=1.4)
ax2.set_title("Tightening schedule " + RHO_T); ax2.set_xlabel("time [s]"); ax2.set_ylabel(RHO_T)
ax2.grid(True, alpha=0.3); ax2.legend(loc='best')
fig2.tight_layout()

# 3) J*(t) with baseline-derived shared anchor
fig3, ax3 = plt.subplots(figsize=(10, 4.6))
ax3.plot(t_inputs_plot[:J_base_plot.shape[0]],  J_base_plot,  label=r'PSF-only $J^\star$ (baseline)', lw=1.4)
ax3.plot(t_inputs_plot[:J_learn_plot.shape[0]], J_learn_plot, label=r'Actor $J^\star$', lw=1.4)
ax3.set_title(r"Optimal value $J^\star$ per PSF solve")
ax3.set_xlabel("time [s]"); ax3.set_ylabel(r"$J^\star$")
ax3.grid(True, alpha=0.3); ax3.legend(loc='best')
fig3.tight_layout()

# 4) Time-series subplots: (theta,omega) | (u_L,u) | (PSF active)
fig4, (ax4a, ax4b, ax4c) = plt.subplots(1, 3, figsize=(16, 4.2))

# (i) theta & omega vs time
ax4a.plot(t_states, theta_learn, label=r'$\theta$')
ax4a.plot(t_states, omega_learn, label=r'$\omega$')
ax4a.axhline(np.pi, ls='--', lw=0.8, color='k', label=r'$\theta^\star$')
ax4a.set_title(r'$\theta$ and $\omega$ (Actor + PSF)')
ax4a.set_xlabel('time [s]'); ax4a.set_ylabel('states')
ax4a.grid(True, alpha=0.3); ax4a.legend(loc='best')

# (ii) u_L vs u (+ bounds)
ax4b.plot(t_inputs, uL_series, label=r'$u_L$')
ax4b.plot(t_inputs, u_series, label=r'$u$')
ax4b.axhline(float(control_upper_bound[0]), ls='--', lw=0.8, color='g', label='u bounds')
ax4b.axhline(float(control_lower_bound[0]), ls='--', lw=0.8, color='g')
ax4b.set_title('Control signals')
ax4b.set_xlabel('time [s]'); ax4b.set_ylabel('u')
ax4b.grid(True, alpha=0.3); ax4b.legend(loc='best')

# (iii) PSF activation indicator
ax4c.step(t_inputs, psf_active, where='post', label='PSF active')
ax4c.set_yticks([0, 1]); ax4c.set_ylim([-0.1, 1.1])
ax4c.set_title('PSF activation'); ax4c.set_xlabel('time [s]'); ax4c.set_ylabel('active')
ax4c.grid(True, alpha=0.3); ax4c.legend(loc='best')

fig4.tight_layout()

plt.show()
