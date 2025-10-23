#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pendulum_gif_single.py

- Loads results/<RUN_FOLDER_NAME>/model_best.pth (best actor)
- Rebuilds env + controller, generates u_L online from the actor
- Runs PSF forward (learned only) for ONE realization (theta0)
- Saves a 2D GIF: pendulum motion + obstacle drawn as 95% PDF disk

Run: python make_pendulum_gif_single.py
"""

# =================== USER SETTINGS (manual, like your plot script) ===================
RUN_FOLDER_NAME    = "PSF_SSM_NS_10_22_12_30_21"  # results/<RUN_FOLDER_NAME>
THETA0             = -0.7+3.14 #1.57079632679                # initial angle [rad]; None -> π/2
LEARNED_RHO_MODE   = "scheduled"                  # "scheduled" or "fixed"
RHO_FIXED          = 0.5                          # used only if LEARNED_RHO_MODE == "fixed"

# PSF knobs (match your code)
rho_bar     = 0.5
rho_max     = 10.0
PSFhorizon  = 20

# Output GIF
GIF_PATH    = "results/gifs/pendulum_single.gif"
FPS         = 20

# ===============================================================================
import os, re, ast
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# Project imports
from mad_controller import MADController
from pendulum_env import PendulumEnv
from obstacles import MovingObstacle
from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

mpl.rcParams['text.usetex'] = False
torch.set_default_dtype(torch.double)

# ---------- Paths ----------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "results", RUN_FOLDER_NAME)
BEST_PATH    = os.path.join(RESULTS_DIR, "model_best.pth")
LOG_PATH     = os.path.join(RESULTS_DIR, "training.log")
os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)

if not os.path.exists(RESULTS_DIR):
    raise FileNotFoundError(f"Results dir not found: {RESULTS_DIR}")
if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(f"Best model not found: {BEST_PATH}")
if not os.path.exists(LOG_PATH):
    print(f"[WARN] training.log not found at {LOG_PATH}. Using defaults.")

# ---------- Tiny log parser ----------
FLOAT_RE = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'
def _parse_last_float(line, default=None):
    ms = re.findall(FLOAT_RE, line); return float(ms[-1]) if ms else default
def _parse_last_int(line, default=None):
    ms = re.findall(r'\d+', line);   return int(ms[-1]) if ms else default
def _parse_tensor_like(line, default=None):
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
            if 'Simulation horizon (T)' in ln: meta['sim_horizon'] = _parse_last_int(ln)
            elif 'Epsilon for PSF' in ln:       meta['epsilon']     = _parse_last_float(ln)
            elif 'RENs dim_internal' in ln:     meta['dim_internal']= _parse_last_int(ln)
            elif 'obs_centers' in ln:           meta['obs_centers'] = _parse_tensor_like(ln)
            elif 'obs_covs' in ln:              meta['obs_covs']    = _parse_tensor_like(ln)
            elif 'obs_vel' in ln:               meta['obs_vel']     = _parse_tensor_like(ln)
            elif 'state_lower_bound' in ln:     meta['state_lower_bound']   = _parse_tensor_like(ln)
            elif 'state_upper_bound' in ln:     meta['state_upper_bound']   = _parse_tensor_like(ln)
            elif 'controller_lower_bound' in ln or 'control_lower_bound' in ln:
                meta['control_lower_bound'] = _parse_tensor_like(ln)
            elif 'controller_upper_bound' in ln or 'control_upper_bound' in ln:
                meta['control_upper_bound'] = _parse_tensor_like(ln)
    return meta

meta = load_meta_from_log(LOG_PATH)

# ---------- Defaults (match training) ----------
sim_horizon = int(meta.get('sim_horizon', 250))
epsilon     = float(meta.get('epsilon', 0.05))
DT          = 0.05

state_lower_bound   = np.array(meta.get('state_lower_bound', [0.5, -20.0]), dtype=float)
state_upper_bound   = np.array(meta.get('state_upper_bound', [2*np.pi - 0.5, 20.0]), dtype=float)
control_lower_bound = np.array(meta.get('control_lower_bound', [-3.0]), dtype=float)
control_upper_bound = np.array(meta.get('control_upper_bound', [ 3.0]), dtype=float)
obs_centers         = np.array(meta.get('obs_centers', [[1.0, 0.5]]), dtype=float).reshape(-1,2)
obs_covs            = np.array(meta.get('obs_covs', [0.005]), dtype=float).reshape(-1)
obs_vel             = np.array(meta.get('obs_vel', [[-0.2, 0.0]]), dtype=float).reshape(-1,2)

if THETA0 is None:
    THETA0 = np.pi/2

# ---------- Auto-detect SSM size from checkpoint ----------
def load_actor_state_dict_from_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 1: return ckpt[0]
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
    candidates = ['m_dynamics.LRUR.B','m_dynamics.model.0.B',
                  'm_dynamics.LRUR.C','m_dynamics.model.0.C',
                  'm_dynamics.LRUR.state','m_dynamics.model.0.state',
                  'm_dynamics.LRUR.nu_log','m_dynamics.model.0.nu_log']
    for key in candidates:
        if key in actor_sd:
            t = actor_sd[key]
            shape = tuple(t.shape) if isinstance(t, torch.Tensor) else tuple(t.size())
            if key.endswith('.B'): return int(shape[0])  # (k,2)
            if key.endswith('.C'): return int(shape[1])  # (1,k)
            return int(shape[0])   # vectors
    return None

actor_sd = load_actor_state_dict_from_checkpoint(BEST_PATH)
detected_dim = detect_dim_internal(actor_sd)
log_dim      = int(meta.get('dim_internal')) if 'dim_internal' in meta else None
if detected_dim is None and log_dim is None:
    num_dynamics_states = 24
    print("[WARN] Could not detect SSM size from checkpoint/log; using default 24.")
elif detected_dim is None:
    num_dynamics_states = log_dim; print(f"[INFO] Using SSM size from log: {num_dynamics_states}")
elif log_dim is None:
    num_dynamics_states = detected_dim; print(f"[INFO] Using SSM size from checkpoint: {num_dynamics_states}")
else:
    num_dynamics_states = detected_dim
    if detected_dim != log_dim:
        print(f"[WARN] Log dim_internal={log_dim}, checkpoint shows {detected_dim}. Using {detected_dim}.")




# ---------- Obstacle signal ----------
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

# ---------- Controller ----------
mad = MADController(
    env=env,
    buffer_capacity=100000,
    target_state=target_positions,
    num_dynamics_states=num_dynamics_states,
    dynamics_input_time_window_length=sim_horizon+1,
    batch_size=64,
    gamma=0.99, tau=0.005,
    actor_lr=5e-4, critic_lr=1e-3,
    std_dev=0.0,  # eval
    control_action_upper_bound=float(control_upper_bound[0]),
    control_action_lower_bound=float(control_lower_bound[0]),
    nn_type='mad'
)
mad.load_model_weight(BEST_PATH)
print(f"Loaded actor from {BEST_PATH}")

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
CHI2_2_95 = 5.991464547107979

def rho_schedule_from_uL(uL_scalar, rho_bar, rho_max, epsilon):
    r2 = float(uL_scalar**2); sigma = r2 / (epsilon*epsilon + r2)
    return float(rho_bar + (rho_max - rho_bar) * sigma)

# ---------- Single-run (LEARNED ONLY): rollout & collect θ ----------
psf_learn = make_psf(LEARNED_RHO_MODE, RHO_FIXED)
plant_single = SinglePendulum(
    xbar=torch.tensor([np.pi, 0.0], dtype=torch.double),
    x_init=torch.tensor([THETA0, 0.0], dtype=torch.double).view(1, -1),
    u_init=torch.zeros(1, 1, dtype=torch.double),
)

thetas = [THETA0]
U_prev = X_prev = None
x_t = torch.tensor([THETA0, 0.0], dtype=torch.double).view(1,1,-1)

# Initialize actor episode windows
mad.set_ep_initial_state(initial_state=x_t.view(1,-1))
mad.reset_ep_timestep()
mad.update_dynamics_input_time_window()
mad.w = torch.zeros_like(x_t.view(-1))

for k in range(sim_horizon):
    x_np = x_t.view(-1).cpu().numpy()
    state_error = (x_t.view(1,-1) - mad.target_state).to(next(mad.actor_model.parameters()).device)
    uL_tensor = mad.learned_policy(
        state_error,
        mad.dynamics_input_time_window,
        mad.dynamics_disturbance_time_window
    )
    uL = np.asarray(uL_tensor, dtype=float).reshape(-1)

    try:
        if k == 0:
            U_sol, X_sol, _ = psf_learn.solve_mpc(x_np, np.array([np.pi,0.0],dtype=float), uL)
        else:
            U_sol, X_sol, _ = psf_learn.solve_mpc(x_np, np.array([np.pi,0.0],dtype=float), uL, U_prev, X_prev)
    except Exception:
        U_sol = X_sol = None

    if U_sol is None:
        u_cmd = uL.reshape(1,1,1); U_prev = X_prev = None
    else:
        u_cmd = U_sol[:,0:1].reshape(1,1,1); U_prev, X_prev = U_sol, X_sol

    # integrate
    x_t = plant_single.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
    thetas.append(float(x_t.view(-1)[0]))

    # advance SSM windows
    mad.update_ep_timestep()
    mad.update_dynamics_input_time_window()

# ---------- Build the GIF (2D pendulum + 95% obstacle disk) ----------
fig, ax = plt.subplots(figsize=(5.2, 5.2))
R_draw = 1.05 * L_link
ax.set_xlim([-R_draw, R_draw]); ax.set_ylim([-R_draw, R_draw])
ax.set_aspect("equal", adjustable="box")
ax.set_title("Pendulum + PSF + Trained Policy")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.grid(True, alpha=0.3)

rod_line,   = ax.plot([], [], lw=2.5, solid_capstyle='round')
tip_trace,  = ax.plot([], [], lw=1.0, ls='--', alpha=0.6)
tip_dot,    = ax.plot([], [], 'o', ms=5)
ax.plot([0.0], [0.0], 'ko', ms=4)  # pivot

obstacle_patch = None
tip_x_hist, tip_y_hist = [], []

# BEFORE the functions
obstacle_patch = None
tip_x_hist, tip_y_hist = [], []

def init():
    rod_line.set_data([], [])
    tip_trace.set_data([], [])
    tip_dot.set_data([], [])
    return rod_line, tip_trace, tip_dot

def frame(i):
    global obstacle_patch   # <-- change from 'nonlocal' to 'global'
    theta = thetas[i]
    x_tip = L_link * np.sin(theta)
    y_tip = -L_link * np.cos(theta)
    tip_x_hist.append(x_tip); tip_y_hist.append(y_tip)

    rod_line.set_data([0.0, x_tip], [0.0, y_tip])
    tip_trace.set_data(tip_x_hist, tip_y_hist)
    tip_dot.set_data([x_tip], [y_tip])

    # obstacle @ 95% PDF disk
    if obstacle_patch is not None:
        obstacle_patch.remove()
        obstacle_patch = None

    pos_t, covs_t = moving_obs.get_obstacles(i)
    cx, cy = float(pos_t[0,0]), float(pos_t[0,1])
    cov    = float(covs_t[0].item() if hasattr(covs_t[0], 'item') else covs_t[0])
    r95    = np.sqrt(CHI2_2_95 * cov)

    obstacle_patch = Circle((cx, cy), r95,
                            facecolor=(0.8, 0.2, 0.2, 0.25),
                            edgecolor=(0.6, 0.0, 0.0, 0.8),
                            lw=1.0, zorder=0)
    ax.add_patch(obstacle_patch)
    return rod_line, tip_trace, tip_dot, obstacle_patch

ani = FuncAnimation(fig, frame, frames=(sim_horizon+1),
                    init_func=init, interval=DT*1000, blit=True, repeat=False)
ani.save(GIF_PATH, writer=PillowWriter(fps=FPS))
plt.close(fig)
print(f"[Saved GIF] {GIF_PATH}")
