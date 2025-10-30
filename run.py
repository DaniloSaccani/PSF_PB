#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, ast
import numpy as np
import torch
import logging
from datetime import datetime
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from mad_controller import MADController
from pendulum_env import PendulumEnv
from obstacles import MovingObstacle
from plot_functions import plot_results
from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

# ----------------------------- 0) MODE & PATHS -----------------------------
START_MODE = "resume"   # "scratch" or "resume"
RESUME_RUN_FOLDER = "PSF_SSM_NS_10_29_12_30_09"

# ---------------- 0bis) Hyperparameters & Reproducibility ------------------
torch.set_default_dtype(torch.double)
torch.manual_seed(1)
np.random.seed(1)
mpl.rcParams['text.usetex'] = False

# --- (A) User-tunable quantities ---
sim_horizon       = 250
sampled_noise_    = 0.1

loss_x_multiplier       = 0.15
loss_u_match_multiplier = 0.01
loss_u_abs_multiplier   = 0.01
loss_obs_multiplier     = 5.0
alpha_cer               = 0.2

epsilon           = 0.05
dim_internal      = 10
num_epochs        = 1500

obs_centers       = torch.tensor([[1.0, 0.5]])
obs_covs          = torch.tensor([0.005])            # isotropic variance σ²
obs_vel           = torch.tensor([[-0.2, 0.0]])

state_lower_bound   = torch.tensor([0.5, -20.0])
state_upper_bound   = torch.tensor([2 * np.pi - 0.5, 20.0])
control_lower_bound = torch.tensor([-3.0])
control_upper_bound = torch.tensor([ 3.0])

target_positions  = torch.tensor([np.pi, 0.0], dtype=torch.double)
start_position    = np.pi / 2

noise_std         = 0.8
theta0_low, theta0_high = 3.14 - 2.3, 3.14 - 1
theta0_low, theta0_high = 3.14 - 2.3, 3.14 +2.3
decayExpNoise     = 0.95
PSFhorizon        = 20

# ---------------- (B) Fixed quantities -------------------------------------
rho      = None
rho_bar  = 0.5
rho_max  = 10.0
nn_type  = 'mad'

q_theta = 50.0
q_theta_dot = 10.0
r_u = 0.1
Qlyapunov = np.diag([q_theta, q_theta_dot]).astype(float)
Rlyapunov = np.array([[r_u]], dtype=float)

DT = 0.05

# ---------------- Helpers: parsing + SSM detection -------------------------
FLOAT_RE = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'
def parse_last_float(line, default=None):
    ms = re.findall(FLOAT_RE, line); return float(ms[-1]) if ms else default
def parse_last_int(line, default=None):
    ms = re.findall(r'\d+', line);   return int(ms[-1]) if ms else default
def parse_tensor_like(line, default=None):
    m = re.search(r'\[(.*)\]', line)
    if not m: return default
    inner = m.group(0)
    try:
        arr = ast.literal_eval(inner); return np.array(arr, dtype=float)
    except Exception:
        vals = re.findall(FLOAT_RE, inner)
        if not vals: return default
        return np.array([float(v) for v in vals], dtype=float)

def load_meta_from_log(log_path):
    meta = {}
    if not os.path.exists(log_path): return meta
    with open(log_path, 'r') as f:
        lines = f.read().splitlines()
    for ln in lines:
        if 'Simulation horizon (T)' in ln: meta['sim_horizon'] = parse_last_int(ln)
        elif 'Epsilon for PSF' in ln:       meta['epsilon'] = parse_last_float(ln)
        elif 'RENs dim_internal' in ln:     meta['dim_internal'] = parse_last_int(ln)
        elif 'obs_centers' in ln:           meta['obs_centers'] = parse_tensor_like(ln)
        elif 'obs_covs' in ln:              meta['obs_covs'] = parse_tensor_like(ln)
        elif 'obs_vel' in ln:               meta['obs_vel'] = parse_tensor_like(ln)
        elif 'state_lower_bound' in ln:     meta['state_lower_bound'] = parse_tensor_like(ln)
        elif 'state_upper_bound' in ln:     meta['state_upper_bound'] = parse_tensor_like(ln)
        elif 'controller_lower_bound' in ln or 'control_lower_bound' in ln:
            meta['control_lower_bound'] = parse_tensor_like(ln)
        elif 'controller_upper_bound' in ln or 'control_upper_bound' in ln:
            meta['control_upper_bound'] = parse_tensor_like(ln)
        elif 'rho value (learned roll-out)' in ln:
            rho_str = ln.split(':')[-1].strip()
            meta['rho'] = None if 'None' in rho_str else parse_last_float(ln, None)
    return meta

def detect_ssm_size_from_checkpoint(ckpt_path: str, fallback: int = None) -> int:
    if not os.path.exists(ckpt_path):
        return int(fallback) if fallback is not None else 24
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 1 and isinstance(ckpt[0], dict):
        actor_sd = ckpt[0]
    elif isinstance(ckpt, dict):
        if 'actor' in ckpt and isinstance(ckpt['actor'], dict):
            actor_sd = ckpt['actor']
        elif 'actor_state_dict' in ckpt and isinstance(ckpt['actor_state_dict'], dict):
            actor_sd = ckpt['actor_state_dict']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            actor_sd = {k.split('actor_model.',1)[-1]: v for k, v in ckpt['state_dict'].items()
                        if k.startswith('actor_model.')}
            if not actor_sd: actor_sd = ckpt['state_dict']
        else:
            actor_sd = ckpt
    else:
        actor_sd = ckpt
    candidates = [
        'm_dynamics.LRUR.B','m_dynamics.model.0.B',
        'm_dynamics.LRUR.C','m_dynamics.model.0.C',
        'm_dynamics.LRUR.state','m_dynamics.model.0.state',
        'm_dynamics.LRUR.nu_log','m_dynamics.model.0.nu_log',
        'm_dynamics.LRUR.theta_log','m_dynamics.model.0.theta_log',
    ]
    for key in candidates:
        if key in actor_sd:
            t = actor_sd[key]
            shape = tuple(getattr(t, 'shape', getattr(t, 'size', lambda: ())) )
            if not shape: shape = tuple(t.size())
            if key.endswith('.B') and len(shape)==2: return int(shape[0])
            if key.endswith('.C') and len(shape)==2: return int(shape[1])
            if len(shape)==1:                        return int(shape[0])
    return int(fallback) if fallback is not None else 24

# ---------------- 1) Create experiment folder ------------------------------
RUN_DIR = Path(__file__).resolve().parent
now = datetime.now().strftime("%m_%d_%H_%M_%S")
folder_name = f'PSF_SSM_NS_{now}'
save_folder = (RUN_DIR / 'results' / folder_name)
save_folder.mkdir(parents=True, exist_ok=True)

# ---------------- 2) Set up logging ---------------------------------------
log_filename = os.path.join(save_folder, 'training.log')
logging.basicConfig(filename=log_filename, format='%(asctime)s %(levelname)s %(message)s', filemode='w')
logger = logging.getLogger(folder_name); logger.setLevel(logging.DEBUG)
class WrapLogger:
    def __init__(self, logger): self._l = logger
    def info(self, msg): print(msg); self._l.info(msg)
logger = WrapLogger(logger)

# ---------------- 2bis) RESUME: import meta + detect SSM -------------------
resume_ckpt_path = None
if START_MODE.lower() == "resume":
    resume_dir = RUN_DIR / 'results' / RESUME_RUN_FOLDER
    resume_ckpt_path = str(resume_dir / 'model_best.pth')
    resume_log_path  = str(resume_dir / 'training.log')
    if not os.path.exists(resume_ckpt_path):
        raise FileNotFoundError(f"[RESUME] Missing checkpoint: {resume_ckpt_path}")
    detected_k = detect_ssm_size_from_checkpoint(resume_ckpt_path, fallback=dim_internal)
    if detected_k != dim_internal:
        logger.info(f"[RESUME] Overriding dim_internal {dim_internal} → {detected_k}")
        dim_internal = int(detected_k)
    meta_old = load_meta_from_log(resume_log_path)
    if 'sim_horizon' in meta_old: sim_horizon = int(meta_old['sim_horizon'])
    if 'epsilon' in meta_old:      epsilon     = float(meta_old['epsilon'])
    def _maybe_tensorize(x):
        if x is None: return None
        arr = np.array(x, dtype=float)
        return torch.tensor(arr, dtype=torch.double)
    if 'obs_centers' in meta_old:       obs_centers = _maybe_tensorize(meta_old['obs_centers'])
    if 'obs_covs' in meta_old:          obs_covs    = _maybe_tensorize(meta_old['obs_covs'])
    if 'obs_vel' in meta_old:           obs_vel     = _maybe_tensorize(meta_old['obs_vel'])
    # do NOT override bounds or rho here

# ---------------- 3) Log metadata -----------------------------------------
logger.info("======= Experiment Metadata =======")
logger.info(f"START_MODE               : {START_MODE}")
if resume_ckpt_path: logger.info(f"RESUME_FROM              : {resume_ckpt_path}")
logger.info(f"Simulation horizon (T)   : {sim_horizon}")
logger.info(f"Epsilon for PSF          : {epsilon}")
logger.info(f"RENs dim_internal        : {dim_internal}")
logger.info(f"obs_centers              : {obs_centers}")
logger.info(f"obs_covs                 : {obs_covs}")
logger.info(f"obs_vel                  : {obs_vel}")
logger.info(f"state_lower_bound        : {state_lower_bound}")
logger.info(f"state_upper_bound        : {state_upper_bound}")
logger.info(f"controller_lower_bound   : {control_lower_bound}")
logger.info(f"controller_upper_bound   : {control_upper_bound}")
logger.info(f"rho value (learned roll-out) : {rho}")
logger.info("===================================")

# ---------------- 4) Define the environment --------------------------------
T_total = sim_horizon + 1 + 500
tvec = torch.arange(T_total, dtype=torch.double)
obs_pos_raw = obs_centers + (tvec[:, None] * DT) * obs_vel  # (T_total, 2)
obs_pos = obs_pos_raw.unsqueeze(0)                           # (1, T_total, 2)
moving_obs = MovingObstacle(obs_pos, obs_covs)

env = PendulumEnv(
    initial_state_low=torch.tensor([theta0_low,  0.0], dtype=torch.double),
    initial_state_high=torch.tensor([theta0_high, 0.0], dtype=torch.double),
    state_limit_low=state_lower_bound.to(torch.double),
    state_limit_high=state_upper_bound.to(torch.double),
    control_limit_low=control_lower_bound.to(torch.double),
    control_limit_high=control_upper_bound.to(torch.double),
    alpha_state=loss_x_multiplier,
    alpha_control=loss_u_match_multiplier,
    alpha_control_abs=loss_u_abs_multiplier,
    alpha_obst=loss_obs_multiplier,
    obstacle=moving_obs,
    obstacle_avoidance=True,
    alpha_cer=alpha_cer,
    control_reward_regularization=True,
    obstacle_avoidance_loss_function="pdf99clip",
    epsilon=epsilon,
    rho=rho,
    rho_bar=rho_bar,
    rho_max=rho_max,
    Qlyapunov=Qlyapunov,
    Rlyapunov=Rlyapunov,
    horizon = PSFhorizon,
    final_convergence_window=(sim_horizon-50, sim_horizon),
    convergence_theta_tol=0.05,
    convergence_omega_tol=0.30,
    convergence_hold_steps=20,
    convergence_bonus=50,#15.0,
    use_ramped_state_weight=False,
    state_weight_warmup=0.20,
    # --- MODIFIED: Pass context info ---
    sim_horizon=sim_horizon,
    obs_vel=obs_vel
    # --- END MODIFIED ---
)

# Precompute centers over time for overlay (cheap; reuse everywhere)
T_band = sim_horizon + 1
t_idx  = torch.arange(T_band, dtype=torch.double)
centers_over_time = obs_centers + (t_idx[:, None] * env.sys.h) * obs_vel  # [T,2] torch
cov_var_scalar = float(obs_covs.item())  # isotropic σ²

# ---------------- 5) MAD controller ----------------------------------------
mad_controller = MADController(
    env=env,
    buffer_capacity=100000,
    target_state=target_positions,
    num_dynamics_states=dim_internal,
    dynamics_input_time_window_length=sim_horizon+1,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    actor_lr=0.0005,
    critic_lr=0.001,
    std_dev=noise_std,
    control_action_upper_bound=float(control_upper_bound.item()),
    control_action_lower_bound=float(control_lower_bound.item()),
    nn_type=nn_type
)

# ---------------- 5bis) Load weights if RESUME -----------------------------
if START_MODE.lower() == "resume":
    mad_controller.load_model_weight(resume_ckpt_path)
    logger.info(f"[RESUME] Loaded weights from {resume_ckpt_path}")

# ---------------- Helpers ---------------------------------------------------
from contextlib import contextmanager
@contextmanager
def deterministic_actor(controller: MADController):
    old_std = getattr(getattr(controller, 'ou_noise', None), 'std_dev', None)
    if old_std is not None: controller.ou_noise.std_dev = 0.0
    was_training = controller.actor_model.training
    controller.actor_model.eval(); torch.set_grad_enabled(False)
    try:
        yield
    finally:
        if old_std is not None: controller.ou_noise.std_dev = old_std
        controller.actor_model.train(was_training); torch.set_grad_enabled(True)

def evaluate_policy(controller: MADController, n_inits: int = 5, timesteps: int = None) -> float:
    if timesteps is None: timesteps = sim_horizon
    scores = []
    with deterministic_actor(controller):
        for _ in range(n_inits):
            th0 = float(np.random.uniform(theta0_low, theta0_high))
            r_list, *_ = controller.get_trajectory(
                torch.tensor([[th0, 0.0]], dtype=torch.double), timesteps=timesteps
            )
            scores.append(float(np.sum(r_list)))
    return float(np.mean(scores)) if scores else -np.inf

# ---------------- 6) Deterministic rollout BEFORE training -----------------
before_prefix = os.path.join(save_folder, 'before_training')
with deterministic_actor(mad_controller):
    rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(
        torch.tensor([[start_position, 0.0]], dtype=torch.double),
        timesteps=sim_horizon
    )

# Time-series figure with overlay ON
plot_results(
    obs_log[:,:sim_horizon+1,:],
    u_log[:,:sim_horizon,:],
    u_L_Log=u_L_log[:,:sim_horizon,:],
    dt=env.sys.h,
    length=env.sys.l,
    plot_trj=False,
    file_path=before_prefix,
    obstacle_centers=None,
    obstacle_covs=None,
    overlay_ts_obstacle=True,
    obstacle_band_centers_t=centers_over_time[:sim_horizon],  # [T,2]
    obstacle_band_cov_var=cov_var_scalar,
    theta_ref=np.pi,
)

# ---------------- 7) Training with BEST checkpoint tracking ----------------
best_score = -float("inf")
best_path  = os.path.join(save_folder, "model_best.pth")

initial_eval = evaluate_policy(mad_controller, n_inits=5, timesteps=sim_horizon)
logger.info(f"[Eval] Before training: mean return over 5 inits = {initial_eval:.2f}")
best_score = initial_eval
mad_controller.save_model_weights(best_path)
logger.info(f"✓ Saved initial 'best' checkpoint → {best_path}")

for epoch in range(num_epochs):
    mad_controller.train(total_episodes=1, episode_length=sim_horizon, logger=logger)
    if hasattr(mad_controller, "ou_noise") and hasattr(mad_controller.ou_noise, "std_dev"):
        mad_controller.ou_noise.std_dev = max(0.1, mad_controller.ou_noise.std_dev * decayExpNoise)

    eval_return = evaluate_policy(mad_controller, n_inits=5, timesteps=sim_horizon)
    logger.info(f"[Eval] Epoch {epoch:03d}: mean return (5 inits) = {eval_return:.2f}")

    if eval_return > best_score:
        best_score = eval_return
        mad_controller.save_model_weights(best_path)
        logger.info(f"✓ New best checkpoint (return={best_score:.2f}) → {best_path}")

    # Optional intermediate plot (deterministic) — with overlay ON
    if epoch % 5 == 0:
        ic = mad_controller.last_episode_ic  # shape (2,)
        epoch_prefix = os.path.join(save_folder, f'epoch_{epoch:03d}')
        with deterministic_actor(mad_controller):
            rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(
                ic.view(1, -1),  # [[theta0, omega0]]
                timesteps=sim_horizon
            )
        plot_results(
            obs_log[:,:sim_horizon+1,:],
            u_log[:,:sim_horizon,:],
            u_L_Log=u_L_log[:,:sim_horizon,:],
            dt=env.sys.h,
            length=env.sys.l,
            plot_trj=False,
            file_path=epoch_prefix,
            obstacle_centers=None,
            obstacle_covs=None,
            overlay_ts_obstacle=True,
            obstacle_band_centers_t=centers_over_time[:sim_horizon],
            obstacle_band_cov_var=cov_var_scalar,
            theta_ref=np.pi,
        )

logger.info(f"Training complete. Best mean return = {best_score:.2f}")
logger.info(f"Best checkpoint path: {best_path}")

# ---------------- 8) Final rewards figure ----------------
fig = plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mad_controller.rewards_state_list) + 1), mad_controller.rewards_state_list, label=r'$Rewards_{x}$')
plt.plot(range(1, len(mad_controller.rewards_control_diff_list) + 1), mad_controller.rewards_control_diff_list, label=r'$Rewards_{u}$')
plt.plot(range(1, len(mad_controller.rewards_obs_list) + 1), mad_controller.rewards_obs_list, label=r'$Rewards_{obs}$')
plt.plot(range(1, len(mad_controller.rewards_list) + 1), mad_controller.rewards_list, label="Total Rewards")
plt.ylim([-4000, 20]); plt.xlabel("Episode"); plt.ylabel("Reward")
plt.title("Training Rewards (per episode)"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
rewards_curve_path = os.path.join(save_folder, 'rewards_curve.png')
fig.savefig(rewards_curve_path, dpi=300, bbox_inches='tight'); plt.close(fig)
logger.info(f"Saved final rewards curve plot → {rewards_curve_path}")

# ---------------- 9) Loss figures ----------------
fig = plt.figure(figsize=(7,4))
plt.plot(mad_controller.actor_loss_list); plt.ylim([-10, 10000])
plt.xlabel("Training step"); plt.ylabel("Actor loss"); plt.title("Actor loss during training"); plt.grid(True, alpha=0.3)
actor_loss_path = os.path.join(save_folder, 'actor_loss.png')
fig.savefig(actor_loss_path, dpi=300, bbox_inches='tight'); plt.close(fig)

fig = plt.figure(figsize=(7,4))
plt.plot(mad_controller.critic_loss_list); plt.ylim([-10, 10000])
plt.xlabel("Training step"); plt.ylabel("Critic loss"); plt.title("Critic loss during training"); plt.grid(True, alpha=0.3)
critic_loss_path = os.path.join(save_folder, 'critic_loss.png')
fig.savefig(critic_loss_path, dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- 10) After-training roll-out for comparison plots (deterministic) ----
mad_controller.load_model_weight(best_path)
after_prefix = os.path.join(save_folder, 'after_training')
with deterministic_actor(mad_controller):
    rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(
        torch.tensor([[start_position, 0.0]]), timesteps=sim_horizon+500
    )
for tN in [86, 500 + sim_horizon]:
    plot_results(
        obs_log[:,:tN+1,:],
        u_log[:,:tN,:],
        u_L_Log=u_L_log[:,:tN,:],
        dt=env.sys.h,
        length=env.sys.l,
        file_path=os.path.join(after_prefix, f't_{tN}'),
        plot_trj=True,
        obstacle_centers=obs_pos_raw[tN, :].unsqueeze(0),
        obstacle_covs=torch.tensor([[obs_covs.item(), obs_covs.item()]], dtype=torch.double),
        state_lower_bound=state_lower_bound.unsqueeze(0),
        state_upper_bound=state_upper_bound.unsqueeze(0),
        control_lower_bound=control_lower_bound.unsqueeze(0),
        control_upper_bound=control_upper_bound.unsqueeze(0),
    )

# ---------------- 11) Save metrics ----------------
rewards            = np.array(mad_controller.rewards_list)
rewards_state      = np.array(mad_controller.rewards_state_list)
rewards_control    = np.array(mad_controller.rewards_control_diff_list)
rewards_obs        = np.array(mad_controller.rewards_obs_list)
rewards_state_normalized = np.array(mad_controller.rewards_state_normalized_list)
rewards_control_normalized = np.array(mad_controller.rewards_control_normalized_list)
rewards_obs_normalized = np.array(mad_controller.rewards_obs_normalized_list)
actor_losses       = np.array(mad_controller.actor_loss_list)
critic_losses      = np.array(mad_controller.critic_loss_list)
out_path = os.path.join(save_folder, 'training_metrics.npz')
np.savez_compressed(
    out_path,
    rewards=rewards,
    rewards_state=rewards_state,
    rewards_control=rewards_control,
    rewards_obs=rewards_obs,
    rewards_state_normalized=rewards_state_normalized,
    rewards_control_normalized=rewards_control_normalized,
    rewards_obs_normalized=rewards_obs_normalized,
    actor_losses=actor_losses,
    critic_losses=critic_losses
)