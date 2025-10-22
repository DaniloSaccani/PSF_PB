import numpy as np
import torch
from mad_controller import MADController
from pendulum_env import PendulumEnv
from obstacles import MovingObstacle
from plot_functions import plot_results
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# NEW: imports for the PSF-only baseline and model geometry
from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

# ---------------- 0) Hyperparameters & Reproducibility ----------------
torch.set_default_dtype(torch.double)
torch.manual_seed(1)
np.random.seed(1)

# --- (A) User‐tunable quantities ---
sim_horizon       = 250   # Simulation horizon
sampled_noise_    = 0.1   # Initial x0 noise
loss_x_multiplier = 0.12   # was 0.03
loss_u_multiplier = 0.35   # was 2
loss_obs_multiplier = 3
epsilon           = 0.05  # PSF threshold
dim_internal      = 10
num_epochs        = 300
obs_centers       = torch.tensor([[1, 0.5]])
obs_covs          = torch.tensor([0.005])
obs_vel           = torch.tensor([[-0.2, 0.0]])
state_lower_bound = torch.tensor([0.5, -20])
state_upper_bound = torch.tensor([2 * np.pi - 0.5, 20])
control_lower_bound = torch.tensor([-3.0])
control_upper_bound = torch.tensor([3.0])
target_positions  = torch.tensor([np.pi, 0.0], dtype=torch.double)
start_position    = np.pi / 2
noise_std         = 0.5  # RL Exploration Noise
alpha_cer         = 0.6    # weight for Δu penalty
theta0_low, theta0_high = 0.6, 1.9
decayExpNoise     = 0.9  # per-epoch decay of exploration noise
# ---------------- (B) Fixed quantities ----------------
# Learned policy rollouts use FIXED rho=0.5; baseline uses scheduler with rhobar=0.5
rho = None
rho_bar = 0.5
rho_max = 2.0
nn_type = 'mad'

q_theta = 50
q_theta_dot = 10
r_u = 0.1
Qlyapunov = np.diag([q_theta, q_theta_dot]).astype(float)   # shape (2,2)
Rlyapunov = np.array([[r_u]], dtype=float)

# Baseline PSF horizon and references
BASELINE_HORIZON = 20
SAFE_TH_LO = 0.5
SAFE_TH_HI = 2*np.pi - 0.5
RHO_MAX_BASELINE = 1.0

# Obstacle shading constants
DT = 0.05
CHI2_2_95 = 5.991464547107979
OBSTACLE_SHADE_COLOR = (0.85, 0.2, 0.2, 0.18)

# ---------------- Helpers ----------------
def to_numpy(x):
    """Return a NumPy array from either a Tensor or an ndarray/list."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def unpack_logs(obs_log, u_L_log, u_log):
    """Make logs NumPy and unify shapes to (T+1,2), (T,1), (T,1)."""
    obs = to_numpy(obs_log)
    uL  = to_numpy(u_L_log)
    u   = to_numpy(u_log)
    if obs.ndim == 3:  # (B, T+1, 2)
        obs = obs[0]
    if uL.ndim == 3:   # (B, T, 1)
        uL = uL[0]
    if u.ndim == 3:    # (B, T, 1)
        u = u[0]
    return obs, uL, u

def wrap_0_2pi(a):
    return (a + 2*np.pi) % (2*np.pi)

def theta_arcs_hit_obstacle(center_xy, link_length, cov_var, chi2_val=CHI2_2_95):
    """
    95% 'collision' arc: pendulum tip p(θ) = [L sin θ, -L cos θ] inside circle
    radius r = sqrt(chi2 * cov) centered at center_xy. Returns list of (θ_lo, θ_hi) in [0, 2π).
    """
    r = np.sqrt(chi2_val * cov_var)
    c = np.asarray(center_xy, dtype=float)
    L = float(link_length)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        return [(0.0, 2*np.pi)] if r >= L else []
    zeta = (L**2 + norm_c**2 - r**2) / (2.0 * L * norm_c)
    if zeta <= -1.0:
        return [(0.0, 2*np.pi)]
    if zeta >=  1.0:
        return []
    delta = np.arccos(zeta)
    phi   = np.arctan2(c[1], c[0])
    th_lo = wrap_0_2pi(phi - delta + np.pi/2)
    th_hi = wrap_0_2pi(phi + delta + np.pi/2)
    if th_lo <= th_hi:
        return [(th_lo, th_hi)]
    else:
        return [(0.0, th_hi), (th_lo, 2*np.pi)]

# ---------------- 1) Create experiment folder ----------------
RUN_DIR = Path(__file__).resolve().parent
now = datetime.now().strftime("%m_%d_%H_%M_%S")
folder_name = f'PSF_SSM_NS_{now}'
save_folder = (RUN_DIR / 'results' / folder_name)
save_folder.mkdir(parents=True, exist_ok=True)
BASE_DIR = str(RUN_DIR)

# ---------------- 2) Set up logging ----------------
log_filename = os.path.join(save_folder, 'training.log')
logging.basicConfig(filename=log_filename, format='%(asctime)s %(levelname)s %(message)s', filemode='w')
logger = logging.getLogger(folder_name)
logger.setLevel(logging.DEBUG)
class WrapLogger:
    def __init__(self, logger): self._l = logger
    def info(self, msg): print(msg); self._l.info(msg)
logger = WrapLogger(logger)

# ---------------- 3) Log metadata ----------------
logger.info("======= Experiment Metadata =======")
logger.info(f"Simulation horizon (T)   : {sim_horizon}")
logger.info(f"Noise amplitude          : {sampled_noise_}")
logger.info(f"Number of epochs         : {num_epochs}")
logger.info(f"Epsilon for PSF          : {epsilon}")
logger.info(f"RENs dim_internal        : {dim_internal}")
logger.info(f"obs_centers              : {obs_centers}")
logger.info(f"obs_covs                 : {obs_covs}")
logger.info(f"obs_vel                  : {obs_vel}")
logger.info(f"Weight for state         : {loss_x_multiplier}")
logger.info(f"Weight for control diff  : {loss_u_multiplier}")
logger.info(f"Weight before loss_obs   : {loss_obs_multiplier}")
logger.info(f"state_lower_bound        : {state_lower_bound}")
logger.info(f"state_upper_bound        : {state_upper_bound}")
logger.info(f"controller_lower_bound   : {control_lower_bound}")
logger.info(f"controller_upper_bound   : {control_upper_bound}")
logger.info(f"noise_std                : {noise_std}")
logger.info(f"rho value (learned roll-out) : {rho}")
logger.info("===================================")

# ---------------- 4) Define the environment ----------------
T_total = sim_horizon + 1 + 500
t = torch.arange(T_total, dtype=obs_centers.dtype)
obs_pos_raw = obs_centers + (t[:, None] * DT) * obs_vel   # (T_total, 2)
obs_pos = obs_pos_raw.unsqueeze(0)                        # (1, T_total, 2)
moving_obs = MovingObstacle(obs_pos, obs_covs)

env = PendulumEnv(
    initial_state_low=torch.tensor([theta0_low,  0.0], dtype=torch.double),
    initial_state_high=torch.tensor([theta0_high, 0.0], dtype=torch.double),
    state_limit_low=state_lower_bound.to(torch.double),
    state_limit_high=state_upper_bound.to(torch.double),
    control_limit_low=control_lower_bound.to(torch.double),
    control_limit_high=control_upper_bound.to(torch.double),
    alpha_state=loss_x_multiplier,
    alpha_control=loss_u_multiplier,
    alpha_obst=loss_obs_multiplier,
    obstacle=moving_obs,
    obstacle_avoidance=True,
    alpha_cer=alpha_cer,                  # Δu penalty weight
    control_reward_regularization=True,   # enable Δu penalty
    epsilon=epsilon,
    rho=rho,              # FIXED rho for learned roll-out
    rho_bar=rho_bar,
    rho_max=rho_max,
    Qlyapunov=Qlyapunov,
    Rlyapunov=Rlyapunov
)

# ---------------- 5) MAD controller ----------------
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

# ---------------- Helper: deterministic evaluation over multiple inits ----------------
def evaluate_policy(controller: MADController, n_inits: int = 5, timesteps: int = None) -> float:
    """
    Return the mean episodic return over n_inits random θ0 ∈ [theta0_low, theta0_high].
    Uses controller.learned_policy (no OU noise) via get_trajectory.
    """
    if timesteps is None:
        timesteps = sim_horizon
    returns = []
    for _ in range(n_inits):
        th0 = float(np.random.uniform(theta0_low, theta0_high))
        r_list, *_ = controller.get_trajectory(torch.tensor([[th0, 0.0]], dtype=torch.double), timesteps=timesteps)
        returns.append(float(np.sum(r_list)))
    return float(np.mean(returns))

# ---------------- 6) (optional) Before-training roll-out ----------------
before_prefix = os.path.join(save_folder, 'before_training')
rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(
    torch.tensor([[start_position, 0.0]]), timesteps=sim_horizon+500
)
plot_instance = [86, 500 + sim_horizon]

# ---------------- 7) Training with BEST checkpoint tracking ----------------
best_score = -float("inf")
best_path  = os.path.join(save_folder, "model_best.pth")

# Initial eval before any training (so we can keep "best" if learning hurts)
initial_eval = evaluate_policy(mad_controller, n_inits=5, timesteps=sim_horizon)
logger.info(f"[Eval] Before training: mean return over 5 inits = {initial_eval:.2f}")
best_score = initial_eval
mad_controller.save_model_weights(best_path)
logger.info(f"✓ Saved initial 'best' checkpoint → {best_path}")

for epoch in range(num_epochs):
    mad_controller.train(total_episodes=1, episode_length=sim_horizon, logger=logger)
    mad_controller.ou_noise.std_dev = max(0.1, mad_controller.ou_noise.std_dev * decayExpNoise)   # decay per epoch of exploration noise

    # Evaluate deterministically (no OU) on a small batch of initial θ0
    eval_return = evaluate_policy(mad_controller, n_inits=5, timesteps=sim_horizon)
    logger.info(f"[Eval] Epoch {epoch:03d}: mean return (5 inits) = {eval_return:.2f}")

    # Keep only the best checkpoint
    if eval_return > best_score:
        best_score = eval_return
        mad_controller.save_model_weights(best_path)
        logger.info(f"✓ New best checkpoint (return={best_score:.2f}) → {best_path}")

    # Optional intermediate plot
    if epoch % 5 == 0:
        epoch_prefix = os.path.join(save_folder, f'epoch_{epoch:03d}')
        rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(
            torch.tensor([[start_position, 0.0]]), timesteps=sim_horizon
        )
        t_plot = min(sim_horizon, obs_pos_raw.shape[0]-1)
        plot_results(
            obs_log[:,:sim_horizon+1,:],
            u_log[:,:sim_horizon,:],
            u_L_Log=u_L_log[:,:sim_horizon,:],
            dt=env.sys.h,
            length=env.sys.l,
            plot_trj=False,
            file_path=epoch_prefix,
            obstacle_centers=obs_pos_raw[t_plot, :].unsqueeze(0),
            obstacle_covs=torch.tensor([[obs_covs.item(), obs_covs.item()]], dtype=torch.double),
            state_lower_bound=state_lower_bound.unsqueeze(0),
            state_upper_bound=state_upper_bound.unsqueeze(0),
            control_lower_bound=control_lower_bound.unsqueeze(0),
            control_upper_bound=control_upper_bound.unsqueeze(0),
        )

logger.info(f"Training complete. Best mean return = {best_score:.2f}")
logger.info(f"Best checkpoint path: {best_path}")

# ---------------- 8) Final rewards figure (with titles/axes) ----------------
fig = plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_state_list, label=r'$Rewards_{x}$')
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_control_diff_list, label=r'$Rewards_{u}$')
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_obs_list, label=r'$Rewards_{obs}$')
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_list, label="Total Rewards")
plt.ylim([-4000, 20])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Rewards (per episode)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
rewards_curve_path = os.path.join(save_folder, 'rewards_curve.png')
fig.savefig(rewards_curve_path, dpi=300, bbox_inches='tight')
plt.close(fig)
logger.info(f"Saved final rewards curve plot → {rewards_curve_path}")
logger.info('------------ Training complete ------------')

# ---------------- 9) Loss figures (titles/axes) ----------------
fig = plt.figure(figsize=(7,4))
plt.plot(mad_controller.actor_loss_list)
plt.ylim([-10, 10000])
plt.xlabel("Training step")
plt.ylabel("Actor loss")
plt.title("Actor loss during training")
plt.grid(True, alpha=0.3)
actor_loss_path = os.path.join(save_folder, 'actor_loss.png')
fig.savefig(actor_loss_path, dpi=300, bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(7,4))
plt.plot(mad_controller.critic_loss_list)
plt.ylim([-10, 10000])
plt.xlabel("Training step")
plt.ylabel("Critic loss")
plt.title("Critic loss during training")
plt.grid(True, alpha=0.3)
critic_loss_path = os.path.join(save_folder, 'critic_loss.png')
fig.savefig(critic_loss_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------- 10) After-training roll-out for comparison plots ----------------
# IMPORTANT: Reload the BEST checkpoint before plotting
mad_controller.load_model_weight(best_path)

after_prefix = os.path.join(save_folder, 'after_training')
rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(
    torch.tensor([[start_position, 0.0]]), timesteps=sim_horizon+500
)
for tN in plot_instance:
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
logger.info(f"Saved 'after training' plots into {after_prefix}/*")

# Save metrics
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

# =====================================================================
# 11) Closed-loop comparisons (forward simulation, logging J*)
#     Plots: θ(t) + obstacle band, ρ_t, J*(t)
# =====================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi
from PSF import MPCPredictSafetyFilter

# --- Make sure Matplotlib does NOT call external LaTeX (fixes crashes) ---
mpl.rcParams['text.usetex'] = False  # use mathtext, not LaTeX

# ---- helpers (same geometry as run_prestabilizedSys) -----------------
CHI2_2_95 = 5.991464547  # 95% quantile for chi-square with 2 dof
OBSTACLE_SHADE_COLOR = (0.95, 0.55, 0.55, 0.35)  # soft red

SAFE_TH_LO = float(state_lower_bound[0].item())
SAFE_TH_HI = float(state_upper_bound[0].item())

DT = float(env.sys.h)
T_STEPS = int(sim_horizon)  # number of input steps (states have T_STEPS+1 samples)
HORIZON = 25                # PSF horizon for these replays (tune if needed)

# Angle wrap to [0, 2π)
def wrap_0_2pi(x):
    return np.mod(x, 2*np.pi)

# Compute θ-intervals that make the tip intersect a circular obstacle (95%)
def theta_arcs_hit_obstacle(center_xy, L, cov_var):
    # 95% radius for isotropic Gaussian with variance 'cov_var'
    r = np.sqrt(CHI2_2_95 * cov_var)
    c = np.asarray(center_xy, dtype=float).reshape(2,)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        return [(0.0, 2*np.pi)]  # centered on pivot -> full circle

    # Tip moves on circle of radius L centered at origin. Obstacle is circle (center c, radius r).
    # Let α be the angle on the tip circle; θ = α + π/2. Intersection condition:
    zeta = (L**2 + norm_c**2 - r**2) / (2.0 * L * norm_c)
    if zeta <= -1.0:
        return [(0.0, 2*np.pi)]
    if zeta >=  1.0:
        return []

    delta = np.arccos(zeta)          # half width in α
    phi   = np.arctan2(c[1], c[0])   # center direction
    alpha_lo, alpha_hi = phi - delta, phi + delta
    # Convert back to θ = α + π/2 and wrap
    th_lo = wrap_0_2pi(alpha_lo + np.pi/2)
    th_hi = wrap_0_2pi(alpha_hi + np.pi/2)
    if th_lo <= th_hi:
        return [(th_lo, th_hi)]
    else:
        return [(0.0, th_hi), (th_lo, 2*np.pi)]

# External copy of the smooth schedule used inside PSF (for plotting ρ_t)
def rho_schedule_from_uL(uL_scalar, rho_bar, rho_max, epsilon):
    # same as PSF.py: sigma = ||uL||^2 / (ε^2 + ||uL||^2)
    r2 = float(uL_scalar**2)
    sigma = r2 / (epsilon*epsilon + r2)
    return float(rho_bar + (rho_max - rho_bar) * sigma)

# ---- build obstacle centers along time (same as your env/moving_obs) --
obs_centers_np = obs_centers.detach().cpu().numpy()
obs_vel_np     = obs_vel.detach().cpu().numpy()
def obstacle_centers_at_step(k: int):
    return obs_centers_np + (k * DT) * obs_vel_np  # shape (n_obs, 2)

# ---- Common PSF model/plant ingredients --------------------------------
xbar = np.array([np.pi, 0.0], dtype=float)
U_MIN = float(control_lower_bound.item())
U_MAX = float(control_upper_bound.item())

Q = Qlyapunov
R = Rlyapunov
EPSILON = float(epsilon)
RHO_MAX = float(rho_max)

def simulate_closed_loop(uL_seq,  # array shape (T_STEPS, 1) OR callable(t, x) -> np.ndarray([1])
                         rho_fixed=None,  # if None -> scheduler ON; else scalar fixed ρ
                         title_suffix=""):
    """
    Runs PSF + plant forward, logs θ, ρ_t (for plotting), J*(t), feasibility.
    If uL_seq is callable, it is evaluated at each step with (t, x_t) to produce u_L,t.
    """
    # CasADi model for PSF + torch plant
    model = SinglePendulumCasadi(xbar=xbar.copy())
    L_link = float(model.l)
    plant = SinglePendulum(
        xbar=torch.tensor(xbar, dtype=torch.double),
        x_init=torch.tensor([start_position, 0.0], dtype=torch.double).view(1, -1),
        u_init=torch.zeros(1, 1, dtype=torch.double),
    )

    psf = MPCPredictSafetyFilter(
        model,
        horizon=HORIZON,
        state_lower_bound=state_lower_bound.numpy().astype(float),
        state_upper_bound=state_upper_bound.numpy().astype(float),
        control_lower_bound=control_lower_bound.numpy().astype(float),
        control_upper_bound=control_upper_bound.numpy().astype(float),
        Q=Q, R=R, solver_opts=None,
        set_lyacon=True,
        epsilon=EPSILON,
        rho=rho_fixed,         # None -> scheduled, scalar -> fixed-ρ
        rho_bar=float(rho_bar),
        rho_max=RHO_MAX,
    )

    theta_log = [float(start_position)]
    rho_log   = []
    Jstar_log = []
    feas_log  = []

    U_prev = None
    X_prev = None
    x_t = torch.tensor([start_position, 0.0], dtype=torch.double).view(1,1,-1)

    for t in range(T_STEPS):
        x_np = x_t.detach().cpu().numpy().reshape(-1)

        # get u_L,t
        if callable(uL_seq):
            uL = np.asarray(uL_seq(t, x_np), dtype=float).reshape(1,)
        else:
            # use provided sequence (from learned rollout)
            uL = np.asarray(uL_seq[t, :], dtype=float).reshape(1,)

        # solve PSF at current state
        try:
            if t == 0:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, xbar, uL)
            else:
                U_sol, X_sol, J_curr = psf.solve_mpc(x_np, xbar, uL, U_prev, X_prev)
        except Exception:
            U_sol, X_sol, J_curr = None, None, None

        if U_sol is None or X_sol is None:
            # Infeasible: pass-through uL; drop warm start
            u_cmd = uL.copy().reshape(1,1,1)
            U_prev, X_prev = None, None
            feas_log.append(False)
            Jstar_log.append(np.nan)
        else:
            # Apply first move and keep warm start for next step
            u_cmd = U_sol[:, 0:1].reshape(1,1,1)
            U_prev, X_prev = U_sol, X_sol
            feas_log.append(True)
            Jstar_log.append(float(J_curr))

        # log ρ_t used (for plotting)
        if rho_fixed is None:
            rho_log.append(rho_schedule_from_uL(uL[0], rho_bar=float(rho_bar), rho_max=RHO_MAX, epsilon=EPSILON))
        else:
            rho_log.append(float(rho_fixed))

        # plant step (filtered control)
        x_t = plant.rk4_integration(x_t, torch.tensor(u_cmd, dtype=torch.double))
        theta_log.append(float(x_t.view(-1)[0]))

    return (np.array(theta_log), np.array(rho_log), np.array(Jstar_log), np.array(feas_log), L_link)

# ---- Build uL sequence for the "learned" run from logs (from best model rollout) ----
# Use the u_L sequence from the AFTER-TRAINING rollout above (best model loaded)
obs_arr_sec, uL_arr_sec, u_arr_sec = unpack_logs(obs_log, u_L_log, u_log)
uL_learn_seq = np.asarray(uL_arr_sec[:int(sim_horizon), :], dtype=float)   # shape (T_STEPS, 1)

# 1) Baseline: fixed rho=0.5, u_L=0
uL_zero = np.zeros_like(uL_learn_seq)
theta_base, rho_base, J_base, feas_base, L_link = simulate_closed_loop(
    uL_seq=uL_zero, rho_fixed=0.5, title_suffix="(baseline)"
)

# 2) Learned: scheduled rho (rho=None), actor’s u_L from the rollout
theta_learn_arr, rho_learn, J_learn, feas_learn, _ = simulate_closed_loop(
    uL_seq=uL_learn_seq, rho_fixed=None, title_suffix="(learned)"
)

# ---- time axes ----------------------------------------------------------
t_states = np.arange(int(sim_horizon) + 1) * DT
t_inputs = np.arange(int(sim_horizon)) * DT

# ---- Figure 1: θ(t) with obstacle bands + reference ---------------------
fig1, ax1 = plt.subplots(figsize=(10, 4.8))
ax1.plot(t_states[:theta_base.shape[0]],  theta_base,  label=r'PSF-only ($\bar{\rho}=0.5$, $u_L\equiv 0$)', lw=1.4)
ax1.plot(t_states[:theta_learn_arr.shape[0]], theta_learn_arr, label=r'Trained policy + PSF (scheduled $\rho_t$)', lw=1.4)
ax1.axhline(np.pi, linestyle='--', linewidth=1.0, color='k', label=r'$\theta^\star=\pi$')
ax1.set_title(r"Pendulum angle $\theta(t)$ with obstacle (95%)")
ax1.set_xlabel("time [s]")
ax1.set_ylabel(r"$\theta$ [rad]")
ax1.set_ylim([SAFE_TH_LO-0.1, SAFE_TH_HI+0.1])
ax1.grid(True, alpha=0.3)

# obstacle shading (95%) per step using arcs
for k in range(int(sim_horizon)):
    centers_k = obstacle_centers_at_step(k)  # (n_obs, 2)
    for i in range(centers_k.shape[0]):
        arcs = theta_arcs_hit_obstacle(centers_k[i], L_link, cov_var=float(obs_covs[i]))
        if not arcs:
            continue
        x_span = [k*DT, (k+1)*DT]
        for (th_lo, th_hi) in arcs:
            ax1.fill_between(x_span, [th_lo, th_lo], [th_hi, th_hi],
                             facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', linewidth=0.0, zorder=0)

ax1.axhline(SAFE_TH_LO, color='k', lw=0.8, ls=':')
ax1.axhline(SAFE_TH_HI, color='k', lw=0.8, ls=':')
handles, labels = ax1.get_legend_handles_labels()
handles.append(Patch(facecolor=OBSTACLE_SHADE_COLOR, edgecolor='none', label='Obstacle (95%)'))
ax1.legend(handles=handles, loc='best')

theta_obs_path = os.path.join(save_folder, 'comparison_theta_obstacle.png')
fig1.tight_layout()
fig1.savefig(theta_obs_path, dpi=300, bbox_inches='tight')
plt.close(fig1)

# ---- Figure 2: ρ_t (learned vs baseline) --------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 4.6))
ax2.plot(t_inputs[:rho_base.shape[0]],  rho_base,  label=r'PSF-only $\rho_t$ (baseline)', lw=1.4)
ax2.plot(t_inputs[:rho_learn.shape[0]], rho_learn, label=r'Learned policy $\rho_t$ (scheduled)', lw=1.4)
ax2.set_title(r"Tightening schedule $\rho_t$")
ax2.set_xlabel("time [s]")
ax2.set_ylabel(r"$\rho_t$")
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')

rho_path = os.path.join(save_folder, 'comparison_rho.png')
fig2.tight_layout()
fig2.savefig(rho_path, dpi=300, bbox_inches='tight')
plt.close(fig2)

# ---- Figure 3: J*(t) (learned vs baseline) ------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 4.6))
ax3.plot(t_inputs[:J_base.shape[0]],  J_base,  label=r'PSF-only $J^\star$ (baseline)', lw=1.4)
ax3.plot(t_inputs[:J_learn.shape[0]], J_learn, label=r'Learned policy $J^\star$', lw=1.4)
ax3.set_title(r"Optimal value $J^\star$ per PSF solve")
ax3.set_xlabel("time [s]")
ax3.set_ylabel(r"$J^\star$")
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')

jstar_path = os.path.join(save_folder, 'comparison_Jstar.png')
fig3.tight_layout()
fig3.savefig(jstar_path, dpi=300, bbox_inches='tight')
plt.close(fig3)

logger.info(f"Saved comparison plots →\n  {theta_obs_path}\n  {rho_path}\n  {jstar_path}")
