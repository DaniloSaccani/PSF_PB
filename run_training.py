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
from pathlib import Path


# —————— 0) Hyperparameters & Reproducibility ——————
torch.set_default_dtype(torch.double)
torch.manual_seed(1)
np.random.seed(1)

# --- (A) User‐tunable quantities ---
sim_horizon       = 250   # Simultion horizon
sampled_noise_    = 0.1   # Initial x0 noise
loss_x_multiplier = 0.03
loss_u_multiplier = 2  # this is the term/weight before loss_uRu; change as needed
loss_obs_multiplier = 3  #
epsilon           = 0.05  # PSF threshold (The bigger the epslon value, the earlier to activate the stability constraint)
dim_internal      = 24     # MAD LRU parameters
num_epochs        = 2#300
obs_centers       = torch.tensor([[1, 0.5]])
obs_covs          = torch.tensor([0.005])
obs_vel           = torch.tensor([[-0.2, 0.0]])
state_lower_bound = torch.tensor([0.5, -20])
state_upper_bound = torch.tensor([2 * np.pi - 0.5, 20])
control_lower_bound = torch.tensor([-3.0])
control_upper_bound = torch.tensor([3.0])
target_positions  = torch.tensor([np.pi, 0.0], dtype=torch.double)
start_position    = np.pi / 2
noise_std         = 1 # RL Exploration Noise
rho = 0.5
rho_bar = 0.5
rho_max = 10.0
nn_type = 'mad'
q_theta = 50
q_theta_dot = 10
r_u = 0.1
Qlyapunov = np.diag([q_theta, q_theta_dot]).astype(float)   # shape (2,2)
Rlyapunov = np.array([[r_u]], dtype=float)


# —————— 1) Create experiment folder (including num_samples in its name) ——————
RUN_DIR = Path(__file__).resolve().parent  # folder that contains run.py
now = datetime.now().strftime("%m_%d_%H_%M_%S")
folder_name = f'PSF_RENs_NS_{now}'
save_folder = (RUN_DIR / 'results' / folder_name)
save_folder.mkdir(parents=True, exist_ok=True)
# (optional) keep BASE_DIR around if other code expects it
BASE_DIR = str(RUN_DIR)

# —————— 2) Set up logging ——————
log_filename = os.path.join(save_folder, 'training.log')
logging.basicConfig(
    filename=log_filename,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='w'
)
logger = logging.getLogger(folder_name)
logger.setLevel(logging.DEBUG)
class WrapLogger:
    def __init__(self, logger): self._l = logger
    def info(self, msg): print(msg); self._l.info(msg)
logger = WrapLogger(logger)

# —————— 3) Log metadata at start ——————
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
#logger.info(f"target_positions         : {target_positions}")
logger.info(f"noise_std                : {noise_std}")
logger.info(f"rho value                : {rho}")
logger.info("===================================")


# —————— 4) Define the environment ——————
T = sim_horizon + 1 + 500
t = torch.arange(T, dtype=obs_centers.dtype)  # shape (T,)
dt = 0.05
obs_pos_raw = obs_centers + (t[:, None] * dt) * obs_vel   # shape (T,2)
obs_pos = obs_pos_raw.unsqueeze(0)    # shape (1, T, 2)
moving_obs = MovingObstacle(obs_pos, obs_covs)

env = PendulumEnv(
    initial_state_low=torch.tensor([start_position - sampled_noise_, 0.0], dtype=torch.double),
    initial_state_high=torch.tensor([start_position + sampled_noise_, 0.0], dtype=torch.double),
    state_limit_low=state_lower_bound.to(torch.double),
    state_limit_high=state_upper_bound.to(torch.double),
    control_limit_low=control_lower_bound.to(torch.double),
    control_limit_high=control_upper_bound.to(torch.double),
    alpha_state=loss_x_multiplier,
    alpha_control=loss_u_multiplier,
    alpha_obst=loss_obs_multiplier,
    obstacle=moving_obs,
    obstacle_avoidance=True,
    epsilon=epsilon,
    rho=rho,
    rho_bar = rho_bar,
    rho_max=rho_max,
    Qlyapunov = Qlyapunov,
    Rlyapunov = Rlyapunov
    )

# —————— 5) Define the MAD controller ——————
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

# mad_controller.load_model_weight(BASE_DIR+f'/experiments/pendulum_ddpg/saved_results/obs_right/PSF_RENs_NS_06_26_11_10_11/model_weights')

# —————— 6) Plot open loop trajectory ——————
before_prefix = os.path.join(save_folder, 'before_training')
rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(torch.tensor([[start_position, 0.0]]),timesteps=sim_horizon+500)
plot_instance = [86, 500 + sim_horizon]
# for t in plot_instance:
#     plot_results(
#             obs_log[:,:t+1,:],
#             u_log[:,:t,:],
#             u_L_Log=u_L_log[:,:t,:],
#             dt=env.sys.h,
#             length=env.sys.l,
#             file_path= os.path.join(before_prefix, f't_{t}'),
#             plot_trj=True,
#             obstacle_centers=obs_pos_raw[t, :].unsqueeze(0),
#             obstacle_covs=torch.tensor([[obs_covs, obs_covs]]),
#             state_lower_bound=state_lower_bound.unsqueeze(0),
#             state_upper_bound=state_upper_bound.unsqueeze(0),
#             control_lower_bound=control_lower_bound.unsqueeze(0),
#             control_upper_bound=control_upper_bound.unsqueeze(0),
#             )
    # logger.info(f"Saved 'before training' plots → {before_prefix}_timeseries.png, "
    #             f"{before_prefix}_trajectory.png")

# —————— 7) Training ——————
for epoch in range(num_epochs):
    mad_controller.train(
        total_episodes=1,
        episode_length=sim_horizon,
        logger=logger
    )

    # (Optional) Save an intermediate plot on test data
    if epoch % 5 == 0:
        print(torch.exp(-torch.exp(mad_controller.actor_model.m_dynamics.LRUR.nu_log)))
        if epoch == 30:
            print(1)
        epoch_prefix = os.path.join(save_folder, f'epoch_{epoch:03d}')
        rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(torch.tensor([[start_position, 0.0]]),

                                                                                      timesteps=sim_horizon)
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
    if epoch > 165 and epoch % 1 == 0:
        mad_controller.save_model_weights(os.path.join(save_folder, f'model_weights_epoch_{epoch:03d}'))
mad_controller.save_model_weights(os.path.join(save_folder, f'model_weights_epoch_{epoch:03d}'))

# —————— 8) After training: save final loss‐curve plot ——————
fig = plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_state_list,     label=r'$Rewards_{x}$')
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_control_diff_list,     label=r'$Rewards_{u}$' )
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_obs_list, label=r'$Rewards_{obs}$')
plt.plot(range(1, num_epochs + 1), mad_controller.rewards_list, label="Total Rewards")
plt.ylim([-4000, 20])
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title("Training Rewards over Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()

rewards_curve_path = os.path.join(save_folder, 'rewards_curve.png')
fig.savefig(rewards_curve_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved final rewards curve plot → {rewards_curve_path}")
plt.close(fig)

logger.info('------------ Training complete ------------')

# —————— 9) Plot open loop trajectory ——————
plt.plot(mad_controller.actor_loss_list)
plt.ylim([-10, 10000])
plt.show()
plt.plot(mad_controller.critic_loss_list)
plt.ylim([-10, 10000])
plt.show()

after_prefix = os.path.join(save_folder, 'after_training')
rewards_list, obs_log, u_L_log, w_list, u_log = mad_controller.get_trajectory(torch.tensor([[start_position, 0.0]]), timesteps=sim_horizon+500)
for t in plot_instance:
    plot_results(
            obs_log[:,:t+1,:],
            u_log[:,:t,:],
            u_L_Log=u_L_log[:,:t,:],
            dt=env.sys.h,
            length=env.sys.l,
            file_path= os.path.join(after_prefix, f't_{t}'),
            plot_trj=True,
            obstacle_centers=obs_pos_raw[t, :].unsqueeze(0),
            obstacle_covs=torch.tensor([[obs_covs.item(), obs_covs.item()]], dtype=torch.double),
            state_lower_bound=state_lower_bound.unsqueeze(0),
            state_upper_bound=state_upper_bound.unsqueeze(0),
            control_lower_bound=control_lower_bound.unsqueeze(0),
            control_upper_bound=control_upper_bound.unsqueeze(0),
            )
    logger.info(f"Saved 'before training' plots → {after_prefix}_timeseries.png, "
                f"{after_prefix}_trajectory.png")

# Convert lists to ndarrays
rewards            = np.array(mad_controller.rewards_list)
rewards_state      = np.array(mad_controller.rewards_state_list)
rewards_control    = np.array(mad_controller.rewards_control_diff_list)
rewards_obs        = np.array(mad_controller.rewards_obs_list)
rewards_state_normalized = np.array(mad_controller.rewards_state_normalized_list)
rewards_control_normalized = np.array(mad_controller.rewards_control_normalized_list)
rewards_obs_normalized = np.array(mad_controller.rewards_obs_normalized_list)
actor_losses       = np.array(mad_controller.actor_loss_list)
critic_losses      = np.array(mad_controller.critic_loss_list)

# Save them all in one compressed file
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