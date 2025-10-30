# pendulum_env.py (top)
import torch
import numpy as np
import torch.nn.functional as F
from types import SimpleNamespace
import math

import loss_function as lf
from PSF import MPCPredictSafetyFilter
from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi


class PendulumEnv:
    def __init__(self,
                 x0=torch.tensor([0.0, 0.0]),
                 target_positions=np.array([np.pi, 0]),
                 obstacle=None,
                 obstacle_avoidance_loss_function="pdf99clip",  # before "pdf"
                 prestabilized=True,
                 disturbance=True,
                 nonlinear_damping=True,
                 obstacle_avoidance=False,
                 collision_avoidance=False,
                 control_reward_regularization=False,
                 initial_state_low=None,
                 initial_state_high=None,
                 state_limit_low=None,
                 state_limit_high=None,
                 control_limit_low=None,
                 control_limit_high=None,
                 epsilon=0.05,
                 alpha_obst=1,
                 alpha_control=1,
                 alpha_control_abs=0.0,
                 alpha_state=1,
                 alpha_ca=1,
                 alpha_cer=1,
                 rho=None,
                 rho_bar=None,
                 rho_max=None,
                 Qlyapunov=None,
                 Rlyapunov=None,
                 Q=None,
                 R=None,
                 horizon=None,
                 final_convergence_window=(220, 250),  # (t0, t1) in sim steps
                 convergence_theta_tol=0.08,  # rad, ~4.6°
                 convergence_omega_tol=0.30,  # rad/s
                 convergence_hold_steps=5,  # require K consecutive steps
                 convergence_bonus=5.0,  # per-step bonus scale
                 # --- OPTIONAL: de-emphasize early tracking ---
                 use_ramped_state_weight=False,
                 state_weight_warmup=0.20,
                 # --- MODIFIED: Added for augmented state ---
                 sim_horizon=250,
                 obs_vel=None
                 ):
        self.n = 2  # physical state dim
        self.m = 1  # control dim

        # --- MODIFIED: Context dims (REMOVED phi_t) ---
        # context = [obs_pos_x, obs_pos_y, obs_vel_x, obs_vel_y]
        self.context_n = 2 + 2
        self.aug_n = self.n + self.context_n  # augmented state dim (now 6)
        # --- END MODIFIED ---

        self.sim_horizon = float(sim_horizon)  # Still used for convergence bonus logic
        if obs_vel is None:
            self.obs_vel = torch.tensor([0.0, 0.0], dtype=torch.double)
        else:
            # Assuming 1 obstacle for simplicity, matching run.py
            self.obs_vel = obs_vel.reshape(-1).to(torch.double)

        self.prestabilized = prestabilized
        self.disturbance = disturbance
        self.nonlinear_damping = nonlinear_damping
        self.obstacle = obstacle
        if rho_bar is None:
            rho_bar = 0.5
        if rho_max is None:
            rho_max = 2.0

        if horizon is None:
            self.horizon = 15
        else:
            self.horizon = horizon

        self.t = 0
        self.dt = 0.05

        self.min_dis = torch.tensor(float('inf'), dtype=torch.double)
        self.obstacle_avoidance = obstacle_avoidance
        self.collision_avoidance = collision_avoidance

        if self.obstacle_avoidance:
            self.obstacle_avoidance_loss_function = getattr(
                lf, f"loss_obstacle_avoidance_{obstacle_avoidance_loss_function}"
            )

        self.alpha_obst = alpha_obst
        self.alpha_control = alpha_control
        self.alpha_control_abs = alpha_control_abs
        self.alpha_state = alpha_state
        self.alpha_ca = alpha_ca
        self.alpha_cer = alpha_cer

        self.step_reward_state_error = None
        self.step_reward_control_effort = None
        self.step_reward_control_effort_regularization = None
        self.step_reward_obstacle_avoidance = None
        self.step_reward_collision_avoidance = None

        self.final_window_start, self.final_window_end = final_convergence_window
        self.theta_tol = float(convergence_theta_tol)
        self.omega_tol = float(convergence_omega_tol)
        self.convergence_hold_steps = int(convergence_hold_steps)
        self.convergence_bonus = float(convergence_bonus)
        self.use_ramped_state_weight = bool(use_ramped_state_weight)
        self.state_weight_warmup = float(state_weight_warmup)

        self.converged_counter = 0
        self.step_reward_convergence = torch.tensor(0.0, dtype=torch.double)

        # weights for loss
        if Q is None:
            self.Q = torch.diag(torch.tensor([30.0, 5.0]))
        else:
            self.Q = Q
        if R is None:
            self.R = torch.eye(self.m)
        else:
            self.R = R

        # weights for the lyapunov
        if Qlyapunov is None:
            self.Qlyapunov = torch.diag(torch.tensor([10.0, 1.0]))
        else:
            self.Qlyapunov = Qlyapunov
        if Rlyapunov is None:
            self.Rlyapunov = torch.eye(self.m)
        else:
            self.Rlyapunov = Rlyapunov

        # Target (equilibrium)
        self.target_positions = torch.tensor(target_positions, dtype=torch.double)

        # Limits (for physical state)
        if initial_state_low is None: initial_state_low = -1 * torch.ones((self.n,), dtype=torch.double)
        if initial_state_high is None: initial_state_high = 1 * torch.ones((self.n,), dtype=torch.double)
        if state_limit_low is None: state_limit_low = torch.tensor([0.5, -np.inf], dtype=torch.double)
        if state_limit_high is None: state_limit_high = torch.tensor([2 * np.pi - 0.5, np.inf], dtype=torch.double)
        if control_limit_low is None: control_limit_low = torch.tensor([-3.0], dtype=torch.double)
        if control_limit_high is None: control_limit_high = torch.tensor([3.0], dtype=torch.double)

        self.initial_state_low = initial_state_low.reshape(-1)
        self.initial_state_high = initial_state_high.reshape(-1)
        self.state_limit_low = state_limit_low.reshape(-1)
        self.state_limit_high = state_limit_high.reshape(-1)
        self.control_limit_low = control_limit_low.reshape(-1)
        self.control_limit_high = control_limit_high.reshape(-1)

        # Minimal gym-ish spaces
        # --- MODIFIED: Observation space is the AUGMENTED state ---
        self.observation_space = SimpleNamespace(shape=(self.aug_n,))
        # --- END MODIFIED ---
        self.action_space = SimpleNamespace(shape=(self.m,))

        # init
        self.state = None  # This will store the PHYSICAL state [theta, omega]
        self.w = None
        self.prev_action = torch.zeros(self.m, dtype=torch.double).view(1, 1, -1)
        self.control_reward_regularization = control_reward_regularization

        # plant + PSF model
        self.sys = SinglePendulum(xbar=torch.tensor(target_positions, dtype=torch.double),
                                  x_init=x0, u_init=self.prev_action)

        sys_casadi = SinglePendulumCasadi(xbar=np.array(target_positions, dtype=float))
        self.PSF = MPCPredictSafetyFilter(
            sys_casadi,
            horizon=self.horizon,
            state_lower_bound=self.state_limit_low.numpy(),  # <— flat
            state_upper_bound=self.state_limit_high.numpy(),  # <— flat
            control_lower_bound=self.control_limit_low.numpy(),
            control_upper_bound=self.control_limit_high.numpy(),
            Q=self.Qlyapunov, R=self.Rlyapunov,
            solver_opts=None,
            set_lyacon=True,
            epsilon=epsilon,
            rho=rho,
            rho_bar=rho_bar,
            rho_max=rho_max
        )

    def f(self, x, u):
        # keep shape (B, 1, n)
        return self.sys.rk4_integration(x, u)

    # --- MODIFIED: New helper function ---
    def _get_augmented_state(self):
        """
        Constructs the augmented state \tilde{x}_t = [x_t, c_t]
        where x_t = [theta, omega]
        and   c_t = [p_obs_x, p_obs_y, v_obs_x, v_obs_y]
        """
        # Ensure state is on the correct device
        default_device = self.state_limit_low.device

        # 1. Physical state (size 2)
        physical_state = self.state.squeeze(0).squeeze(0).to(default_device)

        # 2. Context: phi_t (REMOVED)

        # 3. Context: obs_pos_t (size 2)
        # Clamp time index for obstacle lookup
        max_obs_index = self.obstacle.positions.size(1) - 1
        safe_t_index = min(self.t, max_obs_index)
        obs_pos_t, _ = self.obstacle.get_obstacles(safe_t_index)

        obs_pos_flat = obs_pos_t.reshape(-1).to(torch.double).to(default_device)

        # 4. Context: obs_vel (size 2)
        obs_vel_flat = self.obs_vel.reshape(-1).to(torch.double).to(default_device)

        # --- MODIFIED: torch.cat call (REMOVED phi_t) ---
        aug_state = torch.cat([
            physical_state,
            obs_pos_flat,
            obs_vel_flat
        ])
        # --- END MODIFIED ---

        return aug_state.squeeze()  # Return shape (aug_n,)

    # --- END MODIFIED ---

    def reset(self):
        # Reset PHYSICAL state
        self.state = torch.rand(self.n, dtype=torch.double) * (
                self.initial_state_high - self.initial_state_low) + self.initial_state_low
        self.state = self.state.view(1, 1, -1)  # plant expects (B,1,n)

        # Reset internal trackers
        self.prev_action *= 0.0
        self.t = 0
        self.min_dis = torch.tensor(float('inf'), dtype=self.state.dtype, device=self.state.device)
        self.converged_counter = 0
        self.step_reward_convergence = torch.tensor(0.0, dtype=torch.double)

        # --- MODIFIED: Return AUGMENTED state ---
        return self._get_augmented_state()
        # --- END MODIFIED ---

    def step(self, action, U_prev=None, X_prev=None):
        # PSF expects numpy 1D
        uL = np.array(action, dtype=float).reshape(-1)
        x0 = self.state.detach().cpu().numpy().reshape(-1)  # Use physical state
        xbar = self.target_positions.detach().cpu().numpy().reshape(-1)

        if self.t == 0:
            U, X, _ = self.PSF.solve_mpc(x0, xbar, uL)
        else:
            U, X, _ = self.PSF.solve_mpc(x0, xbar, uL, U_prev, X_prev)

        # fallback (infeasible PSF): pass-through uL
        if U is None or X is None:
            U = np.tile(uL.reshape(-1, 1), (1, self.PSF.horizon))
            X = np.tile(x0.reshape(-1, 1), (1, self.PSF.horizon + 1))

        u = torch.tensor(U[:, 0:1].reshape(1, 1, 1), dtype=torch.double)
        action_t = u  # filtered input

        # plant step (updates self.state)
        self.state = self.f(self.state, action_t)

        # (optional) disturbance
        self.w = self.disturbance_process() if self.disturbance else torch.zeros(self.n)

        # losses (match dtypes/devices)
        loss_states = lf.loss_state_tracking(self.state.squeeze(0).squeeze(0), self.target_positions, self.Q)
        # T_switch = 2/0.05
        # # --- MODIFIED: Time-varying target for learner ---
        # current_physical_state = self.state.squeeze(0).squeeze(0)  # Get physical state [theta, omega]
        #
        # if self.t < T_switch:
        #     # Use temporary target for the first T_switch steps
        #     target_temp = torch.tensor([np.pi - 2.6, 0.0], dtype=torch.double, device=current_physical_state.device)
        #     loss_states = lf.loss_state_tracking(current_physical_state, target_temp, self.Q)
        # else:
        #     # Use the final target after T_switch steps
        #    loss_states = lf.loss_state_tracking(current_physical_state, self.target_positions, self.Q)
        # --- END MODIFIED ---

        # The rest of the reward calculation uses this potentially time-varying loss_states
        alpha_state_eff = self.alpha_state
        # ... (ramped weight logic remains the same) ...
        self.step_reward_state_error = - alpha_state_eff * loss_states

        alpha_state_eff = self.alpha_state
        if self.use_ramped_state_weight:
            t0, t1 = self.final_window_start, self.final_window_end
            # ramp w(t): 0 before t0 → 1 at t1
            w = (self.t - t0) / max(1, (t1 - t0))
            w = max(0.0, min(1.0, float(w)))
            # keep a small fraction early, then ramp up
            alpha_state_eff = self.alpha_state * (self.state_weight_warmup + (1.0 - self.state_weight_warmup) * w)

        self.step_reward_state_error = - alpha_state_eff * loss_states

        uL_torch = torch.tensor(uL, dtype=torch.double)  # for diff penalty
        loss_action = lf.loss_control_effort(action_t.squeeze(0).squeeze(0), self.R, uL_torch)
        self.step_reward_control_effort = - self.alpha_control * loss_action

        loss_abs = lf.loss_control_effort_abs(action_t.squeeze(0).squeeze(0), self.R)
        self.step_reward_control_effort_abs = - self.alpha_control_abs * loss_abs

        self.step_reward_control_effort_regularization = 0
        if self.control_reward_regularization:
            loss_action_regularized = lf.loss_control_effort_regularized(
                u=action_t.squeeze(0).squeeze(0), u_prev=self.prev_action.squeeze(0).squeeze(0), R=self.R
            )
            self.step_reward_control_effort_regularization = - self.alpha_cer * loss_action_regularized
            self.prev_action = action_t

        self.step_reward_obstacle_avoidance = 0
        if self.obstacle_avoidance:
            theta = self.state[0, 0, 0]
            x_tip = self.sys.l * torch.sin(theta)
            y_tip = -self.sys.l * torch.cos(theta)
            pos = torch.stack([x_tip, y_tip])
            loss_obst, min_dis = self.obstacle.get_obstacle_avoidance_loss(pos, self.obstacle_avoidance_loss_function,
                                                                           self.t)
            self.step_reward_obstacle_avoidance = - self.alpha_obst * loss_obst
            min_dis = torch.as_tensor(min_dis, dtype=self.state.dtype, device=self.state.device)
            self.min_dis = torch.minimum(self.min_dis, min_dis)

        # --- Late-window convergence bonus (only matters near the end) ---
        t0, t1 = self.final_window_start, self.final_window_end
        w = (self.t - t0) / max(1, (t1 - t0))  # linear ramp 0→1 on [t0, t1]
        w = max(0.0, min(1.0, float(w)))
        w_t = torch.tensor(w, dtype=torch.double, device=self.state.device)

        theta = self.state[0, 0, 0]
        omega = self.state[0, 0, 1]

        # angle error to π, wrapped to (-π, π]
        theta_err = torch.atan2(torch.sin(theta - math.pi), torch.cos(theta - math.pi))

        inside = (torch.abs(theta_err) <= self.theta_tol) & (torch.abs(omega) <= self.omega_tol)

        # require K consecutive "inside" steps to count as converged
        self.converged_counter = self.converged_counter + 1 if bool(inside) else 0
        hold_ok = self.converged_counter >= self.convergence_hold_steps
        hold_ok_t = torch.tensor(1.0 if hold_ok else 0.0, dtype=torch.double, device=self.state.device)

        self.step_reward_convergence = w_t * hold_ok_t * self.convergence_bonus

        reward = self.step_reward_state_error \
                 + self.step_reward_control_effort \
                 + self.step_reward_control_effort_regularization \
                 + self.step_reward_control_effort_abs \
                 + self.step_reward_obstacle_avoidance \
                 + self.step_reward_convergence

        terminated = False
        truncated = False

        # --- MODIFIED: Increment time *before* getting augmented state ---
        self.t += 1
        aug_state = self._get_augmented_state()
        return aug_state, reward, terminated, truncated, {}, U, X
        # --- END MODIFIED ---

    def disturbance_process(self):
        """
        Generates a random disturbance process that decays over time, simulating external forces
        acting on the mobile robots.
         # w = (x0, 0, 0, ...)
        Returns:
            torch.Tensor: A disturbance vector added to the system dynamics.
        """
        # d = 0.1 * torch.randn(self.n)
        # d[3::4] *= 0.1
        # d[2::4] *= 0.1
        # d *= torch.exp(torch.tensor(-0.05 * self.t))
        return torch.zeros(self.n)  # self.n is physical_n

    def render(self, mode="human"):
        """
        Renders the current state of the environment. Currently, it prints the state to the console.

        Args:
            mode (str, optional): The rendering mode. Default is "human".
        """
        print(f"State: {self.state}")  # Print physical state

    def close(self):
        """
        Closes the environment and releases any resources. Placeholder function.
        """
        pass