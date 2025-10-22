# pendulum_env.py (top)
import torch
import numpy as np
import torch.nn.functional as F
from types import SimpleNamespace  # <— add

import loss_function as lf         # <— fix module name
from PSF import MPCPredictSafetyFilter
from single_pendulum_sys import SinglePendulum, SinglePendulumCasadi

# class PendulumEnv(DiscreteTransferFunctionEnv):   # <— remove base
class PendulumEnv:
    def __init__(self,
        x0 = torch.tensor([0.0, 0.0]),
        target_positions = np.array([np.pi, 0]),
        obstacle=None,
        obstacle_avoidance_loss_function="pdf",
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
        epsilon = 0.05,
        alpha_obst = 1,
        alpha_control = 1,
        alpha_state = 1,
        alpha_ca = 1,
        alpha_cer = 1,
        rho = None,
        rho_bar = None,
        rho_max = None,
        Qlyapunov=None,
        Rlyapunov=None,
        Q = None,
        R = None
    ):
        self.n = 2
        self.m = 1
        self.prestabilized = prestabilized
        self.disturbance = disturbance
        self.nonlinear_damping = nonlinear_damping
        self.obstacle = obstacle
        if rho_bar is None:
            rho_bar = 0.5
        if rho_max is None:
            rho_max = 2.0

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
        self.alpha_state = alpha_state
        self.alpha_ca = alpha_ca
        self.alpha_cer = alpha_cer

        self.step_reward_state_error = None
        self.step_reward_control_effort = None
        self.step_reward_control_effort_regularization = None
        self.step_reward_obstacle_avoidance = None
        self.step_reward_collision_avoidance = None

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

        # Limits
        if initial_state_low  is None: initial_state_low  = -1 * torch.ones((self.n,), dtype=torch.double)
        if initial_state_high is None: initial_state_high =  1 * torch.ones((self.n,), dtype=torch.double)
        if state_limit_low    is None: state_limit_low    = torch.tensor([0.5, -np.inf], dtype=torch.double)
        if state_limit_high   is None: state_limit_high   = torch.tensor([2*np.pi-0.5, np.inf], dtype=torch.double)
        if control_limit_low  is None: control_limit_low  = torch.tensor([-3.0], dtype=torch.double)
        if control_limit_high is None: control_limit_high = torch.tensor([ 3.0], dtype=torch.double)

        self.initial_state_low  = initial_state_low.reshape(-1)
        self.initial_state_high = initial_state_high.reshape(-1)
        self.state_limit_low    = state_limit_low.reshape(-1)
        self.state_limit_high   = state_limit_high.reshape(-1)
        self.control_limit_low  = control_limit_low.reshape(-1)
        self.control_limit_high = control_limit_high.reshape(-1)

        # Minimal gym-ish spaces
        self.observation_space = SimpleNamespace(shape=(self.n,))
        self.action_space      = SimpleNamespace(shape=(self.m,))

        # init
        self.state = None
        self.w = None
        self.prev_action = torch.zeros(self.m, dtype=torch.double).view(1,1,-1)
        self.control_reward_regularization = control_reward_regularization

        # plant + PSF model
        self.sys  = SinglePendulum(xbar=torch.tensor(target_positions, dtype=torch.double),
                                   x_init=x0, u_init=self.prev_action)

        sys_casadi = SinglePendulumCasadi(xbar=np.array(target_positions, dtype=float))
        self.PSF = MPCPredictSafetyFilter(
            sys_casadi,
            horizon=15,
            state_lower_bound=self.state_limit_low.numpy(),   # <— flat
            state_upper_bound=self.state_limit_high.numpy(),  # <— flat
            control_lower_bound=self.control_limit_low.numpy(),
            control_upper_bound=self.control_limit_high.numpy(),
            Q=self.Qlyapunov, R=self.Rlyapunov,
            solver_opts=None,
            set_lyacon=True,
            epsilon=epsilon,
            rho=rho,
            rho_bar = rho_bar,
            rho_max = rho_max
        )

    def f(self, x, u):
        # keep shape (B, 1, n)
        return self.sys.rk4_integration(x, u)

    def reset(self):
        self.state = torch.rand(self.n, dtype=torch.double) * (
                    self.initial_state_high - self.initial_state_low) + self.initial_state_low
        self.state = self.state.view(1, 1, -1)  # plant expects (B,1,n)
        self.prev_action *= 0.0
        self.t = 0
        self.min_dis = torch.tensor(float('inf'), dtype=self.state.dtype, device=self.state.device)
        return self.state.squeeze(0).squeeze(0)  # return (n,) to the agent

    def step(self, action, U_prev=None, X_prev=None):
        # PSF expects numpy 1D
        uL = np.array(action, dtype=float).reshape(-1)
        x0 = self.state.detach().cpu().numpy().reshape(-1)
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

        # plant step
        self.state = self.f(self.state, action_t)

        # (optional) disturbance
        self.w = self.disturbance_process() if self.disturbance else torch.zeros(self.n)

        # losses (match dtypes/devices)
        loss_states = lf.loss_state_tracking(self.state.squeeze(0).squeeze(0), self.target_positions, self.Q)
        self.step_reward_state_error = - self.alpha_state * loss_states

        uL_torch = torch.tensor(uL, dtype=torch.double)  # for diff penalty
        loss_action = lf.loss_control_effort(action_t.squeeze(0).squeeze(0), self.R, uL_torch)
        self.step_reward_control_effort = - self.alpha_control * loss_action

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

        reward = self.step_reward_state_error \
                 + self.step_reward_control_effort \
                 + self.step_reward_control_effort_regularization \
                 + self.step_reward_obstacle_avoidance

        terminated = False
        truncated = False
        self.t += 1
        return self.state.squeeze(0).squeeze(0), reward, terminated, truncated, {}, U, X


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
        return torch.zeros(self.n)

    def render(self, mode="human"):
        """
        Renders the current state of the environment. Currently, it prints the state to the console.

        Args:
            mode (str, optional): The rendering mode. Default is "human".
        """
        print(f"State: {self.state}")

    def close(self):
        """
        Closes the environment and releases any resources. Placeholder function.
        """
        pass