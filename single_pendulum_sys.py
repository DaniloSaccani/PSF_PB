# python
import casadi as ca
import numpy as np
import torch

torch.set_default_dtype(torch.double)  # precision needs to be aligned with CasaDi implementation
import torch.nn as nn
import matplotlib.pyplot as plt


class SinglePendulum(nn.Module):
    def __init__(self, xbar: torch.Tensor, x_init=None, u_init=None):
        """
        Initializes the pendulum parameters.
        m: mass
        l: length
        g: gravitational constant
        h: time step for integration.
        """
        super(SinglePendulum, self).__init__()
        m = 0.2
        l = 0.5
        g = 9.80
        #b = 0.005
        b = 0.02
        h = 0.05

        self.l = l
        self.a = g / l  # g/l, used with sin(theta)
        self.b1 = m * (l ** 2) / b
        self.b2 = m * (l ** 2)  # Moment of inertia
        self.h = h  # Time step stored as attribute

        # initial state
        self.register_buffer('xbar', xbar.reshape(1, -1))  # shape = (1, state_dim)
        x_init = self.xbar.detach().clone() if x_init is None else x_init.reshape(1, -1)  # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, int(self.xbar.shape[1] / 2)) if u_init is None else u_init.reshape(1,
                                                                                                   -1)  # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)

        # system dimensions
        self.state_dim = 2
        self.in_dim = 1
        assert self.xbar.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

    def dynamics(self, x, u):
        """
        Computes the continuous-time dynamics of the pendulum using masks instead of slicing.
        x: state tensor of shape (batch, 1, state_dim) where the last dimension is [theta, omega]
        u: control input (torque)
        Returns tensor of state derivatives with shape (batch, 1, state_dim).
        """
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)

        mask_theta = torch.tensor([1, 0]).view(1, 1, -1)
        mask_omega = torch.tensor([0, 1]).view(1, 1, -1)

        theta = (x * mask_theta).sum(dim=-1)  # (batch, 1)
        omega = (x * mask_omega).sum(dim=-1)  # (batch, 1)

        theta_dot = omega
        omega_dot = - self.a * torch.sin(theta) - omega / self.b1 + u.squeeze(-1) / self.b2

        return torch.stack((theta_dot, omega_dot), dim=-1)  # (batch, 1, state_dim)

    def rk4_integration(self, x, u):
        """
        Integrate one time step using the 4th order Runge-Kutta (RK4) method.
        """
        dt = self.h
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + dt / 2 * k1, u)
        k3 = self.dynamics(x + dt / 2 * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_new

    def noiseless_forward(self, t, x, u):  # TODO clear t
        """
        Computes the next state for a given current state (x) and control input (u)
        using the specified integration method ("rk4").
        """
        return self.rk4_integration(x, u)

    def forward(self, t, x, u, w):  # TODO clean t
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """
        x_new = self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)
        return x_new

    def rollout(self, data, controller, psfLayer=None, train=False):
        """
        Rolls out the state trajectory starting from initial state x0 using a sequence of control inputs.

        Arguments:
            data: [w0, 0, 0, ...]
            controller:
            method: Integration method ("euler" or "rk4"). Defaults to "rk4".
            train: boolean. Defaults to False.
        Returns:
            traj: Tensor containing the state trajectory of shape (T+1, state_dim)
                  (or (batch_size, T+1, state_dim) for batch inputs).
        """
        controller.reset()
        x = self.x_init.detach().clone().repeat(data.shape[0], 1, 1)  # shape = (batch, 1, state_dim)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)  # shape = (batch, 1, in_dim)

        if psfLayer is None:
            # Simulate
            print("----- Simulating with PB controller ----")
            for t in range(data.shape[1]):
                # x = self.forward(t=t, x=x, u=u, w=data[:, t:t+1, :])
                x_noisefree = self.noiseless_forward(t=t, x=x, u=u)
                w = data[:, t:t + 1, :].view(-1, 1, self.state_dim)
                x = x_noisefree + w
                u = controller.emme.forward(w)

                if t == 0:
                    x_log, u_log = x, u
                else:
                    x_log = torch.cat((x_log, x), 1)
                    u_log = torch.cat((u_log, u), 1)

            controller.reset()
            if not train:
                x_log, u_log = x_log.detach(), u_log.detach()

            return x_log, u_log
        else:
            # Simulate
            for t in range(data.shape[1]):
                x_noisefree = self.noiseless_forward(t=t, x=x, u=u)
                w = data[:, t:t + 1, :].view(-1, 1, self.state_dim)
                x = x_noisefree + w
                u_L = controller.emme.forward(w)
                if t == 0:
                    U, X = psfLayer(x, self.xbar, u_L)
                    u = U[:, 0:1].reshape(1, 1, 1)
                    x_log, u_log, u_L_log = x, u, u_L

                else:  # Provide Initial Guess
                    U, X = psfLayer(x, self.xbar, u_L, u_prev=U, x_prev=X)
                    u = U[:, 0:1].reshape(1, 1, 1)
                    x_log = torch.cat((x_log, x), 1)
                    u_log = torch.cat((u_log, u), 1)
                    u_L_log = torch.cat((u_L_log, u_L), 1)

            controller.reset()
            if not train:
                x_log, u_log, u_L_log = x_log.detach(), u_log.detach(), u_L_log.detach()

            return x_log, u_log, u_L_log


class SinglePendulumCasadi:
    def __init__(self, xbar):
        """
        Initializes the pendulum parameters.
        m: mass
        l: length
        g: gravitational constant
        b: damping coefficient
        h: time step for integration.
        """
        m = 0.2
        l = 0.5
        g = 9.80
        b = 0.02
        #b = 0.005
        h = 0.05

        self.l = l
        self.a = g / l  # g/l, used with sin(theta)
        self.b1 = m * (l ** 2) / b
        self.b2 = m * (l ** 2)  # Moment of inertia
        self.h = h  # Time step stored as attribute

        self.state_dim = 2
        self.in_dim = 1
        self.xbar = xbar.reshape(1, -1)  # shape = (1, state_dim)

        self.A_lin = np.eye(2) + h * np.array([[0, 1], [self.a, -1 / self.b1]])  # Linearized dynamics around xbar
        self.B_lin = h * np.array([[0], [1 / self.b2]])  # Linearized dynamics around xbar

    def dynamics(self, x, u):
        """
        Continuous-time dynamics of the pendulum.
        x: state vector [theta, omega]
        u: control input (torque)
        Returns [theta_dot, omega_dot]
        """
        theta = x[0]
        omega = x[1]
        theta_dot = omega
        omega_dot = - self.a * ca.sin(theta) - omega / self.b1 + u / self.b2
        return ca.vertcat(theta_dot, omega_dot)

    def rk4_integration(self, x, u):
        """
        Integrate one time step using 4th order Runge-Kutta (RK4),
        wrapping theta in [0, 2*pi).
        """
        h = self.h
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + h / 2 * k1, u)
        k3 = self.dynamics(x + h / 2 * k2, u)
        k4 = self.dynamics(x + h * k3, u)
        x_new = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_new

    def noiseless_forward(self, x, u):
        """
        Computes the next state given current state (x) and control input (u) using the
        integration method specified by 'method' ("euler" or "rk4"). The time step h is taken from self.h.
        """

        return self.rk4_integration(x, u)

