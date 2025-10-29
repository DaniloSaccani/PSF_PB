import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import scipy


class MPCPredictSafetyFilter:
    def __init__(
            self,
            model,
            horizon,
            state_lower_bound=None,
            state_upper_bound=None,
            control_lower_bound=None,
            control_upper_bound=None,
            Q=None,
            R=None,
            solver_opts=None,
            set_lyacon=False,
            periodic_info=False,
            epsilon=0.05,
            rho=None,
            rho_bar=None,
            rho_max=None,
            # --- NEW ---
            solver_tol=1e-7  # Tolerance for equality constraints
    ):
        self.model_func = model.noiseless_forward
        self.xbar = model.xbar
        self.nx = model.state_dim
        self.nu = model.in_dim
        self.horizon = horizon
        self.xlb = state_lower_bound if state_lower_bound is not None else np.full(self.nx, -np.inf)
        self.xub = state_upper_bound if state_upper_bound is not None else np.full(self.nx, np.inf)
        self.ulb = control_lower_bound if control_lower_bound is not None else np.full(self.nu, -np.inf)
        self.uub = control_upper_bound if control_upper_bound is not None else np.full(self.nu, np.inf)
        self.dt = model.h
        self.periodic_info = periodic_info if periodic_info is not None else {}
        self.epsilon = epsilon
        self.rho = rho
        if rho_bar is None:
            self.rho_bar = 0.5
        else:
            self.rho_bar = rho_bar
        if rho_max is None:
            self.rho_max = 2.0
        else:
            self.rho_max = rho_max

        # --- NEW ---
        self.solver_tol = solver_tol
        # --- END NEW ---

        R = R if R is not None else np.eye(self.nu)
        Q = Q if Q is not None else np.eye(self.nx)
        A = ca.DM(model.A_lin)
        B = ca.DM(model.B_lin)
        P = solve_discrete_are(model.A_lin, model.B_lin, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

        self.P = ca.DM(P)
        self.Q = ca.DM(Q)
        self.R = ca.DM(R)
        self.K = K

        # set constraints
        self.set_lyacon = set_lyacon
        self.set_problem()

    def wrap_casadi(self, x, x_ref):
        """
        x, x_ref: Scalars (CasADi SX/MX)
        return  theta value ∈[-pi, pi]
        needs to be removed if applied to other systems
        """
        phi = x - x_ref
        return ca.atan2(ca.sin(phi), ca.cos(phi))

    def set_problem(self):
        self.opti = ca.Opti()

        # Define the optimization variables
        self.X = self.opti.variable(self.nx, self.horizon + 1)
        self.U = self.opti.variable(self.nu, self.horizon)

        # Define the parameters
        self.x0 = self.opti.parameter(self.nx)
        self.xbar = self.opti.parameter(self.nx)
        self.uL = self.opti.parameter(self.nu)
        self.J_prev = self.opti.parameter(1)
        self.penalty_term = self.opti.parameter(1)

        # Define the intial bounds and terminal bounds
        # --- MODIFIED: Relax equality constraint ---
        self.opti.subject_to(
            self.opti.bounded(-self.solver_tol, self.X[:, 0] - self.x0, self.solver_tol)
        )
        # --- END MODIFIED ---

        # Define the dynamics and constraints
        for k in range(self.horizon):
            x_next = self.model_func(self.X[:, k], self.U[:, k])
            # --- MODIFIED: Relax equality constraint ---
            self.opti.subject_to(
                self.opti.bounded(-self.solver_tol, self.X[:, k + 1] - x_next, self.solver_tol)
            )
            # --- END MODIFIED ---

            self.opti.subject_to(self.opti.bounded(self.ulb, self.U[:, k], self.uub))
            if k > 0:
                self.opti.subject_to(self.opti.bounded(self.xlb, self.X[:, k], self.xub))

        self.opti.subject_to(self.opti.bounded(self.xlb, self.X[:, self.horizon], self.xub))

        # Define the lyapunov constraint
        if self.set_lyacon:
            J_curr = 0
            for i in range(self.horizon):
                J_curr += ca.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
                # state error:
                err_vec = []
                for k in range(self.nx):
                    xk_i = self.X[k, i]
                    xref_k = self.xbar[k]
                    if self.periodic_info and k == 0:
                        # This is a periodic state，use wrap_casadi
                        di = self.wrap_casadi(xk_i, xref_k)
                        err_vec.append(di)
                    else:
                        di = xk_i - xref_k
                        err_vec.append(di)

                err_vec_i = ca.vertcat(*err_vec)
                J_curr += ca.mtimes([err_vec_i.T, self.Q, err_vec_i])

            # At k = horizon
            delta_theta_T = ca.atan2(
                ca.sin(self.X[0, self.horizon] - self.xbar[0]),
                ca.cos(self.X[0, self.horizon] - self.xbar[0])
            )
            error_omega_T = self.X[1, self.horizon] - self.xbar[1]
            err_vec_T = ca.vertcat(delta_theta_T, error_omega_T)
            # J_curr += ca.mtimes([err_vec_T.T, self.P, err_vec_T])

            # Terminal set constraint
            # --- MODIFIED: Relax equality constraint ---
            self.opti.subject_to(
                self.opti.bounded(-self.solver_tol, self.X[:, self.horizon] - self.xbar, self.solver_tol)
            )
            # --- END MODIFIED ---

            # eps_term = 1e-3  # tune
            # self.opti.subject_to(ca.mtimes([err_vec_T.T, self.P, err_vec_T]) <= eps_term)
            # with slack
            # s = self.opti.variable(1)
            # self.opti.subject_to(s >= 0)
            # self.opti.subject_to(ca.mtimes([err_vec_T.T, self.P, err_vec_T]) <= eps_term + s)
            # mu = 1e3

            if self.rho is None:
                # r = ca.norm_2(self.uL)  # ||u_L,t||
                # ratio = (r - self.epsilon) / self.epsilon
                # ratio_clipped = ca.fmax(0, ca.fmin(1, ratio))  # in [0,1]
                # rho = self.rho_bar + (self.rho_max - self.rho_bar) * ratio_clipped

                # --- replace your scheduler block with this ---
                # beta = 5.0  # modest slope; larger is fine now, but start modest

                # def softplus_stable(x):  # smooth and overflow-safe
                #    return ca.if_else(x > 0, x + ca.log(1 + ca.exp(-x)), ca.log(1 + ca.exp(x)))

                # z = (ca.norm_2(self.uL) - self.epsilon) / self.epsilon
                # pos = softplus_stable(beta * z) / beta  # >= 0, no overflow
                # sigma = 1 - ca.exp(-pos)  # in (0,1), stable even for large pos
                # rho = self.rho_bar + (self.rho_max - self.rho_bar) * sigma

                ratio = ca.sumsqr(self.uL) / (self.epsilon * self.epsilon)  # ≥ 0, smooth at 0
                sigma = ratio / (1 + ratio)  # maps [0,∞) → (0,1), C∞
                rho = self.rho_bar + (self.rho_max - self.rho_bar) * sigma
            else:
                rho = self.rho  # fixed-ρ mode
            self.opti.subject_to(J_curr - (self.J_prev - (1 - rho) * self.penalty_term) <= 0)

        # Define the cost function
        reg_u = 1e-4 * ca.sumsqr(self.U)  # tiny!
        reg_x = 1e-6 * ca.sumsqr(self.X)  # tinier!
        stage_cost = ca.sumsqr(self.U[:, 0] - self.uL)  # + reg_u + reg_x
        self.opti.minimize(stage_cost)

        # Define the solver options
        solver_opts = {
            "ipopt.print_level": 0,  # no Ipopt iteration log
            "ipopt.sb": "yes",  # silence Ipopt banner
            "print_time": False,  # no CasADi timing line
            "ipopt.max_iter": 200,  # (optional) keep solves snappy
            "ipopt.mu_strategy": "adaptive",
            "ipopt.bound_relax_factor": 1e-8,
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.tol": 1e-6,
            "ipopt.hessian_approximation": "limited-memory",  # more forgiving
            "ipopt.fixed_variable_treatment": "make_parameter"  # avoids some multipliers issues
        }
        self.opti.solver("ipopt", solver_opts)

    def solve_mpc(self, x0_val, xbar_val, uL_val, u_prev=None, x_prev=None):
        x0_val = x0_val.reshape(self.nx)
        xbar_val = xbar_val.reshape(self.nx)
        uL_val = uL_val.reshape(self.nu)

        self.opti.set_value(self.x0, x0_val)
        self.opti.set_value(self.xbar, xbar_val)
        self.opti.set_value(self.uL, uL_val)

        # PSF.py (inside solve_mpc, where u_prev/x_prev is None)
        if u_prev is None or x_prev is None:
            th0, om0 = x0_val
            thT, omT = xbar_val

            def angdiff(a, b):
                d = (b - a + 2 * np.pi) % (2 * np.pi)
                if d > np.pi: d -= 2 * np.pi
                return d

            dth = angdiff(th0, thT)
            tgrid = np.linspace(0.0, 1.0, self.horizon + 1)
            th_path = th0 + tgrid * dth
            om_path = om0 + tgrid * (omT - om0)
            X_guess = np.vstack([th_path, om_path])  # shape (2, N+1)

            self.opti.set_initial(self.X, X_guess)
            # in solve_mpc() when u_prev/x_prev is None
            self.opti.set_initial(self.U, np.tile(uL_val.reshape(-1, 1), (1, self.horizon)))

            # self.opti.set_initial(self.U, np.tile(self.uub.reshape(-1, 1), (1, self.horizon)))

            J_prev = 1e7
            penalty_term = 0
        else:
            J_prev = self.quadcost(x_prev, u_prev, xbar_val.reshape(2, ))

            th0 = x_prev[0, 0]
            th_ref = xbar_val[0]
            dth0 = np.arctan2(np.sin(th0 - th_ref), np.cos(th0 - th_ref))
            domega0 = x_prev[1, 0] - xbar_val[1]
            err0 = np.array([dth0, domega0])  # shape=(2,)
            penalty_term = err0.T @ np.array(self.Q) @ err0
            u0 = u_prev[:, 0]  # shape=(1,)
            penalty_term += u0.T @ np.array(self.R) @ u0

            x_prev = np.append(x_prev[:, 1:], xbar_val.reshape(2, 1), axis=1)
            u_prev = np.append(u_prev[:, 1:], np.array([[0]]), axis=1)

            self.opti.set_initial(self.X, np.array(x_prev))
            self.opti.set_initial(self.U, np.array(u_prev))

        self.opti.set_value(self.J_prev, J_prev)
        self.opti.set_value(self.penalty_term, penalty_term)

        try:
            sol = self.opti.solve()
            # Use Opti.value(...) after a successful solve
            U_sol = np.asarray(self.opti.value(self.U)).reshape(self.nu, self.horizon)
            X_sol = np.asarray(self.opti.value(self.X)).reshape(self.nx, self.horizon + 1)
            J_curr = self.quadcost(X_sol, U_sol, xbar_val.reshape(2, ))
            return U_sol, X_sol, J_curr
        except RuntimeError as e:
            print("Solver failed:", e)
            self.opti.debug.show_infeasibilities()
            # Safe fallback: pass-through uL, hold state
            U_fallback = np.tile(uL_val.reshape(-1, 1), (1, self.horizon))
            X_fallback = np.tile(x0_val.reshape(-1, 1), (1, self.horizon + 1))
            J_curr = self.quadcost(X_fallback, U_fallback, xbar_val.reshape(2, ))
            return U_fallback, X_fallback, J_curr

    def quadcost(self, X_full, U_full, xbar_val):
        # --- Shape guards (important) ---
        X_full = np.asarray(X_full)
        U_full = np.asarray(U_full)
        if X_full.ndim == 1:
            X_full = X_full.reshape(self.nx, self.horizon + 1)
        if U_full.ndim == 1:
            U_full = U_full.reshape(self.nu, self.horizon)
        # ---------------------------------

        Q_np = np.array(self.Q)
        R_np = np.array(self.R)
        P_np = np.array(self.P)

        cost = 0.0
        for k in range(self.horizon):
            th_k = X_full[0, k]
            th_ref = xbar_val[0]
            dth_k = np.arctan2(np.sin(th_k - th_ref), np.cos(th_k - th_ref))
            domega_k = X_full[1, k] - xbar_val[1]
            errvec_k = np.array([dth_k, domega_k])
            cost += errvec_k.T @ Q_np @ errvec_k
            u_k = U_full[:, k]  # now safe (shape (1,))
            cost += u_k.T @ R_np @ u_k

        th_T = X_full[0, self.horizon]
        th_ref = xbar_val[0]
        dth_T = np.arctan2(np.sin(th_T - th_ref), np.cos(th_T - th_ref))
        domega_T = X_full[1, self.horizon] - xbar_val[1]
        errvec_T = np.array([dth_T, domega_T])
        cost += errvec_T.T @ P_np @ errvec_T
        return cost