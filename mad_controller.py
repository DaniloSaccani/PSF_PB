import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append("../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ornstein-Uhlenbeck noise process
class OUActionNoise:
    """
    Class for implementing the Ornstein-Uhlenbeck noise process, commonly used for exploration in reinforcement learning.

    Attributes:
        mean (numpy.ndarray): The mean of the noise process.
        std_deviation (numpy.ndarray): The standard deviation of the noise process.
        theta (float): The rate of mean reversion, default is 0.15.
        dt (float): Time step used in the process, default is 1e-2.
        x_initial (numpy.ndarray or None): Initial value for the noise process, default is None (sets to zero).
        x_prev (numpy.ndarray): The previous state of the noise process.

    Methods:
        __call__(): Generates the next noise value based on the Ornstein-Uhlenbeck process.
        reset(): Resets the noise process to its initial state.
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """
        Initializes the Ornstein-Uhlenbeck noise process.

        Args:
            mean (numpy.ndarray): The mean of the noise process.
            std_deviation (numpy.ndarray): The standard deviation of the noise process.
            theta (float, optional): The rate of mean reversion, default is 0.15.
            dt (float, optional): Time step for the process, default is 1e-2.
            x_initial (numpy.ndarray or None, optional): Initial state of the noise process, default is None.
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """
        Generates the next value from the Ornstein-Uhlenbeck noise process.

        Returns:
            numpy.ndarray: The generated noise value.
        """
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """
        Resets the noise process to its initial state.

        If no initial state is provided, it is set to zero.
        """
        self.x_prev = (
            self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)
        )


# Replay buffer
class ReplayBuffer:
    """
    A replay buffer for storing and sampling experience tuples used for training reinforcement learning agents.

    --- MODIFIED ---
    - state_buffer now stores the full *augmented* state.
    - ..._buffer (initial_state, disturbance) stores *physical* state sequences.
    --- END MODIFIED ---

    Attributes:
        buffer_capacity (int): The maximum number of samples the buffer can store, default is 100000.
        batch_size (int): The number of samples to return in each batch, default is 64.
        dynamics_input_time_window (int): The time window of dynamics input to store, default is 50.
        buffer_counter (int): Counter for the number of samples added.
        num_states (int): The number of *augmented* state features.
        num_physical_states (int): The number of *physical* state features.
        num_dynamics_states (int): The number of dynamics state features.
        num_actions (int): The number of action features.
        state_buffer (numpy.ndarray): Buffer storing augmented states.
        action_buffer (numpy.ndarray): Buffer storing actions.
        reward_buffer (numpy.ndarray): Buffer storing rewards.
        next_state_buffer (numpy.ndarray): Buffer storing next augmented states.
        initial_state_buffer (numpy.ndarray): Buffer storing initial physical state error time windows (for x0).
        next_initial_state_buffer (numpy.ndarray): Buffer storing next initial physical state error time windows.
        dynamics_states_real_buffer (numpy.ndarray): Buffer storing real dynamics states.
        next_dynamics_states_real_buffer (numpy.ndarray): Buffer storing next dynamics states.
        disturbance_buffer (numpy.ndarray): Buffer storing physical disturbance information.
        next_disturbance_buffer (numpy.ndarray): Buffer storing next physical disturbance information.

    Methods:
        record(obs_tuple): Records a new experience tuple into the buffer.
        sample(): Samples a batch of experience tuples from the buffer.
    """

    def __init__(
            self,
            buffer_capacity=100000,
            batch_size=64,
            num_states=1,  # Augmented state dim (e.g., 7)
            num_physical_states=1,  # Physical state dim (e.g., 2)
            num_dynamics_states=1,
            num_actions=1,
            dynamics_input_time_window=50,
    ):
        """
        Initializes the replay buffer with the given parameters.

        Args:
            buffer_capacity (int, optional): The maximum size of the buffer, default is 100000.
            batch_size (int, optional): The number of samples per batch, default is 64.
            num_states (int, optional): Number of *augmented* state features, default is 1.
            num_physical_states (int, optional): Number of *physical* state features, default is 1.
            num_dynamics_states (int, optional): Number of dynamics state features, default is 1.
            num_actions (int, optional): Number of action features, default is 1.
            dynamics_input_time_window (int, optional): The time window for storing dynamics input, default is 50.
        """
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.dynamics_input_time_window = dynamics_input_time_window
        self.buffer_counter = 0
        self.num_states = num_states  # Augmented dim
        self.num_physical_states = num_physical_states  # Physical dim
        self.num_dynamics_states = num_dynamics_states
        self.num_actions = num_actions

        # Buffers for augmented state
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

        # Buffers for action/reward
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))

        # Buffers for SSM internal state
        self.dynamics_states_real_buffer = np.zeros(
            (self.buffer_capacity, 1 * self.num_dynamics_states)
        )
        self.next_dynamics_states_real_buffer = np.zeros(
            (self.buffer_capacity, 1 * self.num_dynamics_states)
        )

        # --- Buffers for SSM *input* (physical dim) ---
        self.initial_state_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_physical_states)
        )
        self.next_initial_state_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_physical_states)
        )
        self.disturbance_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_physical_states)
        )
        self.next_disturbance_buffer = np.zeros(
            (self.buffer_capacity, self.dynamics_input_time_window, self.num_physical_states)
        )

    def record(self, obs_tuple):
        """
        Records a new experience tuple into the buffer.

        Args:
            obs_tuple (tuple): A tuple containing state, action, reward, next_state,
                                initial_state, next_initial_state, dynamics states,
                                next dynamics states, disturbance, and next disturbance.
        """
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]  # aug_state
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]  # next_aug_state
        self.initial_state_buffer[index] = obs_tuple[4]  # x0 sequence (physical)
        self.next_initial_state_buffer[index] = obs_tuple[5]  # next x0 sequence (physical)
        self.dynamics_states_real_buffer[index] = obs_tuple[6]
        self.next_dynamics_states_real_buffer[index] = obs_tuple[7]
        self.disturbance_buffer[index] = obs_tuple[8]  # w sequence (physical)
        self.next_disturbance_buffer[index] = obs_tuple[9]  # next w sequence (physical)

        self.buffer_counter += 1

    def sample(self):
        """
        Samples a batch of experience tuples from the buffer.

        Returns:
            tuple: A tuple of tensors representing the sampled experience batch.
        """
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = torch.tensor(
            self.state_buffer[batch_indices]
        ).to(device)
        action_batch = torch.tensor(
            self.action_buffer[batch_indices]
        ).to(device)
        reward_batch = torch.tensor(
            self.reward_buffer[batch_indices]
        ).to(device)
        next_state_batch = torch.tensor(
            self.next_state_buffer[batch_indices]
        ).to(device)
        dynamics_input_time_window_batch = torch.tensor(
            self.initial_state_buffer[batch_indices]
        ).to(device)
        next_dynamics_input_time_window_batch = torch.tensor(
            self.next_initial_state_buffer[batch_indices]
        ).to(device)
        dynamics_states_real_batch = torch.tensor(
            self.dynamics_states_real_buffer[batch_indices]
        ).to(device)
        next_dynamics_states_real_batch = torch.tensor(
            self.next_dynamics_states_real_buffer[batch_indices]
        ).to(device)
        dynamics_disturbance_time_window_batch = torch.tensor(
            self.disturbance_buffer[batch_indices]
        ).to(device)
        next_dynamics_disturbance_time_window_batch = torch.tensor(
            self.next_disturbance_buffer[batch_indices]
        ).to(device)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            dynamics_input_time_window_batch,
            next_dynamics_input_time_window_batch,
            dynamics_states_real_batch,
            next_dynamics_states_real_batch,
            dynamics_disturbance_time_window_batch,
            next_dynamics_disturbance_time_window_batch,
        )


# Actor Network
class Actor(nn.Module):
    """
    The Actor network that takes the state and predicts the control action.

    --- MODIFIED ---
    - `num_states` is the *augmented* state dim (e.g., 7) for the direction_mlp.
    - `num_physical_states` is the *physical* state dim (e.g., 2) for the m_dynamics (SSM) input.
    - `forward` now correctly uses `dynamics_input_time_window` for the SSM.
    --- END MODIFIED ---

    Attributes:
        control_action_upper_bound (float): The upper bound for the control action.
        direction_mlp (torch.nn.Sequential): MLP used to predict the direction term of the action.
        m_dynamics (SSM): A state-space model used to handle dynamics.
        num_states (int): Number of input augmented states.
        m_term (torch.Tensor or None): The model term of the action.
        a_term (torch.Tensor or None): The action term.
        d_term (torch.Tensor or None): The disturbance term.

    Methods:
        _initialize_weights(): Initializes the weights and biases of the network.
        forward(state, dynamics_input_time_window, dynamics_disturbance_time_window):
            Forward pass to compute the action.
    """

    def __init__(
            self,
            num_states,  # Augmented state dim
            num_physical_states,  # Physical state dim
            num_actions,
            control_action_upper_bound,
            num_dynamics_states=3,
            hidden_dim=10,
            nn_type='mad'
    ):
        """
        Initializes the Actor network.

        Args:
            num_states (int): Number of input *augmented* states (for direction).
            num_physical_states (int): Number of input *physical* states (for magnitude).
            num_actions (int): Number of output actions.
            control_action_upper_bound (float): The upper bound for the control action.
            num_dynamics_states (int, optional): Number of dynamics states, default is 2.
            hidden_dim (int, optional): The number of hidden units in the MLP, default is 16.
        """
        super(Actor, self).__init__()
        self.control_action_upper_bound = control_action_upper_bound
        self.nn_type = nn_type
        self.dir_gain = nn.Parameter(torch.ones(1, num_actions))

        # --- MODIFIED: input is augmented state ---
        self.direction_mlp = nn.Sequential(
            nn.Linear(num_states, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions, bias=False),
        )

        # LSTM for sequential processing of state
        if nn_type == 'lstm':
            lstm_hidden_dim = 48
            self.lstm = nn.LSTM(
                input_size=num_states,  # Augmented state
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True,
            )

            self.lstm_to_action = nn.Linear(lstm_hidden_dim, num_actions)

        self._initialize_weights()

        # --- MODIFIED: input is physical state ---
        self.m_dynamics = SSM(
            in_features=num_physical_states,  # Input is x0 sequence
            out_features=num_actions,
            state_features=num_dynamics_states,
            scan=True,
        )

        self.num_states = num_states
        self.m_term = None
        self.a_term = None
        self.d_term = None

    def _initialize_weights(self):
        """
        Initializes weights to small random values and sets biases to zero (non-trainable).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    m.bias.requires_grad = False

        # Explicit LSTM init
        if hasattr(self, "lstm") and isinstance(self.lstm, nn.LSTM):
            for name, param in self.lstm.named_parameters():
                if "weight_hh" in name:  # recurrent weights
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:  # input weights
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget-gate bias to +1: gates are [i, f, g, o]
                    hidden = self.lstm.hidden_size
                    param.data[hidden:2 * hidden] = 1.0

    def forward(
            self, state, dynamics_input_time_window, dynamics_disturbance_time_window
    ):
        """
        Forward pass to compute the action based on the state and dynamics input.

        Args:
            state (torch.Tensor): The current *augmented* state.
            dynamics_input_time_window (torch.Tensor): The initial state time window (x0 sequence).
            dynamics_disturbance_time_window (torch.Tensor): The disturbance time window (w sequence).

        Returns:
            torch.Tensor: The computed action.
        """
        if self.nn_type == 'mad':
            # Direction term uses the full augmented state
            direction_term = self.direction_mlp(state)

            # --- MODIFIED: Magnitude term (SSM) uses the x0 sequence ---
            if dynamics_input_time_window.dim() == 2:
                dynamics_input = dynamics_input_time_window.unsqueeze(0)
                dynamics_input_output = self.m_dynamics(dynamics_input)
                m_term = dynamics_input_output.squeeze(0)[-1, :]
                d_term = torch.tanh(direction_term) * self.control_action_upper_bound
                d_term = self.dir_gain * d_term  # <-- NEW: scale direction by learnable gain
                action = (m_term) * d_term
                if action.dim() == 1:
                    action = action.unsqueeze(0)

            elif dynamics_input_time_window.dim() == 3:
                dynamics_input = dynamics_input_time_window
                dynamics_input_output = self.m_dynamics(dynamics_input)
                m_term = dynamics_input_output[:, -1, :]
                d_term = torch.tanh(direction_term) * self.control_action_upper_bound
                action = (m_term) * d_term
                if action.dim() == 1:
                    action = action.unsqueeze(0)
            # --- END MODIFIED ---

            return action

        elif self.nn_type == 'lstm':
            # accept (B, S) or (B, T, S)
            if state.dim() == 2:
                state_seq = state.unsqueeze(1)  # (B, 1, S)
            else:
                state_seq = state  # (B, T, S)
            output, (h_n, c_n) = self.lstm(state_seq)  # (B, T, H)
            last = output[:, -1, :]  # (B, H)
            action = self.lstm_to_action(last)  # (B, A)
            return action
        else:
            print(f'Unknown nn_type {self.nn_type}')


# Critic network
class Critic(nn.Module):
    """
    The Critic network used to estimate the value function.

    --- MODIFIED ---
    - `num_states` is the *augmented* state dim (e.g., 7).
    --- END MODIFIED ---

    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        fc3 (torch.nn.Linear): Third fully connected layer.
        fc4 (torch.nn.Linear): Fourth fully connected layer.
        fc5 (torch.nn.Linear): Fifth fully connected layer.
        fc6 (torch.nn.Linear): Final output layer.

    Methods:
        forward(state, dynamics_states_real, action): Forward pass to compute the value estimate.
    """

    def __init__(self, num_states, num_dynamics_states, num_actions):
        """
        Initializes the Critic network.

        Args:
            num_states (int): Number of input *augmented* states.
            num_dynamics_states (int): Number of dynamics states.
            num_actions (int): Number of actions.
        """
        super(Critic, self).__init__()
        # --- MODIFIED: Input layer uses augmented state size ---
        self.fc1 = nn.Linear(num_states + 1 * num_dynamics_states, 16)
        # --- END MODIFIED ---
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(num_actions, 32)
        self.fc4 = nn.Linear(32 * 2, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, state, dynamics_states_real, action):
        """
        Forward pass to compute the value estimate.

        Args:
            state (torch.Tensor): The *augmented* state tensor.
            dynamics_states_real (torch.Tensor): The real dynamics states tensor.
            action (torch.Tensor): The action tensor.

        Returns:
            torch.Tensor: The computed value estimate.
        """
        state_augmented = torch.cat([state, 2 * dynamics_states_real], dim=-1)
        state_out = F.tanh(self.fc1(state_augmented))
        state_out = F.tanh(self.fc2(state_out))
        action_out = F.tanh(self.fc3(action))
        concat = torch.cat([state_out, action_out], dim=-1)
        x = F.relu(self.fc4(concat))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


# DDPG agent class
class MADController:
    """
    MADController class...

    --- MODIFIED ---
    - Renamed `self.aug_state` to `self.state` (it holds the augmented state).
    - Added `self.num_physical_states`.
    - Corrected `__init__` to slice `self.state` for physical state error calculation.
    - Corrected `get_trajectory` to manually build the initial augmented state.
    - Corrected `set_ep_initial_state` call
    - **Corrected `if` condition in `train` loop**
    --- END MODIFIED ---

    Attributes:
        env (gym.Env): The environment with which the agent interacts.
        num_states (int): The number of *augmented* states in the environment.
        num_physical_states (int): The number of *physical* states in the environment.
        num_actions (int): The number of actions in the environment.
        state (torch.Tensor): Current *augmented* state of the agent.
        target_state (torch.Tensor): Target *physical* state the agent is aiming for.
        state_error (torch.Tensor): Difference between current *physical* state and target state.
        w (torch.Tensor): *Physical* state-related dynamics information (disturbance).
        ...
        ep_initial_aug_state (torch.Tensor): Initial *augmented* state for the episode.
        ep_initial_state_error (torch.Tensor): Initial *physical* state error for the episode.
        ...
    """

    def __init__(
            self,
            env,
            buffer_capacity=100000,
            target_state=None,
            num_dynamics_states=8,
            dynamics_input_time_window_length=500,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            actor_lr=0.0005,  # changed from 0.001 to 0.0005
            critic_lr=0.002,
            std_dev=0.2,
            control_action_upper_bound=1,
            control_action_lower_bound=-1,
            nn_type='mad'
    ):
        """
        Initializes the MADController agent for training in a given environment.
        """

        self.env = env
        # --- MODIFIED: Distinguish physical and augmented state dims ---
        self.num_states = env.observation_space.shape[0]  # Augmented state dim
        self.num_physical_states = env.sys.state_dim  # Physical state dim
        self.num_actions = env.action_space.shape[0]

        self.state = torch.zeros(self.num_states).to(device)  # Current augmented state
        if target_state is None:
            self.target_state = torch.zeros(self.num_physical_states).to(device)
        else:
            self.target_state = target_state.to(device)  # Target is physical

        # State error is physical, computed from slicing the augmented state
        self.state_error = self.state[:self.num_physical_states] - self.target_state
        self.w = torch.zeros(self.num_physical_states).to(device)  # Disturbance is physical

        self.num_dynamics_states = num_dynamics_states
        self.dynamics_states_real = torch.zeros(1 * self.num_dynamics_states).to(device)
        self.dynamics_input_time_window_length = dynamics_input_time_window_length

        self.dynamics_input_time_window = torch.zeros(
            (self.dynamics_input_time_window_length, self.num_physical_states)
        ).to(device)
        self.dynamics_disturbance_time_window = torch.zeros(
            (self.dynamics_input_time_window_length, self.num_physical_states)
        ).to(device)
        self.ep_initial_aug_state = torch.zeros(self.num_states).to(device)
        self.ep_initial_state_error = torch.zeros(self.num_physical_states).to(device)  # Physical
        self.ep_timestep = torch.ones(1).to(device)

        self.actor_model = Actor(
            num_states=self.num_states,
            num_physical_states=self.num_physical_states,
            num_actions=self.num_actions,
            control_action_upper_bound=control_action_upper_bound,
            num_dynamics_states=self.num_dynamics_states,
            nn_type=nn_type
        ).to(device)
        self.target_actor = Actor(
            num_states=self.num_states,
            num_physical_states=self.num_physical_states,
            num_actions=self.num_actions,
            control_action_upper_bound=control_action_upper_bound,
            num_dynamics_states=self.num_dynamics_states,
            nn_type=nn_type
        ).to(device)

        self.critic_model = Critic(
            self.num_states, self.num_dynamics_states, self.num_actions
        ).to(device)
        self.target_critic = Critic(
            self.num_states, self.num_dynamics_states, self.num_actions
        ).to(device)

        self.target_actor.load_state_dict(self.actor_model.state_dict())
        self.target_critic.load_state_dict(self.critic_model.state_dict())

        self.critic_optimizer = optim.AdamW(
            self.critic_model.parameters(), lr=critic_lr
        )
        self.actor_optimizer = optim.AdamW(self.actor_model.parameters(), lr=actor_lr)

        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.num_actions),
            std_deviation=float(std_dev) * np.ones(self.num_actions),
        )

        self.buffer = ReplayBuffer(
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            num_states=self.num_states,
            num_physical_states=self.num_physical_states,
            num_dynamics_states=self.num_dynamics_states,
            num_actions=self.num_actions,
            dynamics_input_time_window=self.dynamics_input_time_window_length,
        )

        self.gamma = gamma
        self.tau = tau
        self.control_action_upper_bound = control_action_upper_bound
        self.control_action_lower_bound = control_action_lower_bound

        self.episode_count = 0
        self.rewards_list = []
        self.rewards_state_list = []
        self.rewards_control_diff_list = []
        self.rewards_obs_list = []
        self.rewards_ma50_list = []
        self.obs_min_dis_list = []

        self.rewards_state_normalized_list = []
        self.rewards_control_normalized_list = []
        self.rewards_obs_normalized_list = []

        self.actor_loss_list = []
        self.critic_loss_list = []

    def set_ep_initial_state(self, initial_aug_state):
        """
        Sets the initial state of the episode relative to the target state.

        --- MODIFIED ---
        - Takes `initial_aug_state` (size 7).
        - Computes and stores `ep_initial_state_error` (size 2).
        --- END MODIFIED ---

        Args:
            initial_aug_state (array-like): The initial *augmented* state for the episode.
        """
        self.ep_initial_aug_state = initial_aug_state.to(
            device
        )
        # Error is based on physical part of the state
        ep_initial_physical_state = initial_aug_state[:self.num_physical_states]
        self.ep_initial_state_error = ep_initial_physical_state - self.target_state

    def update_dynamics_input_time_window(self):
        """
        Updates the dynamics input time window with the current state.

        --- MODIFIED ---
        - Clamped update_index to 1 to prevent OOD all-zero input
          (or index=0) which was not seen during training.
        --- END MODIFIED ---
        """
        self.dynamics_input_time_window *= 0.0

        current_timestep = self.ep_timestep.cpu().item()
        length = self.dynamics_input_time_window_length  # e.g., 251

        update_index = int(length - current_timestep)  # Goes from 250 down to 1, then 0, -1, ...



        if update_index >= 0:
            # This will fire for t=1..251 (indices 250..0)
            self.dynamics_input_time_window[update_index] = self.ep_initial_state_error
        # For t > 251, update_index is < 0, and the window remains all-zeros.

        if self.ep_timestep == 1:
            # Disturbance sequence
            self.dynamics_disturbance_time_window[-1] = self.ep_initial_state_error
        if self.ep_timestep >= self.dynamics_input_time_window_length:
            self.dynamics_disturbance_time_window = (
                self.dynamics_disturbance_time_window
            )
        else:
            temp = torch.roll(self.dynamics_disturbance_time_window, shifts=-1, dims=0)
            temp[-1] = self.w  # self.w is physical (size 2)
            self.dynamics_disturbance_time_window = temp

    def reset_ep_timestep(self):
        """
        Resets the episode timestep and clears the dynamics disturbance time window.
        """
        self.ep_timestep = torch.ones(1).to(device)
        # State error is physical
        self.state_error = self.state[:self.num_physical_states] - self.target_state
        self.dynamics_disturbance_time_window *= 0.0

    def update_ep_timestep(self):
        """
        Increments the episode timestep and updates the state error.
        """
        self.ep_timestep += 1
        # State error is physical
        self.state_error = self.state[:self.num_physical_states] - self.target_state

    def policy(
            self, state, dynamics_input_time_window, dynamics_disturbance_time_window
    ):
        """
        Selects an action based on the current state and dynamics information, adding noise for exploration.

        Args:
            state (array-like): The current *augmented* state of the agent.
            dynamics_input_time_window (array-like): The current dynamics input time window (x0).
            dynamics_disturbance_time_window (array-like): The current dynamics disturbance time window (w).

        Returns:
            np.ndarray: The selected action within the valid control bounds.
        """
        state = state.to(device)  # state is augmented
        sampled_actions = (
            self.actor_model(
                state, dynamics_input_time_window, dynamics_disturbance_time_window
            ).cpu().detach().numpy()
        )
        # ensure 1-D shape (num_actions,) even if num_actions==1
        sampled_actions = np.atleast_1d(sampled_actions).astype(float)

        noise = self.ou_noise()
        # make sure shapes match
        noise = np.asarray(noise, dtype=float).reshape(sampled_actions.shape)

        sampled_actions += noise
        return sampled_actions

    def learned_policy(
            self, state, dynamics_input_time_window, dynamics_disturbance_time_window
    ):
        """
        Selects an action using the learned policy (actor model).


        Args:
            state (array-like): The current *augmented* state of the agent.
            dynamics_input_time_window (array-like): The current dynamics input time window (x0).
            dynamics_disturbance_time_window (array-like): The current dynamics disturbance time window (w).

        Returns:
            np.ndarray: The selected action within the valid control bounds.
        """
        state = state.to(device)  # state is augmented
        sampled_actions = self.actor_model(
            state, dynamics_input_time_window, dynamics_disturbance_time_window
        )

        return sampled_actions.cpu().detach().numpy()

    def update_target(self):
        """
        Soft updates the target actor and target critic networks based on the current actor and critic networks using the tau parameter.
        """
        for target_param, param in zip(
                self.target_actor.parameters(), self.actor_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
                self.target_critic.parameters(), self.critic_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(self, total_episodes=100, episode_length=100, logger=None):
        """
        Trains the agent over multiple episodes using the DDPG algorithm.

        --- modified ---
        - Uses `self.state` to store the augmented observation from the env.
        - Passes `self.state` to the policy.
        - Records `state` and `next_state` (augmented) in the buffer.
        - Corrected call to `set_ep_initial_state`
        - **Corrected `if` condition in `train` loop**
        --- END modified ---
        """

        for ep in range(total_episodes):
            self.episode_count += 1
            self.ou_noise.reset()
            # --- Reset returns augmented state ---
            self.state = self.env.reset().to(device)  # self.state now holds aug_state
            self.last_episode_ic = self.env.state.squeeze(0).squeeze(0).detach().cpu().clone()  # Store physical IC

            self.dynamics_states_real = torch.cat(
                [
                    self.actor_model.m_dynamics.LRUR.states_last.real.squeeze(),
                ]
            )
            self.set_ep_initial_state(initial_aug_state=self.state)  # Pass aug_state
            self.reset_ep_timestep()
            self.update_dynamics_input_time_window()
            episodic_reward = 0
            episodic_state_reward = 0
            episodic_control_reward = 0
            episodic_obs_reward = 0

            while True:
                # --- Policy takes augmented state ---
                action = self.policy(
                    state=self.state,
                    dynamics_input_time_window=self.dynamics_input_time_window,
                    dynamics_disturbance_time_window=self.dynamics_disturbance_time_window,
                )
                old_state = self.state.clone()  # old_aug_state

                old_dynamics_states_real = self.dynamics_states_real.clone()
                old_dynamics_input_time_window = self.dynamics_input_time_window.clone()
                old_dynamics_disturbance_time_window = (
                    self.dynamics_disturbance_time_window.clone()
                )

                if self.env.t == 0:  # Check *before* step. t is 0 on first step.
                    next_state, reward, done, truncated, _, U_prev, X_prev = self.env.step(action)
                else:
                    next_state, reward, done, truncated, _, U_prev, X_prev = self.env.step(action, U_prev, X_prev)

                self.state = next_state.to(device)  # self.state now holds next_aug_state
                self.w = self.env.w  # w is physical

                self.dynamics_states_real = torch.cat(
                    [
                        self.actor_model.m_dynamics.LRUR.states_last.real.squeeze(),
                    ]
                )
                self.update_ep_timestep()  # Computes physical state_error
                self.update_dynamics_input_time_window()  # Uses physical error/w

                # --- Record augmented states ---
                obs_tuple = (
                    old_state.cpu(),
                    action,
                    reward,
                    self.state.cpu(),  # next_aug_state
                    old_dynamics_input_time_window.cpu(),  # physical
                    self.dynamics_input_time_window.cpu(),  # physical
                    old_dynamics_states_real.cpu().detach(),
                    self.dynamics_states_real.cpu().detach(),
                    old_dynamics_disturbance_time_window.cpu(),  # physical
                    self.dynamics_disturbance_time_window.cpu(),  # physical
                )

                self.buffer.record(obs_tuple=obs_tuple)
                episodic_reward += reward
                episodic_state_reward += self.env.step_reward_state_error
                episodic_control_reward += self.env.step_reward_control_effort
                episodic_obs_reward += self.env.step_reward_obstacle_avoidance

                if self.buffer.buffer_counter > self.buffer.batch_size:
                    (
                        state_batch,
                        action_batch,
                        reward_batch,
                        next_state_batch,
                        dynamics_input_time_window_batch,
                        next_dynamics_input_time_window_batch,
                        dynamics_states_real_batch,
                        next_dynamics_states_real_batch,
                        dynamics_disturbance_time_window_batch,
                        next_dynamics_disturbance_time_window_batch,
                    ) = self.buffer.sample()

                    # state_batch and next_state_batch are augmented states
                    with torch.no_grad():
                        target_actions = self.target_actor(
                            state=next_state_batch,
                            dynamics_input_time_window=next_dynamics_input_time_window_batch,
                            dynamics_disturbance_time_window=next_dynamics_disturbance_time_window_batch,
                        )
                        y = reward_batch + self.gamma * self.target_critic(
                            state=next_state_batch,
                            dynamics_states_real=next_dynamics_states_real_batch,
                            action=target_actions,
                        )

                    self.critic_optimizer.zero_grad()
                    critic_value = self.critic_model(
                        state=state_batch,
                        dynamics_states_real=dynamics_states_real_batch,
                        action=action_batch,
                    )
                    critic_loss = F.mse_loss(critic_value, y)
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    self.actor_optimizer.zero_grad()
                    actions = self.actor_model(
                        state=state_batch,
                        dynamics_input_time_window=dynamics_input_time_window_batch,
                        dynamics_disturbance_time_window=dynamics_disturbance_time_window_batch,
                    )
                    actor_loss = -self.critic_model(
                        state=state_batch,
                        dynamics_states_real=dynamics_states_real_batch,
                        action=actions,
                    ).mean()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.update_target()

                    self.actor_loss_list.append(actor_loss.detach().clone())
                    self.critic_loss_list.append(critic_loss.detach().clone())

                if done or truncated or (self.ep_timestep >= episode_length):
                    break

            self.rewards_list.append(episodic_reward)
            self.rewards_state_list.append(episodic_state_reward)
            self.rewards_control_diff_list.append(episodic_control_reward)
            self.rewards_obs_list.append(episodic_obs_reward)
            self.rewards_state_normalized_list.append(episodic_state_reward / self.env.alpha_state)
            self.rewards_control_normalized_list.append(episodic_control_reward / self.env.alpha_control)
            self.rewards_obs_normalized_list.append(episodic_obs_reward / self.env.alpha_obst)

            ma50_rerward = np.mean(self.rewards_list[-50:])
            self.rewards_ma50_list.append(ma50_rerward)
            if logger is not None:
                logger.info(
                    f"Episode {self.episode_count} | "
                    f"Reward_state: {episodic_state_reward:.6f} | "
                    f"Reward_control: {episodic_control_reward:.6f} | "
                    f"Reward_obs: {episodic_obs_reward:.6f} | "
                    f"Total Reward: {episodic_reward:.6f} | "
                    f"Min_dis_obs: {self.env.min_dis.item():.6f} | "
                )

    def get_trajectory(self, initial_state, timesteps=300):
        """
        Generates a trajectory by interacting with the environment from a given initial state.

        --- modified ---
        - `initial_state` is the *physical* state (size 2).
        - Manually resets env and calls `_get_augmented_state` to get the first `self.state`.
        - Passes `self.state` (augmented) to the `learned_policy`.
        - `obs_list` correctly tracks the *physical* state for plotting.
        - Corrected call to `set_ep_initial_state`
        --- END modified ---

        Args:
            initial_state (array-like): The starting *physical* state of the trajectory (size 2).
            timesteps (int, optional): The number of timesteps to simulate. Defaults to 300.

        Returns:
            tuple: A tuple containing:
                - rewards_list: List of rewards received during the trajectory.
                - obs_list: List of observed *physical* states during the trajectory.
                - action_list: List of actions taken during the trajectory.
                - w_list: List of environment states (or any relevant data for the trajectory).
        """

        # --- Manual reset using physical state ---
        _ = self.env.reset()  # Reset to get valid obstacle info
        # Set physical state in env
        self.env.state = initial_state.view(1, 1, -1).to(self.env.state_limit_low.device)
        self.env.t = 0
        self.env.min_dis = torch.tensor(float('inf'), dtype=self.env.state.dtype, device=self.env.state.device)
        self.env.converged_counter = 0

        # Get the initial augmented state from the env
        aug_state = self.env._get_augmented_state().to(device)
        self.state = aug_state  # Set controller's state (self.state is augmented)

        self.set_ep_initial_state(initial_aug_state=self.state)  # Pass augmented state
        self.reset_ep_timestep()
        self.update_dynamics_input_time_window()

        # obs_list tracks PHYSICAL state for plotting
        obs_list = [self.env.state.squeeze(0).squeeze(0).detach().cpu()]

        w_list = []
        rewards_list = []
        u_L_log = []
        u_log = []
        for _ in range(timesteps):
            # --- Policy takes augmented state ---
            action = self.learned_policy(
                self.state,  # self.state is augmented
                self.dynamics_input_time_window,
                self.dynamics_disturbance_time_window,
            )

            if self.env.t == 0:
                obs, reward, done, truncated, info, U_prev, X_prev = self.env.step(action)
            else:
                obs, reward, done, truncated, info, U_prev, X_prev = self.env.step(action, U_prev, X_prev)

            # obs is the next_aug_state
            self.w = self.env.w

            # --- MODIFIED: Store physical state for plotting ---
            obs_list.append(self.env.state.squeeze(0).squeeze(0).detach().cpu())

            w_list.append(self.env.w)
            u_L_log.append(action)
            u_log.append(U_prev[:, 0])

            rewards_list.append(reward)

            # --- MODIFIED: Update controller's augmented state ---
            self.state = obs.to(device)  # self.state is augmented

            self.update_ep_timestep()
            self.update_dynamics_input_time_window()

        # obs_arr is (1, T+1, physical_n)
        obs_arr = np.stack([
            (o.detach().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)).reshape(-1)
            for o in obs_list
        ], axis=0)[None, ...]  # shape (1, T+1, n)

        # u_L_log: list of actions; stack to (T, m) then add batch
        uL_arr = np.stack([
            (a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)).reshape(-1)
            for a in u_L_log
        ], axis=0)[None, ...]  # shape (1, T, m)

        # u_log: you stored U_prev[:,0] (numpy 1D); stack to (T, m) then add batch
        u_arr = np.stack(u_log, axis=0)[None, ...]  # shape (1, T, m)

        # w_list: list of tensors (physical_n,) -> stack to (T, physical_n)
        w_arr = np.stack([
            (w.detach().cpu().numpy() if torch.is_tensor(w) else np.asarray(w)).reshape(-1)
            for w in w_list
        ], axis=0)  # shape (T, n)

        return rewards_list, obs_arr, uL_arr, w_arr, u_arr
        # return rewards_list, np.array(obs_list).transpose(1, 0, 2), np.expand_dims(u_L_log, axis=0), w_list, np.expand_dims(u_log, axis=0)

    def save_model_weights(self, filename):
        """
        Saves the weights of the model (actor, critic, and optimizers) to a file.

        Args:
            filename (str): The path to save the model weights.
        """
        state_dict_list = [
            self.actor_model.state_dict(),
            self.critic_model.state_dict(),
            self.target_actor.state_dict(),
            self.target_critic.state_dict(),
            self.actor_optimizer.state_dict(),
            self.critic_optimizer.state_dict(),
            self.actor_model.m_dynamics.state_dict(),
            self.target_actor.m_dynamics.state_dict(),
        ]
        torch.save(state_dict_list, filename)
        print(f"DDPG Model weights saved at {filename}.")

    def load_model_weight(self, filename):
        """
        Loads the weights of the model (actor, critic, and optimizers) from a file.
        Backward-compatible: new params (e.g., dir_gain) are initialized if missing.
        """
        ckpt = torch.load(filename)

        # 0: actor, 1: critic, 2: target_actor, 3: target_critic,
        # 4: actor_opt, 5: critic_opt, 6: actor.m_dynamics, 7: target_actor.m_dynamics

        # --- MODIFIED: Use strict=False to handle potential shape mismatches ---
        # This will *fail* if you try to load an *old* checkpoint with the *new*
        # architecture, which is expected.
        self.actor_model.load_state_dict(ckpt[0], strict=False)
        self.critic_model.load_state_dict(ckpt[1], strict=False)
        self.target_actor.load_state_dict(ckpt[2], strict=False)
        self.target_critic.load_state_dict(ckpt[3], strict=False)

        # SSM submodules
        if len(ckpt) > 6:
            self.actor_model.m_dynamics.load_state_dict(ckpt[6], strict=False)
        if len(ckpt) > 7:
            self.target_actor.m_dynamics.load_state_dict(ckpt[7], strict=False)

        # Optimizers: best-effort load; if it fails (old runs), re-init silently
        try:
            self.actor_optimizer.load_state_dict(ckpt[4])
        except Exception as e:
            print(f"[load_model_weight] actor optimizer mismatch -> reinit ({e})")

        try:
            self.critic_optimizer.load_state_dict(ckpt[5])
        except Exception as e:
            print(f"[load_model_weight] critic optimizer mismatch -> reinit ({e})")

        print(f"DDPG Model weights loaded from {filename} successfully.")


# Implements of an SSM.
# we have definition of MLP (it should be substituted with the Lipschitz bounded network of Manchester, you can just download it)
# Then we have parallel scan (it should allow a fast implementation)
# LRU are nothing else a linear system parametrized to be stable reference paper: https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf
# SSM are just a combination of LRU + MLP. There is also an additional component that feedforward the input to the output (a linear layer), feel free to use or remove it

import math
import torch
import torch.nn as nn
from sympy import false


class MLP(
    nn.Module
):  # Simple MLP layer used in the SSM scaffolding later on, can be modified
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the model using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=false),  # First layer
            nn.SiLU(),  # Activation after the first layer
            nn.Linear(hidden_size, hidden_size, bias=false),  # Hidden layer
            nn.ReLU(),  # Activation after hidden layer
            nn.Linear(
                hidden_size, output_size, bias=false
            ),  # Output layer (no activation)
        )

    def forward(self, x):
        if x.dim() == 3:
            # x is of shape (batch_size, sequence_length, input_size)
            batch_size, seq_length, input_size = x.size()

            # Flatten the batch and sequence dimensions for the MLP
            x = x.reshape(-1, input_size)  # Use reshape instead of view

            # Apply the MLP to each feature vector
            x = self.model(x)  # Shape: (batch_size * sequence_length, output_size)

            # Reshape back to (batch_size, sequence_length, output_size)
            output_size = x.size(-1)
            x = x.reshape(
                batch_size, seq_length, output_size
            )  # Use reshape instead of view
        else:
            # If x is not 3D, just apply the MLP directly
            x = self.model(x)
        return x


class PScan(torch.autograd.Function):  # Parallel Scan Algorithm
    # Given A is NxTx1 and X is NxTxD, expands A and X in place in O(T),
    # and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))
            A[:, -1].mul_(A[:, -2])

    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        # ppprint(grad_output)
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)


pscan = PScan.apply


class LRU(
    nn.Module
):  # Implements a Linear Recurrent Unit (LRU) following the parametrization of
    # the paper " Resurrecting Linear Recurrences ".
    # The LRU is simulated using Parallel Scan (fast!) when "scan" is set to True (default), otherwise recursively (slow).
    def __init__(
            self,
            in_features,
            out_features,
            state_features,
            scan=True,
            rmin=0.96,
            rmax=0.976,
            max_phase=0.0  # 6.283,
    ):
        super().__init__()
        self.state_features = state_features
        self.in_features = in_features
        self.scan = scan
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        # self.theta_log = torch.log(max_phase * u2).to(self.nu_log.device)
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.register_buffer(
            "state",
            torch.complex(
                torch.zeros(state_features),  # + 0.1 * torch.rand(state_features),
                torch.zeros(state_features),  # + 0.1 * torch.rand(state_features),
            ),
        )
        self.states_last = self.state

    def forward(self, input):
        self.state = self.state
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im)  # Eigenvalues matrix
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )
        # Input must be (Batches,Seq_length, Input size), otherwise adds dummy dimension = 1 for batches
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if self.scan:  # Simulate the LRU with Parallel Scan
            input = input.permute(2, 1, 0)  # (Input size,Seq_length, Batches)
            # Unsqueeze b to make its shape (N, V, 1, 1)
            B_unsqueezed = self.B.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast b along dimensions T and D so it can be multiplied elementwise with u
            B_broadcasted = B_unsqueezed.expand(
                self.state_features, self.in_features, input.shape[1], input.shape[2]
            )
            # Expand u so that it can be multiplied along dimension N, resulting in shape (N, V, T, D)
            input_broadcasted = input.unsqueeze(0).expand(
                self.state_features, self.in_features, input.shape[1], input.shape[2]
            )
            # Elementwise multiplication and then sum over V (the second dimension)
            inputBU = torch.sum(
                B_broadcasted * input_broadcasted, dim=1
            )  # (State size,Seq_length, Batches)

            # Prepare matrix Lambda for scan
            Lambda = Lambda.unsqueeze(1)
            A = torch.tile(Lambda, (1, inputBU.shape[1]))
            # Initial condition
            init = torch.complex(
                torch.zeros(
                    (self.state_features, inputBU.shape[2]), device=self.B.device
                ),
                torch.zeros(
                    (self.state_features, inputBU.shape[2]), device=self.B.device
                ),
            )

            gammas_reshaped = gammas.unsqueeze(2)  # Shape becomes (State size, 1, 1)
            # Element-wise multiplication
            GBU = gammas_reshaped * inputBU

            states = pscan(A, GBU, init)  # dimensions: (State size,Seq_length, Batches)
            if states.shape[-1] == 1:
                self.states_last = states.clone().permute(2, 1, 0)[:, -1, :]

            # Prepare output matrices C and D for sequence and batch handling
            # Unsqueeze C to make its shape (Y, X, 1, 1)
            C_unsqueezed = self.C.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            C_broadcasted = C_unsqueezed.expand(
                self.out_features,
                self.state_features,
                inputBU.shape[1],
                inputBU.shape[2],
            )
            # Elementwise multiplication and then sum over V (the second dimension)
            CX = torch.sum(C_broadcasted * states, dim=1)

            # Unsqueeze D to make its shape (Y, U, 1, 1)
            D_unsqueezed = self.D.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            D_broadcasted = D_unsqueezed.expand(
                self.out_features, self.in_features, input.shape[1], input.shape[2]
            )
            # Elementwise multiplication and then sum over V (the second dimension)
            DU = torch.sum(D_broadcasted * input, dim=1)

            output = 2 * CX.real + DU
            output = output.permute(
                2, 1, 0
            )  # Back to (Batches, Seq length, Input size)
        else:  # Simulate the LRU recursively
            for i, batch in enumerate(input):
                out_seq = torch.empty(input.shape[1], self.out_features)
                for j, step in enumerate(batch):
                    self.state = Lambda * self.state + gammas * self.B @ step.to(
                        dtype=self.B.dtype
                    )
                    out_step = (self.C @ self.state).real + self.D @ step
                    out_seq[j] = out_step
                self.state = torch.complex(
                    torch.zeros_like(self.state.real), torch.zeros_like(self.state.real)
                )
                output[i] = out_seq
        return output  # Shape (Batches,Seq_length, Input size)


class SSM(
    nn.Module
):  # Implements LRU + a user-defined scaffolding, this is our SSM block.
    # Scaffolding can be modified. In this case we have LRU, MLP plus linear skip connection.
    def __init__(
            self,
            in_features,
            out_features,
            state_features,
            scan,
            mlp_hidden_size=15,
            rmin=0.85,
            rmax=0.9,
            max_phase=6.283,
    ):
        super().__init__()
        self.mlp = MLP(out_features, mlp_hidden_size, out_features)
        self.LRUR = LRU(
            in_features, out_features, state_features, scan, rmin, rmax, max_phase
        )
        self.model = nn.Sequential(self.LRUR, self.mlp)
        self.lin = nn.Linear(in_features, out_features, bias=false)

    def set_paramS(self):
        self.LRUR.set_param()

    def forward(self, input):
        result = self.model(input) + self.lin(input)
        return result


class DeepLRU(
    nn.Module
):  # Implements a cascade of N SSMs. Linear pre- and post-processing can be modified
    def __init__(
            self, N, in_features, out_features, mid_features, state_features, scan=True
    ):
        super().__init__()
        self.linin = nn.Linear(in_features, mid_features, bias=false)
        self.linout = nn.Linear(mid_features, out_features, bias=false)
        self.modelt = nn.ModuleList(
            [SSM(mid_features, mid_features, state_features, scan) for j in range(N)]
        )
        self.modelt.insert(0, self.linin)
        self.modelt.append(self.linout)
        self.model = nn.Sequential(*self.modelt)
        self.set_param()

    def set_param(self):
        # Apply the 'custom_method' to all elements except the first and last
        for i in range(1, len(self.modelt) - 1):
            if isinstance(
                    self.modelt[i], SSM
            ):  # Check if it's an instance of CustomModule
                self.modelt[i].set_paramS()  # Call the custom method

    def forward(self, input):
        result = self.model(input)
        return result