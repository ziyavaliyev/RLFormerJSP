import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jsp_il.jsp_instance import (
    build_initial_state,
    state_to_tokens,
    apply_action,
    is_done,
    makespan,
)


class JSPDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        instances,
        reward_mode="dense_terminal",
        reward_scale=1000.0,
        seed=None,
    ):
        super().__init__()

        self.instances = [np.asarray(x, dtype=np.int64) for x in instances]
        self.reward_mode = reward_mode
        self.reward_scale = float(reward_scale)
        self.rng = np.random.default_rng(seed)

        self.instance = None
        self.state = None
        self.prev_makespan = 0

        _, self.n_jobs, self.n_machines = self.instances[0].shape
        self.T = self.n_jobs * self.n_machines

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.T, 16),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.T)

    def action_mask(self):
        _, mask = state_to_tokens(self.instance, self.state)
        return mask.astype(bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = int(self.rng.integers(0, len(self.instances)))
        self.instance = self.instances[idx]
        self.state = build_initial_state(self.instance)
        self.prev_makespan = 0

        tokens, _ = state_to_tokens(self.instance, self.state)
        return tokens.astype(np.float32), {}

    def step(self, action):
        action = int(action)
        mask = self.action_mask()

        if not mask[action]:
            # Should not happen with masking, but keeps env safe.
            tokens, _ = state_to_tokens(self.instance, self.state)
            return tokens.astype(np.float32), -1.0, False, False, {"invalid_action": True}

        old_ms = makespan(self.state)
        self.state = apply_action(self.instance, self.state, action)
        new_ms = makespan(self.state)

        done = is_done(self.instance, self.state)

        if self.reward_mode == "terminal":
            reward = -new_ms / self.reward_scale if done else 0.0

        elif self.reward_mode == "dense":
            reward = -(new_ms - old_ms) / self.reward_scale

        elif self.reward_mode == "dense_terminal":
            reward = -(new_ms - old_ms) / self.reward_scale
            if done:
                reward += -new_ms / self.reward_scale

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        tokens, _ = state_to_tokens(self.instance, self.state)

        info = {}
        if done:
            info["makespan"] = int(new_ms)

        return tokens.astype(np.float32), float(reward), done, False, info