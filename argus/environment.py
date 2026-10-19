"""Gymnasium-compatible sampled environment for ARGUS-Micro."""

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Allows exact solvers/tests to run without Gymnasium installed.
    gym = object  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from .mdp import Action, ReferenceMDP, State, build_reference_mdp


class ArgusMicroEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int | None = None, random_start: bool = False, mdp: ReferenceMDP | None = None):
        self.mdp = mdp or build_reference_mdp()
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.observation_space = spaces.Discrete(self.mdp.n_states)
        self.action_space = spaces.Discrete(self.mdp.n_actions)
        self._rng = np.random.default_rng()
        self.state = int(State.CLEAN)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        starts = [int(State.CLEAN), int(State.SUSPECTED), int(State.COMPROMISED), int(State.QUARANTINED)]
        self.state = int(self._rng.choice(starts)) if self.random_start else int(State.CLEAN)
        self.steps = 0
        return self.state, {"state": State(self.state).name}

    def step(self, action: int):
        if self.mdp.is_terminal(self.state):
            raise RuntimeError("Cannot step a terminated episode; call reset().")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        previous_state = self.state
        self.state, reward = self.mdp.sample_transition(previous_state, action, self._rng)
        self.steps += 1
        terminated = self.mdp.is_terminal(self.state)
        truncated = self.max_episode_steps is not None and self.steps >= self.max_episode_steps and not terminated
        info = {"state": State(self.state).name, "previous_state": State(previous_state).name, "action": Action(action).name}
        return self.state, reward, terminated, truncated, info
