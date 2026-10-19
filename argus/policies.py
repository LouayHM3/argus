"""Policy helpers and named static baselines."""

import numpy as np

from .mdp import Action, State


BASELINES: dict[str, np.ndarray] = {
    "never_act": np.array([Action.NOOP, Action.NOOP, Action.NOOP, Action.NOOP], dtype=int),
    "always_monitor": np.array([Action.MONITOR] * 4, dtype=int),
    "static_playbook": np.array([Action.NOOP, Action.ISOLATE, Action.ISOLATE, Action.NOOP], dtype=int),
}


def epsilon_greedy_action(q_values: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[-1]))
    values = q_values[state]
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def policy_to_names(policy: np.ndarray) -> dict[str, str]:
    return {State(state).name.lower(): Action(int(policy[state])).name.lower() for state in range(4)}
