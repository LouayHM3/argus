"""Exact dynamic-programming solvers and policy evaluation."""

import numpy as np

from .mdp import ReferenceMDP


def policy_evaluation(mdp: ReferenceMDP, policy: np.ndarray, tolerance: float = 1e-13, max_sweeps: int = 100_000) -> np.ndarray:
    values = np.zeros(mdp.n_states, dtype=float)
    for _ in range(max_sweeps):
        updated = values.copy()
        for state in range(mdp.n_states):
            if mdp.is_terminal(state):
                updated[state] = 0.0
                continue
            action = int(policy[state])
            updated[state] = np.sum(mdp.transitions[state, action] * (mdp.rewards[state, action] + mdp.gamma * values))
        if np.max(np.abs(updated - values)) < tolerance:
            return updated
        values = updated
    raise RuntimeError("Policy evaluation did not converge")


def value_iteration(mdp: ReferenceMDP, tolerance: float = 1e-13, max_sweeps: int = 100_000) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.zeros(mdp.n_states, dtype=float)
    for sweep in range(1, max_sweeps + 1):
        action_values = np.zeros((mdp.n_states, mdp.n_actions))
        for state in range(mdp.n_states):
            for action in range(mdp.n_actions):
                action_values[state, action] = np.sum(mdp.transitions[state, action] * (mdp.rewards[state, action] + mdp.gamma * values))
        updated = action_values.max(axis=1)
        updated[list(mdp.terminal_states)] = 0.0
        if np.max(np.abs(updated - values)) < tolerance:
            policy = action_values.argmax(axis=1).astype(int)
            return updated, policy, sweep
        values = updated
    raise RuntimeError("Value iteration did not converge")


def q_from_value(mdp: ReferenceMDP, values: np.ndarray) -> np.ndarray:
    q_values = np.zeros((mdp.n_states, mdp.n_actions))
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            q_values[state, action] = np.sum(mdp.transitions[state, action] * (mdp.rewards[state, action] + mdp.gamma * values))
    q_values[list(mdp.terminal_states)] = 0.0
    return q_values


def evaluate_policy(mdp: ReferenceMDP, policy: np.ndarray) -> float:
    return float(policy_evaluation(mdp, policy)[0])
