"""SARSA and Expected SARSA control."""

import numpy as np

from argus.environment import ArgusMicroEnv
from argus.policies import epsilon_greedy_action


def _epsilon_value(epsilon: float, episode: int, floor: float = 0.02) -> float:
    return max(floor, epsilon if epsilon < 1 else 1.0 / (1.0 + 0.0008 * episode))


def train_sarsa(env: ArgusMicroEnv, episodes: int, alpha: float = 0.15, gamma: float = 0.9, epsilon: float = 1.0, seed: int = 0, random_start: bool = True, expected: bool = False) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.mdp.n_states, env.action_space.n))
    returns, history = [], []
    for episode_number in range(episodes):
        env.random_start = random_start
        state, _ = env.reset(seed=int(rng.integers(2**32 - 1)))
        current_epsilon = _epsilon_value(epsilon, episode_number)
        action = epsilon_greedy_action(q_values, state, current_epsilon, rng)
        total = 0.0
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                target = reward
            elif expected:
                next_values = q_values[next_state]
                probabilities = np.full(env.action_space.n, current_epsilon / env.action_space.n)
                probabilities[next_values == next_values.max()] += (1.0 - current_epsilon) / np.sum(next_values == next_values.max())
                target = reward + gamma * float(np.dot(probabilities, next_values))
            else:
                next_action = epsilon_greedy_action(q_values, next_state, current_epsilon, rng)
                target = reward + gamma * q_values[next_state, next_action]
            q_values[state, action] += alpha * (target - q_values[state, action])
            if terminated or truncated:
                break
            state, action = next_state, (next_action if not expected else epsilon_greedy_action(q_values, next_state, current_epsilon, rng))
        returns.append(total)
        history.append(q_values.copy())
    return q_values, returns, history
