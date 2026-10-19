"""Q-learning with constant or count-based decaying step sizes."""

import numpy as np

from argus.environment import ArgusMicroEnv
from argus.policies import epsilon_greedy_action


def train_q_learning(env: ArgusMicroEnv, episodes: int, alpha: float = 0.15, gamma: float = 0.9, epsilon: float = 1.0, seed: int = 0, random_start: bool = True, decaying_alpha: bool = False) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.mdp.n_states, env.action_space.n))
    counts = np.zeros_like(q_values)
    returns, history = [], []
    for episode_number in range(episodes):
        env.random_start = random_start
        state, _ = env.reset(seed=int(rng.integers(2**32 - 1)))
        episode_epsilon = max(0.02, epsilon if epsilon < 1 else 1.0 / (1.0 + 0.0008 * episode_number))
        total = 0.0
        while True:
            action = epsilon_greedy_action(q_values, state, episode_epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            counts[state, action] += 1
            step_alpha = alpha / (1.0 + 0.002 * counts[state, action]) if decaying_alpha else alpha
            target = reward if terminated or truncated else reward + gamma * np.max(q_values[next_state])
            q_values[state, action] += step_alpha * (target - q_values[state, action])
            total += reward
            if terminated or truncated:
                break
            state = next_state
        returns.append(total)
        history.append(q_values.copy())
    return q_values, returns, history
