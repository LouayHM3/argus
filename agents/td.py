"""TD(0) prediction."""

import numpy as np

from argus.environment import ArgusMicroEnv


def td0_prediction(env: ArgusMicroEnv, policy: np.ndarray, episodes: int, alpha: float = 0.05, gamma: float = 0.9, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    values = np.zeros(env.mdp.n_states)
    history, errors = [], []
    for _ in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2**32 - 1)))
        while True:
            next_state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            target = reward if terminated else reward + gamma * values[next_state]
            error = target - values[state]
            values[state] += alpha * error
            errors.append(float(error))
            if terminated or truncated:
                break
            state = next_state
        history.append(values.copy())
    return values, history, errors
