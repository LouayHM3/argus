"""First-visit Monte Carlo prediction and epsilon-greedy control."""

import numpy as np

from argus.environment import ArgusMicroEnv
from argus.policies import epsilon_greedy_action


def generate_episode(env: ArgusMicroEnv, q_values: np.ndarray | None = None, epsilon: float = 0.1, rng: np.random.Generator | None = None, random_start: bool = False) -> list[tuple[int, int, float, int]]:
    rng = rng or np.random.default_rng()
    env.random_start = random_start
    state, _ = env.reset(seed=int(rng.integers(2**32 - 1)))
    episode = []
    while True:
        action = epsilon_greedy_action(q_values, state, epsilon, rng) if q_values is not None else int(rng.integers(env.action_space.n))
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward, next_state))
        if terminated or truncated:
            return episode
        state = next_state


def first_visit_mc_prediction(env: ArgusMicroEnv, policy: np.ndarray, episodes: int, alpha: float = 0.05, gamma: float = 0.9, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    values = np.zeros(env.mdp.n_states)
    history = []
    for _ in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(2**32 - 1)))
        trajectory = []
        while True:
            next_state, reward, terminated, truncated, _ = env.step(int(policy[state]))
            trajectory.append((state, reward))
            if terminated or truncated:
                break
            state = next_state
        returns = np.zeros(len(trajectory))
        running = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            running = trajectory[index][1] + gamma * running
            returns[index] = running
        visited = set()
        for index, (visited_state, _) in enumerate(trajectory):
            if visited_state not in visited:
                values[visited_state] += alpha * (returns[index] - values[visited_state])
                visited.add(visited_state)
        history.append(values.copy())
    return values, history


def mc_control(env: ArgusMicroEnv, episodes: int, alpha: float = 0.05, gamma: float = 0.9, epsilon: float = 0.1, seed: int = 0, random_start: bool = True) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.mdp.n_states, env.action_space.n))
    returns, history = [], []
    for _ in range(episodes):
        episode = generate_episode(env, q_values, epsilon, rng, random_start)
        total = sum(item[2] for item in episode)
        returns.append(total)
        running = 0.0
        visited = set()
        for state, action, reward, _ in reversed(episode):
            running = reward + gamma * running
            if (state, action) not in visited:
                q_values[state, action] += alpha * (running - q_values[state, action])
                visited.add((state, action))
        history.append(q_values.copy())
    return q_values, returns, history
