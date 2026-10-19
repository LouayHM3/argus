from types import SimpleNamespace

import numpy as np

from agents import monte_carlo
from agents.q_learning import train_q_learning


def test_mc_control_uses_first_visit_return(monkeypatch):
    env = SimpleNamespace(mdp=SimpleNamespace(n_states=2), action_space=SimpleNamespace(n=1))
    monkeypatch.setattr(
        monte_carlo, "generate_episode",
        lambda *args: [(0, 0, 1.0, 0), (0, 0, 2.0, 1)],
    )
    q_values, returns, _ = monte_carlo.mc_control(env, episodes=1, alpha=1.0, gamma=1.0)
    assert q_values[0, 0] == 3.0
    assert returns == [3.0]


def test_decaying_q_learning_respects_alpha():
    class OneStepEnv:
        mdp = SimpleNamespace(n_states=2)
        action_space = SimpleNamespace(n=1)

        def reset(self, **kwargs):
            return 0, {}

        def step(self, action):
            return 1, 2.0, True, False, {}

    q_values, _, _ = train_q_learning(
        OneStepEnv(), episodes=1, alpha=0.4, decaying_alpha=True,
    )
    assert np.isclose(q_values[0, 0], 0.8 / 1.002)
