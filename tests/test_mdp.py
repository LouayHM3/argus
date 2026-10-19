import numpy as np

from argus.evaluation import evaluate_policy, value_iteration
from argus.mdp import Action, State, build_reference_mdp
from argus.policies import BASELINES


def test_transition_probabilities_sum_to_one():
    mdp = build_reference_mdp()
    assert np.allclose(mdp.transitions.sum(axis=2), 1.0)


def test_terminal_states_are_absorbing_and_rewardless():
    mdp = build_reference_mdp()
    for state in mdp.terminal_states:
        assert np.all(mdp.transitions[state, :, state] == 1.0)
        assert np.all(mdp.rewards[state] == 0.0)


def test_terminal_entry_rewards():
    mdp = build_reference_mdp()
    assert mdp.rewards[State.CLEAN, Action.NOOP, State.SECURED] == 3.0
    assert mdp.rewards[State.COMPROMISED, Action.MONITOR, State.BREACHED] == -33.1


def test_reference_values():
    mdp = build_reference_mdp()
    values, policy, sweeps = value_iteration(mdp)
    assert sweeps > 0
    assert policy[:4].tolist() == [Action.NOOP, Action.MONITOR, Action.ISOLATE, Action.MONITOR]
    assert np.isclose(values[State.CLEAN], -0.9671, atol=0.002)
    assert np.isclose(evaluate_policy(mdp, BASELINES["never_act"]), -6.8083, atol=0.002)
    assert np.isclose(evaluate_policy(mdp, BASELINES["always_monitor"]), -3.8346, atol=0.002)
    assert np.isclose(evaluate_policy(mdp, BASELINES["static_playbook"]), -1.4151, atol=0.002)


def test_sampled_transitions_follow_reference_distribution():
    mdp = build_reference_mdp()
    rng = np.random.default_rng(42)
    samples = 5_000
    for state in range(4):
        for action in Action:
            counts = np.zeros(mdp.n_states)
            for _ in range(samples):
                successor, reward = mdp.sample_transition(state, action, rng)
                counts[successor] += 1
                assert reward == mdp.rewards[state, action, successor]
            probabilities = mdp.transitions[state, action]
            tolerance = 6 * np.sqrt(probabilities * (1 - probabilities) / samples) + 1 / samples
            assert np.all(np.abs(counts / samples - probabilities) <= tolerance)
