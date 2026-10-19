from argus.environment import ArgusMicroEnv
from argus.mdp import Action, State


def test_seeded_environment_is_reproducible():
    first = ArgusMicroEnv()
    second = ArgusMicroEnv()
    first.reset(seed=42)
    second.reset(seed=42)
    trace_a, trace_b = [], []
    for action in [Action.NOOP, Action.MONITOR, Action.NOOP]:
        trace_a.append(first.step(action)[:4])
        trace_b.append(second.step(action)[:4])
        if trace_a[-1][2] or trace_a[-1][3]:
            break
    assert trace_a == trace_b


def test_horizon_is_truncation():
    env = ArgusMicroEnv(max_episode_steps=1)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(Action.ISOLATE)
    assert not terminated
    assert truncated


def test_compromised_monitor_breach_reward():
    env = ArgusMicroEnv()
    env.reset(seed=0)
    env.state = int(State.COMPROMISED)
    # The reward table is tested directly; sampling can select either successor.
    assert env.mdp.rewards[State.COMPROMISED, Action.MONITOR, State.BREACHED] == -33.1
