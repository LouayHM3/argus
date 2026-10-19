"""ARGUS-Micro reference model from Table 1, PDF page 9 of the final report."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class State(IntEnum):
    CLEAN = 0
    SUSPECTED = 1
    COMPROMISED = 2
    QUARANTINED = 3
    SECURED = 4
    BREACHED = 5


class Action(IntEnum):
    NOOP = 0
    MONITOR = 1
    ISOLATE = 2


@dataclass
class ReferenceMDP:
    transitions: np.ndarray
    rewards: np.ndarray
    gamma: float = 0.9
    terminal_states: tuple[int, ...] = (State.SECURED, State.BREACHED)

    @property
    def n_states(self) -> int:
        return self.transitions.shape[0]

    @property
    def n_actions(self) -> int:
        return self.transitions.shape[1]

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states

    def sample_transition(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float]:
        next_state = int(rng.choice(self.n_states, p=self.transitions[state, action]))
        return next_state, float(self.rewards[state, action, next_state])


def build_reference_mdp(gamma: float = 0.9) -> ReferenceMDP:
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    transitions = np.zeros((len(State), len(Action), len(State)))
    rewards = np.zeros_like(transitions)
    # Each row specifies a base reward; terminal entry bonuses are added below.
    rows = {
        (State.CLEAN, Action.NOOP): (0.0, {State.CLEAN: .87, State.SUSPECTED: .08, State.SECURED: .05}),
        (State.CLEAN, Action.MONITOR): (-.1, {State.CLEAN: .89, State.SUSPECTED: .06, State.SECURED: .05}),
        (State.CLEAN, Action.ISOLATE): (-6.0, {State.QUARANTINED: 1.0}),
        (State.SUSPECTED, Action.NOOP): (-.6, {State.CLEAN: .10, State.SUSPECTED: .55, State.COMPROMISED: .35}),
        (State.SUSPECTED, Action.MONITOR): (-.7, {State.CLEAN: .20, State.SUSPECTED: .62, State.COMPROMISED: .18}),
        (State.SUSPECTED, Action.ISOLATE): (-2.5, {State.QUARANTINED: 1.0}),
        (State.COMPROMISED, Action.NOOP): (-3.0, {State.COMPROMISED: .72, State.BREACHED: .28}),
        (State.COMPROMISED, Action.MONITOR): (-3.1, {State.COMPROMISED: .80, State.BREACHED: .20}),
        (State.COMPROMISED, Action.ISOLATE): (-1.0, {State.QUARANTINED: .85, State.BREACHED: .15}),
        (State.QUARANTINED, Action.NOOP): (-1.2, {State.QUARANTINED: .70, State.CLEAN: .30}),
        (State.QUARANTINED, Action.MONITOR): (-1.3, {State.QUARANTINED: .60, State.CLEAN: .40}),
        (State.QUARANTINED, Action.ISOLATE): (-1.5, {State.QUARANTINED: 1.0}),
    }
    for (state, action), (base_reward, successors) in rows.items():
        rewards[state, action, :] = base_reward
        rewards[state, action, State.SECURED] += 3.0
        rewards[state, action, State.BREACHED] -= 30.0
        for successor, probability in successors.items():
            transitions[state, action, successor] = probability
    for terminal in (State.SECURED, State.BREACHED):
        transitions[terminal, :, terminal] = 1.0
    return ReferenceMDP(transitions=transitions, rewards=rewards, gamma=gamma)
