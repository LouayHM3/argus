"""Print the independently computed exact reference solution."""

from argus.evaluation import q_from_value, value_iteration
from argus.mdp import Action, State, build_reference_mdp


if __name__ == "__main__":
    mdp = build_reference_mdp()
    values, policy, sweeps = value_iteration(mdp)
    q_values = q_from_value(mdp, values)
    print(f"value_iteration_sweeps={sweeps}")
    for state in State:
        print(f"{state.name.lower()}: value={values[state]:.8f}, action={Action(int(policy[state])).name.lower()}, q={q_values[state].tolist()}")
