"""Run exact, baseline, and multi-seed ARGUS-Micro experiments."""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.monte_carlo import first_visit_mc_prediction, mc_control
from agents.sarsa import train_sarsa
from agents.q_learning import train_q_learning
from agents.td import td0_prediction
from argus.environment import ArgusMicroEnv
from argus.evaluation import evaluate_policy, q_from_value, value_iteration
from argus.mdp import Action, State, build_reference_mdp
from argus.policies import BASELINES, policy_to_names


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"
ASSET_DIR = ROOT / "report_assets"


def moving_average(values: list[float], window: int = 250) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if len(series) < window:
        return np.array([series.mean()]) if len(series) else np.array([])
    return np.convolve(series, np.ones(window) / window, mode="valid")


def write_worked_episode(mdp, path: Path) -> None:
    # A report-friendly feasible path with explicitly selected stochastic outcomes.
    states_actions = [
        (State.CLEAN, Action.MONITOR, State.CLEAN),
        (State.CLEAN, Action.NOOP, State.CLEAN),
        (State.CLEAN, Action.NOOP, State.SUSPECTED),
        (State.SUSPECTED, Action.MONITOR, State.COMPROMISED),
        (State.COMPROMISED, Action.MONITOR, State.BREACHED),
    ]
    rows = []
    total_return = 0.0
    values = np.zeros(mdp.n_states)
    for index in range(len(states_actions) - 1, -1, -1):
        state, action, next_state = states_actions[index]
        reward = float(mdp.rewards[int(state), int(action), int(next_state)])
        total_return = reward + mdp.gamma * total_return
        td_error = reward + (0.0 if mdp.is_terminal(int(next_state)) else mdp.gamma * values[int(next_state)]) - values[int(state)]
        values[int(state)] = total_return
        rows.append({"time": index, "state": State(state).name.lower(), "action": Action(action).name.lower(), "next_state": State(next_state).name.lower(), "reward": reward, "return": total_return, "td_error": td_error, "updated_v": values[int(state)]})
    pd.DataFrame(reversed(rows)).to_csv(path, index=False)


def run(args: argparse.Namespace) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mdp = build_reference_mdp(args.gamma)
    optimal_values, optimal_policy, sweeps = value_iteration(mdp)
    optimal_q = q_from_value(mdp, optimal_values)

    exact_rows = [{"state": State(state).name.lower(), "v_star": optimal_values[state], "optimal_action": Action(int(optimal_policy[state])).name.lower(), "sweeps": sweeps} for state in range(mdp.n_states)]
    pd.DataFrame(exact_rows).to_csv(CSV_DIR / "exact_values.csv", index=False)

    baseline_rows = []
    for name, policy in BASELINES.items():
        baseline_rows.append({"algorithm": name, "v_clean": evaluate_policy(mdp, policy), "policy": policy_to_names(policy)})
    pd.DataFrame(baseline_rows).to_csv(CSV_DIR / "baseline_values.csv", index=False)

    algorithms = {
        "mc": lambda seed: mc_control(ArgusMicroEnv(mdp=mdp), args.episodes, args.alpha, args.gamma, args.epsilon, seed),
        "sarsa": lambda seed: train_sarsa(ArgusMicroEnv(mdp=mdp), args.episodes, args.alpha, args.gamma, args.epsilon, seed, True, False),
        "expected_sarsa": lambda seed: train_sarsa(ArgusMicroEnv(mdp=mdp), args.episodes, args.alpha, args.gamma, args.epsilon, seed, True, True),
        "qlearning_constant": lambda seed: train_q_learning(ArgusMicroEnv(mdp=mdp), args.episodes, args.alpha, args.gamma, args.epsilon, seed, True, False),
        "qlearning_decayed": lambda seed: train_q_learning(ArgusMicroEnv(mdp=mdp), args.episodes, args.alpha, args.gamma, args.epsilon, seed, True, True),
    }
    td_values, td_history, td_errors = td0_prediction(ArgusMicroEnv(mdp=mdp), BASELINES["always_monitor"], args.episodes, args.alpha, args.gamma, args.seeds[0])
    mc_values, mc_history = first_visit_mc_prediction(ArgusMicroEnv(mdp=mdp), BASELINES["always_monitor"], args.episodes, args.alpha, args.gamma, args.seeds[0])
    pd.DataFrame({"episode": np.arange(1, args.episodes + 1), "mc_clean": [values[State.CLEAN] for values in mc_history], "td_clean": [values[State.CLEAN] for values in td_history]}).to_csv(CSV_DIR / "prediction_history.csv", index=False)
    rows, policy_rows, seed_rows = [], [], []
    curves: dict[str, list[np.ndarray]] = {}
    q_error_curves: dict[str, list[np.ndarray]] = {}
    for algorithm, trainer in algorithms.items():
        for seed in args.seeds:
            result = trainer(seed)
            q_values, returns, q_history = result
            policy = q_values.argmax(axis=1).astype(int)
            values = evaluate_policy(mdp, policy)
            q_error = float(np.linalg.norm(q_values - optimal_q))
            rows.append({"algorithm": algorithm, "seed": seed, "episodes": args.episodes, "alpha": args.alpha, "epsilon": args.epsilon, "v_clean": values, "q_error": q_error, "return_mean": np.mean(returns), "return_std": np.std(returns)})
            policy_rows.extend({"algorithm": algorithm, "seed": seed, "state": State(state).name.lower(), "action": Action(int(policy[state])).name.lower()} for state in range(4))
            seed_rows.append({"algorithm": algorithm, "seed": seed, "v_clean": values, "q_error": q_error})
            curves.setdefault(algorithm, []).append(moving_average(returns))
            q_error_curves.setdefault(algorithm, []).append(np.asarray([np.linalg.norm(estimate - optimal_q) for estimate in q_history]))
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(CSV_DIR / "seed_summary.csv", index=False)
    result_frame[result_frame.algorithm.str.startswith("qlearning")].to_csv(CSV_DIR / "qlearning_results.csv", index=False)
    result_frame[result_frame.algorithm == "sarsa"].to_csv(CSV_DIR / "sarsa_results.csv", index=False)
    result_frame[result_frame.algorithm == "expected_sarsa"].to_csv(CSV_DIR / "expected_sarsa_results.csv", index=False)
    result_frame[result_frame.algorithm == "mc"].to_csv(CSV_DIR / "mc_results.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(CSV_DIR / "learned_policies.csv", index=False)
    write_worked_episode(mdp, CSV_DIR / "worked_episode.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 4.5))
    for algorithm, values in curves.items():
        lengths = min(len(item) for item in values)
        plt.plot(np.mean([item[:lengths] for item in values], axis=0), label=algorithm)
    plt.xlabel("Episode"); plt.ylabel("Moving-average return"); plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "qlearning_learning_curve.png", dpi=180); plt.close()

    comparison = result_frame.groupby("algorithm").v_clean.mean().to_dict()
    comparison["optimal"] = optimal_values[State.CLEAN]
    plt.figure(figsize=(8, 4.5)); plt.bar(comparison.keys(), comparison.values()); plt.xticks(rotation=30, ha="right"); plt.ylabel("Exact value from Clean"); plt.tight_layout(); plt.savefig(FIG_DIR / "algorithm_comparison.png", dpi=180); plt.close()

    policy_rows_frame = pd.DataFrame(seed_rows)
    plt.figure(figsize=(8, 4.5));
    for algorithm, group in policy_rows_frame.groupby("algorithm"):
        plt.scatter([algorithm] * len(group), group.v_clean, alpha=0.75)
    plt.axhline(optimal_values[State.CLEAN], color="black", linestyle="--", label="optimal"); plt.ylabel("Exact value from Clean"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(FIG_DIR / "policy_value_by_seed.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.5))
    for algorithm, values in q_error_curves.items():
        lengths = min(len(item) for item in values)
        plt.plot(np.arange(1, lengths + 1), np.mean([item[:lengths] for item in values], axis=0), label=algorithm)
    plt.xlabel("Episode"); plt.ylabel("Reference Q norm"); plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "convergence_to_qstar.png", dpi=180); plt.close()

    prediction_frame = pd.read_csv(CSV_DIR / "prediction_history.csv")
    plt.figure(figsize=(8, 4.5)); plt.plot(prediction_frame.episode, prediction_frame.mc_clean, label="MC"); plt.plot(prediction_frame.episode, prediction_frame.td_clean, label="TD(0)"); plt.axhline(evaluate_policy(mdp, BASELINES["always_monitor"]), color="black", linestyle="--", label="exact"); plt.xlabel("Episode"); plt.ylabel("V(Clean)"); plt.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "value_estimation_mc_vs_td.png", dpi=180); plt.close()

    with (ASSET_DIR / "implementation_summary.md").open("w", encoding="utf-8") as file:
        file.write(f"# Implementation Summary\n\nThe reference MDP was encoded, validated, and solved with value iteration in {sweeps} sweeps. The computed optimal policy is `{policy_to_names(optimal_policy)}` and v*(Clean) is `{optimal_values[State.CLEAN]:.8f}`. Baselines and all configured learners were trained over {len(args.seeds)} seeds and evaluated exactly from Clean. Detailed numeric outputs are in `results/csv`.\n\nThe implementation sequence was: reference MDP; environment validation; dynamic programming ground truth; baselines; Monte Carlo; TD(0); SARSA; Q-learning; step-size comparison; final evaluation against q*. Results are generated by this script and are not hand-edited.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run the complete experiment suite.")
    parser.add_argument("--episodes", type=int, default=25_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
