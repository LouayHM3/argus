# ARGUS-Micro: executed validation results

Model source: Table 1 of ARGUS_Final_Report_Louay_Hammoudi_Taha_Marouani.pdf, PDF page 9 (printed page 8). The missing model module was implemented from this table; the supplied PDF was not modified.

## Validation

- Unit tests: **10 passed**, including 60,000 transition samples and regression tests for first-visit MC and configurable decaying alpha.
- Gymnasium environment checker: passed.
- Exact value iteration: 197 sweeps; V*(Clean) = -0.9670695506.
- Optimal policy (Clean, Suspected, Compromised, Quarantined): noop, monitor, isolate, monitor.
- All 25 control runs have finite summary metrics and four nonterminal policy actions.
- An independent 25,000-episode Q-learning-decayed run with seed 0 reproduced the final policy, exact value, Q-error, return mean, and return standard deviation within 1e-12 tolerance. This verifies that run's summary and policy; it does not claim every method or full trajectory was repeated.

## Configuration

25,000 episodes per method and seed; seeds [0, 1, 2, 3, 4]; gamma=0.9; alpha=0.15. Control starts uniformly across the four nonterminal states; prediction starts in Clean under always-monitor with seed 0. MC uses constant epsilon=1.0. SARSA and Q-learning use max(0.02, epsilon_argument if epsilon_argument < 1 else 1/(1+0.0008*episode_index)). The decaying Q-learning step size is alpha/(1+0.002*visit_count). Detailed configuration and package versions are in run_configuration.json and requirements-lock.txt.

## Exact baseline values

| Policy | V(Clean) | Gap to optimum |
|---|---:|---:|
| never_act | -6.80832867 | 5.84125912 |
| always_monitor | -3.83458166 | 2.86751211 |
| static_playbook | -1.41510335 | 0.44803380 |

## Learned policies across 5 seeds

| Method | Mean V(Clean) | Std across seeds | Mean gap | Optimal policies |
|---|---:|---:|---:|---:|
| expected_sarsa | -1.36706383 | 0.14189240 | 0.39999428 | 0/5 |
| mc | -1.25750568 | 0.17322382 | 0.29043613 | 0/5 |
| qlearning_constant | -1.36193849 | 0.08430797 | 0.39486894 | 0/5 |
| qlearning_decayed | -0.96706955 | 0.00000000 | 0.00000000 | 5/5 |
| sarsa | -1.29206395 | 0.18972115 | 0.32499440 | 1/5 |

Values above are exact evaluations of final greedy policies, not sample training returns. The standard deviation is the sample standard deviation across seeds. The gap is V* minus the policy value. Training returns are undiscounted and use random initial states, so they are not directly comparable to discounted V(Clean). Q-error is the Frobenius norm over the full state-action table, not RMSE.

## Individual runs

| Method | Seed | Clean | Suspected | Compromised | Quarantined | V(Clean) | Q-error |
|---|---:|---|---|---|---|---:|---:|
| mc | 0 | noop | isolate | isolate | noop | -1.41510335 | 38.45866700 |
| mc | 1 | noop | isolate | isolate | monitor | -1.28170266 | 33.83699083 |
| mc | 2 | noop | isolate | isolate | noop | -1.41510335 | 35.35096613 |
| mc | 3 | monitor | monitor | isolate | noop | -1.16582502 | 32.60659034 |
| mc | 4 | noop | monitor | isolate | noop | -1.00979399 | 38.99687995 |
| sarsa | 0 | noop | isolate | isolate | monitor | -1.28170266 | 7.43158077 |
| sarsa | 1 | noop | isolate | isolate | noop | -1.41510335 | 9.13923920 |
| sarsa | 2 | noop | monitor | isolate | monitor | -0.96706955 | 7.30473280 |
| sarsa | 3 | monitor | isolate | isolate | monitor | -1.38134082 | 7.34104352 |
| sarsa | 4 | noop | isolate | isolate | noop | -1.41510335 | 9.66499681 |
| expected_sarsa | 0 | noop | isolate | isolate | monitor | -1.28170266 | 5.57471413 |
| expected_sarsa | 1 | monitor | noop | isolate | monitor | -1.60887034 | 5.38223702 |
| expected_sarsa | 2 | noop | isolate | isolate | monitor | -1.28170266 | 5.96367188 |
| expected_sarsa | 3 | monitor | isolate | isolate | monitor | -1.38134082 | 6.68900856 |
| expected_sarsa | 4 | noop | isolate | isolate | monitor | -1.28170266 | 8.41628127 |
| qlearning_constant | 0 | monitor | isolate | isolate | monitor | -1.38134082 | 7.25272086 |
| qlearning_constant | 1 | monitor | isolate | isolate | noop | -1.48360547 | 4.37788364 |
| qlearning_constant | 2 | noop | isolate | isolate | monitor | -1.28170266 | 7.55939501 |
| qlearning_constant | 3 | noop | isolate | isolate | monitor | -1.28170266 | 6.18498947 |
| qlearning_constant | 4 | monitor | isolate | isolate | monitor | -1.38134082 | 4.63691602 |
| qlearning_decayed | 0 | noop | monitor | isolate | monitor | -0.96706955 | 2.30643252 |
| qlearning_decayed | 1 | noop | monitor | isolate | monitor | -0.96706955 | 3.08447626 |
| qlearning_decayed | 2 | noop | monitor | isolate | monitor | -0.96706955 | 2.24464382 |
| qlearning_decayed | 3 | noop | monitor | isolate | monitor | -0.96706955 | 1.73057501 |
| qlearning_decayed | 4 | noop | monitor | isolate | monitor | -0.96706955 | 0.41871296 |

## Interpretation and remaining report work

The analytical reference values are reproduced by the implemented MDP. Learning results above are measured outputs; a single successful seed is not a guarantee that a method always recovers the optimum. MC keeps epsilon fixed; SARSA and Q-learning reduce epsilon when the epsilon argument is at least 1. This difference limits a direct algorithm-only interpretation of the comparison. Constant step sizes preserve sampling fluctuations; policies can differ between seeds when action values are close.

Figures are in results/figures and full numerical tables are in results/csv. The PDF still contains implementation placeholders and needs these executed results and figures integrated into its final version.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --all --episodes 25000 --seeds 0 1 2 3 4 --alpha 0.15 --gamma 0.9 --epsilon 1.0
.\.venv\Scripts\python.exe -m experiments.validate_results
```
