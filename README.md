# Project ARGUS

## Objective
ARGUS-Micro is a reproducible educational reinforcement-learning simulation for sequential cyber-defence decisions. It studies when an agent should do nothing, monitor, or isolate a single host while balancing damage, false positives, containment cost, and delayed consequences.

This is not production cybersecurity software and does not control real security infrastructure.

## Environment

States are `Clean`, `Suspected`, `Compromised`, `Quarantined`, `Secured`, and `Breached`. Actions are `noop`, `monitor`, and `isolate`. Secured and Breached are terminal. Entering Secured adds `+3`; entering Breached adds `-30` to the base transition reward. The reference discount factor is `gamma=0.9`.

## Implemented algorithms

- Iterative policy evaluation and value iteration
- First-visit Monte Carlo prediction and control
- TD(0) prediction
- SARSA and Expected SARSA
- Q-learning with constant and count-based decaying alpha
- Static baselines and exact policy evaluation

## Installation

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running experiments

### Local interface

On Windows, double-click `lancer_interface.bat`, then open http://127.0.0.1:8501. Alternatively, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The French interface provides a dashboard, seeded step-by-step simulation with manual/baseline/learned policies, configurable training, CSV downloads, saved experiment comparisons and figures. New training results remain in the current browser session and can be exported; they do not replace the saved reference experiments. Simulation episodes are capped at 200 decisions and distinguish truncation from a terminal outcome.

The dashboard includes direct navigation, a policy overview and interactive charts comparing mean policy values, variation across seeds and the exact optimum. The Results page includes baseline comparisons, saved figures and CSV exports. Report preparation is handled outside the interface; its prompt remains in `report_assets/prompt_rapport_final.md`.

### Full experiment suite

```bash
python main.py --all --episodes 25000 --seeds 0 1 2 3 4
pytest -q
```

Outputs are written to `results/csv`, `results/figures`, and `report_assets`. Use `python main.py --help` for configurable seed, episode, alpha, epsilon, and gamma settings.

## Project structure

`argus/` contains the MDP, environment, policy helpers, and exact evaluation. `agents/` contains tabular learners. `experiments/` contains runnable experiment code. `tests/` validates transition, reward, environment, and DP behavior.

## Limitations

The transition probabilities are simulated and the reward coefficients encode assumptions. Real cybersecurity is partially observable, while ARGUS-Micro exposes a compact state directly. Tabular methods do not scale to enterprise environments, and success here demonstrates algorithmic feasibility rather than safe deployment.

## Validation and results

### Local validation status (2026-10-19)

- Python 3.11.9 is installed and `.venv` has been created.
- All dependencies in `requirements.txt` have been installed. On this Windows network, pip needed `--use-feature=truststore` to use the Windows certificate store.
- `python -m compileall -q agents argus experiments tests` completed successfully.
- The missing `argus/mdp.py` was implemented from Table 1 (PDF page 9, printed page 8) of `ARGUS_Final_Report_Louay_Hammoudi_Taha_Marouani.pdf`. It includes all twelve nonterminal state-action distributions, terminal entry rewards, and absorbing terminal states.
- `python -m pytest -q`: **10 passed**, including 60,000 sampled transitions checked against the model.
- Gymnasium's `check_env(ArgusMicroEnv(), skip_render_check=True)` passed.
- Monte Carlo control was corrected to update the first visit to each state-action pair. Decaying Q-learning now respects the supplied `alpha`. Regression tests are in `tests/test_learners.py`.
- Exact value iteration converges in 197 sweeps: `v*(Clean) = -0.967069550599`, with policy `(noop, monitor, isolate, monitor)`, matching the report's reference values.
- The installed dependency versions are recorded in `requirements-lock.txt`.
- The full experiment suite completed: 25 control runs (five methods, five seeds), each with 25,000 episodes, plus MC/TD prediction. Ten CSV files and five figures were generated.
- Decaying-step Q-learning recovered the optimal policy on all five seeds. An independent repeat of seed 0 matched its final policy and summary metrics within 1e-12 tolerance.
- Experiment validation, individual policies, and numerical comparisons are recorded in [report_assets/validation_results.md](report_assets/validation_results.md).

### Reproduction on this Windows installation

Run these commands from the directory containing `main.py`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --all --episodes 25000 --seeds 0 1 2 3 4
.\.venv\Scripts\python.exe -m experiments.validate_results
```

The experiment runner uses a project-local Matplotlib cache and the noninteractive Agg backend so figures can be generated without a desktop window. To recreate the dependency versions, install `requirements-lock.txt` in a new Python 3.11 environment.

### Remaining submission work

Integrate the generated results and figures into the report and replace its placeholder boxes. Git publication requires installing Git and supplying a repository destination; this directory has no configured remote.

Do not treat results as complete until the experiments have been executed. Numerical values must be generated by the code rather than entered manually.
