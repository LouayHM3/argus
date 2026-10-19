# Chronological Design Progress

1. Initial idea: RL-based intrusion detection agent.
2. First limitation: binary malicious/benign classification did not choose responses over time.
3. Sequential reformulation: each response changes future states.
4. Partial observability: true compromise status is hidden in the full concept.
5. POMDP / belief-state design.
6. State decomposition: compromise, posture, and service.
7. Action-space refinement: observation, response, and escalation.
8. Reward redesign: cumulative operational harm instead of detection accuracy.
9. Safety constraints: avoid quarantine-everything behavior through containment costs.
10. ARGUS-Micro: reduce the full system to an exactly solvable learning problem.
11. Dynamic programming: obtain q* as ground truth.
12. MC and TD: study prediction and bootstrapping.
13. SARSA and Q-learning: study control.
14. Final algorithm comparison: evaluate learned policies exactly.
15. Final conclusions: report support or disagreement with the theoretical reference honestly.
16. Executed validation (19 October 2026): installed Python and dependencies, implemented the missing MDP from Table 1 of the supplied report, corrected first-visit MC control and the configurable Q-learning step size, and passed ten tests plus the Gymnasium checker.
17. Executed experiments: completed 25 control runs of 25,000 episodes each; decaying-step Q-learning recovered the optimal policy on all five seeds. An independent seed-0 repeat reproduced its policy and summary metrics. Generated ten CSV files, five figures, and `validation_results.md`; integration into the PDF remains pending.
