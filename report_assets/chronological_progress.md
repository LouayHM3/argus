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
