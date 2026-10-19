# Captures d’écran pour le rapport ARGUS

Ces images proviennent de l’interface locale exécutée dans Microsoft Edge en mode sans fenêtre. Elles montrent de vrais résultats et interactions de l’application.

| Fichier | Légende proposée pour le rapport en anglais |
|---|---|
| `01_vue_ensemble.png` | **ARGUS-Micro dashboard.** Overview of the exact optimal policy, recorded experiment counts, and mean learned-policy values compared with the exact optimum. White segments show one standard deviation across seeds; the dashed gold line marks the optimal value. |
| `02_simulation_episode.png` | **Interactive episode simulation.** Step-by-step execution of the exact optimal policy, starting from Suspected with random seed 42. The interface displays the host state, discounted cumulative return, transition probabilities where applicable, and the decision trace. |
| `03_laboratoire_apprentissage.png` | **Interactive learning laboratory.** A demonstration run of decaying-step Q-learning with 1,000 episodes, seed 0, alpha=0.15 and gamma=0.9. The displayed policy is evaluated exactly from Clean. This demonstration is separate from the five-seed, 25,000-episode reference experiment. |
| `04_comparaison_resultats.png` | **Recorded experimental results.** Comparison of five control methods over five seeds with 25,000 episodes per run, using exact evaluation of the final greedy policies. The dashboard presents mean values, variation across seeds, optimal-policy counts and a saved experimental figure. |

## Utilisation

- Les PNG sont en haute résolution. Conserver leur ratio et éviter de les étirer.
- Pour le corps du rapport, les captures 01 et 02 montrent l’interface et la simulation. Les captures plus longues 03 et 04 peuvent être placées en annexe ou sur une page entière.
- Les figures scientifiques originales dans `results/figures/` restent préférables aux captures d’écran pour analyser la convergence et les valeurs.
- Les paramètres de capture sont enregistrés dans `capture_parameters.json`. Le journal de simulation montre un épisode échantillonné, pas une probabilité de succès mesurée sur plusieurs épisodes.

## Reproduction

Lancer l’application, puis exécuter `python scripts/capture_interface.py` avec Playwright installé et Microsoft Edge disponible. Ce script n’altère pas les CSV des expériences de référence.
