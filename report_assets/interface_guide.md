# Interface locale ARGUS

## Lancement

Depuis le dossier contenant `app.py`, double-cliquer sur `lancer_interface.bat`, ou exécuter :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ouvrir http://127.0.0.1:8501. Le service écoute uniquement sur la machine locale. Arrêter le service avec Ctrl+C dans son terminal.

## Parcours de démonstration

1. Vue d’ensemble : expliquer les six états, les trois actions et la politique optimale à gamma=0,9.
2. Simulation : choisir une graine, un état initial, puis cliquer sur Nouvel épisode. En mode manuel, jouer une action à la fois. En mode politique, utiliser Jouer une étape ou Terminer avec cette stratégie. La distribution de la prochaine transition est consultable.
3. Observer le journal, les récompenses et le retour actualisé. Exporter l’épisode en CSV. Un épisode peut finir en Secured, Breached, ou être interrompu au plafond de 200 décisions ; ces cas sont distingués.
4. Laboratoire : choisir l’algorithme et entraîner sur 100 à 10 000 épisodes. Consulter la politique, sa valeur exacte depuis Clean, l’écart à l’optimum, les retours et la norme d’erreur Q. Exporter l’historique brut et la table Q.
5. Revenir dans Simulation et choisir Mon dernier agent entraîné. Le modèle utilise la même valeur gamma=0,9 que les expériences de référence.
6. Résultats : filtrer les méthodes enregistrées, comparer les moyennes et écarts-types à l’optimum, consulter les baselines et les résultats par graine, puis télécharger les CSV et les figures. La préparation du rapport et le prompt sont conservés en dehors du dashboard.

## Données et portée

Les résultats du laboratoire sont conservés dans la session Streamlit et se téléchargent séparément. Ils ne remplacent pas les CSV des 25 expériences de référence. Les paramètres de départ d’une simulation sont appliqués au clic sur Nouvel épisode ; la stratégie peut changer en cours d’épisode et est enregistrée dans le journal.

L’interface est un démonstrateur de l’environnement simulé Argus-Micro. Elle ne reçoit pas de télémétrie réseau réelle et ne contrôle pas d’équipement de sécurité.

## Validation exécutée

- Streamlit 1.66.0, avec dépendances enregistrées dans requirements-lock.txt.
- `python -m pytest -q` : **13 passed**, soit dix tests du modèle et des agents, plus trois tests d’interface avec Streamlit AppTest.
- Navigation du tableau de bord vers les résultats, y compris une sélection vide de méthodes.
- Action manuelle, épisode guidé, désactivation des actions à la fin et reproduction du journal avec la même graine.
- Entraînement d’un agent sur 100 épisodes et utilisation de sa politique dans la simulation.
- Serveur local : réponse HTTP 200 et `ok` sur `/_stcore/health`.

Les graphiques et les valeurs des expériences de référence proviennent de l’exécution scientifique documentée dans validation_results.md. Aucun nouvel entraînement complet de 25 000 épisodes n’a été nécessaire pour ajouter l’interface.
