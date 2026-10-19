"""Local interactive dashboard for the ARGUS-Micro experiments."""

from pathlib import Path
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from agents.monte_carlo import mc_control
from agents.q_learning import train_q_learning
from agents.sarsa import train_sarsa
from argus.environment import ArgusMicroEnv
from argus.evaluation import evaluate_policy, q_from_value, value_iteration
from argus.mdp import Action, State, build_reference_mdp
from argus.policies import BASELINES


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "csv"
GAMMA = 0.9
MDP = build_reference_mdp(GAMMA)
VALUES, OPTIMAL, _ = value_iteration(MDP)
METHODS = {
    "mc": "Monte Carlo", "sarsa": "SARSA", "expected_sarsa": "Expected SARSA",
    "qlearning_constant": "Q-learning · constant", "qlearning_decayed": "Q-learning · décroissant",
}
BASELINE_NAMES = {"never_act": "Ne jamais intervenir", "always_monitor": "Toujours surveiller", "static_playbook": "Playbook statique"}
ACTION_NAMES = {"noop": "Attendre", "monitor": "Surveiller", "isolate": "Isoler"}

st.set_page_config(page_title="ARGUS · Laboratoire de cyberdéfense", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.stApp {background: radial-gradient(ellipse at 85% 0%, #15304755, transparent 45%), #0b1120; color: #e2e8f0;}
[data-testid="stSidebar"] {background: #101a2e; border-right: 1px solid #26334c;}
.block-container {max-width: 1260px; padding-top: 2.5rem;}
.eyebrow {color:#5eead4; font-size:.8rem; letter-spacing:.18em; font-weight:700;}
.hero {background:linear-gradient(120deg,#132d43,#101c32);border:1px solid #2c435b;border-left:4px solid #2dd4bf;border-radius:20px;padding:32px;margin-bottom:24px;}
.hero h1 {font-size:2.5rem;letter-spacing:-.04em;margin:4px 0 12px;}
.hero p {color:#aebed2;max-width:750px;line-height:1.7;margin:0;}
[data-testid="stMetric"] {background:#132039;border:1px solid #293b55;border-radius:14px;padding:18px;}
[data-testid="stMetricLabel"] {color:#aebed2;}
.stButton button {border-radius:9px;}
.policy-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:12px 0 18px;}
.policy-card {background:#122037;border:1px solid #2b3d56;border-radius:12px;padding:18px;}
.policy-card span {display:block;font-size:.8rem;color:#aebed2;margin-bottom:8px;}
.policy-card strong {font-size:1.1rem;color:#5eead4;}
.terminal-strip {border:1px solid #2b3d56;border-radius:12px;padding:14px 18px;color:#aebed2;font-size:.85rem;}
@media(max-width:640px){.hero{padding:20px;}.hero h1{font-size:1.9rem;}.policy-grid{grid-template-columns:1fr;}}
</style>""", unsafe_allow_html=True)


def read_csv(name):
    path = CSV / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def hero(label, title, description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{label}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


def policy_frame(policy):
    return pd.DataFrame({"État": [s.name.title() for s in list(State)[:4]],
                         "Action": [ACTION_NAMES[Action(int(a)).name.lower()] for a in policy[:4]]})


def navigate(destination):
    st.session_state.dashboard_page = destination


def comparison_chart(summary):
    frame = summary.groupby("algorithm", as_index=False).v_clean.agg(["mean", "std"]).reset_index()
    frame["Méthode"] = frame.algorithm.map(METHODS)
    frame["Écart-type"] = frame["std"].fillna(0)
    frame["Borne basse"] = frame["mean"] - frame["Écart-type"]
    frame["Borne haute"] = frame["mean"] + frame["Écart-type"]
    base = alt.Chart(frame).encode(y=alt.Y("Méthode:N", sort=alt.SortField(field="mean", order="descending"), title=None))
    bars = base.mark_bar(cornerRadiusEnd=5, size=25).encode(
        x=alt.X("mean:Q", title="Valeur exacte depuis Clean · plus élevée = meilleure"),
        color=alt.condition(alt.datum.algorithm == "qlearning_decayed", alt.value("#2dd4bf"), alt.value("#638cff")),
        tooltip=[alt.Tooltip("Méthode:N"), alt.Tooltip("mean:Q", title="Valeur moyenne", format=".5f"), alt.Tooltip("Écart-type:Q", format=".5f")],
    )
    spread = base.mark_rule(color="#e2e8f0", strokeWidth=2).encode(x="Borne basse:Q", x2="Borne haute:Q")
    optimum = alt.Chart(pd.DataFrame({"Valeur": [float(VALUES[0])], "Référence": ["Optimum exact"]})).mark_rule(color="#fbbf24", strokeDash=[6, 4]).encode(x="Valeur:Q", tooltip=["Référence:N", alt.Tooltip("Valeur:Q", format=".5f")])
    return (bars + spread + optimum).properties(height=260).configure_view(stroke=None)


with st.sidebar:
    st.markdown("## 🛡️ ARGUS")
    st.caption("MICRO · REINFORCEMENT LEARNING")
    page = st.radio("Espace de travail", ["Vue d’ensemble", "Simulation", "Laboratoire", "Résultats"], key="dashboard_page")
    st.divider()
    st.caption("Modèle de référence · 6 états · 3 actions")
    st.caption("● Laboratoire local · γ = 0,9")
    st.caption("Simulation pédagogique de cyberdéfense. Les actions portent sur un hôte simulé.")


if page == "Vue d’ensemble":
    hero("DÉCIDER · APPRENDRE · COMPARER", "La bonne action, au bon moment.",
         "Explorez la manière dont un agent apprend à attendre, surveiller ou isoler un hôte. Comparez ses décisions à une solution exacte et aux stratégies de référence.")
    summary = read_csv("seed_summary.csv")
    config_path = ROOT / "report_assets/run_configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    cols = st.columns(4)
    cols[0].metric("Valeur optimale · Clean", f"{VALUES[0]:.4f}")
    cols[1].metric("Méthodes comparées", str(summary.algorithm.nunique()) if not summary.empty else "0")
    cols[2].metric("Exécutions enregistrées", str(len(summary)))
    cols[3].metric("Épisodes par exécution", f"{config['episodes_per_seed']:,}".replace(",", " ") if "episodes_per_seed" in config else "—")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("La politique optimale")
        cards = "".join(f'<div class="policy-card"><span>{State(state).name.title()}</span><strong>{ACTION_NAMES[Action(int(OPTIMAL[state])).name.lower()]}</strong></div>' for state in range(4))
        st.markdown(f'<div class="policy-grid">{cards}</div><div class="terminal-strip">Fin d’épisode : <b>Secured</b> · hôte sécurisé / <b>Breached</b> · brèche</div>', unsafe_allow_html=True)
        st.caption("Politique calculée par itération de la valeur, γ = 0,9. Secured et Breached sont terminaux.")
    with right:
        st.subheader("Les méthodes face à l’optimum")
        if not summary.empty:
            st.altair_chart(comparison_chart(summary), width="stretch")
            st.caption("Barres : moyenne entre graines · traits blancs : ± un écart-type · ligne dorée : optimum exact.")
        else:
            st.info("Aucune expérience enregistrée. Entraînez un agent dans le laboratoire pour commencer.")
    if not summary.empty:
        decayed = summary[summary.algorithm == "qlearning_decayed"]
        successes = int(np.isclose(decayed.v_clean, VALUES[0], atol=1e-10, rtol=0).sum())
        st.success(f"Q-learning décroissant : politique optimale retrouvée sur {successes}/{len(decayed)} graines enregistrées.")
    st.subheader("À vous de jouer")
    for column, title, description, destination in zip(st.columns(3),
            ["01 · Observer", "02 · Entraîner", "03 · Comparer"],
            ["Suivez un épisode et testez vos propres décisions.", "Ajustez l’exploration et entraînez un nouvel agent.", "Explorez les valeurs, les graines et les figures."],
            ["Simulation", "Laboratoire", "Résultats"]):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
            st.button(f"Ouvrir {destination.lower()}", key=f"open_{destination}", on_click=navigate, args=(destination,), width="stretch")

elif page == "Simulation":
    hero("SIMULATION INTERACTIVE", "Une décision à la fois.", "Choisissez une stratégie, puis suivez l’état de l’hôte, les récompenses et le déroulement d’un épisode reproductible.")
    strategies = {"Politique optimale": OPTIMAL}
    strategies.update({BASELINE_NAMES[name]: policy for name, policy in BASELINES.items()})
    learned = read_csv("learned_policies.csv")
    if not learned.empty:
        for (method, seed), group in learned.groupby(["algorithm", "seed"]):
            actions = group.set_index("state").action
            strategies[f"{METHODS.get(method, method)} · graine {seed}"] = np.array([
                int(Action[actions[State(s).name.lower()].upper()]) for s in range(4)
            ])
    if "training" in st.session_state:
        strategies["Mon dernier agent entraîné"] = st.session_state.training["q"].argmax(axis=1)
    strategy = st.selectbox("Stratégie", ["Décisions manuelles", *strategies])
    c1, c2 = st.columns(2)
    seed = int(c1.number_input("Graine de simulation", min_value=0, max_value=2**32-1, value=42))
    initial = c2.selectbox("État initial", list(State)[:4], format_func=lambda state: state.name.title())

    def new_episode():
        env = ArgusMicroEnv(mdp=MDP, max_episode_steps=200)
        env.reset(seed=seed)
        env.state = int(initial)
        st.session_state.simulation = {"env": env, "trace": [], "done": False, "truncated": False,
                                       "seed": seed, "initial": initial.name.title()}

    if "simulation" not in st.session_state:
        new_episode()
    if st.button("Nouvel épisode", key="new_episode"):
        new_episode()
    sim = st.session_state.simulation
    env = sim["env"]
    st.caption(f"Épisode actif : graine {sim['seed']} · départ {sim['initial']}. Les paramètres de départ s’appliquent au prochain nouvel épisode. La stratégie peut changer à chaque décision.")
    manual = st.selectbox("Action manuelle", list(Action), format_func=lambda a: ACTION_NAMES[a.name.lower()], disabled=strategy != "Décisions manuelles")

    def advance():
        previous = env.state
        action = int(manual) if strategy == "Décisions manuelles" else int(strategies[strategy][previous])
        successor, reward, terminal, truncated, _ = env.step(action)
        sim["trace"].append({"Étape": env.steps, "État": State(previous).name.title(), "Action": Action(action).name.lower(),
                             "État suivant": State(successor).name.title(), "Récompense": reward,
                             "Récompense actualisée": reward * GAMMA ** (env.steps - 1), "Stratégie": strategy})
        sim["done"] = terminal or truncated
        sim["truncated"] = truncated

    b1, b2 = st.columns(2)
    if b1.button("Jouer une étape", key="step", type="primary", disabled=sim["done"], width="stretch"):
        advance()
        st.rerun()
    if b2.button("Terminer avec cette stratégie", key="finish", disabled=sim["done"] or strategy == "Décisions manuelles", width="stretch"):
        while not sim["done"]:
            advance()
        st.rerun()
    metrics = st.columns(3)
    metrics[0].metric("État de l’hôte", State(env.state).name.title())
    metrics[1].metric("Décisions", len(sim["trace"]))
    metrics[2].metric("Retour actualisé · γ = 0,9", f"{sum(row['Récompense actualisée'] for row in sim['trace']):.3f}")
    if sim["done"]:
        if sim["truncated"]:
            st.warning("Limite de 200 décisions atteinte : épisode interrompu, sans résultat terminal.")
        elif env.state == State.SECURED:
            st.success("Épisode terminé : hôte sécurisé.")
        else:
            st.error("Épisode terminé : compromission avec brèche.")
    else:
        action = int(manual) if strategy == "Décisions manuelles" else int(strategies[strategy][env.state])
        with st.expander("Comprendre la prochaine décision", expanded=True):
            st.write(f"Action : **{ACTION_NAMES[Action(action).name.lower()]}**. La transition est tirée dans la distribution ci-dessous.")
            st.dataframe(pd.DataFrame([{"État suivant": State(s).name.title(), "Probabilité": float(p), "Récompense": float(MDP.rewards[env.state, action, s])}
                                      for s, p in enumerate(MDP.transitions[env.state, action]) if p > 0]), hide_index=True, width="stretch")
    if sim["trace"]:
        trace = pd.DataFrame(sim["trace"])
        st.subheader("Journal des décisions")
        st.dataframe(trace, hide_index=True, width="stretch")
        st.line_chart(trace.set_index("Étape")[["Récompense actualisée"]].cumsum(), y_label="Retour actualisé cumulé")
        st.download_button("Exporter cet épisode · CSV", trace.to_csv(index=False).encode("utf-8-sig"), "argus_episode.csv", "text/csv")

elif page == "Laboratoire":
    hero("LABORATOIRE D’APPRENTISSAGE", "Entraînez votre propre agent.", "Ajustez les paramètres, observez l’apprentissage et comparez la politique obtenue à l’optimum exact du modèle.")
    with st.form("training_form"):
        c1, c2 = st.columns(2)
        method = c1.selectbox("Algorithme", list(METHODS), format_func=METHODS.get, index=4)
        episodes = c2.slider("Nombre d’épisodes", 100, 10000, 1000, step=100)
        c1, c2, c3 = st.columns(3)
        seed = int(c1.number_input("Graine d’apprentissage", 0, 2**32-1, 0))
        alpha = c2.slider("Pas initial α", 0.01, 1.0, 0.15, step=0.01)
        epsilon = c3.slider("Exploration ε initiale", 0.02, 1.0, 1.0, step=0.01)
        st.caption("γ = 0,9 · départs uniformes dans les quatre états non terminaux. MC garde ε constant. Les autres méthodes réduisent ε si sa valeur initiale vaut 1 ; sinon ε reste constant, avec un plancher à 0,02.")
        submitted = st.form_submit_button("Lancer l’apprentissage", type="primary")
    if submitted:
        with st.spinner(f"Entraînement : {episodes:,} épisodes…"):
            env = ArgusMicroEnv(mdp=MDP)
            if method == "mc":
                result = mc_control(env, episodes, alpha, GAMMA, epsilon, seed)
            elif method in ("sarsa", "expected_sarsa"):
                result = train_sarsa(env, episodes, alpha, GAMMA, epsilon, seed, True, method == "expected_sarsa")
            else:
                result = train_q_learning(env, episodes, alpha, GAMMA, epsilon, seed, True, method == "qlearning_decayed")
            q, returns, history = result
            q_star = q_from_value(MDP, VALUES)
            st.session_state.training = {"q": q, "returns": returns,
                "errors": [float(np.linalg.norm(estimate - q_star)) for estimate in history],
                "method": method, "episodes": episodes, "seed": seed, "alpha": alpha, "epsilon": epsilon}
    if "training" in st.session_state:
        result = st.session_state.training
        policy = result["q"].argmax(axis=1)
        value = evaluate_policy(MDP, policy)
        st.subheader("Résultat de l’entraînement")
        st.caption(f"{METHODS[result['method']]} · {result['episodes']} épisodes · graine {result['seed']} · α={result['alpha']} · ε initial={result['epsilon']}")
        cols = st.columns(3)
        cols[0].metric("Valeur exacte de la politique", f"{value:.5f}")
        cols[1].metric("Écart à l’optimum", f"{max(0., VALUES[0] - value):.5f}")
        cols[2].metric("Erreur Q · norme", f"{result['errors'][-1]:.3f}")
        st.dataframe(policy_frame(policy), hide_index=True, width="stretch")
        history = pd.DataFrame({"Retour brut": result["returns"], "Erreur Q": result["errors"]}, index=pd.RangeIndex(1, result["episodes"] + 1, name="Épisode"))
        left, right = st.columns(2)
        with left:
            st.caption("Retour brut par épisode · moyenne mobile sur 50 épisodes")
            st.line_chart(history[["Retour brut"]].rolling(50, min_periods=1).mean())
        with right:
            st.caption("Distance à Q* · norme de Frobenius")
            st.line_chart(history[["Erreur Q"]])
        st.caption("Les retours d’apprentissage ne sont pas actualisés et partent d’états variés. La valeur exacte est actualisée depuis Clean.")
        st.download_button("Télécharger l’historique brut · CSV", history.to_csv().encode("utf-8-sig"), "argus_training.csv", "text/csv")
        q_table = pd.DataFrame(result["q"], index=[s.name for s in State], columns=[a.name.lower() for a in Action])
        st.download_button("Télécharger la table Q · CSV", q_table.to_csv().encode("utf-8-sig"), "argus_q_table.csv", "text/csv")
        st.info("Cet agent est disponible dans Simulation → Mon dernier agent entraîné. Retrouvez les expériences enregistrées dans Résultats.")

else:
    hero("ANALYSE DES PERFORMANCES", "Comparez les stratégies de défense.", "Explorez les politiques apprises, leur régularité entre graines et leur distance à la solution optimale.")
    summary = read_csv("seed_summary.csv")
    if summary.empty:
        st.info("Les expériences de référence ne sont pas encore disponibles. Le laboratoire permet déjà d’entraîner un agent.")
    else:
        chosen = st.multiselect("Méthodes à comparer", list(METHODS), default=list(METHODS), format_func=METHODS.get)
        filtered = summary[summary.algorithm.isin(chosen)]
        if not filtered.empty:
            metrics = st.columns(3)
            metrics[0].metric("Exécutions sélectionnées", str(len(filtered)))
            best = filtered.groupby("algorithm").v_clean.mean().idxmax()
            metrics[1].metric("Meilleure moyenne", f"{filtered.groupby('algorithm').v_clean.mean().max():.5f}")
            recovered = int(np.isclose(filtered.v_clean, VALUES[0], rtol=0, atol=1e-10).sum())
            metrics[2].metric("Politiques optimales", f"{recovered}/{len(filtered)}")
            st.caption(f"Meilleure méthode sélectionnée en moyenne : {METHODS.get(best, best)}.")
            st.altair_chart(comparison_chart(filtered), width="stretch")
            st.caption("Moyenne ± un écart-type entre graines. La ligne dorée indique l’optimum exact ; les traits ne sont pas des intervalles de confiance.")
            aggregate = filtered.groupby("algorithm").v_clean.agg(["mean", "std"]).rename(columns={"mean": "Valeur moyenne", "std": "Écart-type"})
            aggregate["Écart moyen à l’optimum"] = VALUES[0] - aggregate["Valeur moyenne"]
            aggregate.index = aggregate.index.map(METHODS)
            st.dataframe(aggregate, width="stretch")
            st.caption("Plus la valeur est proche de zéro, meilleure est la politique. Écart-type entre graines ; référence à γ = 0,9.")
            with st.expander("Détail par graine"):
                st.dataframe(filtered, hide_index=True, width="stretch")
            st.download_button("Exporter la sélection · CSV", filtered.to_csv(index=False).encode("utf-8-sig"), "argus_comparison.csv", "text/csv")
        else:
            st.info("Sélectionnez au moins une méthode pour afficher la comparaison.")
        baselines = read_csv("baseline_values.csv")
        if not baselines.empty:
            with st.expander("Comparer aux stratégies de référence"):
                baseline_table = baselines[["algorithm", "v_clean"]].copy()
                baseline_table["algorithm"] = baseline_table.algorithm.map(BASELINE_NAMES)
                baseline_table.columns = ["Stratégie", "Valeur exacte depuis Clean"]
                st.dataframe(baseline_table, hide_index=True, width="stretch")
        figure_paths = sorted((ROOT / "results/figures").glob("*.png"))
        if figure_paths:
            labels = {"algorithm_comparison": "Comparaison des algorithmes", "convergence_to_qstar": "Convergence vers Q*", "policy_value_by_seed": "Valeur par graine", "qlearning_learning_curve": "Courbes d’apprentissage", "value_estimation_mc_vs_td": "Prédiction MC et TD(0)"}
            figure = st.selectbox("Figure enregistrée", figure_paths, format_func=lambda p: labels.get(p.stem, p.stem))
            st.image(str(figure), width="stretch")
            st.download_button("Télécharger cette figure", figure.read_bytes(), figure.name, "image/png")
    configuration = ROOT / "report_assets/run_configuration.json"
    if configuration.exists():
        with st.expander("Configuration des expériences enregistrées"):
            st.json(json.loads(configuration.read_text(encoding="utf-8")))
