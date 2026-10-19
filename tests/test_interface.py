from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_and_report_navigation():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.button(key="open_Résultats").click().run()
    assert not app.exception
    assert len(app.dataframe) >= 1
    assert not app.text_area
    assert all("rapport" not in option.lower() for option in app.sidebar.radio[0].options)
    assert not any("Préparer le rapport" in heading.value for heading in app.subheader)
    app.multiselect[0].set_value([]).run()
    assert not app.exception


def test_simulation_manual_step_and_seeded_replay():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Simulation").run()
    app.button(key="step").click().run()
    assert not app.exception
    assert len(app.session_state["simulation"]["trace"]) == 1
    app.selectbox[0].set_value("Politique optimale").run()
    app.button(key="new_episode").click().run()
    app.button(key="finish").click().run()
    assert not app.exception
    first = list(app.session_state["simulation"]["trace"])
    assert app.session_state["simulation"]["done"]
    assert app.button(key="step").disabled
    app.button(key="new_episode").click().run()
    app.button(key="finish").click().run()
    assert not app.exception
    assert app.session_state["simulation"]["trace"] == first


def test_training_agent_is_available_in_simulation():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Laboratoire").run()
    app.slider[0].set_value(100)
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["training"]
    assert result["q"].shape == (6, 3)
    assert np.isfinite(result["q"]).all()
    assert len(result["returns"]) == 100
    app.sidebar.radio[0].set_value("Simulation").run()
    assert "Mon dernier agent entraîné" in app.selectbox[0].options
    app.selectbox[0].set_value("Mon dernier agent entraîné").run()
    app.button(key="step").click().run()
    assert not app.exception
