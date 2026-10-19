"""Capture the running local Streamlit app using Playwright and installed Edge.

Install the optional capture tool with `python -m pip install playwright`, start
the application, then run `python scripts/capture_interface.py`.
"""

from pathlib import Path
import json
import math
import zipfile

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report_assets" / "screenshots"
OUTPUT.mkdir(parents=True, exist_ok=True)


def capture(page, filename):
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(1200)
    container = page.locator('[data-testid="stMainBlockContainer"]')
    height = math.ceil(container.bounding_box()["height"]) + 24
    page.set_viewport_size({"width": 1600, "height": max(1000, height)})
    page.locator('[data-testid="stMain"]').evaluate("element => element.scrollTop = 0")
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUTPUT / filename), full_page=True, animations="disabled")
    page.set_viewport_size({"width": 1600, "height": 1100})


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1.25)
    page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
    page.get_by_text("La bonne action, au bon moment.", exact=True).wait_for(timeout=30000)
    capture(page, "01_vue_ensemble.png")

    page.get_by_test_id("stSidebar").get_by_text("Simulation", exact=True).click()
    page.get_by_text("Une décision à la fois.", exact=True).wait_for()
    page.get_by_role("combobox").nth(0).click()
    page.get_by_role("option", name="Politique optimale", exact=True).click()
    page.get_by_role("combobox").nth(1).click()
    page.get_by_role("option", name="Suspected", exact=True).click()
    page.get_by_role("button", name="Nouvel épisode", exact=True).click()
    for step in range(1, 5):
        button = page.get_by_role("button", name="Jouer une étape", exact=True)
        if not button.is_enabled():
            break
        button.click()
        expect(page.locator('[data-testid="stMetricValue"]').nth(1)).to_have_text(str(step))
    capture(page, "02_simulation_episode.png")

    page.get_by_test_id("stSidebar").get_by_text("Laboratoire", exact=True).click()
    page.get_by_role("button", name="Lancer l’apprentissage", exact=True).click()
    page.get_by_text("Résultat de l’entraînement", exact=True).wait_for(timeout=120000)
    page.get_by_text("Cet agent est disponible dans Simulation", exact=False).wait_for()
    capture(page, "03_laboratoire_apprentissage.png")
    metadata = {"simulation_seed": 42, "simulation_policy": "optimal", "simulation_initial_state": "Suspected",
                "training_method": "qlearning_decayed", "training_seed": 0, "training_episodes": 1000,
                "alpha": 0.15, "epsilon_initial": 1.0, "gamma": 0.9,
                "training_metrics": page.locator('[data-testid="stMetricValue"]').all_text_contents(),
                "reference_experiments": "25 runs, 25000 episodes per run; saved reference CSVs unchanged"}

    page.get_by_test_id("stSidebar").get_by_text("Résultats", exact=True).click()
    page.get_by_text("Comparez les stratégies de défense.", exact=True).wait_for()
    page.locator('[data-testid="stImage"] img').wait_for()
    capture(page, "04_comparaison_resultats.png")
    (OUTPUT / "capture_parameters.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    browser.close()

with zipfile.ZipFile(OUTPUT.parent / "ARGUS_captures_rapport.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(OUTPUT.iterdir()):
        if file.is_file():
            archive.write(file, file.name)
print(f"Captures enregistrées dans {OUTPUT}")
