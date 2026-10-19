@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Environnement Python absent. Consultez README.md pour l'installation.
    pause
    exit /b 1
)
echo ARGUS : ouvrez http://127.0.0.1:8501 dans votre navigateur.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
