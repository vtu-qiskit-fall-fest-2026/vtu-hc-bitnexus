@echo off
title Q-Rescue Next - Disaster Relief Planner
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo ======================================================================
echo   🚨 Q-RESCUE NEXT: DISASTER RELIEF PLANNER
echo ======================================================================
echo   • Starting local server on port 8501 (UTF-8 Mode)...
echo   • Local browser URL:   http://localhost:8501
echo   • Mobile Wi-Fi URL:    Check sidebar in app
echo ======================================================================
".venv\Scripts\python.exe" -X utf8 -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
