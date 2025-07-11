@echo off
echo.
echo 🎵 Starting Music Genre Classification Web App...
echo.
echo Make sure you have:
echo ✓ Installed requirements: pip install -r requirements.txt
echo ✓ Trained at least one model in models/ directory
echo.
echo The app will open in your default browser at http://localhost:8501
echo.
pause
echo.
echo Starting Streamlit...
echo Current directory: %CD%
echo App directory: %~dp0
cd /d "%~dp0"
echo Changed to: %CD%
streamlit run app.py
