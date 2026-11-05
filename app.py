# HuggingFace Spaces entry point
# Configured via README.md: app_file: app/streamlit_app.py
# This file exists only as a fallback; the Space uses app/streamlit_app.py directly.
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
runpy.run_path(str(Path(__file__).parent / "app" / "streamlit_app.py"), run_name="__main__")
