"""Install the Streamlit workspace dependencies in the active environment."""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    requirements = Path(__file__).with_name("requirements.txt")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True,
    )
    print("Run python -m streamlit run app.py to open the sample workspace.")
