"""Filesystem layout of the repository. Override any location with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("CHRONOS5G_ROOT", Path(__file__).resolve().parent))
GOTSF_DIR = Path(os.environ.get("GOTSF_DIR", ROOT / "data" / "gotsf"))    # contains train/ and test/
IRISH_DIR = Path(os.environ.get("IRISH_DIR", ROOT / "data" / "irish"))    # unzipped 5G-production-dataset
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
FIG_DIR = Path(os.environ.get("FIG_DIR", ROOT / "figures" / "out"))
