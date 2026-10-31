"""Shared locations. Override any of them with the environment variable of the same name."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# extracted per-case scores (output of step 01; input to steps 02-05 and 07)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
# every JSON result the scripts emit (created on first run)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
# a clone of https://github.com/phamquiluan/RCAEval (needed by steps 01 and 06 only)
RCAEVAL_REPO = os.environ.get("RCAEVAL_REPO", os.path.join(ROOT, "RCAEval"))
# directory holding the RE1-OB, RE1-SS and RE1-TT dataset folders (steps 01 and 06 only);
# the default is where RCAEval's download_re1_dataset("datasets") puts them
RE1_ROOT = os.environ.get("RE1_ROOT", os.path.join(ROOT, "datasets", "RE1"))

os.makedirs(RESULTS_DIR, exist_ok=True)
