import sys
from pathlib import Path

# Make the repository root importable (analysis/, hypotheses/, experiments/, ...)
# regardless of the directory pytest is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
