import sys
import numpy as np
from pathlib import Path
import igl 

ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "SRC" / "python"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_PYTHON))

import contouring