import sys
import numpy as np
from pathlib import Path
import polyscope as ps
import igl
import random
from tqdm import tqdm
import argparse
import pandas as pd
import time
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = ROOT / "src" / "python"
UTILITY = ROOT / "utility"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_PYTHON))
sys.path.insert(0, str(UTILITY))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



import contouring
import utility
import gpytoolbox as gpy
import numpy as np
import os
import definitions
