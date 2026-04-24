import os
import sys
from pathlib import Path

import numpy as np

print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("cwd:", os.getcwd())
print("fig dir exists:", Path("reports/fig").exists())