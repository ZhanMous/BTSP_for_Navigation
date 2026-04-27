import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from btsp_nav import GridWorld

env = GridWorld(width=10, height=10, seed=0)

pos = env.reset()
positions = [pos]

for t in range(200):
    action = env.rng.integers(0, 5)
    pos, reward, done = env.step(action)
    positions.append(pos)

positions = np.array(positions)