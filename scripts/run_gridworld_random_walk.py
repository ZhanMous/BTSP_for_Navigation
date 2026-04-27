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

out_path = PROJECT_ROOT / "reports" / "figs" / "gridworld_random_walk.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6, 6))
plt.plot(positions[:, 0], positions[:, 1], marker="o", markersize=3, linewidth=1)
plt.scatter(positions[0, 0], positions[0, 1], s=80, label="start")
plt.scatter(env.goal[0], env.goal[1], s=80, marker="*", label="goal")

plt.xlim(-0.5, env.width - 0.5)
plt.ylim(-0.5, env.height - 0.5)
plt.xticks(np.arange(env.width))
plt.yticks(np.arange(env.height))
plt.grid(True, alpha=0.3)
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("GridWorld Random Walk")
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=160)

print("positions shape:", positions.shape)
print(f"Saved figure to {out_path}")
