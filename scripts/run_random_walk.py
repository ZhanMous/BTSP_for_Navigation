from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(39)
T = 1000
dt = 0.02
x= np.zeros(T)

for t in range(1, T):
    x[t] = x[t-1] + rng.normal(0.0, 0.05)

time = np.arange(T) * dt
Path("reports/figs").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 3))
plt.plot(time, x)
plt.xlabel("time (s)")
plt.ylabel("x")
plt.title("Random Walk")
plt.tight_layout()
plt.savefig("reports/figs/random_walk.png", dpi=160)