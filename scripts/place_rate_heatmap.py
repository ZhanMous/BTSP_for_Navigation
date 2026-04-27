import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def gaussian_place_rates(x, centers, sigma, r_max):

    return r_max * np.exp(
        -0.5 * ((x[:, None] - centers[None, :]) / sigma) ** 2
        )

T = 1000
x = np.linspace(0, 1, T)

N = 50
centers = np.linspace(0, 1, N)

sigma = 0.08
r_max = 20.0

rates = gaussian_place_rates(x, centers, sigma, r_max)

print(rates.shape)

plt.figure(figsize=(10, 6))

plt.imshow(rates, aspect='auto', origin='lower', cmap='viridis')

plt.colorbar(label='Firing Rate')
plt.xlabel('Neuron Index')
plt.ylabel('Time Index')
plt.title('Gaussian Place Cell Firing Rates')

out_path = Path(__file__).resolve().parents[1] / "reports" / "figs" / "place_rate_heatmap.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"Saved figure to {out_path}")
