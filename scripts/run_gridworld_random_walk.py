import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/btsp_nav_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from btsp_nav.envs import GridWorld


def run_random_episode(env, max_steps):
    pos = env.reset()
    trajectory = [pos]

    for step in range(1, max_steps + 1):
        action = env.rng.integers(0, 5)
        pos, reward, done = env.step(action)
        trajectory.append(pos)

        if done:
            return True, step, np.array(trajectory)

    return False, max_steps, np.array(trajectory)


def evaluate_random_policy(num_episodes=100, max_steps=300):
    successes = []
    steps = []

    for episode in range(num_episodes):
        env = GridWorld(width=10, height=10, seed=episode)
        success, step_count, trajectory = run_random_episode(env, max_steps)
        successes.append(success)
        steps.append(step_count)

    successes = np.array(successes)
    steps = np.array(steps)
    success_steps = steps[successes]

    success_rate = successes.mean()
    avg_steps_success = success_steps.mean() if len(success_steps) > 0 else np.nan
    avg_steps_all = steps.mean()

    return success_rate, avg_steps_success, avg_steps_all


def plot_trajectory(trajectory, env, out_path):
    plt.figure(figsize=(6, 6))
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="o",
        markersize=3,
        linewidth=1,
        label="trajectory",
    )
    plt.scatter(trajectory[0, 0], trajectory[0, 1], s=80, label="start")
    plt.scatter(env.goal[0], env.goal[1], s=120, marker="*", label="goal")

    plt.xlim(-0.5, env.width - 0.5)
    plt.ylim(-0.5, env.height - 0.5)
    plt.xticks(np.arange(env.width))
    plt.yticks(np.arange(env.height))
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("GridWorld Random Policy Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


num_episodes = 100
max_steps = 300

success_rate, avg_steps_success, avg_steps_all = evaluate_random_policy(
    num_episodes=num_episodes,
    max_steps=max_steps,
)

figure_dir = PROJECT_ROOT / "reports" / "figs"
figure_dir.mkdir(parents=True, exist_ok=True)
out_path = figure_dir / "gridworld_trajectory.png"

plot_env = GridWorld(width=10, height=10, seed=0)
success, step_count, trajectory = run_random_episode(plot_env, max_steps)
plot_trajectory(trajectory, plot_env, out_path)

print("| episodes | max_steps | success_rate | avg_steps_success | avg_steps_all |")
print("|---:|---:|---:|---:|---:|")
print(
    f"| {num_episodes} | {max_steps} | {success_rate:.2f} | "
    f"{avg_steps_success:.1f} | {avg_steps_all:.1f} |"
)
print(f"trajectory_success: {success}")
print(f"trajectory_steps: {step_count}")
print(f"saved: {out_path}")
