import numpy as np

class GridWorld:
    def __init__(self, width = 10, height = 10, seed = 0):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.pos = np.array([0, 0], dtype=int)
        self.goal = np.array([self.width - 1, self.height - 1], dtype=int)
        return self.pos.copy()
    
    def step(self, action):
        moves = {
            0: np.array([0, 0]),
            1: np.array([0, 1]),   # up
            2: np.array([0, -1]),  # down
            3: np.array([-1, 0]),  # left
            4: np.array([1, 0])    # right
        }
        new_pos = self.pos + moves[action]
        new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)
        self.pos = new_pos
        done = bool(np.array_equal(self.pos, self.goal))
        reward = 1.0 if done else -0.01
        return self.pos.copy(), reward, done