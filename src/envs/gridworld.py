"""
A tiny GridWorld environment (no external RL deps).

State: agent position on a W x H grid.
Actions: 0=up, 1=right, 2=down, 3=left
Reward: -1 per step, +10 at goal.
Episode ends when agent reaches goal or after max_steps.
"""
from typing import Tuple, List
import numpy as np

class GridWorld:
    def __init__(self, width: int = 5, height: int = 5, start=(0,0), goal=None, max_steps: int = 50, obstacles: List[Tuple[int,int]] = None):
        self.width = width
        self.height = height
        self.start = tuple(start)
        self.goal = tuple(goal) if goal else (width-1, height-1)
        self.max_steps = max_steps
        self.obstacles = set(obstacles or [])
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # Map (x,y) -> state id
        x, y = self.pos
        return x + y * self.width

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and (x,y) not in self.obstacles

    def step(self, action: int):
        # Actions: 0=up,1=right,2=down,3=left
        x, y = self.pos
        if action == 0:
            nx, ny = x, y-1
        elif action == 1:
            nx, ny = x+1, y
        elif action == 2:
            nx, ny = x, y+1
        elif action == 3:
            nx, ny = x-1, y
        else:
            nx, ny = x, y

        if self._in_bounds(nx, ny):
            self.pos = (nx, ny)  # move
        # else: invalid move -> stay in place

        self.steps += 1
        done = False
        reward = -1.0
        if self.pos == self.goal:
            done = True
            reward = 10.0
        elif self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {"pos": self.pos}

    @property
    def n_states(self):
        return self.width * self.height

    @property
    def n_actions(self):
        return 4

    def render_ascii(self):
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]
        gx, gy = self.goal
        sx, sy = self.start
        px, py = self.pos
        grid[gy][gx] = " G"
        grid[sy][sx] = " S"
        grid[py][px] = " A"
        for (ox, oy) in self.obstacles:
            grid[oy][ox] = " X"
        lines = ["".join(row) for row in grid]
        # Print from top row (y=0 at top)
        return "\n".join(lines)
