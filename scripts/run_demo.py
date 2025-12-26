"""
Run a single demo episode using the saved Q-table (if available),
otherwise run random actions for a short demo.
"""
import os
import time
import numpy as np
from envs.gridworld import GridWorld
from agents.q_learning import QLearningAgent

def run():
    env = GridWorld(width=5, height=5, max_steps=50)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, epsilon=0.0)  # greedy by default
    qpath = "outputs/q_table.npy"
    if os.path.exists(qpath):
        agent.load(qpath)
        print("Loaded Q-table from", qpath)
    else:
        print("No Q-table found, running random policy demo (try training first).")

    state = env.reset()
    print(env.render_ascii())
    time.sleep(0.5)
    for _ in range(env.max_steps):
        action = agent.select_action(state, greedy=True) if os.path.exists(qpath) else agent.rng.randint(env.n_actions)
        next_state, reward, done, info = env.step(action)
        state = next_state
        print(env.render_ascii())
        print(f"Action: {action} Reward: {reward}")
        time.sleep(0.25)
        if done:
            print("Episode finished. Reward:", reward)
            break

if __name__ == "__main__":
    run()
