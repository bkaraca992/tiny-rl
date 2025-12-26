"""
Train script for the tiny GridWorld with Q-learning.

Example:
python src/train.py --episodes 2000 --width 6 --height 6
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridWorld
from agents.q_learning import QLearningAgent

def train(env, agent, episodes=1000, max_steps_per_episode=50, render_interval=0):
    rewards = []
    for ep in range(1, episodes+1):
        state = env.reset()
        total = 0.0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total += reward
            if done:
                break
        rewards.append(total)
        # Decay epsilon slowly
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        if render_interval and ep % render_interval == 0:
            print(f"Ep {ep}/{episodes}: reward={total:.2f} eps={agent.epsilon:.3f}")
    return rewards

def plot_learning_curve(rewards, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6,4))
    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid') if len(rewards) >= 50 else rewards
    plt.plot(rewards, alpha=0.3, label='episodic reward')
    plt.plot(range(len(smoothed)), smoothed, label='smoothed (50)', color='C1')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_policy_text(agent, env, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    action_map = {0:'^', 1:'>', 2:'v', 3:'<'}
    lines = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            s = x + y * env.width
            if (x,y) in env.obstacles:
                row.append(' X ')
            elif (x,y) == env.goal:
                row.append(' G ')
            elif (x,y) == env.start:
                row.append(' S ')
            else:
                a = agent.select_action(s, greedy=True)
                row.append(f' {action_map[a]} ')
        lines.append("".join(row))
    # Write grid rows (y from top to bottom)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[::-1]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--height", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    env = GridWorld(width=args.width, height=args.height, max_steps=args.max_steps, start=(0,0), goal=(args.width-1, args.height-1))
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, lr=0.1, gamma=0.99, epsilon=0.3, seed=args.seed)

    rewards = train(env, agent, episodes=args.episodes, max_steps_per_episode=args.max_steps, render_interval= max(1, args.episodes//10))

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    plot_learning_curve(rewards, os.path.join(out_dir, "learning_curve.png"))
    agent.save(os.path.join(out_dir, "q_table.npy"))
    save_policy_text(agent, env, os.path.join(out_dir, "final_policy.txt"))
    print("Training finished. Outputs saved to outputs/")

if __name__ == "__main__":
    main()
