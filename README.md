# tiny-rl — Tiny Reinforcement Learning Playground

[![Status: Prototype](https://img.shields.io/badge/status-prototype-orange.svg)](./README.md) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]

One-line: A compact, self-contained RL playground that demonstrates tabular Q-learning on a tiny GridWorld — easy to run, inspect, and extend.

Project status: Prototype — educational demo. Designed to be lightweight and runnable in minutes. Not intended as production RL code or a research baseline.

Quick start (5–10 minutes)
1. Clone the repo and enter the folder:
   ```bash
   git clone https://github.com/<your-user>/tiny-rl.git
   cd tiny-rl
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train a Q-learning agent (quick run with default small sizes):
   ```bash
   python src/train.py --episodes 1000
   ```
   - After training you'll find `outputs/learning_curve.png` and `outputs/final_policy.txt`.

4. Run a demo episode using the learned policy:
   ```bash
   python scripts/run_demo.py
   ```

What you get
- Minimal GridWorld environment (no Gym dependency).
- Tabular Q-learning agent with epsilon-greedy exploration.
- Training script that logs rewards and saves a learning curve + final policy.
- Simple ASCII + matplotlib visualizations to include in README or portfolio.

Why this repo
- Quick to run in interviews or demo screenshares.
- Easy to extend: swap in different environments, policies (SARSA, DQN), or add visualization.
- Honest, educational: the README and code explain trade-offs and limitations.

Files & folders
- src/ — environment, agent, and training script
- scripts/ — small helper to run a saved policy
- notebooks/ — experiment notes
- outputs/ — generated plots and policy files (gitignored)
- requirements.txt, Dockerfile, .github/workflows/ci.yml, LICENSE

Notes & next steps
- Want a version that uses OpenAI Gym + Stable-Baselines3 (DQN / PPO)? I can add that as an optional branch.
- Want a one-line gh/git push script to create a public repo under your account? Tell me the repo name and I’ll produce exact commands.

License
-------
MIT © 2025 <Your Name>
