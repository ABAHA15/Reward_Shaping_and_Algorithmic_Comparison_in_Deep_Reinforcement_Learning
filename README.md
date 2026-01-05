Reward Shaping and Algorithmic Comparison in Deep Reinforcement Learning


📌 Project Overview:

This repository contains the complete implementation and experimental study for the project “Reward Shaping and Algorithmic Comparison in Deep Reinforcement Learning”, conducted as part of an academic course on Reinforcement Learning.

The project is divided into two major parts:

Part 1: Comparative analysis of multiple Deep Reinforcement Learning (DRL) algorithms on a standard benchmark environment (CartPole-v1), with a focused study on the impact of reward shaping.

Part 2: Design and evaluation of custom Gym-compatible environments with graphical user interfaces (GUI), highlighting agent behavior in deterministic and dynamic settings.

The project emphasizes understanding, interpretation, and critical analysis, rather than blind application of libraries.


🎯 Objectives:

Compare value-based, policy-gradient, and actor–critic algorithms under identical conditions.
Analyze how reward shaping affects learning speed, stability, and convergence.
Design custom non-trivial environments that require genuine decision-making.
Demonstrate learned policies through visual rollouts and GUI-based demos.
Analyze both successes and failures of Deep RL algorithms.


🧠 Algorithms Implemented:

Part 1 – Benchmark Environment (CartPole-v1):

DQN
Double DQN
REINFORCE (from scratch, PyTorch)
Advantage Actor–Critic (A2C)
Proximal Policy Optimization (PPO)
Each algorithm is trained using:
Baseline rewards
Shaped rewards
Learning curves are generated and analyzed for all configurations.


Part 2 – Custom Environments:

GridWorld (5×5) with fixed obstacles
DQN with sparse rewards
DQN with shaped rewards
Custom GUI visualization
Pacman-style dynamic environment
Moving adversary (ghost)
Stochastic dynamics
PPO agent with adaptive behavior
Custom GUI visualization



🏗️ Repository Structure:
.
├── cartpole/
│   ├── train_dqn_baseline.py
│   ├── train_dqn_shaped.py
│   ├── train_double_dqn_baseline.py
│   ├── train_double_dqn_shaped.py
│   ├── train_reinforce_baseline.py
│   ├── train_reinforce_shaped.py
│   ├── train_a2c.py
│   ├── train_ppo.py
│   └── plots/
│       ├── baseline_reward_curve_*.png
│       └── shaped_reward_curve_*.png
│
├── gridworld/
│   ├── gridworld_env.py
│   ├── train_dqn_sparse.py
│   ├── train_dqn_shaped.py
│   ├── gridworld_gui.py
│   └── plots/
│       ├── gridworld_sparse_reward_dqn.png
│       └── gridworld_shaped_reward_dqn.png
│
├── pacman_env/
│   ├── pacman_env.py
│   ├── train_ppo_pacman.py
│   ├── pacman_gui.py
│   └── plots/
│       └── reward_curve_ppo_pacman.png
│
├── models/
│   ├── dqn_cartpole.zip
│   ├── double_dqn_cartpole.zip
│   ├── ppo_cartpole.zip
│   └── ppo_pacman.zip
│
├── report/
│   ├── Project_Report.pdf
│   └── latex_source/
│
├── README.md
└── requirements.txt


⚙️ Setup Instructions:


1️⃣ Create Virtual Environment

python -m venv rl_env
rl_env\Scripts\activate

2️⃣ Install Dependencies

pip install -r requirements.txt

Key Libraries Used:

Python 3.9
Gymnasium
Stable-Baselines3
PyTorch
NumPy
Matplotlib
Pygame (for GUI)

▶️ Running the Experiments:

CartPole (Example)
python train_double_dqn_baseline.py
python train_double_dqn_shaped.py

GridWorld GUI
python gridworld_gui.py

Pacman-Style Environment GUI
python pacman_gui.py


Pre-trained models are provided to enable real-time demos without retraining.


📊 Results and Analysis:

Learning curves compare baseline vs shaped rewards.
Sample efficiency, stability, and convergence are analyzed.
Failures (instability, suboptimal convergence, conservative policies) are explicitly discussed.
GUI rollouts provide qualitative insight into learned behavior.
For detailed analysis, refer to the Project Report (PDF) in the report/ directory.


🎥 Demos:

The project includes:

Saved trained models for live demos.
GUI-based visualizations for custom environments.
Video recordings of trained agents (for presentation use).


📌 Key Takeaways:

Reward shaping improves early learning but may introduce bias.
PPO provides the most stable performance across environments.
No single RL algorithm is universally optimal.
Custom environment design is as important as algorithm choice.


🚀 Future Extensions:

Continuous control (SAC, TD3)
Partial observability and memory-based agents
Multi-agent extensions
Automated reward design
Robustness and generalization studies


📜 License:

This project is intended for academic and educational use.

