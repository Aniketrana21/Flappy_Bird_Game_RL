# 🐦 Flappy Bird RL Agent (Double Deep Q-Network)

An autonomous reinforcement learning agent that learns to play **Flappy Bird** using **Double Deep Q-Networks (DDQN)** with PyTorch and [flappy-bird-gymnasium](https://github.com/Talilar/flappy-bird-gymnasium).

---

## 📌 Overview

This project implements a Deep Q-Learning agent capable of achieving high scores (crossing 60+ pipes) in the classic Flappy Bird environment. By leveraging a **Double DQN (DDQN)** architecture, the agent overcomes Q-value overestimation bias and stably navigates through pipe gaps.

### Key Highlights
- **Double DQN (DDQN)**: Decouples action selection from action evaluation to eliminate maximization bias.
- **Soft Target Updates (Polyak Averaging)**: Stable and smooth target network parameter updates ($\tau = 0.005$).
- **Experience Replay**: Stores up to 100,000 transitions in a FIFO memory buffer for decorrelated mini-batch training.
- **Layer Normalization & Kaiming Init**: 3-hidden-layer MLP with LayerNorm and Kaiming normal initialization for robust gradient flow.
- **Adaptive Exploration & Learning**: Epsilon-greedy schedule (decaying over 3,000 episodes) paired with PyTorch's `CosineAnnealingLR` scheduler.
- **Intelligent Checkpointing**: Resumes automatically from checkpoints, tracking and saving models for best 100-episode rolling average, best single episode reward, and best pipe count record.
- **Live Metrics**: Real-time logging of rewards, pipes crossed, loss, epsilon, and learning rate via **TensorBoard**.

---

## 📁 Project Structure

```text
Flappy_Bird_game/
├── agent.py               # Main agent script (training, evaluation, checkpointing, DDQN loop)
├── dqn.py                 # PyTorch neural network architecture (DQN with LayerNorm)
├── Experience_replay.py   # Experience replay buffer (FIFO queue with random sampling)
├── flappy_game.py         # Playable human mode using Pygame & Gymnasium
├── parameters.yaml        # Hyperparameter configuration file
├── test_env.py            # Diagnostic script to test environment observations and pipe scoring
├── runs/                  # Saved model checkpoints (.pt), logs, and TensorBoard summaries
└── README.md              # Project documentation
```

---

## 🧠 Model Architecture & RL Formulation

### State Space (12 Continuous Features)
The environment provides a 12-dimensional vector containing relative positions and velocities:
- Bird's vertical position and velocity
- Horizontal and vertical distances to the next upper and lower pipes
- Distance to the next-next pair of pipes

### Action Space (Discrete)
- `0`: Idle / Do nothing (bird falls under gravity)
- `1`: Flap (upward thrust)

### Network Architecture (`dqn.py`)
```text
Input (12) ──> Linear(128) ──> LayerNorm ──> ReLU
           ──> Linear(128) ──> LayerNorm ──> ReLU
           ──> Linear(128) ──> LayerNorm ──> ReLU
           ──> Linear(2)   ──> Q(s, a)
```

### Loss Function
Huber Loss (`nn.SmoothL1Loss`) with gradient clipping (`max_norm = 1.0`):
$$\mathcal{L}_{Huber}(Q(s, a), y)$$
where the target $y$ is computed via Double DQN:
$$y = r + \gamma (1 - d) \, Q_{\text{target}}\left(s', \arg\max_{a'} Q_{\text{policy}}(s', a')\right)$$

---

## ⚙️ Hyperparameters (`parameters.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `env_id` | `FlappyBird-v0` | Gymnasium environment identifier |
| `replay_memeory_size` | `100,000` | Maximum capacity of experience replay buffer |
| `mini_batch_size` | `128` | Number of transition samples per training step |
| `alpha` (Initial LR) | `0.0003` | Adam optimizer initial learning rate |
| `gamma` (Discount factor) | `0.99` | Discount factor for future rewards |
| `epsilon_init` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.01` | Minimum exploration rate |
| `epsilon_decay_episodes` | `3,000` | Episodes across which epsilon linearly decays |
| `learning_start` | `1,000` | Steps in replay memory before training starts |
| `train_every` | `4` | Number of environment steps between training updates |
| `tau` | `0.005` | Target network soft update parameter (Polyak averaging) |

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```bash
# Clone the repository and enter the directory
cd Flappy_Bird_game

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Training the Agent

To start or resume training using the settings in `parameters.yaml`:

```bash
python agent.py flappybirdv0 --train
```

- If a checkpoint (`runs/flappybirdv0.pt`) exists, the agent automatically loads weights, optimizer states, epsilon, and continues training.
- Checkpoints saved in `runs/`:
  - `flappybirdv0.pt`: Best 100-episode rolling average model
  - `flappybirdv0_best_episode.pt`: Highest single-episode reward model
  - `flappybirdv0_best_pipes.pt`: Highest pipe-count record model

### 3. Evaluating / Watching the Trained Agent

Run the agent in inference mode with human rendering:

```bash
python agent.py flappybirdv0
```

### 4. Interactive Human Play

Test the environment yourself using the spacebar to flap:

```bash
python flappy_game.py
```

### 5. Monitoring with TensorBoard

Track training metrics (reward curve, 100-episode moving average, pipes passed, loss, learning rate decay):

```bash
tensorboard --logdir runs
```
Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 📊 Benchmark & Performance

- **Peak Pipe Record**: **64 pipes crossed** in a single episode (Episode Reward: `302.40`).
- **Rolling Average (100 episodes)**: Over **44+** average reward at convergence.
- **Exploration Schedule**: Smooth transition from purely random actions ($\epsilon = 1.0$) to high-precision flight ($\epsilon = 0.01$).

---

## 📜 References & Acknowledgments
- [Gymnasium](https://gymnasium.farama.org/)
- [flappy-bird-gymnasium](https://github.com/Talilar/flappy-bird-gymnasium)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2015)](https://arxiv.org/abs/1509.06461)
