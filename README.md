# QMaze: Reinforcement Learning with Rust Optimization

A Q-learning agent that learns to navigate a complex maze environment featuring minotaurs, traps, and teleportation portals. This project demonstrates reinforcement learning fundamentals and performance optimization through Rust integration.

## Project Overview

The agent must navigate a 15×15 maze to reach the goal while:
- **Avoiding or defeating minotaurs** (each with unique strength/keenness stats)
- **Evading traps** that damage health
- **Strategically using portals** for fast travel
- **Operating under partial observability** (5×5 visible window)

### Environment Features

- **Grid**: 15×15 maze with fixed wall layout
- **Enemies**: 3-5 minotaurs per episode (randomly placed)
  - M1: High strength (0.7), Low keenness (0.2)
  - M2: Balanced (0.5 strength, 0.5 keenness)
  - M3: Low strength (0.3), High keenness (0.8)
- **Obstacles**: 5-10 randomly placed traps
- **Portals**: 2-4 color-coded portal pairs (red, blue, green)
- **Health System**: Full → Injured → Critical → Defeat
- **Actions**: UP, DOWN, LEFT, RIGHT, FIGHT, HIDE, HEAL, WAIT, ENTER_PORTAL

### Learning Algorithm

**Q-Learning with:**
- Epsilon-greedy exploration strategy
- Adaptive learning rate: η = 1/(1 + updates)
- Discount factor: γ = 0.9
- State space hashing for efficient tabular representation

## Key Features

### 1. Partial Observability
Agent only sees 5×5 window around current position, requiring it to learn general navigation strategies rather than memorizing specific paths.

### 2. State Hashing
Complex observations (5×5 grid + health + minotaur identity) encoded into unique integers using base-9 encoding:
- 25 cells × 9 possible states per cell
- 3 health levels
- 6 minotaur identity states
- **Theoretical state space: ~141 quadrillion states**
- **Practical encountered states: 50K-100K**

### 3. Rust Optimization
Critical performance bottleneck (state hashing) reimplemented in Rust using PyO3:
- **27% faster training** (100K episodes: 512s → 404s)
- Hash function called millions of times during training
- Demonstrates Python-Rust interoperability

## Project Structure
```
QMaze/
├── maze.py                 # Fixed maze layout generation
├── maze_env.py            # Gymnasium environment implementation
├── state_hash.py          # Python state encoding
├── q_learning.py          # Q-learning algorithm & evaluation
├── state_hash_rust/       # Rust optimization module
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs         # Rust hash function (PyO3)
├── rewards_plot_*.png     # Training reward curves
├── action_heatmap_*.png   # Learned action distributions
└── Q_table_*.pickle       # Trained Q-tables
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Rust 1.70+ (for optimization module)

### Install Python Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install gymnasium numpy matplotlib pygame seaborn tqdm maturin
```

### Build Rust Module (Optional)
```bash
cd state_hash_rust
maturin develop --release
cd ..
```

## Usage

### Training
```bash
# Configure in q_learning.py:
num_episodes = 100000
decay_rate = 0.99999

# Train agent
python q_learning.py train
```

Outputs:
- `Q_table_{episodes}_{decay}.pickle` - Trained Q-table
- `rewards_plot_{episodes}_{decay}.png` - Training progress

### Evaluation
```bash
# Runs 10,000 test episodes
python q_learning.py
```

Outputs:
- Performance metrics (avg reward, episode length, etc.)
- `action_heatmap_{episodes}_{decay}.png` - Action distributions

### Manual Play
```bash
python vis_gym.py
```
Controls: W/A/S/D (move), H (hide), F (fight), E (heal), Space (wait), R (reset)

## Performance Optimization

### Profiling Results (1,000 episodes)
```
get_observation():     36% of time (2.64s)
hash_observation():    17% of time (1.21s)
move_minotaurs():       9% of time (0.68s)
Other:                 38% of time (2.78s)
```

### Rust Integration Impact
- **Micro-benchmark**: 1.1x speedup on hash function alone
- **Training benchmark** (100K episodes): 1.27x overall speedup
- **Time saved** (1M episodes): ~18 minutes

### Why Rust Helps
- Hash function called millions of times during training
- Python → Rust: Dictionary operations → compiled code
- Eliminates Python interpreter overhead for critical loop

## Technical Details

### State Encoding
```python
state_id = health * HEALTH_SPACE + minotaur_id * MINOTAUR_SPACE + window_hash

where:
  WINDOW_SPACE = 9^25
  MINOTAUR_SPACE = WINDOW_SPACE
  HEALTH_SPACE = MINOTAUR_SPACE * 6
```

### Q-Learning Update
```python
η = 1 / (1 + num_updates[s][a])
Q_new(s,a) = (1-η)Q_old(s,a) + η[R(s,a,s') + γ max Q_old(s',a')]
```

### Epsilon Decay
```python
epsilon = epsilon * decay_rate  # After each episode
```

## Learning Outcomes

This project demonstrates:
1. **Reinforcement Learning**: Q-learning in partially observable environment
2. **State Space Management**: Hashing for tractable tabular RL
3. **Exploration/Exploitation**: Epsilon-greedy with decay schedules
4. **Performance Optimization**: Profiling and Rust integration via PyO3
5. **Experimental Design**: Systematic hyperparameter evaluation

## Future Improvements

- **Deep Q-Network (DQN)**: Replace tabular Q-learning with neural network
- **Optimize get_observation()**: Rewrite in Rust (36% of runtime)
- **Multi-agent scenarios**: Multiple cooperative/competitive agents
- **Procedural maze generation**: Random mazes each episode
- **Curriculum learning**: Progressive difficulty increase

## Acknowledgments

- Environment design inspired by "Escape the Castle" CS assignment
- Rust-Python integration using PyO3
- Visualization with Pygame and Matplotlib

---

**Built with:** Python 3.13 | Rust 1.70 | PyO3 | Gymnasium | NumPy | Matplotlib
