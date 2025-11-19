import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from maze_env import MazeEscapeEnv
from state_hash import hash_observation

try:
    from state_hash_rust import hash_observation_rust as hash_observation
    print("✅ Using Rust hash function!")
except ImportError:
    from state_hash import hash_observation
    print("⚠️ Using Python hash function")


# ANSI escape sequences for formatting
BOLD = '\033[1m'
RESET = '\033[0m'

# Parse command line arguments
train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

# Initialize environment
env = MazeEscapeEnv()


def q_learning(num_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.9999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run
    - gamma (float): Discount factor for future rewards
    - epsilon (float): Initial exploration rate
    - decay_rate (float): Rate at which epsilon decays after each episode

    Returns:
    - Q_table (dict): Dictionary mapping state_id -> numpy array of Q-values
    - episode_rewards (list): List of total rewards per episode
    """
    Q_table = {}  # Dict: state_id -> np.array of Q-values for each action
    num_updates = {}  # Dict: state_id -> np.array of update counts for each action
    episode_rewards = []  # Track total reward per episode

    print(f"\n{BOLD}Starting Q-Learning Training{RESET}")
    print(f"Episodes: {num_episodes}, Gamma: {gamma}, Initial Epsilon: {epsilon}, Decay: {decay_rate}\n")

    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get current state hash
            state = hash_observation(obs)

            # Initialize Q-values and update counts for new states
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)
                num_updates[state] = np.zeros(env.action_space.n)

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: choose random action
                action = env.action_space.sample()
            else:
                # Exploit: choose action with highest Q-value
                action = np.argmax(Q_table[state])

            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Get next state hash
            next_state = hash_observation(next_obs)

            # Initialize Q-values for next state if new
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n)
                num_updates[next_state] = np.zeros(env.action_space.n)

            # Calculate learning rate (decreases with more updates)
            eta = 1.0 / (1.0 + num_updates[state][action])

            # Q-learning update (Bellman equation)
            # Q_new(s,a) = (1-η)Q_old(s,a) + η[R(s,a,s') + γ max Q_old(s',a')]
            old_q = Q_table[state][action]
            max_next_q = np.max(Q_table[next_state])
            new_q = (1 - eta) * old_q + eta * (reward + gamma * max_next_q)

            Q_table[state][action] = new_q
            num_updates[state][action] += 1

            # Move to next state
            obs = next_obs

        # Record episode reward
        episode_rewards.append(total_reward)

        # Decay epsilon
        epsilon *= decay_rate

    print(f"\n{BOLD}Training Complete!{RESET}")
    print(f"Total unique states encountered: {len(Q_table)}")
    print(f"Final epsilon: {epsilon:.6f}")

    # Generate reward plot
    plot_rewards(episode_rewards, num_episodes, decay_rate)

    return Q_table, episode_rewards


def plot_rewards(episode_rewards, num_episodes, decay_rate):
    """
    Plot episode rewards with moving average.

    Parameters:
    - episode_rewards (list): List of total rewards per episode
    - num_episodes (int): Number of training episodes
    - decay_rate (float): Epsilon decay rate used
    """
    plt.figure(figsize=(12, 6))

    # Plot raw rewards
    plt.plot(episode_rewards, alpha=0.6, linewidth=0.5, label='Episode Reward', color='blue')

    # Calculate and plot moving average
    window_size = max(1, num_episodes // 100)
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(episode_rewards)), moving_avg,
                 color='red', linewidth=2, label=f'Moving Average (window={window_size})')

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title(f'Training Rewards (episodes={num_episodes}, decay={decay_rate})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = f'rewards_plot_{num_episodes}_{decay_rate}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Reward plot saved as: {filename}")
    plt.close()


'''
Configuration: Set training hyperparameters
'''

# Experiment hyperparameters
num_episodes = 100000
decay_rate = 0.99999

'''
Training Mode: Train agent and save Q-table
'''

if train_flag:
    print(f"\n{BOLD}={'=' * 60}{RESET}")
    print(f"{BOLD}TRAINING MODE{RESET}")
    print(f"{BOLD}={'=' * 60}{RESET}")

    # Run Q-learning
    Q_table, episode_rewards = q_learning(
        num_episodes=num_episodes,
        gamma=0.9,
        epsilon=1.0,
        decay_rate=decay_rate
    )

    # Save Q-table to pickle file
    filename = f'Q_table_{num_episodes}_{decay_rate}.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(Q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{BOLD}Q-table saved as: {filename}{RESET}")
    print(f"{BOLD}={'=' * 60}{RESET}\n")

'''
Evaluation Mode: Load Q-table and evaluate agent
'''


def softmax(x, temp=1.0):
    """Softmax function for action selection during evaluation"""
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)


if not train_flag:
    print(f"\n{BOLD}={'=' * 60}{RESET}")
    print(f"{BOLD}EVALUATION MODE{RESET}")
    print(f"{BOLD}={'=' * 60}{RESET}\n")

    # Load Q-table
    filename = f'Q_table_{num_episodes}_{decay_rate}.pickle'
    try:
        input(f"Loading Q-table from {BOLD}{filename}{RESET}\nPress Enter to confirm, or Ctrl+C to cancel...\n")
        Q_table = pickle.load(open(filename, 'rb'))
        print(f"✓ Q-table loaded successfully!\n")
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found!")
        print(f"Please train the model first using: python q_learning.py train\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled.")
        sys.exit(0)

    # Evaluation metrics
    rewards = []
    episode_lengths = []
    states_in_qtable = set(Q_table.keys())
    new_states_in_eval = set()
    actions_from_qtable = 0
    random_actions_due_to_missing = 0
    total_actions = 0

    # Track actions in specific situations for heatmap
    action_counts = {
        'portal_red': np.zeros(env.action_space.n),
        'portal_blue': np.zeros(env.action_space.n),
        'portal_green': np.zeros(env.action_space.n),
        'minotaur_M1': np.zeros(env.action_space.n),
        'minotaur_M2': np.zeros(env.action_space.n),
        'minotaur_M3': np.zeros(env.action_space.n),
    }

    print(f"{BOLD}EVALUATION METRICS{RESET}")
    print(f"{'=' * 60}")
    print(f"Number of unique states in Q-table: {len(states_in_qtable)}")

    # Start timing
    start_time = time.time()

    # Run evaluation episodes
    num_eval_episodes = 10000
    print(f"\nRunning {num_eval_episodes} evaluation episodes...")

    for episode in tqdm(range(num_eval_episodes), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        episode_length = 0
        episode_reward = 0

        while not done:
            state = hash_observation(obs)
            episode_length += 1
            total_actions += 1

            # Track if state was seen during training
            if state in states_in_qtable:
                actions_from_qtable += 1
            else:
                new_states_in_eval.add(state)
                random_actions_due_to_missing += 1

            # Select action using softmax over Q-values (or random if state not in Q-table)
            try:
                action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))
            except KeyError:
                action = env.action_space.sample()

            # Track actions in specific situations
            # Portal usage
            if obs.get('at_portal'):
                portal_color = obs.get('portal_color')
                if portal_color in ['red', 'blue', 'green']:
                    action_counts[f'portal_{portal_color}'][action] += 1

            # Minotaur encounters
            minotaur_in_cell = obs.get('minotaur_in_cell')
            if minotaur_in_cell:
                # Get minotaur type (M1, M2, M3)
                try:
                    minotaur_type = env.current_state['minotaurs'][minotaur_in_cell]['type']
                    action_counts[f'minotaur_{minotaur_type}'][action] += 1
                except:
                    pass

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    avg_reward = np.mean(rewards)
    avg_episode_length = np.mean(episode_lengths)
    num_new_states = len(new_states_in_eval)
    pct_actions_from_qtable = (actions_from_qtable / total_actions) * 100 if total_actions > 0 else 0
    pct_random_actions = (random_actions_due_to_missing / total_actions) * 100 if total_actions > 0 else 0

    # Print metrics
    print(f"\n{'=' * 60}")
    print(f"Average reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    print(f"Average episode length: {avg_episode_length:.2f} actions")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"New states encountered (not in Q-table): {num_new_states}")
    print(f"% Actions from Q-table (seen states): {pct_actions_from_qtable:.2f}%")
    print(f"% Random actions (unseen states): {pct_random_actions:.2f}%")
    print(f"{'=' * 60}\n")

    # Generate action distribution heatmap
    print("Generating action distribution heatmap...")

    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT', 'PORTAL']
    situations = ['Red Portal', 'Blue Portal', 'Green Portal', 'Minotaur M1', 'Minotaur M2', 'Minotaur M3']

    # Prepare heatmap data (normalize each row)
    heatmap_data = []
    situation_keys = ['portal_red', 'portal_blue', 'portal_green', 'minotaur_M1', 'minotaur_M2', 'minotaur_M3']

    for key in situation_keys:
        counts = action_counts[key]
        total = counts.sum()
        if total > 0:
            normalized = counts / total
        else:
            normalized = counts
        heatmap_data.append(normalized)

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=action_names, yticklabels=situations,
                vmin=0, vmax=1, cbar_kws={'label': 'Normalized Frequency'}, ax=ax)

    ax.set_title(f'Action Distribution Heatmap (episodes={num_episodes}, decay={decay_rate})',
                 fontsize=14, pad=20)
    ax.set_xlabel('Actions', fontsize=12)
    ax.set_ylabel('Situations', fontsize=12)

    plt.tight_layout()

    # Save heatmap
    heatmap_filename = f'action_heatmap_{num_episodes}_{decay_rate}.png'
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Action heatmap saved as: {heatmap_filename}")
    plt.close()

    print(f"\n{BOLD}Evaluation complete!{RESET}\n")

if __name__ == '__main__':
    if not train_flag and not gui_flag:
        pass  # Evaluation mode already handled above
    elif gui_flag:
        print("\n⚠️  GUI mode not yet implemented for this project.")
        print("Use 'python q_learning.py train' to train or 'python q_learning.py' to evaluate.\n")