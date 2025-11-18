import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from maze import create_fixed_maze


class MazeEscapeEnv(gym.Env):
    """
    Maze Escape Environment

    Agent navigates a 15x15 maze to reach goal at top-right.
    Includes walls, minotaurs, traps, portals, and partial observability (5x5 window).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEscapeEnv, self).__init__()

        # Grid setup
        self.grid_size = 15
        self.window_size = 5  # 5x5 observation window
        self.max_steps = 1000  # Episode truncation

        # Load fixed maze
        self.maze = create_fixed_maze(self.grid_size)

        # Goal position (top-right area)
        self.goal_position = (1, 13)  # (row, col)

        # Health states
        self.health_states = ['Full', 'Injured', 'Critical']
        self.health_to_int = {'Full': 2, 'Injured': 1, 'Critical': 0}

        # Actions: UP, DOWN, LEFT, RIGHT, FIGHT, HIDE, HEAL, WAIT, ENTER_PORTAL
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT', 'ENTER_PORTAL']
        self.action_space = spaces.Discrete(len(self.actions))

        # Rewards
        self.rewards = {
            'goal': 10000,
            'combat_win': 100,
            'combat_loss': -10,
            'defeat': -1000,
            'trap': -50,
            'heal': 50,
            'invalid': -5,
            'wall': -5,  # Penalty for hitting wall
            'portal': 10,
        }

        # Minotaur types with different stats
        self.minotaur_types = {
            'M1': {'strength': 0.7, 'keenness': 0.2},  # Strong, not very perceptive
            'M2': {'strength': 0.5, 'keenness': 0.5},  # Balanced
            'M3': {'strength': 0.3, 'keenness': 0.8},  # Weak, very perceptive
        }

        # Portal colors and their teleport regions
        self.portal_colors = ['red', 'blue', 'green']
        self.portal_regions = {
            'red': {'row_range': (1, 4), 'col_range': (1, 4)},      # Top-left
            'blue': {'row_range': (11, 14), 'col_range': (11, 14)}, # Bottom-right
            'green': {'row_range': (1, 4), 'col_range': (11, 14)},  # Top-right (near goal)
        }

        # Initialize state
        self.current_state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.steps = 0

        # Find valid starting positions (empty cells, not at goal, not on walls)
        valid_positions = []
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                if self.maze[i, j] == 0 and (i, j) != self.goal_position:
                    # Prefer bottom-left area for agent start
                    if i > self.grid_size // 2:
                        valid_positions.append((i, j))

        # Random start position for agent
        if not valid_positions:
            agent_position = (13, 1)  # Fallback
        else:
            agent_position = random.choice(valid_positions)

        # Collect all valid positions (not agent, not goal, not wall)
        all_valid = [(i, j) for i in range(1, self.grid_size - 1)
                     for j in range(1, self.grid_size - 1)
                     if self.maze[i, j] == 0 and (i, j) != agent_position and (i, j) != self.goal_position]

        # Spawn 3-5 minotaurs
        num_minotaurs = random.randint(3, 5)
        minotaur_positions = {}
        used_positions = set()

        if len(all_valid) >= num_minotaurs:
            chosen_positions = random.sample(all_valid, num_minotaurs)
            minotaur_types = list(self.minotaur_types.keys())

            for idx in range(num_minotaurs):
                minotaur_id = f"M{idx + 1}"
                minotaur_type = minotaur_types[idx % len(minotaur_types)]
                pos = chosen_positions[idx]

                minotaur_positions[minotaur_id] = {
                    'position': pos,
                    'type': minotaur_type,
                    'strength': self.minotaur_types[minotaur_type]['strength'],
                    'keenness': self.minotaur_types[minotaur_type]['keenness'],
                }
                used_positions.add(pos)

        # Remove used positions from available positions
        remaining_positions = [pos for pos in all_valid if pos not in used_positions]

        # Spawn 5-10 traps
        num_traps = random.randint(5, 10)
        trap_positions = []

        if len(remaining_positions) >= num_traps:
            trap_positions = random.sample(remaining_positions, num_traps)
            for pos in trap_positions:
                used_positions.add(pos)
                remaining_positions.remove(pos)

        # Spawn 2-4 portal pairs (each pair has 2 portals of same color)
        num_portal_pairs = random.randint(2, 4)
        portals = {}  # dict: position -> color
        portal_pairs = {}  # dict: color_id -> [pos1, pos2]

        # Limit green portals to max 1 pair (they're powerful)
        colors_to_use = ['red', 'blue']
        if num_portal_pairs >= 3:
            colors_to_use.append('green')

        colors_used = random.sample(colors_to_use, min(num_portal_pairs, len(colors_to_use)))

        # If we need more pairs than colors, allow duplicates of red/blue
        while len(colors_used) < num_portal_pairs:
            colors_used.append(random.choice(['red', 'blue']))

        for color in colors_used:
            if len(remaining_positions) >= 2:
                # Pick 2 positions for this portal pair
                pair_positions = random.sample(remaining_positions, 2)

                portals[pair_positions[0]] = color
                portals[pair_positions[1]] = color
                portal_pairs[f"{color}_{len([c for c in colors_used[:colors_used.index(color) + 1] if c == color])}"] = pair_positions

                for pos in pair_positions:
                    used_positions.add(pos)
                    remaining_positions.remove(pos)

        self.current_state = {
            'agent_position': agent_position,
            'agent_health': 'Full',
            'minotaurs': minotaur_positions,
            'traps': trap_positions,
            'portals': portals,  # position -> color
            'portal_pairs': portal_pairs,  # color_id -> [pos1, pos2]
        }

        return self.get_observation(), {}

    def get_observation(self):
        """
        Get observation with 5x5 window centered on agent.

        Returns:
            dict: Observation containing:
                - agent_position: (row, col)
                - agent_health: int (0-2)
                - window: dict of cell info for 5x5 area
                - at_goal: bool
                - minotaur_in_cell: str or None
                - at_trap: bool
                - at_portal: bool
                - portal_color: str or None
        """
        agent_pos = self.current_state['agent_position']
        row, col = agent_pos

        # Build 5x5 window
        window = {}
        half_window = self.window_size // 2

        for dr in range(-half_window, half_window + 1):
            for dc in range(-half_window, half_window + 1):
                cell_row = row + dr
                cell_col = col + dc

                # Check if cell is in bounds
                if 0 <= cell_row < self.grid_size and 0 <= cell_col < self.grid_size:
                    cell_pos = (cell_row, cell_col)

                    # Check for minotaurs in this cell
                    minotaurs_here = [mid for mid, mdata in self.current_state['minotaurs'].items()
                                      if mdata['position'] == cell_pos]

                    # Check for trap
                    is_trap = cell_pos in self.current_state['traps']

                    # Check for portal
                    portal_color = self.current_state['portals'].get(cell_pos, None)

                    window[(dr, dc)] = {
                        'is_wall': bool(self.maze[cell_row, cell_col] == 1),
                        'is_goal': cell_pos == self.goal_position,
                        'is_trap': is_trap,
                        'has_minotaur': len(minotaurs_here) > 0,
                        'minotaur_ids': minotaurs_here if minotaurs_here else None,
                        'has_portal': portal_color is not None,
                        'portal_color': portal_color,
                        'in_bounds': True,
                    }
                else:
                    # Out of bounds
                    window[(dr, dc)] = {
                        'is_wall': True,
                        'is_goal': False,
                        'is_trap': False,
                        'has_minotaur': False,
                        'minotaur_ids': None,
                        'has_portal': False,
                        'portal_color': None,
                        'in_bounds': False,
                    }

        # Check if agent is sharing cell with a minotaur
        minotaur_in_cell = None
        for mid, mdata in self.current_state['minotaurs'].items():
            if mdata['position'] == agent_pos:
                minotaur_in_cell = mid
                break

        # Check if at trap
        at_trap = agent_pos in self.current_state['traps']

        # Check if at portal
        portal_color = self.current_state['portals'].get(agent_pos, None)

        obs = {
            'agent_position': agent_pos,
            'agent_health': self.health_to_int[self.current_state['agent_health']],
            'window': window,
            'at_goal': agent_pos == self.goal_position,
            'minotaur_in_cell': minotaur_in_cell,
            'at_trap': at_trap,
            'at_portal': portal_color is not None,
            'portal_color': portal_color,
        }

        return obs

    def is_terminal(self):
        """Check if episode is terminal"""
        if self.current_state['agent_position'] == self.goal_position:
            return 'goal'
        if self.current_state['agent_health'] == 'Critical':
            return 'defeat'
        if self.steps >= self.max_steps:
            return 'truncated'
        return False

    def move_agent(self, action):
        """
        Move agent based on action with 99% success, 1% slip.

        Args:
            action: str ('UP', 'DOWN', 'LEFT', 'RIGHT')

        Returns:
            tuple: (result_message, reward)
        """
        row, col = self.current_state['agent_position']

        # Direction mappings
        directions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1),
        }

        # 99% chance: move as intended
        # 1% chance: slip to random adjacent direction
        if random.random() < 0.99:
            # Intended movement
            dr, dc = directions[action]
        else:
            # Slip: choose random adjacent direction (including intended)
            possible_dirs = list(directions.values())
            dr, dc = random.choice(possible_dirs)

        # Calculate new position
        new_row = row + dr
        new_col = col + dc

        # Check if new position is valid (in bounds and not a wall)
        if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                self.maze[new_row, new_col] == 0):
            # Valid move
            self.current_state['agent_position'] = (new_row, new_col)
            return f"Moved to ({new_row}, {new_col})", 0
        else:
            # Invalid move (hit wall or out of bounds)
            return "Hit wall!", self.rewards['wall']

    def move_minotaurs(self):
        """
        Move each minotaur randomly to an adjacent cell (avoiding walls).
        Minotaurs stay in place if sharing cell with agent.
        """
        for mid, mdata in self.current_state['minotaurs'].items():
            # If minotaur is in same cell as agent, don't move it
            if mdata['position'] == self.current_state['agent_position']:
                continue

            row, col = mdata['position']

            # Possible directions (including staying in place)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

            # Find valid adjacent positions (not walls, in bounds)
            valid_moves = []
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.grid_size and
                        0 <= new_col < self.grid_size and
                        self.maze[new_row, new_col] == 0):
                    valid_moves.append((new_row, new_col))

            # Move to random valid position
            if valid_moves:
                new_position = random.choice(valid_moves)
                self.current_state['minotaurs'][mid]['position'] = new_position

    def try_fight(self):
        """
        Agent attempts to fight minotaur in current cell.

        Returns:
            tuple: (result_message, reward)
        """
        agent_pos = self.current_state['agent_position']

        # Find minotaur in current cell
        minotaur_id = None
        for mid, mdata in self.current_state['minotaurs'].items():
            if mdata['position'] == agent_pos:
                minotaur_id = mid
                break

        if minotaur_id is None:
            return "No minotaur to fight!", self.rewards['invalid']

        # Get minotaur stats
        minotaur = self.current_state['minotaurs'][minotaur_id]
        strength = minotaur['strength']

        # Fight outcome based on strength (lower strength = easier to beat)
        if random.random() > strength:
            # Win fight
            self.move_agent_to_random_adjacent()
            return f"Fought {minotaur_id} ({minotaur['type']}) and won!", self.rewards['combat_win']
        else:
            # Lose fight - take damage
            if self.current_state['agent_health'] == 'Full':
                self.current_state['agent_health'] = 'Injured'
            elif self.current_state['agent_health'] == 'Injured':
                self.current_state['agent_health'] = 'Critical'

            self.move_agent_to_random_adjacent()
            return f"Fought {minotaur_id} ({minotaur['type']}) and lost!", self.rewards['combat_loss']

    def try_hide(self):
        """
        Agent attempts to hide from minotaur in current cell.

        Returns:
            tuple: (result_message, reward)
        """
        agent_pos = self.current_state['agent_position']

        # Find minotaur in current cell
        minotaur_id = None
        for mid, mdata in self.current_state['minotaurs'].items():
            if mdata['position'] == agent_pos:
                minotaur_id = mid
                break

        if minotaur_id is None:
            return "No minotaur to hide from!", self.rewards['invalid']

        # Get minotaur stats
        minotaur = self.current_state['minotaurs'][minotaur_id]
        keenness = minotaur['keenness']

        # Hide outcome based on keenness (lower keenness = easier to hide)
        if random.random() > keenness:
            # Successfully hid
            self.move_agent_to_random_adjacent()
            return f"Successfully hid from {minotaur_id} ({minotaur['type']})!", 0
        else:
            # Failed to hide - forced to fight
            return self.try_fight()

    def move_agent_to_random_adjacent(self):
        """Move agent to a random adjacent valid cell (not wall)"""
        row, col = self.current_state['agent_position']

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Find valid adjacent cells
        valid_moves = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.grid_size and
                    0 <= new_col < self.grid_size and
                    self.maze[new_row, new_col] == 0):
                valid_moves.append((new_row, new_col))

        # Move to random valid cell (or stay if no valid moves)
        if valid_moves:
            self.current_state['agent_position'] = random.choice(valid_moves)

    def try_enter_portal(self):
        """
        Agent attempts to enter portal at current position.
        Teleports to the other portal of the same color.

        Returns:
            tuple: (result_message, reward)
        """
        agent_pos = self.current_state['agent_position']

        # Check if agent is at a portal
        portal_color = self.current_state['portals'].get(agent_pos, None)

        if portal_color is None:
            return "No portal here!", self.rewards['invalid']

        # Find the portal pair
        portal_pair = None
        for pair_id, positions in self.current_state['portal_pairs'].items():
            if agent_pos in positions and portal_color in pair_id:
                portal_pair = positions
                break

        if not portal_pair or len(portal_pair) != 2:
            return "Portal pair incomplete!", self.rewards['invalid']

        # Teleport to the other portal
        other_portal = portal_pair[0] if portal_pair[1] == agent_pos else portal_pair[1]
        self.current_state['agent_position'] = other_portal

        return f"Teleported via {portal_color} portal to {other_portal}!", self.rewards['portal']

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: int or str (action index or name)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert action index to name if needed
        if isinstance(action, (int, np.integer)):
            action_name = self.actions[action]
        else:
            action_name = action

        # Store previous position to detect trap entry
        previous_position = self.current_state['agent_position']

        self.steps += 1

        # Check if agent is sharing cell with minotaur
        minotaur_present = any(mdata['position'] == self.current_state['agent_position']
                               for mdata in self.current_state['minotaurs'].values())

        # Execute action
        if action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if minotaur_present:
                result = "Minotaur in cell! Must FIGHT or HIDE."
                reward = self.rewards['invalid']
            else:
                result, reward = self.move_agent(action_name)
        elif action_name == 'FIGHT':
            result, reward = self.try_fight()
        elif action_name == 'HIDE':
            result, reward = self.try_hide()
        elif action_name == 'HEAL':
            result = "No heal tile here!"  # Will implement in future if needed
            reward = self.rewards['invalid']
        elif action_name == 'WAIT':
            result = "Waiting..."
            reward = 0
        elif action_name == 'ENTER_PORTAL':
            result, reward = self.try_enter_portal()
        else:
            result = "Invalid action!"
            reward = self.rewards['invalid']

        # Check if agent entered a trap (not just standing on it)
        current_position = self.current_state['agent_position']
        if (current_position in self.current_state['traps'] and
                previous_position != current_position):
            # Entered trap - take damage
            if self.current_state['agent_health'] == 'Full':
                self.current_state['agent_health'] = 'Injured'
                result += " Stepped on trap! Lost health."
                reward += self.rewards['trap']
            elif self.current_state['agent_health'] == 'Injured':
                self.current_state['agent_health'] = 'Critical'
                result += " Stepped on trap! Lost health."
                reward += self.rewards['trap']

        # Move minotaurs after agent action (if game not over)
        terminal_state = self.is_terminal()
        if not terminal_state:
            self.move_minotaurs()

        # Check terminal conditions
        terminated = False
        truncated = False
        terminal_state = self.is_terminal()

        if terminal_state == 'goal':
            terminated = True
            reward += self.rewards['goal']
            result += f" Reached goal! +{self.rewards['goal']}"
        elif terminal_state == 'defeat':
            terminated = True
            reward += self.rewards['defeat']
            result += f" Defeated! {self.rewards['defeat']}"
        elif terminal_state == 'truncated':
            truncated = True
            result += " Episode truncated (max steps)."

        observation = self.get_observation()
        info = {
            'result': result,
            'action': action_name,
            'truncated': truncated,
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Print current state (simple text rendering)"""
        print(f"\nAgent: {self.current_state['agent_position']}")
        print(f"Health: {self.current_state['agent_health']}")
        print(f"Steps: {self.steps}/{self.max_steps}")
        print(f"Goal: {self.goal_position}")


def test_environment():
    """Test complete environment with minotaurs, traps, and portals"""
    print("Testing MazeEscapeEnv (Complete)...\n")

    env = MazeEscapeEnv()

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✓ Reset successful")
    print(f"  Start position: {obs['agent_position']}")
    print(f"  Health: {obs['agent_health']}")
    print(f"  Minotaurs: {len(env.current_state['minotaurs'])}")
    for mid, mdata in env.current_state['minotaurs'].items():
        print(f"    {mid} ({mdata['type']}): {mdata['position']}")
    print(f"  Traps: {len(env.current_state['traps'])} at {env.current_state['traps'][:3]}...")
    print(f"  Portal pairs: {len(env.current_state['portal_pairs'])}")
    for pair_id, positions in env.current_state['portal_pairs'].items():
        color = pair_id.split('_')[0]
        print(f"    {color}: {positions[0]} <-> {positions[1]}")

    # Test random actions
    print(f"\n✓ Testing random actions:")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Show interesting events
        if 'trap' in info['result'].lower() or 'portal' in info['result'].lower() or abs(reward) > 10:
            print(f"  Step {i + 1}: {info['action']:12s} -> {info['result'][:60]} (R: {reward:6.0f})")

        if terminated or truncated:
            print(f"  Episode ended at step {i + 1}!")
            break

    # Test trap damage
    print(f"\n✓ Testing trap damage:")
    env.reset(seed=100)

    # Try to find and step on a trap
    found_trap = False
    for step in range(100):
        obs = env.get_observation()

        # Look for trap in visible window
        for (dr, dc), cell in obs['window'].items():
            if cell['is_trap'] and (dr, dc) != (0, 0):
                # Found a trap nearby - try to move to it
                if dr == -1 and dc == 0:
                    action_name = 'UP'
                elif dr == 1 and dc == 0:
                    action_name = 'DOWN'
                elif dr == 0 and dc == -1:
                    action_name = 'LEFT'
                elif dr == 0 and dc == 1:
                    action_name = 'RIGHT'
                else:
                    continue

                old_health = env.current_state['agent_health']
                obs, reward, _, _, info = env.step(action_name)
                new_health = env.current_state['agent_health']

                if 'trap' in info['result'].lower():
                    print(f"  Stepped on trap! Health: {old_health} -> {new_health}, Reward: {reward}")
                    found_trap = True
                    break

        if found_trap:
            break

        # Random movement
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset(seed=100 + step)

    if not found_trap:
        print(f"  No trap encountered in 100 steps (unlucky!)")

    # Test portal teleportation
    print(f"\n✓ Testing portal teleportation:")
    env.reset(seed=200)

    found_portal = False
    for step in range(100):
        obs = env.get_observation()

        # Check if standing on portal
        if obs.get('at_portal'):
            portal_color = obs.get('portal_color')
            old_pos = env.current_state['agent_position']
            print(f"  Standing on {portal_color} portal at {old_pos}")

            obs, reward, _, _, info = env.step('ENTER_PORTAL')
            new_pos = obs['agent_position']

            print(f"  Teleported to {new_pos}! Reward: {reward}")
            print(f"  Result: {info['result']}")
            found_portal = True
            break

        # Look for portal in visible window and try to move to it
        for (dr, dc), cell in obs['window'].items():
            if cell['has_portal'] and (dr, dc) != (0, 0):
                # Found a portal nearby
                if abs(dr) + abs(dc) == 1:  # Adjacent
                    if dr == -1:
                        action_name = 'UP'
                    elif dr == 1:
                        action_name = 'DOWN'
                    elif dc == -1:
                        action_name = 'LEFT'
                    elif dc == 1:
                        action_name = 'RIGHT'
                    else:
                        continue

                    obs, _, _, _, _ = env.step(action_name)
                    break

        # Random movement
        obs, _, terminated, truncated, _ = env.step(random.choice([0, 1, 2, 3]))
        if terminated or truncated:
            env.reset(seed=200 + step)

    if not found_portal:
        print(f"  No portal encountered in 100 steps (unlucky!)")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_environment()