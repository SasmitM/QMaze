from maze_env import MazeEscapeEnv
from state_hash import hash_observation

env = MazeEscapeEnv()
obs, _ = env.reset(seed=42)

# Manually compute like Python does
health = obs['agent_health']
window = obs['window']

# Build cell values
cell_values = []
for dr in range(-2, 3):
    for dc in range(-2, 3):
        cell = window.get((dr, dc))
        if cell is None or not cell.get('in_bounds', False):
            cell_values.append(8)
        elif cell.get('is_wall', False):
            cell_values.append(1)
        elif cell.get('is_goal', False):
            cell_values.append(7)
        elif cell.get('has_portal', False):
            portal_color = cell.get('portal_color', None)
            if portal_color == 'red':
                cell_values.append(4)
            elif portal_color == 'blue':
                cell_values.append(5)
            elif portal_color == 'green':
                cell_values.append(6)
            else:
                cell_values.append(0)
        elif cell.get('has_minotaur', False):
            cell_values.append(3)
        elif cell.get('is_trap', False):
            cell_values.append(2)
        else:
            cell_values.append(0)

# Pack into base-9
window_hash = 0
base = 1
for value in cell_values:
    window_hash += value * base
    base *= 9

minotaur_id = 0
if obs.get('minotaur_in_cell'):
    try:
        minotaur_id = int(obs['minotaur_in_cell'][1:])
    except:
        minotaur_id = 0

WINDOW_SPACE = 9 ** 25
MINOTAUR_SPACE = WINDOW_SPACE
HEALTH_SPACE = MINOTAUR_SPACE * 6

print(f"Health: {health}")
print(f"Minotaur ID: {minotaur_id}")
print(f"Cell values (first 10): {cell_values[:10]}")
print(f"Window hash: {window_hash}")
print(f"\nWINDOW_SPACE: {WINDOW_SPACE}")
print(f"MINOTAUR_SPACE: {MINOTAUR_SPACE}")
print(f"HEALTH_SPACE: {HEALTH_SPACE}")
print(f"\nFinal calculation:")
print(f"  health * HEALTH_SPACE = {health} * {HEALTH_SPACE} = {health * HEALTH_SPACE}")
print(f"  minotaur_id * MINOTAUR_SPACE = {minotaur_id} * {MINOTAUR_SPACE} = {minotaur_id * MINOTAUR_SPACE}")
print(f"  window_hash = {window_hash}")
print(f"  TOTAL = {health * HEALTH_SPACE + minotaur_id * MINOTAUR_SPACE + window_hash}")

# Compare to function
print(f"\nFunction result: {hash_observation(obs)}")