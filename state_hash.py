import numpy as np


def hash_observation(obs):
    """
    Encode observation into a unique integer state ID for Q-learning.

    The observation contains:
    - agent_health: int (0=Critical, 1=Injured, 2=Full)
    - window: 5x5 grid centered on agent
    - minotaur_in_cell: str or None (minotaur ID if present)

    Each cell in the 5x5 window is encoded as a single digit (0-8):
        0: empty
        1: wall
        2: trap
        3: minotaur present
        4: red portal
        5: blue portal
        6: green portal
        7: goal
        8: out of bounds

    The 25 cell values are packed into a base-9 integer (window_hash).

    If agent shares cell with minotaur, we add minotaur identity:
        - No minotaur: 0
        - M1: 1
        - M2: 2
        - M3: 3
        - M4: 4
        - M5: 5

    Final state_id = health * HEALTH_SPACE + minotaur_id * MINOTAUR_SPACE + window_hash

    Args:
        obs (dict): Observation from environment

    Returns:
        int: Unique state identifier
    """
    # Get health (0-2)
    health = int(obs.get('agent_health', 0))

    # Get 5x5 window
    window = obs.get('window', {})

    # Build cell values in stable order: row by row, top to bottom, left to right
    # (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
    # (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
    # ...
    # (2,-2),  (2,-1),  (2,0),  (2,1),  (2,2)

    cell_values = []
    for dr in range(-2, 3):  # -2, -1, 0, 1, 2 (rows)
        for dc in range(-2, 3):  # -2, -1, 0, 1, 2 (cols)
            cell = window.get((dr, dc))

            if cell is None or not cell.get('in_bounds', False):
                # Out of bounds
                cell_values.append(8)
                continue

            # Determine what's in this cell (priority order matters!)

            # First check if it's a wall (walls override everything)
            if cell.get('is_wall', False):
                cell_value = 1
            # Then check for goal
            elif cell.get('is_goal', False):
                cell_value = 7
            # Then check for portals
            elif cell.get('has_portal', False):
                portal_color = cell.get('portal_color', None)
                if portal_color == 'red':
                    cell_value = 4
                elif portal_color == 'blue':
                    cell_value = 5
                elif portal_color == 'green':
                    cell_value = 6
                else:
                    cell_value = 0  # Unknown portal, treat as empty
            # Then check for minotaurs
            elif cell.get('has_minotaur', False):
                cell_value = 3
            # Then check for traps
            elif cell.get('is_trap', False):
                cell_value = 2
            # Otherwise empty
            else:
                cell_value = 0

            cell_values.append(cell_value)

    # Pack cell values into base-9 integer
    window_hash = 0
    base = 1
    for value in cell_values:
        window_hash += value * base
        base *= 9

    # Include minotaur identity if agent is sharing cell with one
    minotaur_in_cell = obs.get('minotaur_in_cell', None)
    if minotaur_in_cell:
        # Extract number from minotaur ID (e.g., 'M1' -> 1, 'M2' -> 2)
        try:
            minotaur_id = int(minotaur_in_cell[1:])  # Get number after 'M'
        except:
            minotaur_id = 0
    else:
        minotaur_id = 0

    # Calculate state spaces
    WINDOW_SPACE = 9 ** 25  # 25 cells, each with 9 possible values
    MINOTAUR_SPACE = WINDOW_SPACE  # Space for minotaur identity (0-5)
    HEALTH_SPACE = MINOTAUR_SPACE * 6  # 6 possible minotaur IDs (0=none, 1-5=M1-M5)

    # Combine all components into final state ID
    state_id = int(health) * HEALTH_SPACE + int(minotaur_id) * MINOTAUR_SPACE + window_hash

    return state_id


def test_hash():
    """Test the hash function with sample observations"""
    print("Testing state hashing...\n")

    # Test 1: Simple observation
    obs1 = {
        'agent_health': 2,  # Full
        'agent_position': (9, 7),
        'window': {},
        'minotaur_in_cell': None,
        'at_trap': False,
        'at_portal': False,
    }

    # Fill window with empty cells
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            obs1['window'][(dr, dc)] = {
                'is_wall': False,
                'is_goal': False,
                'is_trap': False,
                'has_minotaur': False,
                'has_portal': False,
                'portal_color': None,
                'in_bounds': True,
            }

    hash1 = hash_observation(obs1)
    print(f"✓ Test 1 - All empty cells, full health:")
    print(f"  State ID: {hash1}")

    # Test 2: Same observation should give same hash
    hash1_repeat = hash_observation(obs1)
    assert hash1 == hash1_repeat, "Same observation should give same hash!"
    print(f"  Consistency check: ✅ Same hash on repeat")

    # Test 3: Different health should give different hash
    obs2 = obs1.copy()
    obs2['agent_health'] = 1  # Injured
    obs2['window'] = obs1['window'].copy()
    hash2 = hash_observation(obs2)
    assert hash1 != hash2, "Different health should give different hash!"
    print(f"\n✓ Test 2 - Injured health:")
    print(f"  State ID: {hash2}")
    print(f"  Different from Test 1: ✅")

    # Test 4: Wall in window should change hash
    obs3 = obs1.copy()
    obs3['window'] = {}
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            obs3['window'][(dr, dc)] = obs1['window'][(dr, dc)].copy()
    obs3['window'][(0, 1)]['is_wall'] = True  # Wall to the right
    hash3 = hash_observation(obs3)
    assert hash3 != hash1, "Wall should change hash!"
    print(f"\n✓ Test 3 - Wall to the right:")
    print(f"  State ID: {hash3}")
    print(f"  Different from Test 1: ✅")

    # Test 5: Minotaur in cell
    obs4 = obs1.copy()
    obs4['window'] = {}
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            obs4['window'][(dr, dc)] = obs1['window'][(dr, dc)].copy()
    obs4['window'][(1, 0)]['has_minotaur'] = True  # Minotaur below
    obs4['minotaur_in_cell'] = 'M1'  # Sharing cell with M1
    hash4 = hash_observation(obs4)
    assert hash4 != hash1, "Minotaur should change hash!"
    print(f"\n✓ Test 4 - Minotaur M1 in cell:")
    print(f"  State ID: {hash4}")
    print(f"  Different from Test 1: ✅")

    # Test 6: Portal in window
    obs5 = obs1.copy()
    obs5['window'] = {}
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            obs5['window'][(dr, dc)] = obs1['window'][(dr, dc)].copy()
    obs5['window'][(-1, 0)]['has_portal'] = True  # Portal above
    obs5['window'][(-1, 0)]['portal_color'] = 'red'
    hash5 = hash_observation(obs5)
    assert hash5 != hash1, "Portal should change hash!"
    print(f"\n✓ Test 5 - Red portal above:")
    print(f"  State ID: {hash5}")
    print(f"  Different from Test 1: ✅")

    # Test 7: Trap in window
    obs6 = obs1.copy()
    obs6['window'] = {}
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            obs6['window'][(dr, dc)] = obs1['window'][(dr, dc)].copy()
    obs6['window'][(0, -1)]['is_trap'] = True  # Trap to the left
    hash6 = hash_observation(obs6)
    assert hash6 != hash1, "Trap should change hash!"
    print(f"\n✓ Test 6 - Trap to the left:")
    print(f"  State ID: {hash6}")
    print(f"  Different from Test 1: ✅")

    # Test 8: Goal in window
    obs7 = obs1.copy()
    obs7['window'] = {}
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            obs7['window'][(dr, dc)] = obs1['window'][(dr, dc)].copy()
    obs7['window'][(2, 2)]['is_goal'] = True  # Goal at bottom-right of window
    hash7 = hash_observation(obs7)
    assert hash7 != hash1, "Goal should change hash!"
    print(f"\n✓ Test 7 - Goal visible:")
    print(f"  State ID: {hash7}")
    print(f"  Different from Test 1: ✅")

    # Test 9: Out of bounds cells
    obs8 = {
        'agent_health': 2,
        'window': {},
        'minotaur_in_cell': None,
    }
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == -2:  # Top row out of bounds
                obs8['window'][(dr, dc)] = {
                    'in_bounds': False,
                }
            else:
                obs8['window'][(dr, dc)] = obs1['window'][(0, 0)].copy()
    hash8 = hash_observation(obs8)
    assert hash8 != hash1, "Out of bounds should change hash!"
    print(f"\n✓ Test 8 - Top row out of bounds:")
    print(f"  State ID: {hash8}")
    print(f"  Different from Test 1: ✅")

    print(f"\n{'=' * 50}")
    print(f"✅ All hash tests passed!")
    print(f"{'=' * 50}")

    # Show state space size
    print(f"\nState space analysis:")
    print(f"  Window configurations: 9^25 = {9 ** 25:,}")
    print(f"  Minotaur IDs: 6 (0=none, 1-5=M1-M5)")
    print(f"  Health states: 3 (Critical, Injured, Full)")
    print(f"  Theoretical max states: {9 ** 25 * 6 * 3:,}")
    print(f"  (Most will never be encountered in practice)")


if __name__ == '__main__':
    test_hash()