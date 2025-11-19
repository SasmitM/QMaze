from maze_env import MazeEscapeEnv

env = MazeEscapeEnv()
obs, _ = env.reset(seed=42)

# Check what the window looks like
print("Sample observation:")
print(f"Health: {obs['agent_health']}")
print(f"Minotaur in cell: {obs.get('minotaur_in_cell')}")
print(f"\nFirst few window cells:")
for i, (key, val) in enumerate(list(obs['window'].items())[:5]):
    print(f"  {key}: {val}")