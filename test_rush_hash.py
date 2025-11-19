from maze_env import MazeEscapeEnv
from state_hash import hash_observation as hash_python
from state_hash_rust import hash_observation_rust as hash_rust

env = MazeEscapeEnv()
obs, _ = env.reset(seed=42)

# Test both produce same result
hash_py = hash_python(obs)
hash_rs = hash_rust(obs)

print(f"Python hash: {hash_py}")
print(f"Rust hash:   {hash_rs}")
print(f"Match: {hash_py == hash_rs}")

if hash_py == hash_rs:
    print("✅ Rust hash function is correct!")
else:
    print("❌ Hashes don't match - something's wrong!")