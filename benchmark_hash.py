import time
from maze_env import MazeEscapeEnv
from state_hash import hash_observation as hash_python
from state_hash_rust import hash_observation_rust as hash_rust

env = MazeEscapeEnv()

# Generate test observations
observations = []
for i in range(10000):
    obs, _ = env.reset(seed=i)
    observations.append(obs)

print("Benchmarking hash functions with 10,000 observations...\n")

# Benchmark Python
start = time.time()
for obs in observations:
    _ = hash_python(obs)
py_time = time.time() - start

# Benchmark Rust
start = time.time()
for obs in observations:
    _ = hash_rust(obs)
rust_time = time.time() - start

print(f"Python: {py_time:.3f} seconds")
print(f"Rust:   {rust_time:.3f} seconds")
print(f"Speedup: {py_time/rust_time:.1f}x faster! 🚀")