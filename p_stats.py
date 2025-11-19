import cProfile
import pstats
from q_learning import q_learning

# Profile a shorter training run
print("Profiling 1,000 episodes...\n")

profiler = cProfile.Profile()
profiler.enable()

# Run training
q_learning(num_episodes=1000, gamma=0.9, epsilon=1.0, decay_rate=0.999)

profiler.disable()

# Save and analyze results
profiler.dump_stats('profile_stats.prof')

# Print top 30 functions by cumulative time
print("\n" + "="*80)
print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
print("="*80 + "\n")

stats = pstats.Stats('profile_stats.prof')
stats.sort_stats('cumulative')
stats.print_stats(30)

print("\n" + "="*80)
print("TOP 30 FUNCTIONS BY TOTAL TIME (excluding subcalls)")
print("="*80 + "\n")

stats.sort_stats('time')
stats.print_stats(30)