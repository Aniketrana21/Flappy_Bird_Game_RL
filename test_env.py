import flappy_bird_gymnasium
import gymnasium as gym

# Test if simple observations (12-dim) are available
print("=== Testing observation modes ===\n")

# Test 1: Default (LIDAR)
env = gym.make('FlappyBird-v0')
s, _ = env.reset()
print(f"Default obs shape: {s.shape}")
env.close()

# Test 2: Try simple observations
try:
    env = gym.make('FlappyBird-v0', use_lidar=False)
    s, _ = env.reset()
    print(f"Simple obs (use_lidar=False) shape: {s.shape}")
    print(f"Simple obs sample: {s}")
    env.close()
except Exception as e:
    print(f"use_lidar=False failed: {e}")

# Test 3: Try observation_type
try:
    env = gym.make('FlappyBird-v0', observation_type='simple')
    s, _ = env.reset()
    print(f"Simple obs (observation_type) shape: {s.shape}")
    env.close()
except Exception as e:
    print(f"observation_type='simple' failed: {e}")

# Test 4: Check what the score info looks like when crossing pipes
print("\n=== Testing pipe crossing detection ===")
env = gym.make('FlappyBird-v0')
s, _ = env.reset()
prev_score = 0
for i in range(500):
    # Alternate flap/no-flap randomly
    import random
    action = random.choice([0, 0, 0, 1])  # mostly no-flap
    ns, r, t, _, info = env.step(action)
    if info.get('score', 0) > prev_score:
        print(f"  PIPE CROSSED at step {i}: reward={r:.3f}, score={info['score']}")
        prev_score = info['score']
    if t:
        print(f"  Died at step {i}, score={info.get('score', 0)}")
        break
env.close()
