"""
LIVE DEMO – SELF-DRIVING CAR
Automatically selects the best model (highest average reward) and runs one full episode.
Displays detailed statistics and saves a summary report.
"""
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from metadrive import MetaDriveEnv
from stable_baselines3 import PPO
import glob
import time
import numpy as np
import os
from datetime import datetime

print("="*70)
print("SELF-DRIVING CAR – BEST MODEL LIVE DEMO")
print("="*70)

# ----------------------------------------------------------------------
# 1. Find all trained models and evaluate them quickly
# ----------------------------------------------------------------------
model_files = glob.glob("models_saved/self_driving_model_*.zip")
if not model_files:
    print("ERROR: No trained models found. Run 'python main.py' first.")
    exit(1)

print(f"Found {len(model_files)} models. Evaluating each on 3 episodes...")

best_model_path = None
best_reward = -float('inf')
eval_env = MetaDriveEnv({"use_render": False, "traffic_density": 0.03, "num_scenarios": 1})

for model_path in model_files:
    try:
        model = PPO.load(model_path)
        rewards = []
        for _ in range(3):  # 3 quick episodes
            obs, _ = eval_env.reset()          # ← FIX: unpack tuple
            ep_reward = 0.0
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
        avg_reward = np.mean(rewards)
        print(f"  {os.path.basename(model_path):45} avg reward = {avg_reward:8.2f}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_path = model_path
    except Exception as e:
        print(f"  Error loading {os.path.basename(model_path)}: {e}")

eval_env.close()

if best_model_path is None:
    print("\nWARNING: No model could be evaluated. Trying to load the latest model anyway...")
    # Fallback: use the most recent model
    best_model_path = sorted(model_files)[-1]
    print(f"Using fallback model: {os.path.basename(best_model_path)}")
else:
    print(f"\n✅ BEST MODEL SELECTED: {os.path.basename(best_model_path)} (avg reward = {best_reward:.2f})")

print("Loading and starting live demo...\n")

# ----------------------------------------------------------------------
# 2. Load the best model and run the live demo
# ----------------------------------------------------------------------
model = PPO.load(best_model_path)

# Create environment for live demo
env = MetaDriveEnv({
    "use_render": True,
    "traffic_density": 0.05,
    "num_scenarios": 1,
    "window_size": (1024, 768),
})

print("\nStarting live demo (one episode). Watch the window.\n")
print("The car will drive until it crashes or reaches destination.\n")

obs, _ = env.reset()          # ← FIX: unpack tuple
total_reward = 0.0
step = 0
steering_history = []
throttle_history = []
reward_history = []
start_time = time.time()
reason = ""

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        steering_history.append(action[0])
        throttle_history.append(action[1])
        reward_history.append(reward)

        if step % 50 == 0:
            print(f"  Step {step}: reward = {reward:.2f}, total = {total_reward:.2f}")

        if terminated or truncated:
            if 'arrive_dest' in str(info):
                reason = "Reached destination (SUCCESS)"
            elif info.get('crash_vehicle', False):
                reason = "Crashed into another vehicle"
            elif info.get('crash_object', False):
                reason = "Crashed into object"
            elif info.get('out_of_road', False):
                reason = "Drove out of road"
            else:
                reason = "Episode ended"
            break
except KeyboardInterrupt:
    reason = "Demo stopped by user"
finally:
    env.close()

duration = time.time() - start_time

# ----------------------------------------------------------------------
# 3. Print detailed summary to console
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("DEMO SUMMARY")
print("="*70)
print(f"""
Episode Statistics:
  - Steps:          {step}
  - Total reward:   {total_reward:.2f}
  - Average reward: {total_reward/step if step>0 else 0:.2f}
  - Duration:       {duration:.1f} seconds
  - Termination:    {reason}

Action Statistics:
  - Mean steering:  {np.mean(steering_history):.4f}
  - Std steering:   {np.std(steering_history):.4f}
  - Mean throttle:  {np.mean(throttle_history):.4f}
  - Std throttle:   {np.std(throttle_history):.4f}

Reward Statistics:
  - Max reward:     {max(reward_history) if reward_history else 0:.2f}
  - Min reward:     {min(reward_history) if reward_history else 0:.2f}
""")
print("="*70)

# ----------------------------------------------------------------------
# 4. Save a summary report (optional)
# ----------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"videos/live_demo_report_{timestamp}.txt"
os.makedirs("videos", exist_ok=True)

with open(report_path, "w") as f:
    f.write("="*70 + "\n")
    f.write("SELF-DRIVING CAR – LIVE DEMO REPORT\n")
    f.write(f"Model: {os.path.basename(best_model_path)}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n\n")
    f.write(f"Steps: {step}\n")
    f.write(f"Total reward: {total_reward:.2f}\n")
    f.write(f"Average reward: {total_reward/step if step>0 else 0:.2f}\n")
    f.write(f"Duration: {duration:.1f} seconds\n")
    f.write(f"Termination reason: {reason}\n\n")
    f.write(f"Mean steering: {np.mean(steering_history):.4f}\n")
    f.write(f"Mean throttle: {np.mean(throttle_history):.4f}\n")
    f.write(f"Max reward: {max(reward_history) if reward_history else 0:.2f}\n")
    f.write(f"Min reward: {min(reward_history) if reward_history else 0:.2f}\n")

print(f"\n✅ Live demo report saved: {report_path}")
print("Demo complete. Close the window or press Ctrl+C.")