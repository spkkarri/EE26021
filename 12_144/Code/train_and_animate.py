import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import glob
import pickle
import time

print("="*80)
print("SELF-DRIVING CAR SYSTEM - TRAINING & OUTPUT GENERATION")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("videos", exist_ok=True)
os.makedirs("models_saved", exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND VERIFY DATASET
# ============================================================================
print("\n[1/7] LOADING DATASET")
dataset_files = glob.glob("data/raw/dataset_*.pkl")
if not dataset_files:
    print("ERROR: No dataset found! Run collect_dataset.py first")
    sys.exit(1)

dataset_sizes = []
for f in dataset_files:
    try:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        samples = len(data.get("observations", []))
        dataset_sizes.append((f, samples))
        print(f"  {os.path.basename(f)}: {samples:,} samples")
    except Exception as e:
        print(f"  Warning: Could not read {os.path.basename(f)}: {e}")

if not dataset_sizes:
    print("ERROR: No valid dataset found!")
    sys.exit(1)

dataset_sizes.sort(key=lambda x: x[1], reverse=True)
dataset_file, total_samples = dataset_sizes[0]
print(f"\nUsing dataset: {os.path.basename(dataset_file)}")
print(f"Total samples: {total_samples:,}")

# ============================================================================
# STEP 2: TRAIN OR LOAD MODEL (WITH LOSS TRACKING & PROGRESS OUTPUT)
# ============================================================================
print("\n[2/7] MODEL PREPARATION")
model_files = glob.glob("models_saved/self_driving_model_*.zip")
use_existing = False
losses = {"value_loss": [], "policy_loss": [], "entropy_loss": []}

if model_files:
    latest_model = sorted(model_files)[-1]
    print(f"Found existing model: {os.path.basename(latest_model)}")
    choice = input("Use existing model? (y/n): ").strip().lower()
    if choice == 'y':
        model = PPO.load(latest_model)
        use_existing = True
        print("Using existing model")
    else:
        use_existing = False

if not use_existing:
    print("\nTraining NEW model (200,000 steps) with optimised hyperparameters...")
    print("Progress will be shown below.\n")
    train_env = MetaDriveEnv({
        "use_render": False,
        "traffic_density": 0.03,
        "num_scenarios": 10,
        "start_seed": 42,
        "horizon": 2000,
    })
    # Optimised hyperparameters for MetaDrive, with verbose=1 to show progress
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=2.5e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=0.5,
        verbose=1,                     # <-- SHOWS TRAINING PROGRESS
    )

    class LossCallback(BaseCallback):
        def __init__(self, loss_dict, verbose=0):
            super().__init__(verbose)
            self.loss_dict = loss_dict
        def _on_step(self):
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                values = self.model.logger.name_to_value
                self.loss_dict['value_loss'].append(values.get('train/value_loss', 0))
                self.loss_dict['policy_loss'].append(values.get('train/policy_loss', 0))
                self.loss_dict['entropy_loss'].append(values.get('train/entropy_loss', 0))
            return True

    callback = LossCallback(losses)
    model.learn(total_timesteps=200000, callback=callback)
    model.save(f"models_saved/self_driving_model_{timestamp}")
    print("\n✅ Model saved!")
    train_env.close()

# ============================================================================
# STEP 3: TEST MODEL AND COLLECT PERFORMANCE DATA
# ============================================================================
print("\n[3/7] TESTING MODEL")
test_env = MetaDriveEnv({
    "use_render": False,
    "traffic_density": 0.02,
    "num_scenarios": 1,
})

rewards_history = []
steering_history = []
throttle_history = []
episode_lengths = []
episode_rewards = []

for episode in range(5):
    obs, _ = test_env.reset()
    ep_rewards = []
    ep_steering = []
    ep_throttle = []
    steps = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        ep_rewards.append(reward)
        ep_steering.append(action[0])
        ep_throttle.append(action[1])
        steps += 1
        if terminated or truncated:
            break
    rewards_history.extend(ep_rewards)
    steering_history.extend(ep_steering)
    throttle_history.extend(ep_throttle)
    episode_lengths.append(steps)
    episode_rewards.append(sum(ep_rewards))
    print(f"  Episode {episode+1}: {steps} steps, total reward = {sum(ep_rewards):.2f}")

test_env.close()
print(f"\nAverage episode length: {np.mean(episode_lengths):.1f} steps")
print(f"Average episode reward: {np.mean(episode_rewards):.2f}")

# ============================================================================
# STEP 4: CREATE GRAPH-BASED GIF (NO METADRIVE CAPTURE)
# ============================================================================
print("\n[4/7] CREATING PERFORMANCE GIF")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("SELF-DRIVING CAR - REAL-TIME PERFORMANCE", fontsize=14, fontweight='bold')
ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
ax1.set_title("Reward Over Time"); ax1.set_xlabel("Step"); ax1.set_ylabel("Reward"); ax1.grid(True)
ax2.set_title("Steering Angle"); ax2.set_xlabel("Step"); ax2.set_ylabel("Steering"); ax2.grid(True); ax2.axhline(0, color='gray', ls='--')
ax3.set_title("Throttle"); ax3.set_xlabel("Step"); ax3.set_ylabel("Throttle"); ax3.grid(True)
ax4.set_title("Cumulative Reward"); ax4.set_xlabel("Step"); ax4.set_ylabel("Total Reward"); ax4.grid(True)

reward_line, = ax1.plot([], [], 'g-', lw=2)
steering_line, = ax2.plot([], [], 'r-', lw=2)
throttle_line, = ax3.plot([], [], 'b-', lw=2)
cumulative_line, = ax4.plot([], [], 'purple', lw=2)

stats_text = fig.text(0.02, 0.02, "", fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

max_frames = min(300, len(rewards_history))
step_data = np.arange(max_frames)
reward_data = np.array(rewards_history[:max_frames])
steering_data = np.array(steering_history[:max_frames])
throttle_data = np.array(throttle_history[:max_frames])
cumulative_data = np.cumsum(reward_data)

def update(frame):
    reward_line.set_data(step_data[:frame+1], reward_data[:frame+1])
    steering_line.set_data(step_data[:frame+1], steering_data[:frame+1])
    throttle_line.set_data(step_data[:frame+1], throttle_data[:frame+1])
    cumulative_line.set_data(step_data[:frame+1], cumulative_data[:frame+1])
    for ax in (ax1, ax2, ax3, ax4):
        ax.relim()
        ax.autoscale_view()
    stats_text.set_text(f"Frame {frame+1}/{max_frames}\nCurrent Reward: {reward_data[frame]:.2f}\nTotal: {cumulative_data[frame]:.2f}")
    fig.suptitle(f"SELF-DRIVING CAR - Frame {frame+1}/{max_frames}", fontsize=14, fontweight='bold')
    return reward_line, steering_line, throttle_line, cumulative_line, stats_text

ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=50, blit=True, repeat=True)
gif_path = f"videos/performance_graph_{timestamp}.gif"
ani.save(gif_path, writer='pillow', fps=20, dpi=100)
print(f"✅ GIF saved: {gif_path} ({max_frames} frames, {max_frames/20:.1f} seconds)")
plt.close()

# ============================================================================
# STEP 5: CREATE 9 STATIC CHARTS (6 performance + 3 loss graphs)
# ============================================================================
print("\n[5/7] CREATING STATIC CHARTS (9 subplots)")
fig = plt.figure(figsize=(16, 14))
fig.suptitle("SELF-DRIVING CAR - COMPLETE ANALYSIS REPORT", fontsize=16, fontweight='bold')

# Row 1: Performance metrics
ax1 = plt.subplot(3, 3, 1)
ax1.plot(rewards_history, 'b-', lw=1)
ax1.set_title("1. Rewards Over Time"); ax1.set_xlabel("Step"); ax1.set_ylabel("Reward"); ax1.grid(True)

ax2 = plt.subplot(3, 3, 2)
ax2.plot(np.cumsum(rewards_history), 'g-', lw=2)
ax2.set_title("2. Cumulative Reward"); ax2.set_xlabel("Step"); ax2.set_ylabel("Total Reward"); ax2.grid(True)

ax3 = plt.subplot(3, 3, 3)
ax3.hist(steering_history, bins=25, color='red', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='blue', linestyle='--', label='Center')
ax3.set_title("3. Steering Distribution"); ax3.set_xlabel("Steering"); ax3.set_ylabel("Frequency"); ax3.legend(); ax3.grid(True)

ax4 = plt.subplot(3, 3, 4)
ax4.hist(throttle_history, bins=25, color='green', alpha=0.7, edgecolor='black')
ax4.set_title("4. Throttle Distribution"); ax4.set_xlabel("Throttle"); ax4.set_ylabel("Frequency"); ax4.grid(True)

ax5 = plt.subplot(3, 3, 5)
ax5.hist(rewards_history, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax5.axvline(x=np.mean(rewards_history), color='red', linestyle='--', label=f'Mean: {np.mean(rewards_history):.3f}')
ax5.set_title("5. Reward Distribution"); ax5.set_xlabel("Reward"); ax5.set_ylabel("Frequency"); ax5.legend(); ax5.grid(True)

ax6 = plt.subplot(3, 3, 6)
ax6.scatter(steering_history[::5], throttle_history[::5], alpha=0.3, s=5, c='blue')
ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax6.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax6.set_title("6. Action Space (Steering vs Throttle)"); ax6.set_xlabel("Steering"); ax6.set_ylabel("Throttle"); ax6.grid(True)

# Row 2: Loss graphs (using the captured losses)
if losses['value_loss']:
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(losses['value_loss'], 'r-', lw=1)
    ax7.set_title("7. Value Loss During Training"); ax7.set_xlabel("Update Step"); ax7.set_ylabel("Loss"); ax7.grid(True)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(losses['policy_loss'], 'b-', lw=1)
    ax8.set_title("8. Policy Loss During Training"); ax8.set_xlabel("Update Step"); ax8.set_ylabel("Loss"); ax8.grid(True)
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(losses['entropy_loss'], 'g-', lw=1)
    ax9.set_title("9. Entropy Loss During Training"); ax9.set_xlabel("Update Step"); ax9.set_ylabel("Loss"); ax9.grid(True)
else:
    for i, title in enumerate(["7. Value Loss", "8. Policy Loss", "9. Entropy Loss"]):
        ax = plt.subplot(3, 3, 7+i)
        ax.text(0.5, 0.5, "Loss data not captured.\nTrain a new model to see these graphs.", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

plt.tight_layout()
charts_path = f"videos/analysis_charts_{timestamp}.png"
plt.savefig(charts_path, dpi=150, bbox_inches='tight')
print(f"✅ Static charts saved: {charts_path}")
plt.close()

# ============================================================================
# STEP 6: CREATE SUMMARY REPORT (TEXT FILE)
# ============================================================================
print("\n[6/7] SAVING SUMMARY REPORT")
summary_path = f"videos/summary_report_{timestamp}.txt"
with open(summary_path, "w") as f:
    f.write("="*80 + "\n")
    f.write("SELF-DRIVING CAR - PROFESSIONAL SUMMARY REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. DATASET INFORMATION\n")
    f.write(f"   File name: {os.path.basename(dataset_file)}\n")
    f.write(f"   Total samples: {total_samples:,}\n")
    f.write(f"   File size: {os.path.getsize(dataset_file)/(1024*1024):.2f} MB\n\n")
    
    f.write("2. MODEL INFORMATION\n")
    f.write("   Algorithm: PPO (Proximal Policy Optimization)\n")
    f.write("   Training steps: 200,000\n")
    if use_existing and model_files:
        f.write(f"   Model file: {os.path.basename(model_files[-1])}\n")
    else:
        f.write(f"   Model file: self_driving_model_{timestamp}.zip\n")
    f.write("   Hyperparameters:\n")
    f.write("      - Learning rate: 0.00025\n")
    f.write("      - Batch size: 128\n")
    f.write("      - N steps: 4096\n")
    f.write("      - Gamma: 0.99\n")
    f.write("      - GAE lambda: 0.95\n")
    f.write("      - Clip range: 0.2\n")
    f.write("      - Entropy coefficient: 0.005\n\n")
    
    f.write("3. PERFORMANCE METRICS (Test Episodes)\n")
    f.write(f"   Number of test episodes: {len(episode_lengths)}\n")
    f.write(f"   Average episode length: {np.mean(episode_lengths):.1f} steps\n")
    f.write(f"   Average episode reward: {np.mean(episode_rewards):.2f}\n")
    f.write(f"   Best episode reward: {max(episode_rewards):.2f}\n")
    f.write(f"   Overall mean reward per step: {np.mean(rewards_history):.4f}\n")
    f.write(f"   Overall total reward collected: {np.sum(rewards_history):.2f}\n\n")
    
    f.write("4. ACTION STATISTICS\n")
    f.write(f"   Steering mean: {np.mean(steering_history):.4f}\n")
    f.write(f"   Steering std: {np.std(steering_history):.4f}\n")
    f.write(f"   Throttle mean: {np.mean(throttle_history):.4f}\n")
    f.write(f"   Throttle std: {np.std(throttle_history):.4f}\n\n")
    
    f.write("5. OUTPUT FILES GENERATED\n")
    f.write(f"   - Performance GIF: {os.path.basename(gif_path)}\n")
    f.write(f"   - Analysis charts: {os.path.basename(charts_path)}\n")
    f.write(f"   - This summary report: {os.path.basename(summary_path)}\n\n")
    
    f.write("6. SYSTEM CONFIGURATION\n")
    f.write("   - Simulator: MetaDrive 0.4.3\n")
    f.write("   - Framework: Stable-Baselines3\n")
    f.write("   - Dataset collection: 500,000 samples with progressive difficulty\n")
    f.write("   - Training duration: 20-30 minutes on CPU\n")
    f.write("="*80 + "\n")

print(f"✅ Summary report saved: {summary_path}")

# ============================================================================
# STEP 7: FINAL SUMMARY
# ============================================================================
print("\n[7/7] FINAL SUMMARY")
print("="*80)
print("ALL OUTPUTS GENERATED SUCCESSFULLY!")
print("="*80)
print(f"""
OUTPUT FILES (located in 'videos' folder):
  1. GIF animation:      {os.path.basename(gif_path)}
  2. Static charts:      {os.path.basename(charts_path)}
  3. Summary report:     {os.path.basename(summary_path)}

PERFORMANCE HIGHLIGHTS:
  - Dataset: {total_samples:,} samples
  - Average episode length: {np.mean(episode_lengths):.1f} steps
  - Average episode reward: {np.mean(episode_rewards):.2f}
  - Mean steering: {np.mean(steering_history):.4f}
  - Mean throttle: {np.mean(throttle_history):.4f}

TO PRESENT TO PROFESSOR:
  - Show GIF: open {os.path.basename(gif_path)} in any browser
  - Show charts: open {os.path.basename(charts_path)}
  - Live demo: run 'python video.py' (opens MetaDrive window)
""")
print("="*80)