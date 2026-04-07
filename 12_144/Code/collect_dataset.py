import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
import os
import glob
from metadrive import MetaDriveEnv

print("="*70)
print("PROFESSIONAL DATASET COLLECTOR - 500,000 SAMPLES")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

SAVE_DIR = "./data/raw"
TARGET_SAMPLES = 500000
NUM_SCENARIOS = 100
TRAFFIC_DENSITY = 0.1
START_SEED = 42
HORIZON = 1000

# ============================================================================
# INITIALIZATION
# ============================================================================

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n[CONFIGURATION]")
print(f"  Target Samples: {TARGET_SAMPLES:,}")
print(f"  Save Directory: {SAVE_DIR}")
print(f"  Num Scenarios: {NUM_SCENARIOS}")
print(f"  Traffic Density: {TRAFFIC_DENSITY}")
print(f"  Start Seed: {START_SEED}")

# ============================================================================
# LOAD EXISTING DATASET
# ============================================================================

print("\n[STEP 1] LOADING EXISTING DATASET")

existing_files = glob.glob(f"{SAVE_DIR}/dataset_*.pkl")
start_count = 0
dataset = None
existing_path = None

if existing_files:
    existing_path = sorted(existing_files)[-1]
    print(f"Found existing dataset: {os.path.basename(existing_path)}")
    try:
        with open(existing_path, "rb") as f:
            dataset = pickle.load(f)
        start_count = len(dataset["observations"])
        print(f"Existing samples: {start_count:,}")
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        print("Starting fresh...")
        dataset = None
        existing_path = None

if dataset is None:
    print("No existing dataset found. Creating new one...")
    dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminals": [],
        "episode_count": 0,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "target_samples": TARGET_SAMPLES,
            "version": "2.0"
        }
    }
    start_count = 0
    existing_path = None

# ============================================================================
# CONFIGURE ENVIRONMENT
# ============================================================================

print("\n[STEP 2] CONFIGURING ENVIRONMENT")

config = {
    "use_render": False,
    "traffic_density": TRAFFIC_DENSITY,
    "num_scenarios": NUM_SCENARIOS,
    "start_seed": START_SEED,
    "horizon": HORIZON,
}

# Suppress the large config dump; just a short message
print("Initializing MetaDrive environment...")
env = MetaDriveEnv(config)
print("Environment ready!")

# ============================================================================
# RESUME COLLECTION
# ============================================================================

print("\n[STEP 3] RESUMING COLLECTION")

if start_count > 0:
    print(f"Resuming from {start_count:,} samples...")
    obs, _ = env.reset()
    # Advance to reasonable state (no verbose per‑iteration prints)
    for _ in range(min(100, start_count // 10)):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
else:
    print("Starting fresh collection...")
    obs, _ = env.reset()

collected = start_count
episode_count = dataset.get("episode_count", 0)

print(f"\nStarting collection. Target: {TARGET_SAMPLES:,} samples")
print(f"Current progress: {collected:,} ({collected/TARGET_SAMPLES*100:.1f}%)")

# ============================================================================
# PROGRESS BAR
# ============================================================================

pbar = tqdm(total=TARGET_SAMPLES, initial=start_count, desc="Collecting samples", unit="samples")

# ============================================================================
# MAIN COLLECTION LOOP
# ============================================================================

print("\n[STEP 4] COLLECTING SAMPLES")

while collected < TARGET_SAMPLES:
    progress = collected / TARGET_SAMPLES
    
    # Progressive difficulty based on collection progress
    if progress < 0.2:
        action = env.action_space.sample()
    elif progress < 0.4:
        if np.random.random() > 0.7:
            action = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(0.3, 0.7)])
        else:
            action = np.array([0.0, 0.5])
    elif progress < 0.6:
        action = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(0.2, 0.6)])
    elif progress < 0.8:
        action = np.array([np.random.uniform(-0.7, 0.7), np.random.uniform(0.2, 0.5)])
    else:
        if np.random.random() > 0.5:
            action = np.array([np.random.uniform(-0.9, 0.9), np.random.uniform(0.1, 0.4)])
        else:
            action = np.array([0.0, np.random.uniform(0.3, 0.6)])

    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    dataset["observations"].append(obs)
    dataset["actions"].append(action)
    dataset["rewards"].append(reward)
    dataset["next_observations"].append(next_obs)
    dataset["terminals"].append(done)

    obs = next_obs
    collected += 1
    pbar.update(1)

    if done:
        obs, _ = env.reset()
        episode_count += 1
        if episode_count % 50 == 0:
            avg_reward = np.mean(dataset["rewards"][-1000:]) if len(dataset["rewards"]) >= 1000 else np.mean(dataset["rewards"])
            print(f"\n   Episode {episode_count} | Samples: {collected:,} | Avg Reward: {avg_reward:.2f}")

    if collected % 50000 == 0 and collected > start_count:
        checkpoint_file = f"{SAVE_DIR}/checkpoint_{collected}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(dataset, f)
        print(f"\n   Checkpoint saved: {collected:,} samples")

pbar.close()
env.close()

# ============================================================================
# CLEANUP AND SAVE
# ============================================================================

print("\n[STEP 5] SAVING DATASET")

dataset["episode_count"] = episode_count
dataset["metadata"]["total_samples"] = collected
dataset["metadata"]["episodes"] = episode_count
dataset["metadata"]["mean_reward"] = float(np.mean(dataset["rewards"]))
dataset["metadata"]["last_updated"] = datetime.now().isoformat()
dataset["metadata"]["completion_time"] = datetime.now().isoformat()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"dataset_{TARGET_SAMPLES}_{timestamp}.pkl"
filepath = os.path.join(SAVE_DIR, filename)

with open(filepath, "wb") as f:
    pickle.dump(dataset, f)

if existing_path and existing_path != filepath:
    try:
        os.remove(existing_path)
        print(f"Removed old dataset: {os.path.basename(existing_path)}")
    except:
        pass

for cf in glob.glob(f"{SAVE_DIR}/checkpoint_*.pkl"):
    try:
        os.remove(cf)
        print(f"Removed checkpoint: {os.path.basename(cf)}")
    except:
        pass

file_size = os.path.getsize(filepath) / (1024 * 1024)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DATASET COLLECTION COMPLETE!")
print("="*70)
print(f"""
SUMMARY:
  - File:        {filename}
  - Location:    {filepath}
  - Size:        {file_size:.2f} MB
  - Samples:     {collected:,}
  - Episodes:    {episode_count}
  - Mean Reward: {dataset['metadata']['mean_reward']:.3f}
  - New Samples: {collected - start_count:,}

STATUS: SUCCESS
""")
print("="*70)

print("\n[VERIFICATION]")
print(f"Dataset loaded successfully: {len(dataset['observations']):,} samples")
print(f"Actions shape: {np.array(dataset['actions']).shape}")
print(f"Rewards range: [{np.min(dataset['rewards']):.2f}, {np.max(dataset['rewards']):.2f}]")

print("\nReady for training!")