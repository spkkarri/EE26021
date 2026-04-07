import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import glob
import pickle
import subprocess
import time
from datetime import datetime


# CONFIGURATION

TARGET_SAMPLES = 500000
TRAINING_STEPS = 200000
GIF_FRAMES = 300
GIF_FPS = 20

print("="*80)
print("PROFESSIONAL SELF-DRIVING CAR SYSTEM")
print("="*80)
print("""
This system will:
  1. Manage your 500,000 sample dataset
  2. Train a PPO model on your data
  3. Generate professional analysis charts
  4. Create a performance graph GIF
  5. Prepare live demo (video.py) for presentation
""")
print("="*80)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("models_saved", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("logs", exist_ok=True)

start_time = time.time()

# STEP 1: ADVANCED DATASET MANAGEMENT

print("\n" + "="*80)
print("STEP 1: DATASET MANAGEMENT")
print("="*80)

existing_datasets = glob.glob("data/raw/dataset_*.pkl")

if existing_datasets:
    dataset_sizes = []
    for f in existing_datasets:
        try:
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            samples = len(data.get("observations", []))
            dataset_sizes.append((f, samples))
            print(f"  Found: {os.path.basename(f)} -> {samples:,} samples")
        except Exception as e:
            print(f"  Warning: Could not read {os.path.basename(f)}: {e}")

    if dataset_sizes:
        dataset_sizes.sort(key=lambda x: x[1], reverse=True)
        latest_dataset, sample_count = dataset_sizes[0]
        print(f"\nSelected dataset: {os.path.basename(latest_dataset)}")
        print(f"Current samples: {sample_count:,}")
        print(f"Target samples:  {TARGET_SAMPLES:,}")
        print(f"Progress:        {sample_count/TARGET_SAMPLES*100:.1f}%")

        print("\n" + "-"*50)
        print("OPTIONS:")
        print("-"*50)
        print("  [1] Use existing dataset (recommended - saves time)")
        print("  [2] Collect additional samples to reach 500,000")
        print("  [3] Start fresh with new dataset (deletes existing)")
        print("-"*50)

        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '2':
            remaining = TARGET_SAMPLES - sample_count
            if remaining > 0:
                print(f"\nNeed {remaining:,} more samples to reach {TARGET_SAMPLES:,}")
                print("Starting incremental dataset collection...")
                print("This will take 2-3 hours.\n")
                result = subprocess.run([sys.executable, "collect_dataset.py"])
                if result.returncode != 0:
                    print("\nERROR: Dataset collection failed!")
                    sys.exit(1)
            else:
                print("\nDataset already meets target size.")

        elif choice == '3':
            print("\nWARNING: This will delete all existing datasets!")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm == 'CONFIRM':
                for f in existing_datasets:
                    try:
                        os.remove(f)
                        print(f"Deleted: {os.path.basename(f)}")
                    except Exception as e:
                        print(f"Could not delete {os.path.basename(f)}: {e}")
                print("\nStarting fresh dataset collection...")
                result = subprocess.run([sys.executable, "collect_dataset.py"])
                if result.returncode != 0:
                    print("\nERROR: Dataset collection failed!")
                    sys.exit(1)
            else:
                print("\nKeeping existing dataset.")
        else:
            print("\nUsing existing dataset.")
    else:
        print("\nNo readable dataset found. Starting fresh collection...")
        result = subprocess.run([sys.executable, "collect_dataset.py"])
        if result.returncode != 0:
            print("\nERROR: Dataset collection failed!")
            sys.exit(1)
else:
    print("\nNo existing dataset found.")
    print("Starting fresh dataset collection (500,000 samples)...")
    print("This will take 2-3 hours.\n")
    result = subprocess.run([sys.executable, "collect_dataset.py"])
    if result.returncode != 0:
        print("\nERROR: Dataset collection failed!")
        sys.exit(1)

# Verify dataset after collection

dataset_files = glob.glob("data/raw/dataset_*.pkl")
if not dataset_files:
    print("\nERROR: Dataset collection failed - no dataset files found!")
    sys.exit(1)

dataset_sizes = []
for f in dataset_files:
    try:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        samples = len(data.get("observations", []))
        dataset_sizes.append((f, samples))
    except Exception as e:
        print(f"Warning: Could not read {os.path.basename(f)}: {e}")

if dataset_sizes:
    dataset_sizes.sort(key=lambda x: x[1], reverse=True)
    latest_dataset, final_sample_count = dataset_sizes[0]
else:
    print("\nERROR: Could not read any dataset file!")
    sys.exit(1)

print(f"\nDataset ready: {os.path.basename(latest_dataset)}")
print(f"Total samples: {final_sample_count:,}")

if final_sample_count < TARGET_SAMPLES:
    print(f"WARNING: Only {final_sample_count:,} samples. Recommended: {TARGET_SAMPLES:,}")
    response = input("Continue with training? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# STEP 2: TRAIN MODEL AND CREATE OUTPUTS

print("\n" + "="*80)
print("STEP 2: MODEL TRAINING AND OUTPUT GENERATION")
print("="*80)
print(f"""
Training Configuration:
  - Algorithm: PPO (Proximal Policy Optimization)
  - Training steps: {TRAINING_STEPS:,}
  - Environment maps: 10 different scenarios
  - Traffic density: 0.03 (light traffic)

Output Configuration:
  - GIF duration: {GIF_FRAMES/GIF_FPS:.0f} seconds
  - Charts: 6 professional analysis charts
  - Live demo: video.py (ready for presentation)
""")
print("-"*80)

print("\nStarting training and output generation...")
print("This will take 20-30 minutes. Progress shown below.\n")

training_start = time.time()
result = subprocess.run([sys.executable, "train_and_animate.py"])
training_time = time.time() - training_start

if result.returncode != 0:
    print("\nERROR: Training failed!")
    sys.exit(1)

print(f"\nTraining completed in {training_time/60:.1f} minutes")


# STEP 3: RESULTS SUMMARY

print("\n" + "="*80)
print("STEP 3: RESULTS SUMMARY")
print("="*80)

gif_files = glob.glob("videos/*.gif")
chart_files = glob.glob("videos/analysis_charts_*.png")
summary_files = glob.glob("videos/summary_report_*.txt")
model_files = glob.glob("models_saved/self_driving_model_*.zip")

total_time = time.time() - start_time

print(f"\nTotal execution time: {total_time/60:.1f} minutes")

print("\n" + "-"*80)
print("GENERATED FILES")
print("-"*80)

if gif_files:
    for f in gif_files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  [GIF]      {os.path.basename(f)} ({size:.2f} MB)")
else:
    print("  [GIF]      No GIF file found")

if chart_files:
    for f in chart_files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  [CHARTS]   {os.path.basename(f)} ({size:.2f} MB)")
else:
    print("  [CHARTS]   No charts found")

if summary_files:
    for f in summary_files:
        size = os.path.getsize(f) / (1024)
        print(f"  [SUMMARY]  {os.path.basename(f)} ({size:.2f} KB)")
else:
    print("  [SUMMARY]  No summary report found")

if model_files:
    latest_model = sorted(model_files)[-1]
    size = os.path.getsize(latest_model) / (1024)
    print(f"  [MODEL]    {os.path.basename(latest_model)} ({size:.2f} KB)")
else:
    print("  [MODEL]    No model found")

print("\n" + "-"*80)
print("DATASET INFORMATION")
print("-"*80)
print(f"  File:     {os.path.basename(latest_dataset)}")
print(f"  Samples:  {final_sample_count:,}")
print(f"  Size:     {os.path.getsize(latest_dataset)/(1024*1024):.2f} MB")


# STEP 4: LIVE DEMO (OPTIONAL)

print("\n" + "="*80)
print("STEP 4: LIVE DEMO (OPTIONAL)")
print("="*80)
choice = input("\nRun live demo (video.py) now? (y/n): ")
if choice.lower() == 'y':
    print("\nStarting live demo...")
    subprocess.run([sys.executable, "video.py"])
else:
    print("\nYou can run live demo later with: python video.py")


# STEP 5: PRESENTATION GUIDE

print("\n" + "="*80)
print("PRESENTATION GUIDE")
print("="*80)

print(f"""
TO PRESENT TO YOUR PROFESSOR:

1. LIVE DEMO (Recommended - Shows car driving in real-time):
   Run: python video.py
   - A MetaDrive window will open
   - The car will drive automatically
   - Watch the car navigate the road

2. SHOW GIF ANIMATION:
   Open the GIF file in the 'videos' folder
   - Double-click to play in browser

3. SHOW PERFORMANCE CHARTS:
   Open analysis_charts.png in the 'videos' folder
   - Shows 6 professional metrics:
     * Rewards over time
     * Cumulative reward
     * Steering distribution
     * Throttle distribution
     * Reward distribution
     * Action space scatter

4. SHOW SUMMARY REPORT:
   Open summary_report_*.txt in the 'videos' folder
   - Contains all metrics, configuration, and results

5. SHOW DATASET:
   Location: data/raw/
   - {final_sample_count:,} samples collected
   - Provenance: MetaDrive simulator

6. SHOW MODEL:
   Location: models_saved/
   - PPO algorithm
   - Trained for {TRAINING_STEPS:,} steps
""")

print("\n" + "="*80)
print("PROCESS COMPLETE")
print("="*80)
print(f"""
SUMMARY:
  - Dataset:     {final_sample_count:,} samples
  - Model:       Trained for {TRAINING_STEPS:,} steps
  - GIF:         {GIF_FRAMES/GIF_FPS:.0f} second animation
  - Charts:      6 professional analysis charts
  - Report:      Detailed summary
  - Total time:  {total_time/60:.1f} minutes

OUTPUT LOCATION:
  - GIF & Charts & Report: ./videos/
  - Model:                 ./models_saved/
  - Dataset:               ./data/raw/

READY FOR PRESENTATION:
  Run: python video.py
""")
print("="*80)