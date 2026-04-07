"""
Professional Dataset Collector for MetaDrive
Collects 5000+ driving samples for training
"""

import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
import os
from metadrive import MetaDriveEnv

class ProfessionalDatasetCollector:
    """
    Professional dataset collector with:
    - Progress bars
    - Metadata tracking
    - Automatic file naming
    """
    
    def __init__(self, save_dir="./data/raw"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Dataset Collector Ready → {save_dir}")
    
    def collect(self, target_samples=5000):
        """
        Collect target_samples number of driving transitions
        """
        print(f"\n{'='*50}")
        print(f"📊 COLLECTING {target_samples} SAMPLES")
        print(f"{'='*50}")
        
        # Configure MetaDrive
        config = {
            "use_render": False,
            "traffic_density": 0.1,
            "num_scenarios": 20,
            "start_seed": 42,
        }
        
        env = MetaDriveEnv(config)
        
        # Initialize dataset storage
        dataset = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "terminals": [],
            "metadata": {
                "target_samples": target_samples,
                "collection_time": datetime.now().isoformat(),
                "config": config
            }
        }
        
        obs, _ = env.reset()
        collected = 0
        episode_count = 0
        
        # Progress bar
        pbar = tqdm(total=target_samples, desc="Collecting samples")
        
        while collected < target_samples:
            # Different driving strategies for diversity
            if collected < 1000:
                action = env.action_space.sample()  # Random exploration
            elif collected < 3000:
                action = np.array([0.0, 0.5])  # Straight driving
            else:
                action = np.array([0.3, 0.3])  # Curved driving
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            dataset["observations"].append(obs)
            dataset["actions"].append(action)
            dataset["rewards"].append(reward)
            dataset["next_observations"].append(next_obs)
            dataset["terminals"].append(done)
            
            # Update
            obs = next_obs
            collected += 1
            pbar.update(1)
            
            if done:
                obs, _ = env.reset()
                episode_count += 1
        
        pbar.close()
        env.close()
        
        # Add statistics to metadata
        dataset["metadata"]["actual_samples"] = collected
        dataset["metadata"]["episodes"] = episode_count
        dataset["metadata"]["mean_reward"] = float(np.mean(dataset["rewards"]))
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{target_samples}_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"\n✅ DATASET SAVED!")
        print(f"   File: {filename}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Samples: {collected}")
        print(f"   Episodes: {episode_count}")
        
        return dataset, filepath


# Run directly to collect dataset
if __name__ == "__main__":
    collector = ProfessionalDatasetCollector()
    collector.collect(5000)