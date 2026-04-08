"""
Professional Dataset Manager
Handles dataset versioning, loading, and train/val/test splits
"""

import os
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

class DatasetManager:
    """
    Professional dataset manager with:
    - Version control
    - Reproducible splits
    - Metadata tracking
    """
    
    def __init__(self, base_dir="./data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.datasets_dir = os.path.join(base_dir, "datasets")
        self.splits_dir = os.path.join(base_dir, "splits")
        
        # Create all directories
        for d in [self.raw_dir, self.processed_dir, self.datasets_dir, self.splits_dir]:
            os.makedirs(d, exist_ok=True)
        
        print("✅ Dataset Manager Ready")
        print(f"   Raw: {self.raw_dir}")
        print(f"   Splits: {self.splits_dir}")
    
    def load_dataset(self, filepath):
        """Load a dataset from pickle file"""
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
        print(f"📂 Loaded: {os.path.basename(filepath)}")
        print(f"   Samples: {len(dataset['observations'])}")
        return dataset
    
    def create_splits(self, dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Create reproducible train/val/test splits
        
        Args:
            dataset_path: Path to dataset pickle file
            train_ratio: 70% for training
            val_ratio: 15% for validation
            test_ratio: 15% for testing
            random_seed: Fixed seed for reproducibility
        """
        print(f"\n{'='*50}")
        print("📊 CREATING TRAIN/VAL/TEST SPLITS")
        print(f"{'='*50}")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        n = len(dataset["observations"])
        indices = np.arange(n)
        
        # First split: train vs temp (val+test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val vs test from temp
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            random_state=random_seed,
            shuffle=True
        )
        
        # Create splits dictionary
        splits = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "metadata": {
                "dataset_source": dataset_path,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "random_seed": random_seed,
                "total_samples": n,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Save splits to JSON
        dataset_name = os.path.basename(dataset_path).replace(".pkl", "")
        split_path = os.path.join(self.splits_dir, f"{dataset_name}_splits.json")
        
        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)
        
        print(f"\n✅ Splits saved to: {split_path}")
        print(f"   Train: {len(train_idx)} samples ({train_ratio*100:.0f}%)")
        print(f"   Validation: {len(val_idx)} samples ({val_ratio*100:.0f}%)")
        print(f"   Test: {len(test_idx)} samples ({test_ratio*100:.0f}%)")
        
        return splits
    
    def get_split_data(self, dataset_path, split_name="train"):
        """
        Get a specific split (train/val/test) as a dataset
        
        Usage:
            train_data = manager.get_split_data("dataset.pkl", "train")
            val_data = manager.get_split_data("dataset.pkl", "val")
            test_data = manager.get_split_data("dataset.pkl", "test")
        """
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Load splits
        dataset_name = os.path.basename(dataset_path).replace(".pkl", "")
        split_path = os.path.join(self.splits_dir, f"{dataset_name}_splits.json")
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"No splits found. Run create_splits() first.")
        
        with open(split_path, "r") as f:
            splits = json.load(f)
        
        # Get indices for requested split
        indices = splits[f"{split_name}_indices"]
        
        # Create subset dataset
        subset = {
            "observations": [dataset["observations"][i] for i in indices],
            "actions": [dataset["actions"][i] for i in indices],
            "rewards": [dataset["rewards"][i] for i in indices],
            "next_observations": [dataset["next_observations"][i] for i in indices],
            "terminals": [dataset["terminals"][i] for i in indices],
            "metadata": {
                **dataset["metadata"],
                "split": split_name,
                "split_size": len(indices)
            }
        }
        
        print(f"📊 Loaded {split_name} split: {len(indices)} samples")
        return subset
    
    def list_datasets(self):
        """Show all available datasets"""
        print("\n📁 AVAILABLE DATASETS:")
        print("-" * 40)
        
        # Raw datasets
        raw_files = [f for f in os.listdir(self.raw_dir) if f.endswith(".pkl")]
        if raw_files:
            print("\n📂 Raw datasets:")
            for f in raw_files:
                size = os.path.getsize(os.path.join(self.raw_dir, f)) / (1024 * 1024)
                print(f"   • {f} ({size:.2f} MB)")
        
        # Split files
        split_files = [f for f in os.listdir(self.splits_dir) if f.endswith(".json")]
        if split_files:
            print("\n📊 Available splits:")
            for f in split_files:
                print(f"   • {f}")


# Run directly to test
if __name__ == "__main__":
    manager = DatasetManager()
    manager.list_datasets()