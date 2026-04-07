"""
Training Logger for Experiment Tracking
Logs metrics, saves checkpoints, and tracks training progress
"""

import json
import os
from datetime import datetime
import numpy as np

class TrainingLogger:
    """
    Professional training logger
    
    Features:
    - Logs episode rewards, losses, and metrics
    - Saves to JSON format for analysis
    - Tracks best performing models
    - Provides training summaries
    """
    
    def __init__(self, log_dir="./logs", experiment_name=None):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create experiment name
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize log storage
        self.logs = {
            "episodes": [],
            "rewards": [],
            "lengths": [],
            "losses": [],
            "timestamps": [],
            "metadata": {
                "experiment_name": experiment_name,
                "start_time": datetime.now().isoformat(),
                "config": {}
            }
        }
        
        # Tracking best model
        self.best_reward = -float('inf')
        self.best_episode = 0
        
        print(f"✅ TrainingLogger initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Log directory: {self.experiment_dir}")
    
    def log_episode(self, episode, reward, length, loss=None):
        """
        Log episode metrics
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length in steps
            loss: Optional training loss
        """
        self.logs["episodes"].append(episode)
        self.logs["rewards"].append(reward)
        self.logs["lengths"].append(length)
        self.logs["timestamps"].append(datetime.now().isoformat())
        
        if loss is not None:
            self.logs["losses"].append(loss)
        
        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode = episode
        
        # Print progress
        avg_reward = np.mean(self.logs["rewards"][-10:]) if len(self.logs["rewards"]) >= 10 else reward
        
        print(f"📊 Episode {episode:>4} | Reward: {reward:>7.2f} | "
              f"Length: {length:>4} | Avg(10): {avg_reward:>7.2f} | "
              f"Best: {self.best_reward:>7.2f}")
    
    def log_metrics(self, metrics, step=None):
        """
        Log custom metrics
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        timestamp = datetime.now().isoformat()
        
        if "metrics" not in self.logs:
            self.logs["metrics"] = []
        
        entry = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        if step is not None:
            entry["step"] = step
        
        self.logs["metrics"].append(entry)
    
    def log_config(self, config):
        """
        Log experiment configuration
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.logs["metadata"]["config"] = config
        print("📋 Configuration logged")
    
    def save_checkpoint(self, model, optimizer, episode, save_best=False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer to save
            episode: Current episode number
            save_best: Whether this is the best model
        """
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_reward': self.best_reward,
            'logs': self.logs
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if save_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved at episode {episode} (reward: {self.best_reward:.2f})")
    
    def save(self):
        """Save all logs to JSON file"""
        log_path = os.path.join(self.experiment_dir, "training_logs.json")
        
        with open(log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"💾 Logs saved to {log_path}")
        return log_path
    
    def get_summary(self):
        """
        Get training summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.logs["rewards"]:
            return {"error": "No episodes logged yet"}
        
        rewards = self.logs["rewards"]
        lengths = self.logs["lengths"]
        
        summary = {
            "total_episodes": len(rewards),
            "total_steps": sum(lengths),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "best_reward": max(rewards),
            "best_episode": self.logs["episodes"][np.argmax(rewards)],
            "mean_length": np.mean(lengths),
            "final_reward": rewards[-1] if rewards else 0,
            "improvement": rewards[-1] - rewards[0] if len(rewards) > 1 else 0
        }
        
        return summary
    
    def plot_summary(self, save_path=None):
        """
        Create summary plots of training progress
        
        Args:
            save_path: Path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Reward plot
        axes[0, 0].plot(self.logs["episodes"], self.logs["rewards"], 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title("Episode Rewards", fontsize=12)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Moving average
        if len(self.logs["rewards"]) > 10:
            window = 10
            moving_avg = np.convolve(self.logs["rewards"], np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.logs["rewards"])), moving_avg, 
                           'r-', linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 0].legend()
        
        # Episode length plot
        axes[0, 1].plot(self.logs["episodes"], self.logs["lengths"], 'g-', alpha=0.7, linewidth=1)
        axes[0, 1].set_title("Episode Lengths", fontsize=12)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward histogram
        axes[1, 0].hist(self.logs["rewards"], bins=20, edgecolor='black', alpha=0.7, color='blue')
        axes[1, 0].set_title("Reward Distribution", fontsize=12)
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].axvline(x=np.mean(self.logs["rewards"]), color='r', 
                          linestyle='--', label=f'Mean: {np.mean(self.logs["rewards"]):.2f}')
        axes[1, 0].legend()
        
        # Loss plot (if available)
        if self.logs["losses"]:
            axes[1, 1].plot(range(len(self.logs["losses"])), self.logs["losses"], 'r-', alpha=0.7, linewidth=1)
            axes[1, 1].set_title("Training Loss", fontsize=12)
            axes[1, 1].set_xlabel("Update Step")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No loss data available", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Training Loss", fontsize=12)
        
        plt.suptitle(f"Training Summary - {self.experiment_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, "training_summary.png")
        
        plt.savefig(save_path, dpi=150, facecolor='white')
        print(f"📊 Summary plot saved to {save_path}")
        plt.close()
        
        return save_path
    
    def print_summary(self):
        """Print training summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("📊 TRAINING SUMMARY")
        print("="*50)
        print(f"Experiment: {self.experiment_name}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Best Reward: {summary['best_reward']:.2f} (Episode {summary['best_episode']})")
        print(f"Final Reward: {summary['final_reward']:.2f}")
        print(f"Improvement: {summary['improvement']:.2f}")
        print("="*50)
        
        return summary


# Simple console logger for quick logging
class ConsoleLogger:
    """
    Simple console logger for lightweight logging
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.episode_rewards = []
        self.episode_lengths = []
    
    def log(self, episode, reward, length, loss=None):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if self.verbose and episode % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"Episode {episode:>4} | Reward: {reward:>7.2f} | Avg(10): {avg_reward:>7.2f}")
    
    def get_stats(self):
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "best_reward": max(self.episode_rewards),
            "total_episodes": len(self.episode_rewards)
        }


# Test the logger
if __name__ == "__main__":
    print("Testing TrainingLogger...")
    
    # Create logger
    logger = TrainingLogger(experiment_name="test_experiment")
    
    # Log configuration
    logger.log_config({
        "algorithm": "PPO",
        "learning_rate": 0.0003,
        "total_timesteps": 10000
    })
    
    # Simulate training
    for episode in range(1, 51):
        reward = 10 + np.random.randn() * 5 + episode * 0.1
        length = 500 + int(np.random.randn() * 100)
        loss = max(0, 1.0 - episode * 0.01 + np.random.randn() * 0.1)
        
        logger.log_episode(episode, reward, length, loss)
    
    # Save logs
    logger.save()
    
    # Print summary
    logger.print_summary()
    
    # Create summary plot
    logger.plot_summary()
    
    print("\n✅ Logger works correctly!")
    print(f"   Logs saved to: {logger.experiment_dir}")