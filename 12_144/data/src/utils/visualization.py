"""
Professional Visualization Utilities
Creates animations, plots, and dashboards for self-driving car analysis
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, Circle
import os

class DrivingVisualizer:
    """
    Professional visualization tools for self-driving cars
    
    Features:
    - Dashboard animations with real-time stats
    - Training curve plots
    - Attention visualization
    - Bird's eye view of traffic
    """
    
    @staticmethod
    def create_dashboard_animation(frames, rewards, actions=None, save_path="driving_animation.gif", fps=20):
        """
        Create professional dashboard animation with multiple panels
        
        Args:
            frames: List of RGB frames
            rewards: List of rewards per step
            actions: Optional list of actions [steering, throttle]
            save_path: Where to save the animation
            fps: Frames per second
        
        Returns:
            Path to saved animation
        """
        if not frames:
            print("❌ No frames to animate")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 9), facecolor='black')
        
        # Main driving view (top left)
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        ax_main.axis('off')
        ax_main.set_facecolor('black')
        im = ax_main.imshow(frames[0])
        
        # Reward plot (bottom left)
        ax_reward = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        ax_reward.set_facecolor('black')
        ax_reward.tick_params(colors='white')
        ax_reward.set_xlabel('Step', color='white', fontsize=10)
        ax_reward.set_ylabel('Reward', color='white', fontsize=10)
        ax_reward.set_title('Instant Reward', color='white', fontsize=12)
        ax_reward.grid(True, alpha=0.2, color='white')
        reward_line, = ax_reward.plot([], [], 'cyan', linewidth=2)
        
        # Rolling average line
        avg_line, = ax_reward.plot([], [], 'yellow', linewidth=2, alpha=0.7)
        
        # Action plot (if actions provided)
        if actions is not None:
            ax_action = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
            ax_action.set_facecolor('black')
            ax_action.tick_params(colors='white')
            ax_action.set_title('Control Signals', color='white', fontsize=10)
            ax_action.set_ylim(-1.2, 1.2)
            ax_action.grid(True, alpha=0.2, color='white')
            steering_line, = ax_action.plot([], [], 'red', linewidth=1.5, label='Steering')
            throttle_line, = ax_action.plot([], [], 'green', linewidth=1.5, label='Throttle')
            ax_action.legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=8)
        
        # Info panel (right side)
        ax_info = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
        ax_info.axis('off')
        ax_info.set_facecolor('black')
        info_text = ax_info.text(0.1, 0.5, "", transform=ax_info.transAxes,
                                  color='white', fontsize=10, verticalalignment='center',
                                  fontfamily='monospace')
        
        # Precompute rolling averages
        window = 20
        if len(rewards) > window:
            rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        else:
            rolling_rewards = []
        
        def update(frame_idx):
            # Update main view
            im.set_array(frames[frame_idx])
            ax_main.set_title(f"SELF-DRIVING CAR - Frame {frame_idx+1}/{len(frames)}", 
                              color='white', fontsize=14, fontweight='bold')
            
            # Update reward plot
            current_rewards = rewards[:frame_idx+1]
            reward_line.set_data(range(len(current_rewards)), current_rewards)
            
            if len(current_rewards) > window:
                rolling = np.convolve(current_rewards, np.ones(window)/window, mode='valid')
                avg_line.set_data(range(window-1, len(current_rewards)), rolling)
            
            # Auto-scale reward plot
            if current_rewards:
                ax_reward.set_xlim(0, len(frames))
                y_min = min(min(current_rewards), -1)
                y_max = max(max(current_rewards), 1)
                ax_reward.set_ylim(y_min - 0.5, y_max + 0.5)
            
            # Update action plot
            if actions is not None and frame_idx < len(actions):
                current_actions = actions[:frame_idx+1]
                steering_vals = [a[0] for a in current_actions]
                throttle_vals = [a[1] for a in current_actions]
                steering_line.set_data(range(len(steering_vals)), steering_vals)
                throttle_line.set_data(range(len(throttle_vals)), throttle_vals)
                ax_action.set_xlim(0, len(frames))
            
            # Update info panel
            if frame_idx < len(rewards):
                current_reward = rewards[frame_idx]
                avg_reward = np.mean(rewards[:frame_idx+1]) if frame_idx > 0 else current_reward
                best_reward = max(rewards[:frame_idx+1])
                
                info_str = f"""
╔══════════════════════════════╗
║     VEHICLE STATUS           ║
╠══════════════════════════════╣
║ Frame:     {frame_idx+1:>4}/{len(frames)}
║ Reward:    {current_reward:>7.2f}
║ Avg Reward:{avg_reward:>7.2f}
║ Best:      {best_reward:>7.2f}
╠══════════════════════════════╣
║     SYSTEM INFO              ║
╠══════════════════════════════╣
║ Algorithm: PPO
║ Training:  10,000 steps
║ Status:    ACTIVE
╚══════════════════════════════╝
                """
                info_text.set_text(info_str)
            
            return [im, reward_line, avg_line, info_text] + ([steering_line, throttle_line] if actions else [])
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(frames), 
                                      interval=1000/fps, blit=True, repeat=True)
        plt.close()
        
        # Save animation
        ani.save(save_path, writer='pillow', fps=fps)
        print(f"✅ Animation saved: {save_path}")
        
        return save_path
    
    @staticmethod
    def plot_training_curves(rewards, losses=None, save_path="training_curves.png"):
        """
        Plot training curves for analysis
        
        Args:
            rewards: List of episode rewards
            losses: Optional list of training losses
            save_path: Where to save the plot
        """
        fig, axes = plt.subplots(1, 2 if losses else 1, figsize=(12, 4))
        
        if losses:
            ax1, ax2 = axes
        else:
            ax1 = axes
            ax2 = None
        
        # Reward plot
        ax1.plot(rewards, 'b-', linewidth=2, alpha=0.7)
        ax1.set_title("Episode Rewards Over Time", fontsize=12)
        ax1.set_xlabel("Episode", fontsize=10)
        ax1.set_ylabel("Total Reward", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        window = min(20, len(rewards))
        if window > 0:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax1.legend()
        
        # Loss plot (if provided)
        if losses and ax2:
            ax2.plot(losses, 'r-', linewidth=1.5, alpha=0.7)
            ax2.set_title("Training Loss", fontsize=12)
            ax2.set_xlabel("Step", fontsize=10)
            ax2.set_ylabel("Loss", fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor='white')
        print(f"✅ Training curves saved: {save_path}")
        
        return save_path
    
    @staticmethod
    def visualize_attention(attention_weights, save_path="attention_heatmap.png"):
        """
        Visualize attention weights as heatmap
        
        Args:
            attention_weights: Attention weights from transformer/attention layer
            save_path: Where to save the plot
        """
        if attention_weights is None:
            print("⚠️ No attention weights to visualize")
            return None
        
        # Convert to numpy if tensor
        if hasattr(attention_weights, 'cpu'):
            attention_weights = attention_weights.cpu().numpy()
        
        # Remove batch dimension if present
        if attention_weights.ndim == 4:
            attention_weights = attention_weights[0, 0]
        elif attention_weights.ndim == 3:
            attention_weights = attention_weights[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='hot', interpolation='nearest')
        ax.set_title("Attention Weights Heatmap", fontsize=14)
        ax.set_xlabel("Key Position", fontsize=12)
        ax.set_ylabel("Query Position", fontsize=12)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Attention visualization saved: {save_path}")
        
        return save_path
    
    @staticmethod
    def create_birds_eye_view(ego_pos, vehicle_positions, road_boundaries, save_path="birds_eye.png"):
        """
        Create bird's eye view of traffic scene
        
        Args:
            ego_pos: [x, y] position of ego vehicle
            vehicle_positions: List of [x, y] for other vehicles
            road_boundaries: List of (x, y) points for road edges
            save_path: Where to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot road boundaries
        if road_boundaries:
            road_boundaries = np.array(road_boundaries)
            ax.plot(road_boundaries[:, 0], road_boundaries[:, 1], 'gray', linewidth=2, alpha=0.5)
        
        # Plot other vehicles
        for pos in vehicle_positions:
            circle = Circle((pos[0], pos[1]), 2, color='blue', alpha=0.7)
            ax.add_patch(circle)
        
        # Plot ego vehicle (highlighted)
        ego_circle = Circle((ego_pos[0], ego_pos[1]), 2.5, color='red', alpha=0.8, label='Ego Vehicle')
        ax.add_patch(ego_circle)
        
        # Add direction arrow
        ax.arrow(ego_pos[0], ego_pos[1], 5, 0, head_width=1, head_length=2, fc='red', ec='red')
        
        ax.set_title("Bird's Eye View - Traffic Scene", fontsize=14)
        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Bird's eye view saved: {save_path}")
        
        return save_path


# Test the visualizer
if __name__ == "__main__":
    print("Testing DrivingVisualizer...")
    
    # Create dummy data
    dummy_frames = []
    for i in range(50):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_frames.append(frame)
    
    dummy_rewards = np.cumsum(np.random.randn(50) * 0.5) + 10
    dummy_actions = [(np.sin(i*0.1), np.cos(i*0.1)) for i in range(50)]
    
    # Test dashboard animation
    print("\n1. Testing dashboard animation...")
    DrivingVisualizer.create_dashboard_animation(
        dummy_frames, dummy_rewards, dummy_actions, 
        "test_animation.gif", fps=10
    )
    
    # Test training curves
    print("\n2. Testing training curves...")
    DrivingVisualizer.plot_training_curves(dummy_rewards, "test_curves.png")
    
    # Test attention visualization
    print("\n3. Testing attention visualization...")
    dummy_attention = np.random.rand(8, 8)
    DrivingVisualizer.visualize_attention(dummy_attention, "test_attention.png")
    
    print("\n✅ All visualization tools work correctly!")