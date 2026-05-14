"""
Soft Actor-Critic (SAC) Implementation
State-of-the-art reinforcement learning for continuous control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class SoftQNetwork(nn.Module):
    """
    Soft Q-Network for SAC
    
    Uses two Q-networks to reduce overestimation bias
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network (twin)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """
        Get Q-values for state-action pairs
        
        Args:
            state: State tensor (batch, state_dim)
            action: Action tensor (batch, action_dim)
        
        Returns:
            q1, q2: Two Q-value estimates
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state, action):
        """Get only Q1 (used for policy optimization)"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy for SAC
    
    Outputs mean and standard deviation of action distribution
    Uses reparameterization trick for differentiable sampling
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state, deterministic=False, with_logprob=True):
        """
        Get action and log probability
        
        Args:
            state: State tensor (batch, state_dim)
            deterministic: If True, return mean without sampling
            with_logprob: If True, compute log probability
        
        Returns:
            action: Action tensor (batch, action_dim)
            log_prob: Log probability (batch, 1)
        """
        mean = self.mean_net(state)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
            normal = torch.distributions.Normal(mean, std)
            
            # Reparameterization trick
            z = normal.rsample()
            action = torch.tanh(z)
            
            if with_logprob:
                # Compute log probability with change of variables
                log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = None
        
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        """Get action without log probability (for inference)"""
        action, _ = self.forward(state, deterministic, with_logprob=False)
        return action


class SoftActorCritic:
    """
    Complete Soft Actor-Critic Implementation
    
    Features:
    - Automatic entropy tuning
    - Twin Q-networks
    - Target networks for stability
    """
    
    def __init__(self, state_dim, action_dim, 
                 learning_rate=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 target_entropy=None,
                 device='cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Target entropy (auto-tuning)
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        # Create networks
        self.critic = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        
        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Learnable temperature parameter
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Replay buffer (will be set externally)
        self.replay_buffer = None
        
        # Statistics
        self.train_stats = {
            'critic_loss': [],
            'policy_loss': [],
            'alpha_loss': [],
            'alpha': []
        }
    
    @property
    def alpha(self):
        """Get current temperature value"""
        return self.log_alpha.exp()
    
    def select_action(self, state, deterministic=False):
        """
        Select action for environment interaction
        
        Args:
            state: numpy array or tensor
            deterministic: If True, no exploration noise
        
        Returns:
            action: numpy array
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.policy.get_action(state, deterministic)
        
        return action.cpu().numpy()[0]
    
    def update(self, batch):
        """
        Update networks using batch from replay buffer
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones
        
        Returns:
            stats: Dictionary of loss values
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        weights = batch.get('weights', None)
        
        if weights is not None:
            weights = weights.to(self.device)
        
        # ============================================================
        # Update Critic (Q-Networks)
        # ============================================================
        with torch.no_grad():
            # Sample next actions from policy
            next_actions, next_log_probs = self.policy.forward(next_states)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss (weighted if using PER)
        if weights is not None:
            critic_loss = (weights * F.mse_loss(current_q1, target_value, reduction='none')).mean()
            critic_loss += (weights * F.mse_loss(current_q2, target_value, reduction='none')).mean()
        else:
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ============================================================
        # Update Policy (Actor)
        # ============================================================
        new_actions, log_probs = self.policy.forward(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha.detach() * log_probs - q_new).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # ============================================================
        # Update Temperature (Alpha)
        # ============================================================
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ============================================================
        # Soft Update Target Networks
        # ============================================================
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Store statistics
        self.train_stats['critic_loss'].append(critic_loss.item())
        self.train_stats['policy_loss'].append(policy_loss.item())
        self.train_stats['alpha_loss'].append(alpha_loss.item())
        self.train_stats['alpha'].append(self.alpha.item())
        
        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'policy': self.policy.state_dict(),
            'log_alpha': self.log_alpha,
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'train_stats': self.train_stats
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.log_alpha = checkpoint['log_alpha']
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.train_stats = checkpoint['train_stats']
        print(f"📂 Model loaded from {path}")


# Test the SAC implementation
if __name__ == "__main__":
    print("Testing Soft Actor-Critic...")
    
    # Create SAC agent
    sac = SoftActorCritic(state_dim=100, action_dim=2, device='cpu')
    
    # Create dummy batch
    batch = {
        'states': torch.randn(32, 100),
        'actions': torch.randn(32, 2),
        'rewards': torch.randn(32, 1),
        'next_states': torch.randn(32, 100),
        'dones': torch.zeros(32, 1)
    }
    
    # Test update
    stats = sac.update(batch)
    
    print(f"\n✅ SAC works!")
    print(f"   Critic loss: {stats['critic_loss']:.4f}")
    print(f"   Policy loss: {stats['policy_loss']:.4f}")
    print(f"   Alpha: {stats['alpha']:.4f}")
    
    # Test action selection
    dummy_state = np.random.randn(100)
    action = sac.select_action(dummy_state)
    print(f"   Action shape: {action.shape}")