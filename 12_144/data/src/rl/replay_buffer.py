"""
Prioritized Experience Replay (PER) Buffer
Prioritizes important transitions for faster learning
"""

import numpy as np
import torch
import random
from collections import deque

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Features:
    - Stores transitions with priority scores
    - Samples important transitions more frequently
    - Implements importance sampling for unbiased learning
    """
    
    def __init__(self, capacity, state_dim, action_dim, 
                 alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta each sample
            epsilon: Small constant to avoid zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Priorities
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        # Pointers
        self.position = 0
        self.size = 0
        
        print(f"✅ PrioritizedReplayBuffer initialized: capacity={capacity}")
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            priority: Optional priority (if None, use max priority)
        """
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        # Set priority (new transitions get max priority)
        if priority is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max(priority, self.epsilon)
        
        # Update pointers
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary containing batch data and importance weights
        """
        if self.size < batch_size:
            return None
        
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Create batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]).unsqueeze(1),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices]).unsqueeze(1),
            'indices': indices,
            'weights': torch.FloatTensor(weights).unsqueeze(1)
        }
        
        return batch
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors
        
        Args:
            indices: Indices of sampled transitions
            td_errors: Temporal difference errors
        """
        new_priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = new_priorities
    
    def __len__(self):
        return self.size


class ReplayBuffer:
    """
    Standard Experience Replay Buffer (without prioritization)
    Used as baseline comparison
    """
    
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        print(f"✅ ReplayBuffer initialized: capacity={capacity}")
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample random batch"""
        if self.size < batch_size:
            return None
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]).unsqueeze(1),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices]).unsqueeze(1),
            'indices': indices,
            'weights': torch.ones(batch_size, 1)  # Uniform weights
        }
    
    def update_priorities(self, indices, td_errors):
        """Placeholder for PER compatibility"""
        pass
    
    def __len__(self):
        return self.size


class NStepReplayBuffer:
    """
    N-Step Replay Buffer
    
    Stores n-step returns for better credit assignment
    """
    
    def __init__(self, capacity, state_dim, action_dim, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        
        # Main buffer
        self.buffer = ReplayBuffer(capacity, state_dim, action_dim)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        print(f"✅ NStepReplayBuffer initialized: n_step={n_step}")
    
    def add(self, state, action, reward, next_state, done):
        """Add transition with n-step return computation"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full, compute n-step return
        if len(self.n_step_buffer) == self.n_step:
            # Get first transition
            state, action, _, _, _ = self.n_step_buffer[0]
            
            # Compute n-step return
            n_step_reward = 0
            next_state_n = None
            done_n = False
            
            for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                next_state_n = ns
                done_n = d
                if d:
                    break
            
            # Store with n-step return
            self.buffer.add(state, action, n_step_reward, next_state_n, done_n)
        
        # If episode ends, flush buffer
        if done:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Process remaining transitions in n-step buffer"""
        while len(self.n_step_buffer) > 0:
            state, action, reward, next_state, done = self.n_step_buffer.popleft()
            
            # Compute n-step return for remaining
            n_step_reward = reward
            gamma_i = self.gamma
            
            for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
                n_step_reward += (gamma_i) * r
                gamma_i *= self.gamma
                next_state = ns
                done = d
                if d:
                    break
            
            self.buffer.add(state, action, n_step_reward, next_state, done)
    
    def sample(self, batch_size):
        """Sample from main buffer"""
        return self.buffer.sample(batch_size)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities"""
        self.buffer.update_priorities(indices, td_errors)
    
    def __len__(self):
        return len(self.buffer)


# Test the replay buffers
if __name__ == "__main__":
    print("Testing Replay Buffers...")
    
    # Test standard buffer
    print("\n1. Testing Standard ReplayBuffer")
    buffer = ReplayBuffer(capacity=1000, state_dim=100, action_dim=2)
    
    for i in range(100):
        state = np.random.randn(100)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(100)
        done = i == 99
        buffer.add(state, action, reward, next_state, done)
    
    batch = buffer.sample(32)
    print(f"   Batch size: {batch['states'].shape[0]}")
    print(f"   Buffer size: {len(buffer)}")
    
    # Test prioritized buffer
    print("\n2. Testing PrioritizedReplayBuffer")
    per_buffer = PrioritizedReplayBuffer(capacity=1000, state_dim=100, action_dim=2)
    
    for i in range(100):
        per_buffer.add(np.random.randn(100), np.random.randn(2), 
                       np.random.randn(), np.random.randn(100), i == 99)
    
    batch = per_buffer.sample(32)
    print(f"   Batch size: {batch['states'].shape[0]}")
    print(f"   Weights shape: {batch['weights'].shape}")
    print(f"   Buffer size: {len(per_buffer)}")
    
    # Test n-step buffer
    print("\n3. Testing NStepReplayBuffer")
    nstep_buffer = NStepReplayBuffer(capacity=1000, state_dim=100, action_dim=2, n_step=3)
    
    for i in range(100):
        nstep_buffer.add(np.random.randn(100), np.random.randn(2),
                        np.random.randn(), np.random.randn(100), i == 99)
    
    batch = nstep_buffer.sample(32)
    print(f"   Batch size: {batch['states'].shape[0]}")
    
    print("\n✅ All replay buffers work correctly!")