"""
Attention-Based Neural Networks for Self-Driving Cars
Multi-Head Attention mechanism for better scene understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    
    Allows the network to focus on different parts of the input
    simultaneously, improving scene understanding.
    """
    
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        Forward pass of attention
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional mask for padding
        
        Returns:
            output: Attended features (batch, seq_len, d_model)
            attention: Attention weights for visualization
        """
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.W_o(context)
        
        return output, attention


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    
    Adds information about position in sequence since
    attention doesn't naturally understand order.
    """
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AdvancedPolicyNetwork(nn.Module):
    """
    Advanced Policy Network with Attention Mechanism
    
    Architecture:
    1. Input encoding
    2. Multi-Head Attention
    3. Actor head (outputs actions)
    4. Critic head (outputs value)
    """
    
    def __init__(self, obs_dim=100, action_dim=2, hidden_dim=256, n_heads=4):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Positional encoding for sequence
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(hidden_dim, n_heads)
        
        # Layer normalization after attention
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Actor network (outputs actions)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Critic network (outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Attention storage for visualization
        self.last_attention = None
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Observation tensor (batch, obs_dim)
        
        Returns:
            action: Action to take (batch, action_dim)
            value: State value estimate (batch, 1)
            attention: Attention weights for visualization
        """
        # Encode input
        encoded = self.encoder(x)
        
        # Add sequence dimension and positional encoding
        encoded = encoded.unsqueeze(1)  # (batch, 1, hidden_dim)
        encoded = self.pos_encoding(encoded)
        
        # Apply attention
        attended, attention = self.attention(encoded)
        self.last_attention = attention.detach()
        
        # Add residual connection and normalize
        attended = self.norm(encoded + attended)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Get action and value
        action = self.actor(attended)
        value = self.critic(attended)
        
        return action, value, attention
    
    def get_action(self, x, deterministic=True):
        """
        Get action for inference
        
        Args:
            x: Observation tensor
            deterministic: If True, no exploration noise
        """
        action, _, _ = self.forward(x)
        return action
    
    def get_attention_weights(self):
        """Get last attention weights for visualization"""
        return self.last_attention


class SimplePolicyNetwork(nn.Module):
    """
    Simpler policy network without attention (faster)
    Used as baseline comparison
    """
    
    def __init__(self, obs_dim=100, action_dim=2, hidden_dim=256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        action = self.actor(shared)
        value = self.critic(shared)
        return action, value, None


# Test the network
if __name__ == "__main__":
    print("Testing AdvancedPolicyNetwork...")
    
    # Create network
    network = AdvancedPolicyNetwork(obs_dim=100, action_dim=2)
    
    # Test forward pass
    dummy_input = torch.randn(4, 100)  # Batch of 4 observations
    action, value, attention = network(dummy_input)
    
    print(f"✅ Network works!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Action shape: {action.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Attention shape: {attention.shape}")