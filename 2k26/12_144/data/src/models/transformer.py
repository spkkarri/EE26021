"""
Transformer for Temporal Sequence Processing
Remembers past observations to understand traffic patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    
    Adds information about the temporal order of observations.
    Without this, the transformer wouldn't know which observation
    happened 1 second ago vs 5 seconds ago.
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
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer for processing temporal sequences
    
    Features:
    - Remembers past observations (up to max_len)
    - Self-attention to find important past moments
    - Outputs enhanced current observation with temporal context
    """
    
    def __init__(self, input_dim=64, d_model=128, n_heads=4, n_layers=3, max_len=50, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Memory buffer for history
        self.history_buffer = deque(maxlen=max_len)
        self.attention_weights = None
    
    def forward(self, current_obs, history=None, return_attention=False):
        """
        Process current observation with temporal context
        
        Args:
            current_obs: Current observation (batch, input_dim)
            history: Optional historical observations (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            enhanced: Enhanced observation with temporal context
            attention: Attention weights (if return_attention=True)
        """
        batch_size = current_obs.shape[0]
        
        # Use provided history or buffer
        if history is None:
            if len(self.history_buffer) > 0:
                # Convert buffer to tensor
                history_list = list(self.history_buffer)
                history = torch.stack(history_list, dim=1)
            else:
                # Create dummy history (repeat current)
                history = current_obs.unsqueeze(1).repeat(1, 1, 1)
        
        # Combine current and history
        seq_len = history.shape[1]
        if seq_len < self.max_len:
            # Pad with current observation
            padding = current_obs.unsqueeze(1).repeat(1, self.max_len - seq_len, 1)
            seq = torch.cat([history, padding], dim=1)
        else:
            seq = history[:, -self.max_len:, :]
        
        # Project to d_model
        seq = self.input_projection(seq)
        
        # Add positional encoding
        seq = self.pos_encoding(seq)
        
        # Apply transformer
        output = self.transformer_encoder(seq)
        
        # Project output
        output = self.output_projection(output)
        
        # Extract current timestep output (last in sequence)
        enhanced = output[:, -1, :]
        
        # Store attention if requested
        if return_attention:
            # Note: Extracting attention requires modifying transformer
            self.attention_weights = None
        
        # Update memory buffer
        self.history_buffer.append(current_obs.detach())
        
        if return_attention:
            return enhanced, self.attention_weights
        return enhanced
    
    def reset_memory(self):
        """Clear memory buffer for new episode"""
        self.history_buffer.clear()
    
    def get_memory_length(self):
        """Get current memory buffer size"""
        return len(self.history_buffer)


class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer for fusing multiple sensor inputs
    
    Fuses information from:
    - LiDAR (distance to obstacles)
    - Camera (visual information)
    - Vehicle state (speed, steering)
    """
    
    def __init__(self, lidar_dim=240, state_dim=10, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        
        # Modal-specific encoders
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, lidar, state):
        """
        Fuse LiDAR and state information
        
        Args:
            lidar: LiDAR observation (batch, lidar_dim)
            state: Vehicle state (batch, state_dim)
        
        Returns:
            fused: Fused representation (batch, d_model)
            attention: Cross-attention weights
        """
        # Encode each modality
        lidar_encoded = self.lidar_encoder(lidar).unsqueeze(1)  # (batch, 1, d_model)
        state_encoded = self.state_encoder(state).unsqueeze(1)   # (batch, 1, d_model)
        
        # Concatenate along sequence dimension
        combined = torch.cat([lidar_encoded, state_encoded], dim=1)  # (batch, 2, d_model)
        
        # Apply self-attention across modalities
        attended, attention_weights = self.cross_attention(combined, combined, combined)
        
        # Global pooling (average over sequence)
        fused = attended.mean(dim=1)
        
        # Final fusion with original state
        output = self.fusion(torch.cat([fused, state_encoded.squeeze(1)], dim=-1))
        output = self.output_proj(output)
        
        return output, attention_weights


class SimpleTemporalModel(nn.Module):
    """
    Simpler temporal model using LSTM (for comparison)
    """
    
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Memory
        self.hidden_state = None
    
    def forward(self, sequence):
        """
        Process sequence with LSTM
        
        Args:
            sequence: Input sequence (batch, seq_len, input_dim)
        
        Returns:
            output: Last timestep output (batch, hidden_dim)
        """
        output, self.hidden_state = self.lstm(sequence, self.hidden_state)
        last_output = output[:, -1, :]
        return self.output_proj(last_output)
    
    def reset_memory(self):
        """Reset hidden state"""
        self.hidden_state = None


# Test the transformer
if __name__ == "__main__":
    print("Testing TemporalTransformer...")
    
    # Create transformer
    transformer = TemporalTransformer(input_dim=64, d_model=128, max_len=50)
    
    # Simulate sequence of observations
    batch_size = 4
    for i in range(60):  # More than max_len
        current_obs = torch.randn(batch_size, 64)
        enhanced = transformer(current_obs)
        
        if i == 0:
            print(f"First enhanced output shape: {enhanced.shape}")
    
    print(f"Memory buffer size: {transformer.get_memory_length()}")
    
    # Reset and test again
    transformer.reset_memory()
    print(f"After reset: {transformer.get_memory_length()}")
    
    print("\n✅ Transformer works!")
    print("   - Processes temporal sequences")
    print("   - Maintains memory of past observations")
    print("   - Can reset for new episodes")
    
    # Test cross-modal transformer
    print("\nTesting CrossModalTransformer...")
    cross_transformer = CrossModalTransformer(lidar_dim=240, state_dim=10)
    
    dummy_lidar = torch.randn(batch_size, 240)
    dummy_state = torch.randn(batch_size, 10)
    
    fused, attention = cross_transformer(dummy_lidar, dummy_state)
    print(f"Fused output shape: {fused.shape}")
    print(f"Attention shape: {attention.shape}")