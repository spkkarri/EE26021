"""
Graph Neural Network for Vehicle Interaction Modeling
Models how vehicles influence each other in traffic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer
    
    Each vehicle attends to neighboring vehicles to understand
    how they will affect its driving decisions.
    """
    
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Weight matrix for feature transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, adj):
        """
        Forward pass
        
        Args:
            h: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            h_prime: Updated node features
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)  # (num_nodes, out_features)
        
        # Compute attention coefficients
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])
        
        # Combine for each edge
        e = self.leakyrelu(Wh1 + Wh2.T)
        
        # Mask with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_prime = torch.mm(attention, Wh)
        h_prime = F.elu(h_prime)
        
        return h_prime, attention


class VehicleInteractionGNN(nn.Module):
    """
    Graph Neural Network for Vehicle Interactions
    
    Models the traffic scene as a graph where:
    - Nodes: Vehicles (including ego)
    - Edges: Proximity-based interactions
    - Features: Position, velocity, heading
    """
    
    def __init__(self, node_features=10, hidden_dim=64, output_dim=32, num_layers=2):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Graph attention layers
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.attentions.append(GraphAttentionLayer(in_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Ego vehicle attention (focus on self)
        self.ego_attention = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
        self.last_attention = None
    
    def forward(self, node_features, adjacency_matrix):
        """
        Process traffic graph
        
        Args:
            node_features: Features for each vehicle (num_vehicles, node_features)
            adjacency_matrix: Which vehicles can interact (num_vehicles, num_vehicles)
        
        Returns:
            ego_features: Enhanced features for ego vehicle
            all_features: Features for all vehicles
            attention: Attention weights for visualization
        """
        # Project input features
        h = F.relu(self.input_proj(node_features))
        
        # Apply graph attention layers
        attention_maps = []
        for attention_layer in self.attentions:
            h, attn = attention_layer(h, adjacency_matrix)
            attention_maps.append(attn)
        
        # Project to output dimension
        all_features = self.output_proj(h)
        
        # Extract ego vehicle features (first node is always ego)
        ego_features = all_features[0:1, :]
        
        # Apply ego attention
        ego_weight = self.ego_attention(ego_features)
        ego_features = ego_features * ego_weight
        
        # Store attention for visualization
        self.last_attention = attention_maps[-1]
        
        return ego_features, all_features, attention_maps


class TrafficSceneGraphBuilder:
    """
    Builds graph representation from traffic scene
    
    Converts raw vehicle positions and states into
    node features and adjacency matrix for GNN.
    """
    
    def __init__(self, communication_range=30.0, max_vehicles=20):
        self.communication_range = communication_range
        self.max_vehicles = max_vehicles
    
    def build_graph(self, ego_state, vehicle_states, vehicle_positions):
        """
        Build graph from traffic scene
        
        Args:
            ego_state: Ego vehicle state [x, y, vx, vy, heading]
            vehicle_states: List of other vehicle states
            vehicle_positions: List of (x, y) positions
        
        Returns:
            node_features: Tensor of node features
            adjacency_matrix: Tensor of adjacency matrix
        """
        # Combine all vehicles (ego first)
        all_vehicles = [ego_state] + vehicle_states[:self.max_vehicles - 1]
        all_positions = [ego_state[:2]] + vehicle_positions[:self.max_vehicles - 1]
        
        num_vehicles = len(all_vehicles)
        
        # Build node features
        node_features = []
        for i, vehicle in enumerate(all_vehicles):
            is_ego = 1.0 if i == 0 else 0.0
            
            features = [
                vehicle[0] / 100.0,      # x position (normalized)
                vehicle[1] / 100.0,      # y position (normalized)
                vehicle[2] / 30.0,       # vx velocity (normalized)
                vehicle[3] / 30.0,       # vy velocity (normalized)
                np.sin(vehicle[4]),      # sin(heading)
                np.cos(vehicle[4]),      # cos(heading)
                is_ego,                  # is ego flag
            ]
            node_features.append(features)
        
        node_features = torch.FloatTensor(node_features)
        
        # Build adjacency matrix based on proximity
        adjacency = torch.zeros(num_vehicles, num_vehicles)
        
        for i in range(num_vehicles):
            for j in range(num_vehicles):
                if i != j:
                    dx = all_positions[i][0] - all_positions[j][0]
                    dy = all_positions[i][1] - all_positions[j][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < self.communication_range:
                        # Edge weight inversely proportional to distance
                        adjacency[i, j] = 1.0 / (distance + 1.0)
        
        # Normalize adjacency matrix
        row_sum = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / (row_sum + 1e-6)
        
        return node_features, adjacency


# Test the GNN
if __name__ == "__main__":
    print("Testing VehicleInteractionGNN...")
    
    # Create dummy vehicles
    ego = [0, 0, 10, 0, 0]  # x, y, vx, vy, heading
    vehicles = [
        [5, 2, 9, 0, 0],   # Vehicle 1
        [-3, 1, 11, 0, 0], # Vehicle 2
        [10, -1, 10, 0, 0] # Vehicle 3
    ]
    positions = [(5, 2), (-3, 1), (10, -1)]
    
    # Build graph
    builder = TrafficSceneGraphBuilder()
    node_features, adjacency = builder.build_graph(ego, vehicles, positions)
    
    print(f"Node features shape: {node_features.shape}")
    print(f"Adjacency shape: {adjacency.shape}")
    
    # Create GNN
    gnn = VehicleInteractionGNN(node_features=7, hidden_dim=32, output_dim=16)
    
    # Forward pass
    ego_features, all_features, attention = gnn(node_features, adjacency)
    
    print(f"\n✅ GNN works!")
    print(f"   Ego features shape: {ego_features.shape}")
    print(f"   All features shape: {all_features.shape}")
    print(f"   Attention maps: {len(attention)}")