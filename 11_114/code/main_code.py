import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. DATA PREPARATION (NASA SOH DATA)
# ==========================================
SEQUENCE_LENGTH = 30  
FEATURE_COLS = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'cycle_id']
NUM_FEATURES = len(FEATURE_COLS)

def load_raw_data(folder_path):
    features_list = []
    targets_list = []
    all_capacities = []
    
    # First pass: Find the absolute max capacity across ALL files
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            all_capacities.append(df['Capacity'].max())
    
    GLOBAL_MAX_CAPACITY = max(all_capacities) # This is our 100% SOH baseline

    # Second pass: Process data
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            cycle_data = df.groupby('cycle_id').agg({
                'Voltage_measured': 'mean',
                'Current_measured': 'mean',
                'Temperature_measured': 'mean',
                'Capacity': 'first',
                'cycle_id': 'first'
            }).reset_index(drop=True)
            
            cycle_data = cycle_data.dropna()
            
            # SOH is now relative to the same baseline for every battery
            
            cycle_data['SOH'] = cycle_data['Capacity'] / GLOBAL_MAX_CAPACITY
# Add this line:
            cycle_data = cycle_data[cycle_data['SOH'] > 0.1]
            
            features_list.append(cycle_data[FEATURE_COLS].values)
            targets_list.append(cycle_data['SOH'].values.reshape(-1, 1))
            
    return features_list, targets_list

# Load the raw, unscaled data
CSV_FOLDER = "C:/Users/Sanyasinaidu ch/OneDrive/Desktop/dataset/csv_output"
print("Loading NASA Battery Data...")
features_list, targets_list = load_raw_data(CSV_FOLDER)

# Split by file (80% of batteries for training, 20% for validation)
# This prevents data leakage and forces the model to learn to generalize to NEW batteries!
split_idx = int(len(features_list) * 0.8)
train_feats, train_targs = features_list[:split_idx], targets_list[:split_idx]
val_feats, val_targs = features_list[split_idx:], targets_list[split_idx:]

# Fit scalers on TRAINING data only
feature_scaler = MinMaxScaler()


# Concatenate all training lists to fit the scalers
feature_scaler.fit(np.concatenate(train_feats, axis=0))



# Create sliding windows per file to maintain boundary integrity
def create_windows(feats_list, targs_list, f_scaler, seq_length):
    X, y = [], []
    for feats, targs in zip(feats_list, targs_list):
        # Scale ONLY the features
        feats_scaled = f_scaler.transform(feats)
        
        for i in range(len(feats_scaled) - seq_length):
            X.append(feats_scaled[i : i + seq_length])
            y.append(targs[i + seq_length]) # <-- Keep targets as raw SOH!
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

X_train, y_train = create_windows(train_feats, train_targs, feature_scaler, SEQUENCE_LENGTH)
X_val, y_val = create_windows(val_feats, val_targs, feature_scaler, SEQUENCE_LENGTH)



print(f"Loaded {len(X_train)} total training sequences.")
print(f"Loaded {len(X_val)} total validation sequences.")

# Create Datasets and DataLoaders
class BatteryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# CHANGE 3: Reduced batch size for better convergence
BATCH_SIZE = 16 
train_loader = DataLoader(BatteryDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BatteryDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output), attention_weights
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x)
        out1 = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(out1)
        return self.norm2(out1 + self.dropout2(ff_output))
    
class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_size, lstm_units, embed_dim, num_heads, ff_dim, num_blocks, dropout_rate, seq_len, num_lstm_layers=2):
        super(HybridLSTMTransformer, self).__init__()
        
        self.lstm = nn.LSTM(input_size, lstm_units, num_layers=num_lstm_layers, batch_first=True, dropout=dropout_rate)
        self.batch_norm_lstm = nn.BatchNorm1d(lstm_units)
        
        self.projection = nn.Linear(lstm_units, embed_dim) if lstm_units != embed_dim else nn.Identity()
        self.positional_encoding = nn.Embedding(seq_len, embed_dim)
        self.dropout_pe = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_blocks)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1) 

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # LSTM -> BatchNorm -> Projection
        lstm_out, _ = self.lstm(x) 
        lstm_out_bn = self.batch_norm_lstm(lstm_out.transpose(1, 2)).transpose(1, 2)
        x = self.projection(lstm_out_bn)

        # Positional Encoding + Transformer
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        x = self.dropout_pe(x + self.positional_encoding(positions))
        for block in self.transformer_blocks: x = block(x)

        # Pooling -> Regression Head
        x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1) 
        x = self.dropout_fc(self.relu(self.fc1(x)))
        # Change this in your model's forward method:
# return self.fc2(x) 
        return self.fc2(x)
        


# ==========================================
# 3. SETUP & TRAINING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

model = HybridLSTMTransformer(
    input_size=NUM_FEATURES, lstm_units=64, embed_dim=64, num_heads=4, 
    ff_dim=128, num_blocks=2, dropout_rate=0.3, seq_len=SEQUENCE_LENGTH, num_lstm_layers=2
).to(device)


loss_fn = nn.MSELoss()
# In Section 3
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) # AdamW is better for Transformers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def mae_metric(preds, targets): return torch.mean(torch.abs(preds - targets))
def rmse_metric(preds, targets): return torch.sqrt(torch.mean((preds - targets)**2))

NUM_EPOCHS = 100
PATIENCE = 20
best_val_loss = float('inf')
epochs_no_improve = 0
history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}

print("\n--- Training Model ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = loss_fn(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss, total_val_mae, total_val_rmse = 0, 0, 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            predictions = model(batch_features)
            total_val_loss += loss_fn(predictions, batch_labels).item()
            total_val_mae += mae_metric(predictions, batch_labels).item()
            total_val_rmse += rmse_metric(predictions, batch_labels).item()

    avg_val_loss = total_val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_mae'].append(total_val_mae / len(val_loader))
    history['val_rmse'].append(total_val_rmse / len(val_loader))

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_hybrid_soh_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"--> Early stopping at epoch {epoch+1}")
            break

# ==========================================
# 4. EVALUATION & VISUALIZATION
# ==========================================
print("\n--- Evaluation ---")
try:
    model.load_state_dict(torch.load('best_hybrid_soh_model.pth', weights_only=True))
    model.eval()

   # 1. Pass the ENTIRE validation set through the model
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_pred_real = model(X_val_tensor).cpu().numpy() # <-- Already real SOH!

    # 2. Get the actual values directly
    y_true_real = y_val # <-- Already real SOH!
    
    # C. Keep clipping just as a physical safety net
    y_pred_real = np.clip(y_pred_real, 0.0, 1.0)

    # 3. Print a small snippet of the results
    print(f"Predicted first 5 SOH: {y_pred_real[:5].flatten()}")
    print(f"Actual first 5 SOH:    {y_true_real[:5].flatten()}")

    # 4. Plot the degradation curve for the validation set!
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_real, label='Actual SOH', color='blue', linewidth=2)
    plt.plot(y_pred_real, label='Predicted SOH', color='orange', linestyle='--', linewidth=2)
    plt.title('Actual vs Predicted State of Health (Validation Set)')
    plt.xlabel('Sequence Index')
    plt.ylabel('State of Health (SOH)')
    plt.legend()
    plt.grid(True)
    plt.show()


     # Plotting metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss (MSE)')
    plt.plot(history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.plot(history['val_rmse'], label='Validation RMSE')
    plt.title('Error Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Error: Could not load the model.")
