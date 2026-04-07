"""
Temporal Fusion Transformer (TFT) OOF Training — All 3 Tracks
- Paper: Lim et al. 2020 (arxiv 2012.15671)
- Multi-head attention over temporal sequences
- Handles multivariate input natively (no flattening)
- 5-fold walk-forward CV (no leakage)
- Per-fold normalization on train only
- GPU accelerated (RTX 5070)
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = ROOT / "data" / "processed" / "oof" / "tft"
MODELS_DIR = ROOT / "data" / "processed" / "models" / "tft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "input_window"   : 168,
    "horizon"        : 24,
    "sequence_stride": 24,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
}

# Track-specific configs
TRACK_CONFIGS = {
    "load": {
        "n_splits"               : 5,
        "d_model"                : 128,   # doubled — more representational power
        "n_heads"                : 4,
        "n_encoder_layers"       : 2,
        "d_ff"                   : 512,   # doubled
        "dropout"                : 0.05,  # less dropout
        "learning_rate"          : 0.0002,# lower — fold 4 was stopping at epoch 20
        "batch_size"             : 64,    # smaller — better gradient estimates
        "epochs"                 : 150,
        "early_stopping_patience": 15,    # more patience — fold 4 stopped too early
        "min_delta"              : 0.00005,
        "blown_threshold"        : 5.0,
    },
    "solar": {
        "n_splits"               : 3,
        "d_model"                : 64,    # doubled
        "n_heads"                : 4,     # doubled
        "n_encoder_layers"       : 2,
        "d_ff"                   : 256,   # doubled
        "dropout"                : 0.1,
        "learning_rate"          : 0.0001,
        "batch_size"             : 32,
        "epochs"                 : 200,
        "early_stopping_patience": 20,
        "min_delta"              : 0.00003,
        "blown_threshold"        : 3.0,
    },
    "wind": {
        "n_splits"               : 3,
        "d_model"                : 64,    # doubled
        "n_heads"                : 4,     # doubled
        "n_encoder_layers"       : 2,
        "d_ff"                   : 256,   # doubled
        "dropout"                : 0.1,
        "learning_rate"          : 0.0001,# lower — more stable
        "batch_size"             : 32,
        "epochs"                 : 200,
        "early_stopping_patience": 20,
        "min_delta"              : 0.00003,
        "blown_threshold"        : 3.0,
    },
}

TRACKS = {
    "load" : "load_track_train.parquet",
    "solar": "solar_track_train.parquet",
    "wind" : "wind_track_train.parquet",
}

print(f"Device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────
# TFT ARCHITECTURE
# ─────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout=0.1, context_size=None):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.drop = nn.Dropout(dropout)

        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size, bias=False)
        else:
            self.skip = None

    def forward(self, x, context=None):
        h = self.fc1(x)
        if context is not None and hasattr(self, 'context_fc'):
            h = h + self.context_fc(context)
        h = torch.nn.functional.elu(h)
        h = self.drop(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        h    = gate * h
        residual = self.skip(x) if self.skip is not None else x
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.n_features  = n_features
        self.d_model     = d_model

        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        self.weight_grn = GatedResidualNetwork(
            n_features, d_model, n_features, dropout
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        B, T, n_feat = x.shape

        # Process each feature independently
        feature_outputs = []
        for i in range(n_feat):
            feat = x[:, :, i:i+1].reshape(B * T, 1)   # (B*T, 1)
            out  = self.feature_grns[i](feat)           # (B*T, d_model)
            feature_outputs.append(out.reshape(B, T, self.d_model))

        stacked = torch.stack(feature_outputs, dim=2)   # (B, T, n_feat, d_model)

        flat    = x.reshape(B * T, n_feat)
        weights = self.weight_grn(flat)                 # (B*T, n_feat)
        weights = torch.softmax(weights, dim=-1)
        weights = weights.reshape(B, T, n_feat, 1)

        output = (stacked * weights).sum(dim=2)         # (B, T, d_model)
        return output, weights.squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4,
                 n_encoder_layers=2, d_ff=256,
                 input_window=168, horizon=24, dropout=0.1):
        super().__init__()

        self.n_features   = n_features
        self.d_model      = d_model
        self.input_window = input_window
        self.horizon      = horizon

        self.vsn = VariableSelectionNetwork(n_features, d_model, dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder_layers,
            dropout=dropout if n_encoder_layers > 1 else 0.0,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.post_attn_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.output_grn    = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.output_fc     = nn.Linear(d_model, horizon)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # x: (batch, seq_len, n_features)

        vsn_out, _ = self.vsn(x)                        # (B, T, d_model)

        lstm_out, _ = self.lstm(vsn_out)                # (B, T, d_model)
        lstm_out    = self.lstm_norm(lstm_out + vsn_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_drop(attn_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)

        post_out = self.post_attn_grn(attn_out)         # (B, T, d_model)

        last = post_out[:, -1, :]                       # (B, d_model)
        out  = self.output_grn(last)
        out  = self.output_fc(out)                      # (B, horizon)

        return out
# ─────────────────────────────────────────────
# STEP 1: BUILD SEQUENCES
# ─────────────────────────────────────────────
def build_sequences(df, input_window, horizon, stride):
    target_col = "load_norm"
    ignore     = {"asset_id", "timestamp", target_col}
    feat_cols  = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    all_X, all_y, all_assets = [], [], []

    for asset_id, asset_df in df.groupby("asset_id"):
        asset_df = asset_df.sort_values("timestamp").reset_index(drop=True)
        values_X = asset_df[feat_cols].values.astype(np.float32)
        values_y = asset_df[target_col].values.astype(np.float32)

        max_start = len(asset_df) - input_window - horizon + 1
        if max_start <= 0:
            print(f"  ⚠ Skipping {asset_id} — not enough data")
            continue

        for s in range(0, max_start, stride):
            all_X.append(values_X[s: s + input_window])
            all_y.append(values_y[s + input_window: s + input_window + horizon])
            all_assets.append(asset_id)

    X      = np.stack(all_X)   # (N, T, F) — keep 3D for TFT
    y      = np.stack(all_y)   # (N, 24)
    assets = np.array(all_assets)

    print(f"  Sequences: X {X.shape} | y {y.shape}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")
    return X, y, assets, feat_cols


# ─────────────────────────────────────────────
# STEP 2: NAIVE BASELINE
# ─────────────────────────────────────────────
def naive_baseline_oof(X, y, feat_cols):
    if "lag_24" in feat_cols:
        idx      = feat_cols.index("lag_24")
        last_lag = X[:, -1, idx]
    elif "lag_1" in feat_cols:
        idx      = feat_cols.index("lag_1")
        last_lag = X[:, -1, idx]
        print("  ⚠ lag_24 not found, using lag_1")
    else:
        last_lag = np.zeros(len(y))

    preds = np.tile(last_lag.reshape(-1, 1), (1, y.shape[1]))
    mae   = np.mean(np.abs(y - preds))
    return mae, preds


# ─────────────────────────────────────────────
# STEP 3: EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────
# STEP 4: TFT OOF
# ─────────────────────────────────────────────
def run_tft_oof(X, y, global_config, track_cfg, track_name, naive_mae):
    """
    Walk-forward OOF with TFT.
    X: (N, T, F) — 3D, NOT flattened (TFT processes sequences natively)
    y: (N, 24)
    """
    device          = global_config["device"]
    horizon         = global_config["horizon"]
    n_splits        = track_cfg["n_splits"]
    blown_threshold = naive_mae * track_cfg["blown_threshold"]

    tscv       = TimeSeriesSplit(n_splits=n_splits)
    oof_preds  = np.full((len(X), horizon), np.nan, dtype=np.float32)
    valid_mask = np.zeros(len(X), dtype=bool)

    n_features = X.shape[2]
    print(f"  n_features: {n_features} | seq_len: {X.shape[1]}")
    print(f"  n_splits: {n_splits} | blown threshold: {blown_threshold:.4f}")

    best_fold_mae   = float("inf")
    best_fold_model = None
    best_scaler     = None
    blown_folds     = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold+1}/{n_splits} | "
              f"train {len(train_idx)} | val {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Per-fold normalization — fit on train only
        # X shape: (N, T, F) — normalize per feature F
        X_tr_flat = X_train.reshape(-1, X_train.shape[-1])  # (N*T, F)
        mean = X_tr_flat.mean(axis=0)                        # (F,)
        std  = X_tr_flat.std(axis=0)                         # (F,)
        std  = np.where(std < 1e-8, 1.0, std)

        X_train_n = (X_train - mean) / std
        X_val_n   = (X_val   - mean) / std

        # Clip extremes
        X_train_n = np.clip(X_train_n, -10, 10)
        X_val_n   = np.clip(X_val_n,   -10, 10)

        # Tensors — keep 3D for TFT
        X_tr_t = torch.tensor(X_train_n, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train,   dtype=torch.float32)
        X_vl_t = torch.tensor(X_val_n,   dtype=torch.float32)
        y_vl_t = torch.tensor(y_val,     dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=track_cfg["batch_size"],
            shuffle=False,
            pin_memory=True if device == "cuda" else False,
            num_workers=0
        )
        val_loader = DataLoader(
            TensorDataset(X_vl_t, y_vl_t),
            batch_size=track_cfg["batch_size"],
            shuffle=False,
            pin_memory=True if device == "cuda" else False,
            num_workers=0
        )

        # Build model
        model = TFTModel(
            n_features    =n_features,
            d_model       =track_cfg["d_model"],
            n_heads       =track_cfg["n_heads"],
            n_encoder_layers=track_cfg["n_encoder_layers"],
            d_ff          =track_cfg["d_ff"],
            input_window  =global_config["input_window"],
            horizon       =horizon,
            dropout       =track_cfg["dropout"],
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {total_params:,}")

        optimizer  = torch.optim.Adam(
            model.parameters(),
            lr=track_cfg["learning_rate"],
            weight_decay=1e-5
        )
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6
        )
        early_stop = EarlyStopping(
            patience=track_cfg["early_stopping_patience"],
            min_delta=track_cfg["min_delta"]
        )
        criterion  = nn.MSELoss()

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(track_cfg["epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item()
            avg_val = val_loss / len(val_loader)

            # Divergence check
            if avg_val > 1000 or np.isnan(avg_val):
                print(f"    ⚠ Diverged at epoch {epoch+1} — stopping fold")
                break

            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | "
                      f"train={avg_train:.4f} | "
                      f"val={avg_val:.4f} | "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

            if early_stop.step(avg_val):
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if best_state is None or best_val_loss > 100:
            print(f"  ⚠ Fold {fold+1} failed — skipping")
            blown_folds += 1
            continue

        # OOF predictions
        model.load_state_dict(best_state)
        model.eval()

        all_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                all_preds.append(model(xb.to(device)).cpu().numpy())

        fold_preds = np.concatenate(all_preds, axis=0)
        fold_mae   = np.mean(np.abs(y_val - fold_preds))

        if fold_mae > blown_threshold:
            print(f"  ⚠ Fold {fold+1} blown MAE={fold_mae:.4f} "
                  f"> {blown_threshold:.4f} — skipping")
            blown_folds += 1
            continue

        oof_preds[val_idx]  = fold_preds
        valid_mask[val_idx] = True

        print(f"  ✅ Fold {fold+1} MAE: {fold_mae:.4f} | "
              f"val loss: {best_val_loss:.4f}")

        if fold_mae < best_fold_mae:
            best_fold_mae   = fold_mae
            best_fold_model = best_state
            best_scaler     = {"mean": mean, "std": std}

        # Clear GPU cache between folds
        torch.cuda.empty_cache()

    print(f"\n  Blown folds: {blown_folds}/{n_splits}")
    print(f"  Valid windows: {valid_mask.sum()}/{len(X)}")

    if best_fold_model is None:
        print("  ❌ All folds failed")
        return oof_preds, valid_mask

    # Save best fold
    track_model_dir = MODELS_DIR / track_name
    track_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_fold_model, track_model_dir / "tft_best.pth")
    np.save(track_model_dir / "scaler_mean.npy", best_scaler["mean"])
    np.save(track_model_dir / "scaler_std.npy",  best_scaler["std"])
    print(f"  ✅ Best model saved (MAE={best_fold_mae:.4f})")

    return oof_preds, valid_mask


# ─────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred, naive_pred, mask, track_name):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_n = naive_pred[mask]

    def metrics(a, b):
        mae   = np.mean(np.abs(a - b))
        rmse  = np.sqrt(np.mean((a - b) ** 2))
        d     = (np.abs(a) + np.abs(b)) / 2
        m     = d > 1e-8
        smape = np.mean(np.abs(a[m] - b[m]) / d[m]) * 100
        ss_r  = np.sum((a - b) ** 2)
        ss_t  = np.sum((a - a.mean()) ** 2)
        r2    = 1 - ss_r / ss_t
        corr  = np.corrcoef(a.flatten(), b.flatten())[0, 1]
        return mae, rmse, smape, r2, corr

    lm, lr, ls, lr2, lc = metrics(y_t.flatten(), y_p.flatten())
    nm, nr, ns, nr2, nc = metrics(y_t.flatten(), y_n.flatten())
    improve = (nm - lm) / nm * 100

    print(f"\n{'='*58}")
    print(f"  {track_name.upper()} — TFT EVALUATION")
    print(f"{'='*58}")
    print(f"  {'Metric':<12} {'TFT':>10} {'Naive':>10} {'Improve':>10}")
    print(f"  {'-'*50}")
    print(f"  {'MAE':<12} {lm:>10.4f} {nm:>10.4f} {improve:>9.1f}%")
    print(f"  {'RMSE':<12} {lr:>10.4f} {nr:>10.4f}")
    print(f"  {'sMAPE':<12} {ls:>9.1f}% {ns:>9.1f}%")
    print(f"  {'R²':<12} {lr2:>10.4f} {nr2:>10.4f}")
    print(f"  {'Pearson r':<12} {lc:>10.4f} {nc:>10.4f}")

    print(f"\n  R² per forecast hour:")
    for h in range(24):
        ss_r = np.sum((y_t[:, h] - y_p[:, h]) ** 2)
        ss_t = np.sum((y_t[:, h] - y_t[:, h].mean()) ** 2)
        r2_h = 1 - ss_r / ss_t
        bar  = "█" * max(0, int(r2_h * 20))
        print(f"    H+{h+1:02d}: {r2_h:.4f}  {bar}")

    return {
        "track"    : track_name,
        "tft_mae"  : lm,
        "tft_rmse" : lr,
        "tft_smape": ls,
        "tft_r2"   : lr2,
        "tft_corr" : lc,
        "naive_mae": nm,
        "improve"  : improve,
    }


# ─────────────────────────────────────────────
# STEP 6: STATISTICAL SUMMARY
# ─────────────────────────────────────────────
def print_statistical_summary(y_true, y_pred, naive_pred, mask, track_name):
    y_t = y_true[mask].flatten()
    y_p = y_pred[mask].flatten()
    y_n = naive_pred[mask].flatten()

    print(f"\n{'='*58}")
    print(f"  STATISTICAL SUMMARY — {track_name.upper()}")
    print(f"{'='*58}")
    print(f"  N windows:     {mask.sum()}")
    print(f"  N predictions: {len(y_t)}")

    print(f"\n  TRUE values:")
    print(f"    mean={y_t.mean():.4f}  std={y_t.std():.4f}")
    print(f"    min={y_t.min():.4f}  max={y_t.max():.4f}")
    print(f"    p25={np.percentile(y_t,25):.4f}  "
          f"p50={np.percentile(y_t,50):.4f}  "
          f"p75={np.percentile(y_t,75):.4f}")

    for label, pred in [("TFT", y_p), ("Naive", y_n)]:
        res = y_t - pred
        print(f"\n  {label} predictions:")
        print(f"    mean={pred.mean():.4f}  std={pred.std():.4f}")
        print(f"    bias={res.mean():.4f}  residual std={res.std():.4f}")
        for thresh in [0.05, 0.10, 0.20]:
            pct = np.mean(np.abs(res) < thresh) * 100
            print(f"    % within ±{thresh:.2f}: {pct:.1f}%")


# ─────────────────────────────────────────────
# STEP 7: PLOTS
# ─────────────────────────────────────────────
def plot_results(y_true, y_pred, naive_pred, mask,
                 track_name, metrics, output_dir):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_n = naive_pred[mask]

    horizon_mae_tft   = np.mean(np.abs(y_t - y_p), axis=0)
    horizon_mae_naive = np.mean(np.abs(y_t - y_n), axis=0)
    horizon_r2 = []
    for h in range(24):
        ss_r = np.sum((y_t[:, h] - y_p[:, h]) ** 2)
        ss_t = np.sum((y_t[:, h] - y_t[:, h].mean()) ** 2)
        horizon_r2.append(max(0, 1 - ss_r / ss_t))

    window_mae_tft   = np.mean(np.abs(y_t - y_p), axis=1)
    window_mae_naive = np.mean(np.abs(y_t - y_n), axis=1)
    sorted_idx = np.argsort(window_mae_tft)
    best_idx   = sorted_idx[0]
    med_idx    = sorted_idx[len(sorted_idx) // 2]
    worst_idx  = sorted_idx[-1]
    hours      = np.arange(1, 25)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"TFT vs Naive Baseline — {track_name.upper()} Track\n"
        f"TFT:   MAE={metrics['tft_mae']:.4f} | "
        f"RMSE={metrics['tft_rmse']:.4f} | "
        f"R²={metrics['tft_r2']:.4f} | "
        f"r={metrics['tft_corr']:.4f}\n"
        f"Naive: MAE={metrics['naive_mae']:.4f} | "
        f"Improvement={metrics['improve']:.1f}%",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # Plot 1: MAE per horizon
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hours, horizon_mae_tft,   color="#2196F3", linewidth=2,
             marker="o", markersize=3, label="TFT")
    ax1.plot(hours, horizon_mae_naive, color="#e74c3c", linewidth=2,
             linestyle="--", marker="x", markersize=3, label="Naive")
    ax1.fill_between(hours, horizon_mae_tft, horizon_mae_naive,
                     alpha=0.15, color="#2196F3", label="TFT advantage")
    ax1.set_title("MAE per Forecast Hour", fontweight="bold")
    ax1.set_xlabel("H+n")
    ax1.set_ylabel("MAE")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot 2: R² per horizon
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(hours, horizon_r2, color="#9b59b6", alpha=0.85)
    ax2.axhline(0.85, color="red", linestyle="--",
                linewidth=1.5, label="Target R²=0.85")
    mean_r2 = np.mean(horizon_r2)
    ax2.axhline(mean_r2, color="green", linestyle="--",
                linewidth=1.5, label=f"Mean R²={mean_r2:.3f}")
    ax2.set_title("R² per Forecast Hour", fontweight="bold")
    ax2.set_xlabel("H+n")
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: MAE distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(window_mae_tft,   bins=50, color="#2196F3",
             alpha=0.6, label="TFT")
    ax3.hist(window_mae_naive, bins=50, color="#e74c3c",
             alpha=0.6, label="Naive")
    ax3.axvline(window_mae_tft.mean(),   color="#2196F3", linewidth=2,
                linestyle="--",
                label=f"TFT={window_mae_tft.mean():.3f}")
    ax3.axvline(window_mae_naive.mean(), color="#e74c3c", linewidth=2,
                linestyle="--",
                label=f"Naive={window_mae_naive.mean():.3f}")
    ax3.set_title("MAE Distribution", fontweight="bold")
    ax3.set_xlabel("MAE")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # Plot 4: Scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = np.random.choice(len(y_t), min(2000, len(y_t)), replace=False)
    ax4.scatter(y_t[sample].flatten(), y_p[sample].flatten(),
                alpha=0.2, s=1, color="#2196F3", label="TFT")
    ax4.scatter(y_t[sample].flatten(), y_n[sample].flatten(),
                alpha=0.15, s=1, color="#e74c3c", label="Naive")
    lims = [y_t.min(), y_t.max()]
    ax4.plot(lims, lims, "k--", linewidth=1.5, label="Perfect")
    ax4.set_title("Predicted vs Actual", fontweight="bold")
    ax4.set_xlabel("Actual load_norm")
    ax4.set_ylabel("Predicted load_norm")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # Plots 5, 6, 7: Best / Median / Worst
    for ax, idx, label in [
        (fig.add_subplot(gs[1, 1]), best_idx,  "Best"),
        (fig.add_subplot(gs[1, 2]), med_idx,   "Median"),
        (fig.add_subplot(gs[2, 0]), worst_idx, "Worst"),
    ]:
        mae_tft   = np.mean(np.abs(y_t[idx] - y_p[idx]))
        mae_naive = np.mean(np.abs(y_t[idx] - y_n[idx]))
        ax.plot(hours, y_t[idx], color="black",   linewidth=2,
                marker="o", markersize=3, label="Actual")
        ax.plot(hours, y_p[idx], color="#2196F3", linewidth=2,
                linestyle="--", marker="x", markersize=3, label="TFT")
        ax.plot(hours, y_n[idx], color="#e74c3c", linewidth=1.5,
                linestyle=":", marker="s", markersize=2, label="Naive")
        ax.set_title(
            f"{label} Window\n"
            f"TFT={mae_tft:.3f} | Naive={mae_naive:.3f}",
            fontweight="bold", fontsize=9
        )
        ax.set_xlabel("H+n")
        ax.set_ylabel("load_norm")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Plot 8: Residuals
    ax8 = fig.add_subplot(gs[2, 1])
    res_tft   = (y_t - y_p).flatten()
    res_naive = (y_t - y_n).flatten()
    ax8.hist(np.clip(res_tft, -1, 1),   bins=60, color="#2196F3",
             alpha=0.6,
             label=f"TFT (bias={res_tft.mean():.4f})")
    ax8.hist(res_naive, bins=60, color="#e74c3c",
             alpha=0.6,
             label=f"Naive (bias={res_naive.mean():.4f})")
    ax8.axvline(0, color="black", linewidth=2)
    ax8.set_title("Residual Distribution", fontweight="bold")
    ax8.set_xlabel("Actual - Predicted")
    ax8.set_ylabel("Count")
    ax8.legend(fontsize=7)
    ax8.grid(alpha=0.3)

    # Plot 9: Cumulative MAE
    ax9 = fig.add_subplot(gs[2, 2])
    sorted_tft   = np.sort(window_mae_tft)
    sorted_naive = np.sort(window_mae_naive)
    cum = np.arange(1, len(sorted_tft) + 1) / len(sorted_tft)
    ax9.plot(sorted_tft,   cum, color="#2196F3", linewidth=2,
             label=f"TFT (mean={window_mae_tft.mean():.3f})")
    ax9.plot(sorted_naive, cum, color="#e74c3c", linewidth=2,
             linestyle="--",
             label=f"Naive (mean={window_mae_naive.mean():.3f})")
    ax9.set_title("Cumulative MAE Distribution", fontweight="bold")
    ax9.set_xlabel("MAE")
    ax9.set_ylabel("Fraction of Windows")
    ax9.legend(fontsize=8)
    ax9.grid(alpha=0.3)

    save_path = output_dir / f"tft_{track_name}_oof_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    all_metrics = []

    for track_name, fname in TRACKS.items():
        print(f"\n{'='*60}")
        print(f"TRACK: {track_name.upper()}")
        print(f"{'='*60}")

        df = pd.read_parquet(SPLITS_DIR / fname)
        print(f"  Loaded: {df.shape} | "
              f"{df['timestamp'].min().date()} → "
              f"{df['timestamp'].max().date()}")
        print(f"  Assets: {df['asset_id'].nunique()}")

        print(f"\n  Building sequences...")
        X, y, assets, feat_cols = build_sequences(
            df,
            input_window=CONFIG["input_window"],
            horizon=CONFIG["horizon"],
            stride=CONFIG["sequence_stride"],
        )

        naive_mae, naive_preds = naive_baseline_oof(X, y, feat_cols)
        print(f"  Naive baseline MAE: {naive_mae:.4f}")

        track_cfg = TRACK_CONFIGS[track_name]
        print(f"\n  Running TFT OOF on {CONFIG['device'].upper()} "
              f"[n_splits={track_cfg['n_splits']} | "
              f"d_model={track_cfg['d_model']} | "
              f"n_heads={track_cfg['n_heads']} | "
              f"lr={track_cfg['learning_rate']}]...")

        oof_preds, valid_mask = run_tft_oof(
            X, y, CONFIG, track_cfg, track_name, naive_mae
        )

        if valid_mask.sum() == 0:
            print(f"  ❌ No valid predictions — skipping")
            continue

        metrics = evaluate(y, oof_preds, naive_preds, valid_mask, track_name)
        all_metrics.append(metrics)

        print_statistical_summary(
            y, oof_preds, naive_preds, valid_mask, track_name
        )

        print(f"\n  Plotting...")
        plot_results(
            y, oof_preds, naive_preds, valid_mask,
            track_name, metrics, OUTPUT_DIR
        )

        # Save OOF
        track_out = OUTPUT_DIR / track_name
        track_out.mkdir(parents=True, exist_ok=True)
        np.save(track_out / "tft_oof.npy",   oof_preds)
        np.save(track_out / "tft_mask.npy",  valid_mask)
        np.save(track_out / "naive_oof.npy", naive_preds)
        np.save(track_out / "y_true.npy",    y)
        np.save(track_out / "assets.npy",    assets)
        print(f"  OOF saved to {track_out}")

        torch.cuda.empty_cache()

    # Final summary
    if all_metrics:
        print(f"\n{'='*68}")
        print(f"FINAL SUMMARY — TFT vs Naive")
        print(f"{'='*68}")
        print(f"{'Track':<8} {'MAE':>8} {'Naive':>8} "
              f"{'R²':>8} {'sMAPE':>8} {'Improve':>10}")
        print("-" * 68)
        for m in all_metrics:
            print(f"{m['track']:<8} {m['tft_mae']:>8.4f} "
                  f"{m['naive_mae']:>8.4f} {m['tft_r2']:>8.4f} "
                  f"{m['tft_smape']:>7.1f}% {m['improve']:>9.1f}%")


if __name__ == "__main__":
    main()