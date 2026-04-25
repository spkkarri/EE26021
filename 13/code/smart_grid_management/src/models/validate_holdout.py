"""
Holdout Validation — Nov-Dec 2024
Runs all saved models on unseen validation data and compares.
Models: LightGBM, TFT, N-HiTS, Meta-Ensemble
Tracks: Load, Solar, Wind
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "data" / "processed" / "models"
OUTPUT_DIR = ROOT / "data" / "processed" / "holdout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "input_window"   : 168,
    "horizon"        : 24,
    "sequence_stride": 24,
}

TRACKS = ["load", "solar", "wind"]


# ─────────────────────────────────────────────
# STEP 1: BUILD SEQUENCES FROM HOLDOUT DATA
# ─────────────────────────────────────────────
def build_sequences(df, input_window, horizon, stride):
    target_col = "load_norm"
    ignore     = {"asset_id", "timestamp", target_col}
    feat_cols  = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    all_X, all_y, all_assets, all_ts = [], [], [], []

    for asset_id, asset_df in df.groupby("asset_id"):
        asset_df = asset_df.sort_values("timestamp").reset_index(drop=True)
        values_X = asset_df[feat_cols].values.astype(np.float32)
        values_y = asset_df[target_col].values.astype(np.float32)
        timestamps = asset_df["timestamp"].values

        max_start = len(asset_df) - input_window - horizon + 1
        if max_start <= 0:
            continue

        for s in range(0, max_start, stride):
            all_X.append(values_X[s: s + input_window])
            all_y.append(values_y[s + input_window: s + input_window + horizon])
            all_assets.append(asset_id)
            all_ts.append(timestamps[s + input_window])

    X      = np.stack(all_X)
    y      = np.stack(all_y)
    assets = np.array(all_assets)
    ts     = np.array(all_ts)

    print(f"  Holdout sequences: X {X.shape} | y {y.shape}")
    return X, y, assets, ts, feat_cols


# ─────────────────────────────────────────────
# STEP 2: NAIVE BASELINE
# ─────────────────────────────────────────────
def naive_predict(X, feat_cols):
    if "lag_24" in feat_cols:
        idx = feat_cols.index("lag_24")
        last_lag = X[:, -1, idx]
    elif "lag_1" in feat_cols:
        idx = feat_cols.index("lag_1")
        last_lag = X[:, -1, idx]
    else:
        last_lag = np.zeros(len(X))
    return np.tile(last_lag.reshape(-1, 1), (1, 24))


# ─────────────────────────────────────────────
# STEP 3: LIGHTGBM INFERENCE
# ─────────────────────────────────────────────
def lgbm_predict(X, track_name):
    model_dir = MODELS_DIR / "lightgbm" / track_name
    mean = np.load(model_dir / "scaler_mean.npy")
    std  = np.load(model_dir / "scaler_std.npy")
    std  = np.where(std < 1e-8, 1.0, std)

    X_norm = (X - mean) / std
    X_norm = np.clip(X_norm, -10, 10)
    X_2d   = np.ascontiguousarray(X_norm.reshape(len(X_norm), -1))

    preds = np.zeros((len(X), 24), dtype=np.float32)
    for h in range(24):
        with open(model_dir / f"lgbm_h{h+1:02d}.pkl", "rb") as f:
            model = pickle.load(f)
        preds[:, h] = model.predict(X_2d)

    return preds


# ─────────────────────────────────────────────
# STEP 4: N-HiTS ARCHITECTURE (for loading)
# ─────────────────────────────────────────────
class NHiTSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 pool_kernel=1, expressiveness_ratio=1.0, dropout=0.1):
        super().__init__()
        self.output_size       = output_size
        self.pool_kernel       = pool_kernel
        self.n_forecast_coeffs = max(1, int(expressiveness_ratio * output_size))
        self.n_backcast_coeffs = max(1, input_size // pool_kernel)
        pooled_input_size      = input_size // pool_kernel

        self.mlp = nn.Sequential(
            nn.Linear(pooled_input_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.forecast_proj = nn.Linear(hidden_size, self.n_forecast_coeffs)
        self.backcast_proj = nn.Linear(hidden_size, self.n_backcast_coeffs)

    def forward(self, x):
        if self.pool_kernel > 1:
            x_pool = F.max_pool1d(
                x.unsqueeze(1), kernel_size=self.pool_kernel,
                stride=self.pool_kernel
            ).squeeze(1)
        else:
            x_pool = x
        h = self.mlp(x_pool)
        fc = self.forecast_proj(h)
        bc = self.backcast_proj(h)
        forecast = F.interpolate(fc.unsqueeze(1), size=self.output_size,
                                 mode="linear", align_corners=True).squeeze(1)
        backcast = F.interpolate(bc.unsqueeze(1), size=x.size(1),
                                 mode="linear", align_corners=True).squeeze(1)
        return backcast, forecast


class NHiTSModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=24,
                 n_blocks=3, dropout=0.1):
        super().__init__()
        self.output_size = output_size
        pool_kernels          = [24, 4, 1]
        expressiveness_ratios = [0.25, 0.5, 1.0]

        safe_kernels = []
        for k in pool_kernels[:n_blocks]:
            safe_kernels.append(k if input_size % k == 0 else 1)
        while len(safe_kernels) < n_blocks:
            safe_kernels.append(1)

        ratios = expressiveness_ratios[:n_blocks]
        while len(ratios) < n_blocks:
            ratios.append(1.0)

        self.blocks = nn.ModuleList([
            NHiTSBlock(input_size, hidden_size, output_size,
                       safe_kernels[i], ratios[i], dropout)
            for i in range(n_blocks)
        ])

    def forward(self, x):
        residual     = x
        forecast_sum = torch.zeros(x.size(0), self.output_size,
                                   device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual           = residual - backcast
            forecast_sum       = forecast_sum + forecast
        return forecast_sum


def nhits_predict(X, track_name):
    model_dir = MODELS_DIR / "nhits" / track_name
    mean = np.load(model_dir / "scaler_mean.npy")
    std  = np.load(model_dir / "scaler_std.npy")
    std  = np.where(std < 1e-8, 1.0, std)

    X_norm = np.clip((X - mean) / std, -10, 10)
    X_flat = np.ascontiguousarray(X_norm.reshape(len(X_norm), -1))
    input_size = X_flat.shape[1]

    # Detect config from saved model
    state = torch.load(model_dir / "nhits_best.pth", map_location="cpu")
    # Infer hidden_size from first layer weight
    hidden_size = state["blocks.0.mlp.0.weight"].shape[0]
    n_blocks    = sum(1 for k in state if k.startswith("blocks.") 
                      and k.endswith("mlp.0.weight"))

    model = NHiTSModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=24,
        n_blocks=n_blocks,
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    X_t = torch.tensor(X_flat, dtype=torch.float32)
    preds = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size].to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────
# STEP 5: TFT ARCHITECTURE (for loading)
# ─────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout=0.1, context_size=None):
        super().__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.drop = nn.Dropout(dropout)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        self.skip = nn.Linear(input_size, output_size, bias=False) \
            if input_size != output_size else None

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
        B, T, n_feat = x.shape
        feature_outputs = []
        for i in range(n_feat):
            feat = x[:, :, i:i+1].reshape(B * T, 1)
            out  = self.feature_grns[i](feat)
            feature_outputs.append(out.reshape(B, T, self.d_model))
        stacked = torch.stack(feature_outputs, dim=2)
        flat    = x.reshape(B * T, n_feat)
        weights = self.weight_grn(flat)
        weights = torch.softmax(weights, dim=-1)
        weights = weights.reshape(B, T, n_feat, 1)
        output  = (stacked * weights).sum(dim=2)
        return output, weights.squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4,
                 n_encoder_layers=2, d_ff=256,
                 input_window=168, horizon=24, dropout=0.1):
        super().__init__()
        self.vsn       = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm      = nn.LSTM(d_model, d_model, n_encoder_layers,
                                 dropout=dropout if n_encoder_layers > 1 else 0.0,
                                 batch_first=True)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.output_grn    = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.output_fc     = nn.Linear(d_model, horizon)

    def forward(self, x):
        vsn_out, _  = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        lstm_out    = self.lstm_norm(lstm_out + vsn_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_drop(attn_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)
        post_out    = self.post_attn_grn(attn_out)
        last        = post_out[:, -1, :]
        out         = self.output_grn(last)
        return self.output_fc(out)


def tft_predict(X, track_name):
    model_dir = MODELS_DIR / "tft" / track_name
    mean = np.load(model_dir / "scaler_mean.npy")
    std  = np.load(model_dir / "scaler_std.npy")
    std  = np.where(std < 1e-8, 1.0, std)

    X_norm = np.clip((X - mean) / std, -10, 10)
    n_features = X.shape[2]

    # Infer d_model from saved weights
    state   = torch.load(model_dir / "tft_best.pth", map_location="cpu")
    d_model = state["lstm.weight_ih_l0"].shape[0] // 4

    # Infer n_heads — find from attention
    # Default to saved config values
    track_heads = {"load": 4, "solar": 2, "wind": 2}
    track_layers = {"load": 2, "solar": 2, "wind": 2}
    track_dff    = {"load": 256, "solar": 128, "wind": 128}

    model = TFTModel(
        n_features=n_features,
        d_model=d_model,
        n_heads=track_heads[track_name],
        n_encoder_layers=track_layers[track_name],
        d_ff=track_dff[track_name],
        input_window=168,
        horizon=24,
        dropout=0.0  # no dropout at inference
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size].to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────
# STEP 6: META-MODEL INFERENCE
# ─────────────────────────────────────────────
def meta_predict(lgbm_preds, tft_preds, nhits_preds, ts, track_name):
    model_dir = MODELS_DIR / "meta" / track_name

    # Build same meta features as training
    lgbm_tft   = lgbm_preds  - tft_preds
    lgbm_nhits = lgbm_preds  - nhits_preds
    tft_nhits  = tft_preds   - nhits_preds
    mean_pred  = (lgbm_preds + tft_preds + nhits_preds) / 3.0

    # Context features from timestamps
    ts_dt = pd.to_datetime(ts)
    context = np.column_stack([
        ts_dt.hour,
        ts_dt.month,
        ts_dt.dayofweek,
        (ts_dt.dayofweek >= 5).astype(int),
        np.sin(2 * np.pi * ts_dt.hour / 24),
        np.cos(2 * np.pi * ts_dt.hour / 24),
        np.sin(2 * np.pi * ts_dt.month / 12),
        np.cos(2 * np.pi * ts_dt.month / 12),
    ]).astype(np.float32)

    X_meta = np.concatenate([
        lgbm_preds, tft_preds, nhits_preds,
        lgbm_tft, lgbm_nhits, tft_nhits,
        mean_pred, context
    ], axis=1)

    preds = np.zeros((len(X_meta), 24), dtype=np.float32)
    for h in range(24):
        with open(model_dir / f"meta_h{h+1:02d}.pkl", "rb") as f:
            model = pickle.load(f)
        preds[:, h] = model.predict(X_meta)

    return preds


# ─────────────────────────────────────────────
# STEP 7: METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name=""):
    a    = y_true.flatten()
    b    = y_pred.flatten()
    mae  = np.mean(np.abs(a - b))
    rmse = np.sqrt(np.mean((a - b) ** 2))
    ss_r = np.sum((a - b) ** 2)
    ss_t = np.sum((a - a.mean()) ** 2)
    r2   = 1 - ss_r / ss_t
    corr = np.corrcoef(a, b)[0, 1]

    # Per-horizon R²
    h_r2 = []
    for h in range(24):
        ss_r_h = np.sum((y_true[:, h] - y_pred[:, h]) ** 2)
        ss_t_h = np.sum((y_true[:, h] - y_true[:, h].mean()) ** 2)
        h_r2.append(1 - ss_r_h / ss_t_h)

    return {
        "name": name, "mae": mae, "rmse": rmse,
        "r2": r2, "corr": corr, "h_r2": h_r2
    }


# ─────────────────────────────────────────────
# STEP 8: PLOTS
# ─────────────────────────────────────────────
def plot_holdout(y_true, all_preds, track_name, output_dir):
    hours = np.arange(1, 25)

    # Model colors
    colors = {
        "Meta"     : "black",
        "LightGBM" : "#2196F3",
        "TFT"      : "#9b59b6",
        "N-HiTS"   : "#e74c3c",
        "Naive"    : "gray",
    }

    fig = plt.figure(figsize=(22, 20))
    meta_m = all_preds["Meta"]
    fig.suptitle(
        f"Holdout Validation (Nov-Dec 2024) — {track_name.upper()} Track\n"
        f"Meta: MAE={meta_m['mae']:.4f} | R²={meta_m['r2']:.4f} | r={meta_m['corr']:.4f}\n"
        f"LightGBM: MAE={all_preds['LightGBM']['mae']:.4f} | "
        f"TFT: MAE={all_preds['TFT']['mae']:.4f} | "
        f"N-HiTS: MAE={all_preds['N-HiTS']['mae']:.4f}",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    # Plot 1: MAE per horizon
    ax1 = fig.add_subplot(gs[0, 0])
    for name, m in all_preds.items():
        lw = 2.5 if name == "Meta" else 1.5
        ls = "-" if name == "Meta" else "--"
        if name == "Naive":
            ls = ":"
            lw = 1
        mae_h = np.mean(np.abs(y_true - m["preds"]), axis=0)
        ax1.plot(hours, mae_h, color=colors[name], linewidth=lw,
                 linestyle=ls, label=f"{name} ({m['mae']:.3f})")
    ax1.set_title("MAE per Forecast Hour (Holdout)", fontweight="bold")
    ax1.set_xlabel("H+n")
    ax1.set_ylabel("MAE")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # Plot 2: R² per horizon
    ax2 = fig.add_subplot(gs[0, 1])
    for name, m in all_preds.items():
        lw = 2.5 if name == "Meta" else 1.5
        ls = "-" if name == "Meta" else "--"
        if name == "Naive":
            continue
        ax2.plot(hours, m["h_r2"], color=colors[name], linewidth=lw,
                 linestyle=ls, label=f"{name} ({m['r2']:.3f})")
    ax2.axhline(0.85, color="red", linestyle="--",
                linewidth=1.5, label="Target R²=0.85")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("R² per Forecast Hour (Holdout)", fontweight="bold")
    ax2.set_xlabel("H+n")
    ax2.set_ylabel("R²")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # Plot 3: Summary bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    names = [n for n in all_preds if n != "Naive"]
    maes  = [all_preds[n]["mae"]  for n in names]
    r2s   = [all_preds[n]["r2"]   for n in names]
    x     = np.arange(len(names))
    bars  = ax3.bar(x, maes, color=[colors[n] for n in names], alpha=0.8)
    ax3.set_title("MAE Comparison (Holdout)", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, fontsize=8)
    ax3.set_ylabel("MAE")
    for bar, mae in zip(bars, maes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{mae:.4f}", ha="center", va="bottom", fontsize=7)
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Scatter — Meta vs Actual
    ax4 = fig.add_subplot(gs[1, 0])
    meta_preds = all_preds["Meta"]["preds"]
    lgbm_preds = all_preds["LightGBM"]["preds"]
    sample = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
    ax4.scatter(y_true[sample].flatten(), meta_preds[sample].flatten(),
                alpha=0.2, s=1, color="black", label="Meta")
    ax4.scatter(y_true[sample].flatten(), lgbm_preds[sample].flatten(),
                alpha=0.1, s=1, color="#2196F3", label="LightGBM")
    lims = [y_true.min(), y_true.max()]
    ax4.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
    ax4.set_title("Predicted vs Actual (Holdout)", fontweight="bold")
    ax4.set_xlabel("Actual load_norm")
    ax4.set_ylabel("Predicted load_norm")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # Plot 5: R² bar comparison
    ax5 = fig.add_subplot(gs[1, 1])
    r2_vals = [all_preds[n]["r2"] for n in names]
    bars5 = ax5.bar(x, r2_vals, color=[colors[n] for n in names], alpha=0.8)
    ax5.axhline(0.85, color="red", linestyle="--",
                linewidth=1.5, label="Target=0.85")
    ax5.set_title("R² Comparison (Holdout)", fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(names, rotation=15, fontsize=8)
    ax5.set_ylabel("R²")
    ax5.set_ylim(0, 1)
    for bar, r2 in zip(bars5, r2_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{r2:.3f}", ha="center", va="bottom", fontsize=7)
    ax5.legend(fontsize=7)
    ax5.grid(axis="y", alpha=0.3)

    # Plot 6: Residual distributions
    ax6 = fig.add_subplot(gs[1, 2])
    for name, m in all_preds.items():
        if name == "Naive":
            continue
        res = (y_true - m["preds"]).flatten()
        ax6.hist(np.clip(res, -1, 1), bins=60, alpha=0.4,
                 color=colors[name], label=f"{name} (bias={res.mean():.4f})")
    ax6.axvline(0, color="black", linewidth=2)
    ax6.set_title("Residual Distribution (Holdout)", fontweight="bold")
    ax6.set_xlabel("Actual - Predicted")
    ax6.set_ylabel("Count")
    ax6.legend(fontsize=6)
    ax6.grid(alpha=0.3)

    # Plots 7-9: Best / Median / Worst windows (Meta)
    window_mae = np.mean(np.abs(y_true - meta_preds), axis=1)
    sorted_idx = np.argsort(window_mae)
    for ax, idx, label in [
        (fig.add_subplot(gs[2, 0]), sorted_idx[0],                 "Best"),
        (fig.add_subplot(gs[2, 1]), sorted_idx[len(sorted_idx)//2],"Median"),
        (fig.add_subplot(gs[2, 2]), sorted_idx[-1],                "Worst"),
    ]:
        ax.plot(hours, y_true[idx],    color="black",   linewidth=2,
                marker="o", markersize=3, label="Actual")
        ax.plot(hours, meta_preds[idx], color="black",   linewidth=2,
                linestyle="--", marker="x", markersize=3, label="Meta")
        ax.plot(hours, lgbm_preds[idx], color="#2196F3", linewidth=1,
                linestyle=":", label="LightGBM")
        mae_v = np.mean(np.abs(y_true[idx] - meta_preds[idx]))
        ax.set_title(f"{label} Window (Meta MAE={mae_v:.3f})",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("H+n")
        ax.set_ylabel("load_norm")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    # Plot 10-12: % within thresholds
    ax10 = fig.add_subplot(gs[3, 0])
    thresholds = [0.05, 0.10, 0.15, 0.20]
    for name, m in all_preds.items():
        if name == "Naive":
            continue
        res = np.abs(y_true - m["preds"]).flatten()
        pcts = [np.mean(res < t) * 100 for t in thresholds]
        ax10.plot(thresholds, pcts, color=colors[name], linewidth=2,
                  marker="o", markersize=4, label=name)
    ax10.set_title("% Predictions Within Threshold", fontweight="bold")
    ax10.set_xlabel("Threshold")
    ax10.set_ylabel("% of predictions")
    ax10.legend(fontsize=7)
    ax10.grid(alpha=0.3)

    # Plot 11: OOF vs Holdout R² comparison
    ax11 = fig.add_subplot(gs[3, 1])
    oof_r2 = {
        "LightGBM": {"load": 0.767, "solar": 0.782, "wind": 0.724},
        "TFT"     : {"load": 0.678, "solar": 0.693, "wind": 0.541},
        "N-HiTS"  : {"load": 0.676, "solar": 0.601, "wind": 0.217},
        "Meta"    : {"load": 0.756, "solar": 0.777, "wind": 0.727},
    }
    x2 = np.arange(len(names))
    w  = 0.35
    oof_vals  = [oof_r2.get(n, {}).get(track_name, 0) for n in names]
    hold_vals = [all_preds[n]["r2"] for n in names]
    ax11.bar(x2 - w/2, oof_vals,  w, color=[colors[n] for n in names],
             alpha=0.5, label="OOF (train)")
    ax11.bar(x2 + w/2, hold_vals, w, color=[colors[n] for n in names],
             alpha=0.9, label="Holdout (test)")
    ax11.set_title("OOF vs Holdout R²", fontweight="bold")
    ax11.set_xticks(x2)
    ax11.set_xticklabels(names, rotation=15, fontsize=8)
    ax11.set_ylabel("R²")
    ax11.set_ylim(0, 1)
    ax11.axhline(0.85, color="red", linestyle="--", linewidth=1)
    ax11.legend(fontsize=7)
    ax11.grid(axis="y", alpha=0.3)

    # Plot 12: MAE by asset
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.set_title("Summary", fontweight="bold")
    summary_text = "HOLDOUT RESULTS\n" + "="*30 + "\n"
    for name, m in all_preds.items():
        summary_text += (f"{name:<10} MAE={m['mae']:.4f} "
                         f"R²={m['r2']:.4f}\n")
    summary_text += "\nH+1 R² per model:\n"
    for name, m in all_preds.items():
        if name == "Naive":
            continue
        summary_text += f"  {name:<10} {m['h_r2'][0]:.4f}\n"
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
              fontsize=8, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax12.axis("off")

    save_path = output_dir / f"holdout_{track_name}_validation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {save_path}")


# ─────────────────────────────────────────────
# STEP 9: STATISTICAL SUMMARY
# ─────────────────────────────────────────────
def print_statistical_summary(y_true, all_preds, track_name):
    print(f"\n{'='*65}")
    print(f"  {track_name.upper()} — HOLDOUT STATISTICAL SUMMARY")
    print(f"{'='*65}")
    print(f"  Holdout windows: {len(y_true)}")
    print(f"  Holdout predictions: {len(y_true) * 24}")
    print(f"\n  TRUE values (holdout):")
    yt = y_true.flatten()
    print(f"    mean={yt.mean():.4f}  std={yt.std():.4f}")
    print(f"    min={yt.min():.4f}  max={yt.max():.4f}")
    print(f"    p25={np.percentile(yt,25):.4f}  "
          f"p50={np.percentile(yt,50):.4f}  "
          f"p75={np.percentile(yt,75):.4f}")

    print(f"\n  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
          f"{'r':>8} {'±0.05':>8} {'±0.10':>8} {'±0.20':>8}")
    print(f"  {'-'*72}")
    for name, m in all_preds.items():
        res = np.abs(yt - m["preds"].flatten())
        p05 = np.mean(res < 0.05) * 100
        p10 = np.mean(res < 0.10) * 100
        p20 = np.mean(res < 0.20) * 100
        marker = " ★" if name == "Meta" else ""
        print(f"  {name:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['r2']:>8.4f} {m['corr']:>8.4f} "
              f"{p05:>7.1f}% {p10:>7.1f}% {p20:>7.1f}%{marker}")

    print(f"\n  R² per forecast hour (Meta vs LightGBM):")
    print(f"  {'H+n':<6} {'Meta':>8} {'LightGBM':>10} {'Δ':>8}")
    print(f"  {'-'*35}")
    for h in range(24):
        meta_r2 = all_preds["Meta"]["h_r2"][h]
        lgbm_r2 = all_preds["LightGBM"]["h_r2"][h]
        delta   = meta_r2 - lgbm_r2
        marker  = "✓" if meta_r2 >= 0.85 else " "
        print(f"  H+{h+1:02d}   {meta_r2:>8.4f} {lgbm_r2:>10.4f} "
              f"{delta:>+8.4f} {marker}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    all_track_results = {}

    for track_name in TRACKS:
        print(f"\n{'='*60}")
        print(f"TRACK: {track_name.upper()} — HOLDOUT VALIDATION")
        print(f"{'='*60}")

        # Load holdout data
        df = pd.read_parquet(
            SPLITS_DIR / f"{track_name}_track_benchmark.parquet"
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"  Holdout: {df.shape} | "
              f"{df['timestamp'].min().date()} → "
              f"{df['timestamp'].max().date()}")

        # Build sequences
        X, y, assets, ts, feat_cols = build_sequences(
            df,
            input_window=CONFIG["input_window"],
            horizon=CONFIG["horizon"],
            stride=CONFIG["sequence_stride"],
        )

        print(f"\n  Running inference...")

        # Naive
        naive_preds = naive_predict(X, feat_cols)
        print(f"  ✅ Naive done")

        # LightGBM
        lgbm_preds = lgbm_predict(X, track_name)
        print(f"  ✅ LightGBM done")

        # N-HiTS
        nhits_preds = nhits_predict(X, track_name)
        print(f"  ✅ N-HiTS done")

        # TFT
        tft_preds = tft_predict(X, track_name)
        print(f"  ✅ TFT done")

        # Meta
        ensemble_preds = meta_predict(
            lgbm_preds, tft_preds, nhits_preds, ts, track_name
        )
        print(f"  ✅ Meta done")

        # Compute metrics
        all_preds = {}
        for name, preds in [
            ("Naive",     naive_preds),
            ("LightGBM",  lgbm_preds),
            ("TFT",       tft_preds),
            ("N-HiTS",    nhits_preds),
            ("Meta",      ensemble_preds),
        ]:
            m = compute_metrics(y, preds, name)
            m["preds"] = preds
            all_preds[name] = m

        # Print summary
        print_statistical_summary(y, all_preds, track_name)

        # Plot
        print(f"\n  Plotting...")
        plot_holdout(y, all_preds, track_name, OUTPUT_DIR)

        # Save predictions
        track_out = OUTPUT_DIR / track_name
        track_out.mkdir(parents=True, exist_ok=True)
        np.save(track_out / "y_true.npy",        y)
        np.save(track_out / "naive_preds.npy",   naive_preds)
        np.save(track_out / "lgbm_preds.npy",    lgbm_preds)
        np.save(track_out / "tft_preds.npy",     tft_preds)
        np.save(track_out / "nhits_preds.npy",   nhits_preds)
        np.save(track_out / "meta_preds.npy",    ensemble_preds)
        print(f"  Predictions saved to {track_out}")

        all_track_results[track_name] = {
            name: {"mae": m["mae"], "r2": m["r2"]}
            for name, m in all_preds.items()
        }

    # Final cross-track summary
    print(f"\n{'='*75}")
    print(f"FINAL HOLDOUT SUMMARY — ALL TRACKS")
    print(f"{'='*75}")
    print(f"{'Track':<8} {'Model':<12} {'MAE':>8} {'R²':>8}")
    print("-" * 75)
    for track, results in all_track_results.items():
        for model, m in results.items():
            marker = " ★" if model == "Meta" else ""
            print(f"{track:<8} {model:<12} "
                  f"{m['mae']:>8.4f} {m['r2']:>8.4f}{marker}")
        print()


if __name__ == "__main__":
    main()