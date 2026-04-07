"""
Smart Grid Decision Engine
- Uses Meta model predictions (best 10 forecast hours)
- Net Load = Load - Solar - Wind
- Stress Index = Load / upper_limit
- Renewable % = (Solar + Wind) / Load
- Congestion thresholds: CRITICAL(0.90), WARNING(0.85), WATCH(0.70)
- Confidence bands calibrated from holdout residuals
- Cost signal: oversupply / normal / high
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "processed" / "splits"
MODELS_DIR  = ROOT / "data" / "processed" / "models"
HOLDOUT_DIR = ROOT / "data" / "processed" / "holdout"
OOF_DIR     = ROOT / "data" / "processed" / "oof"
OUTPUT_DIR  = ROOT / "data" / "processed" / "decision_engine"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON      = 10
FULL_HORIZON = 24

CONGESTION_CRITICAL = 0.90
CONGESTION_WARNING  = 0.85
CONGESTION_WATCH    = 0.70
CONF_HIGH = 0.05
CONF_MED  = 0.10
CONF_LOW  = 0.20


# ─────────────────────────────────────────────
# N-HiTS MODEL
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
            nn.Linear(pooled_input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),       nn.ReLU(), nn.Dropout(dropout),
        )
        self.forecast_proj = nn.Linear(hidden_size, self.n_forecast_coeffs)
        self.backcast_proj = nn.Linear(hidden_size, self.n_backcast_coeffs)

    def forward(self, x):
        if self.pool_kernel > 1:
            x_pool = F.max_pool1d(x.unsqueeze(1), kernel_size=self.pool_kernel,
                                  stride=self.pool_kernel).squeeze(1)
        else:
            x_pool = x
        h               = self.mlp(x_pool)
        forecast_coeffs = self.forecast_proj(h)
        backcast_coeffs = self.backcast_proj(h)
        forecast = F.interpolate(forecast_coeffs.unsqueeze(1), size=self.output_size,
                                 mode="linear", align_corners=True).squeeze(1)
        backcast = F.interpolate(backcast_coeffs.unsqueeze(1), size=x.size(1),
                                 mode="linear", align_corners=True).squeeze(1)
        return backcast, forecast


class NHiTSModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=24,
                 n_blocks=3, dropout=0.1):
        super().__init__()
        self.output_size      = output_size
        pool_kernels          = [24, 4, 1]
        expressiveness_ratios = [0.25, 0.5, 1.0]
        safe_kernels = [k if input_size % k == 0 else 1 for k in pool_kernels[:n_blocks]]
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


# ─────────────────────────────────────────────
# TFT MODEL
# ─────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(input_size, output_size, bias=False) \
            if input_size != output_size else None

    def forward(self, x, context=None):
        h    = F.elu(self.fc1(x))
        h    = self.drop(h)
        h    = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        h    = gate * h
        res  = self.skip(x) if self.skip is not None else x
        return self.norm(h + res)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.n_features   = n_features
        self.d_model      = d_model
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        self.weight_grn = GatedResidualNetwork(n_features, d_model, n_features, dropout)

    def forward(self, x):
        B, T, n_feat = x.shape
        feats = []
        for i in range(n_feat):
            feat = x[:, :, i:i+1].reshape(B * T, 1)
            feats.append(self.feature_grns[i](feat).reshape(B, T, self.d_model))
        stacked = torch.stack(feats, dim=2)
        weights = torch.softmax(self.weight_grn(x.reshape(B * T, n_feat)), dim=-1)
        weights = weights.reshape(B, T, n_feat, 1)
        return (stacked * weights).sum(dim=2), weights.squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4,
                 n_encoder_layers=2, d_ff=256,
                 input_window=168, horizon=24, dropout=0.1):
        super().__init__()
        self.vsn           = VariableSelectionNetwork(n_features, d_model, dropout)
        self.lstm          = nn.LSTM(d_model, d_model, n_encoder_layers,
                                     dropout=dropout if n_encoder_layers > 1 else 0.0,
                                     batch_first=True)
        self.lstm_norm     = nn.LayerNorm(d_model)
        self.attention     = nn.MultiheadAttention(d_model, n_heads,
                                                   dropout=dropout, batch_first=True)
        self.attn_norm     = nn.LayerNorm(d_model)
        self.attn_drop     = nn.Dropout(dropout)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.output_grn    = GatedResidualNetwork(d_model, d_ff, d_model, dropout)
        self.output_fc     = nn.Linear(d_model, horizon)

    def forward(self, x):
        vsn_out, _  = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        lstm_out    = self.lstm_norm(lstm_out + vsn_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(self.attn_drop(attn_out) + lstm_out)
        last        = self.post_attn_grn(attn_out)[:, -1, :]
        return self.output_fc(self.output_grn(last))


# ─────────────────────────────────────────────
# STEP 1: CALIBRATION
# ─────────────────────────────────────────────
def calibrate_confidence(track_name):
    pred_dir = HOLDOUT_DIR / track_name
    try:
        meta_preds  = np.load(pred_dir / "meta_preds.npy")
        y_true      = np.load(pred_dir / "y_true.npy")
        residuals   = y_true - meta_preds
        horizon_std = np.std(residuals, axis=0)
        print(f"    [{track_name}] Calibrated from holdout ({len(y_true)} windows)")
    except FileNotFoundError:
        meta_preds  = np.load(OOF_DIR / "meta" / track_name / f"meta_{track_name}_oof.npy")
        y_true      = np.load(OOF_DIR / "lightgbm" / track_name / "y_true.npy")
        mask        = ~np.isnan(meta_preds[:, 0])
        residuals   = y_true[mask] - meta_preds[mask]
        horizon_std = np.std(residuals, axis=0)
        print(f"    [{track_name}] Calibrated from OOF (holdout not found)")
    return horizon_std[:HORIZON]


def get_confidence_label(horizon_std_val):
    if horizon_std_val <= CONF_HIGH:
        pct = max(50, int(100 - (horizon_std_val / CONF_HIGH) * 50))
        return "HIGH", pct
    elif horizon_std_val <= CONF_MED:
        pct = max(35, int(75 - (horizon_std_val / CONF_MED) * 25))
        return "MED ", pct
    elif horizon_std_val <= CONF_LOW:
        pct = max(20, int(55 - (horizon_std_val / CONF_LOW) * 20))
        return "LOW ", pct
    else:
        return "POOR", 15


# ─────────────────────────────────────────────
# STEP 2: LOAD LATEST WINDOW
# ─────────────────────────────────────────────
def get_latest_window(track_name):
    df = pd.read_parquet(SPLITS_DIR / f"{track_name}_track_benchmark.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    target_col = "load_norm"
    ignore     = {"asset_id", "timestamp", target_col}
    feat_cols  = [c for c in df.columns
                  if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]

    asset    = df["asset_id"].unique()[0]
    asset_df = df[df["asset_id"] == asset].sort_values("timestamp").reset_index(drop=True)

    end_idx   = len(asset_df) - 24 - 24
    start_idx = end_idx - 168
    window_df = asset_df.iloc[start_idx:end_idx]

    X_window   = window_df[feat_cols].values.astype(np.float32)
    timestamp  = pd.Timestamp(window_df["timestamp"].iloc[-1])
    # Replace:
    upper_mean = float(asset_df["upper_limit"].mean()) \
        if "upper_limit" in asset_df.columns else 1.0

    # With:
    upper_mean = 1.0  # load_norm is already normalized 0-1, so use 1.0 as capacity

    return X_window, timestamp, upper_mean, feat_cols


# ─────────────────────────────────────────────
# STEP 3: BASE MODEL INFERENCE
# ─────────────────────────────────────────────
def lgbm_predict_window(X_window, track_name):
    model_dir = MODELS_DIR / "lightgbm" / track_name
    mean = np.load(model_dir / "scaler_mean.npy")
    std  = np.where(np.load(model_dir / "scaler_std.npy") < 1e-8, 1.0,
                    np.load(model_dir / "scaler_std.npy"))
    X_flat = np.clip((X_window - mean) / std, -10, 10).reshape(1, -1)
    preds  = []
    for h in range(FULL_HORIZON):
        with open(model_dir / f"lgbm_h{h+1:02d}.pkl", "rb") as f:
            preds.append(pickle.load(f).predict(X_flat)[0])
    return np.array(preds)


def nhits_predict_window(X_window, track_name):
    model_dir = MODELS_DIR / "nhits" / track_name
    mean = np.load(model_dir / "scaler_mean.npy")
    std  = np.where(np.load(model_dir / "scaler_std.npy") < 1e-8, 1.0,
                    np.load(model_dir / "scaler_std.npy"))
    X_flat = np.clip((X_window - mean) / std, -10, 10).reshape(1, -1).astype(np.float32)

    state       = torch.load(model_dir / "nhits_best.pth", map_location=DEVICE)
    hidden_size = state["blocks.0.mlp.0.weight"].shape[0]
    n_blocks    = sum(1 for k in state if k.startswith("blocks.")
                      and k.endswith("mlp.0.weight"))

    model = NHiTSModel(X_flat.shape[1], hidden_size, FULL_HORIZON, n_blocks, 0.0).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_flat).to(DEVICE)).cpu().numpy()[0]


def tft_predict_window(X_window, track_name):
    model_dir = MODELS_DIR / "tft" / track_name
    mean = np.load(model_dir / "scaler_mean.npy")
    std  = np.where(np.load(model_dir / "scaler_std.npy") < 1e-8, 1.0,
                    np.load(model_dir / "scaler_std.npy"))
    X_norm   = np.clip((X_window - mean) / std, -10, 10)
    X_tensor = torch.tensor(X_norm[np.newaxis], dtype=torch.float32).to(DEVICE)

    state   = torch.load(model_dir / "tft_best.pth", map_location=DEVICE)
    d_model = state["lstm.weight_ih_l0"].shape[0] // 4
    d_ff    = state["post_attn_grn.fc1.weight"].shape[0]
    n_heads = {"load": 4, "solar": 4, "wind": 4}[track_name]

    model = TFTModel(X_window.shape[1], d_model, n_heads, 2, d_ff,
                     168, FULL_HORIZON, 0.0).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(X_tensor).cpu().numpy()[0]


def meta_predict_window(lgbm_pred, tft_pred, nhits_pred, timestamp, track_name):
    lgbm  = lgbm_pred.reshape(1, -1)
    tft   = tft_pred.reshape(1, -1)
    nhits = nhits_pred.reshape(1, -1)
    ts    = pd.Timestamp(timestamp)

    ctx = np.array([[
        ts.hour, ts.month, ts.dayofweek, int(ts.dayofweek >= 5),
        np.sin(2 * np.pi * ts.hour / 24), np.cos(2 * np.pi * ts.hour / 24),
        np.sin(2 * np.pi * ts.month / 12), np.cos(2 * np.pi * ts.month / 12),
    ]], dtype=np.float32)

    X_meta = np.concatenate([
        lgbm, tft, nhits,
        lgbm - tft, lgbm - nhits, tft - nhits,
        (lgbm + tft + nhits) / 3.0, ctx
    ], axis=1)

    model_dir = MODELS_DIR / "meta" / track_name
    preds = []
    for h in range(FULL_HORIZON):
        with open(model_dir / f"meta_h{h+1:02d}.pkl", "rb") as f:
            preds.append(pickle.load(f).predict(X_meta)[0])
    return np.array(preds)


# ─────────────────────────────────────────────
# STEP 4: GRID METRICS
# ─────────────────────────────────────────────
def compute_grid_metrics(load_fc, solar_fc, wind_fc, upper_limit):
    metrics = []
    for h in range(HORIZON):
        load  = float(load_fc[h])
        solar = float(solar_fc[h])
        wind  = float(wind_fc[h])

        net_load      = max(0.0, load - solar - wind)
        stress_index  = load / upper_limit if upper_limit > 0.01 else load
        renewable_pct = min(100.0, max(0.0,
                            (solar + wind) / load * 100 if load > 0.01 else 0.0))

        if stress_index >= CONGESTION_CRITICAL:
            congestion, alert = "CRITICAL", "🔴 CRITICAL — Immediate action required"
        elif stress_index >= CONGESTION_WARNING:
            congestion, alert = "WARNING",  "🟠 WARNING  — Prepare demand response"
        elif stress_index >= CONGESTION_WATCH:
            congestion, alert = "WATCH",    "🟡 WATCH    — Monitor closely"
        else:
            congestion, alert = "NORMAL",   "✅ NORMAL"

        if renewable_pct > 80:
            cost_signal = "OVERSUPPLY"
        elif stress_index > CONGESTION_WARNING:
            cost_signal = "HIGH COST "
        else:
            cost_signal = "NORMAL    "

        metrics.append({
            "horizon": h + 1, "load": load, "solar": solar, "wind": wind,
            "net_load": net_load, "stress_index": stress_index,
            "renewable_pct": renewable_pct, "congestion": congestion,
            "alert": alert, "cost_signal": cost_signal,
        })
    return metrics


# ─────────────────────────────────────────────
# STEP 5: RECOMMENDED ACTIONS
# ─────────────────────────────────────────────
def generate_actions(metrics, load_conf, solar_conf, wind_conf):
    actions       = []
    critical_hrs  = [m["horizon"] for m in metrics if m["congestion"] == "CRITICAL"]
    warning_hrs   = [m["horizon"] for m in metrics if m["congestion"] == "WARNING"]
    oversupply_hrs = [m["horizon"] for m in metrics if m["renewable_pct"] > 80]
    low_conf_hrs  = [h + 1 for h, c in enumerate(load_conf) if c[0].strip() == "LOW"]

    if critical_hrs:
        actions.append(f"🔴 H+{critical_hrs[0]}–H+{critical_hrs[-1]}: "
                       f"IMMEDIATE demand curtailment — stress index exceeds 90%")
    if warning_hrs:
        actions.append(f"🟠 H+{warning_hrs[0]}–H+{warning_hrs[-1]}: "
                       f"Activate demand response programs")
    if oversupply_hrs:
        actions.append(f"♻️  H+{oversupply_hrs[0]}–H+{oversupply_hrs[-1]}: "
                       f"Renewable oversupply — consider storage charging or export")
    if low_conf_hrs:
        actions.append(f"⚠️  H+{low_conf_hrs}: "
                       f"Low forecast confidence — maintain operational reserve")

    stress_vals = [m["stress_index"] for m in metrics]
    if stress_vals[-1] > stress_vals[0] + 0.10:
        actions.append("📈 Load stress trending UP — prepare contingency reserves")
    elif stress_vals[-1] < stress_vals[0] - 0.10:
        actions.append("📉 Load stress trending DOWN — reduce reserve gradually")

    if not actions:
        actions.append("✅ All systems nominal — no action required")
    return actions


# ─────────────────────────────────────────────
# STEP 6: PRINT REPORT
# ─────────────────────────────────────────────
def print_report(timestamp, metrics, load_preds, solar_preds, wind_preds,
                 load_conf, solar_conf, wind_conf, actions,
                 load_calib, solar_calib, wind_calib):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SMART GRID DECISION ENGINE")
    print(f"  Reference Time : {timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Forecast Window: Next {HORIZON} hours (H+1 to H+{HORIZON})")
    print(f"  Device         : {DEVICE.upper()}")
    print(sep)

    print(f"\n{'─'*70}")
    print(f"  📊 LOAD TRACK — Next {HORIZON} Hours")
    print(f"{'─'*70}")
    print(f"  {'H+n':<5} {'Forecast':>9} {'Stress':>8} {'Conf':>13} {'Alert'}")
    print(f"  {'─'*65}")
    for i, m in enumerate(metrics):
        conf_label, conf_pct = load_conf[i]
        print(f"  H+{m['horizon']:<3} {m['load']:>9.3f} "
              f"{m['stress_index']:>8.3f} "
              f"{conf_label} ({conf_pct:2d}%)   {m['alert']}")

    print(f"\n{'─'*70}")
    print(f"  ☀️  SOLAR TRACK — Next {HORIZON} Hours")
    print(f"{'─'*70}")
    print(f"  {'H+n':<5} {'Forecast':>9} {'Conf':>13}")
    print(f"  {'─'*30}")
    for i in range(HORIZON):
        conf_label, conf_pct = solar_conf[i]
        print(f"  H+{i+1:<3} {solar_preds[i]:>9.3f} {conf_label} ({conf_pct:2d}%)")

    print(f"\n{'─'*70}")
    print(f"  💨 WIND TRACK — Next {HORIZON} Hours")
    print(f"{'─'*70}")
    print(f"  {'H+n':<5} {'Forecast':>9} {'Conf':>13}")
    print(f"  {'─'*30}")
    for i in range(HORIZON):
        conf_label, conf_pct = wind_conf[i]
        print(f"  H+{i+1:<3} {wind_preds[i]:>9.3f} {conf_label} ({conf_pct:2d}%)")

    print(f"\n{'─'*70}")
    print(f"  ⚡ NET GRID SUMMARY — Next {HORIZON} Hours")
    print(f"{'─'*70}")
    print(f"  {'H+n':<5} {'Net Load':>9} {'Renew%':>8} "
          f"{'Cost Signal':>12} {'Congestion'}")
    print(f"  {'─'*60}")
    for m in metrics:
        print(f"  H+{m['horizon']:<3} {m['net_load']:>9.3f} "
              f"{m['renewable_pct']:>7.1f}% "
              f"{m['cost_signal']:>12}  {m['congestion']}")

    print(f"\n{'─'*70}")
    print(f"  🎯 RECOMMENDED ACTIONS")
    print(f"{'─'*70}")
    for action in actions:
        print(f"  → {action}")

    print(f"\n{'─'*70}")
    print(f"  📋 CONFIDENCE LEGEND  (calibrated from Nov-Dec 2024 holdout)")
    print(f"{'─'*70}")
    print(f"  HIGH — Residual std ≤ 0.05  |  Within ±0.05 ~85%+ of time")
    print(f"  MED  — Residual std ≤ 0.10  |  Within ±0.10 ~70%+ of time")
    print(f"  LOW  — Residual std ≤ 0.20  |  Within ±0.20 ~55%+ of time")
    print(f"  POOR — High uncertainty      |  Use with caution")
    print(f"\n  Per-horizon residual std (H+1..H+{HORIZON}):")
    print(f"  Load : {load_calib.round(4).tolist()}")
    print(f"  Solar: {solar_calib.round(4).tolist()}")
    print(f"  Wind : {wind_calib.round(4).tolist()}")
    print(sep)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"\nDevice: {DEVICE}")

    print("\nStep 1/4 — Calibrating confidence from holdout residuals...")
    load_calib  = calibrate_confidence("load")
    solar_calib = calibrate_confidence("solar")
    wind_calib  = calibrate_confidence("wind")

    print("\nStep 2/4 — Loading latest input windows...")
    X_load,  ts_load,  ul_load,  _ = get_latest_window("load")
    X_solar, ts_solar, ul_solar, _ = get_latest_window("solar")
    X_wind,  ts_wind,  ul_wind,  _ = get_latest_window("wind")
    timestamp = ts_load
    print(f"  Reference timestamp : {timestamp}")
    print(f"  Load  window shape  : {X_load.shape}  upper_limit={ul_load:.4f}")
    print(f"  Solar window shape  : {X_solar.shape}")
    print(f"  Wind  window shape  : {X_wind.shape}")

    print("\nStep 3/4 — Running model inference...")
    print("  LightGBM...", end=" ", flush=True)
    lgbm_load  = lgbm_predict_window(X_load,  "load")
    lgbm_solar = lgbm_predict_window(X_solar, "solar")
    lgbm_wind  = lgbm_predict_window(X_wind,  "wind")
    print("✅")

    print("  N-HiTS...", end=" ", flush=True)
    nhits_load  = nhits_predict_window(X_load,  "load")
    nhits_solar = nhits_predict_window(X_solar, "solar")
    nhits_wind  = nhits_predict_window(X_wind,  "wind")
    print("✅")

    print("  TFT...", end=" ", flush=True)
    tft_load  = tft_predict_window(X_load,  "load")
    tft_solar = tft_predict_window(X_solar, "solar")
    tft_wind  = tft_predict_window(X_wind,  "wind")
    print("✅")

    print("  Meta ensemble...", end=" ", flush=True)
    meta_load  = meta_predict_window(lgbm_load,  tft_load,  nhits_load,  timestamp, "load")
    meta_solar = meta_predict_window(lgbm_solar, tft_solar, nhits_solar, timestamp, "solar")
    meta_wind  = meta_predict_window(lgbm_wind,  tft_wind,  nhits_wind,  timestamp, "wind")
    print("✅")

    load_preds  = meta_load[:HORIZON]
    solar_preds = meta_solar[:HORIZON]
    wind_preds  = meta_wind[:HORIZON]

    print("\nStep 4/4 — Computing grid metrics and generating report...")
    load_conf  = [get_confidence_label(load_calib[h])  for h in range(HORIZON)]
    solar_conf = [get_confidence_label(solar_calib[h]) for h in range(HORIZON)]
    wind_conf  = [get_confidence_label(wind_calib[h])  for h in range(HORIZON)]

    metrics = compute_grid_metrics(load_preds, solar_preds, wind_preds, ul_load)
    actions = generate_actions(metrics, load_conf, solar_conf, wind_conf)

    print_report(timestamp, metrics, load_preds, solar_preds, wind_preds,
                 load_conf, solar_conf, wind_conf, actions,
                 load_calib, solar_calib, wind_calib)

    # Save CSV
    out_rows = [{
        "horizon"         : m["horizon"],
        "load_forecast"   : round(load_preds[i], 4),
        "solar_forecast"  : round(solar_preds[i], 4),
        "wind_forecast"   : round(wind_preds[i], 4),
        "net_load"        : round(m["net_load"], 4),
        "stress_index"    : round(m["stress_index"], 4),
        "renewable_pct"   : round(m["renewable_pct"], 2),
        "congestion"      : m["congestion"],
        "load_confidence" : load_conf[i][0].strip(),
        "load_conf_pct"   : load_conf[i][1],
        "solar_confidence": solar_conf[i][0].strip(),
        "wind_confidence" : wind_conf[i][0].strip(),
        "cost_signal"     : m["cost_signal"].strip(),
        "alert"           : m["alert"],
    } for i, m in enumerate(metrics)]

    out_df = pd.DataFrame(out_rows)
    csv_path = OUTPUT_DIR / "decision_output.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\n  📁 Output saved: {csv_path}")
    print("\n" + out_df.to_string(index=False))


if __name__ == "__main__":
    main()