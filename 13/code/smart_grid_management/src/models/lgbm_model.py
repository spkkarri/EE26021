"""
LightGBM OOF Training — All 3 Tracks
- Loads from data/processed/splits/
- 5-fold walk-forward CV (no leakage)
- Trains 24 models per fold (one per horizon)
- Saves OOF predictions + trained models
- Full evaluation + plots comparing LightGBM vs Naive
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR  = ROOT / "data" / "processed" / "oof" / "lightgbm"
MODELS_DIR  = ROOT / "data" / "processed" / "models" / "lightgbm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "input_window"   : 168,
    "horizon"        : 24,
    "sequence_stride": 24,
    "n_splits"       : 5,
    "lgbm": {
        "n_estimators"    : 500,
        "learning_rate"   : 0.05,
        "max_depth"       : 6,
        "num_leaves"      : 31,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "verbose"         : -1,
    }
}

TRACKS = {
    "load" : "load_track_train.parquet",
    "solar": "solar_track_train.parquet",
    "wind" : "wind_track_train.parquet",
}


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

    X      = np.stack(all_X)
    y      = np.stack(all_y)
    assets = np.array(all_assets)

    print(f"  Sequences: X {X.shape} | y {y.shape}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")
    return X, y, assets, feat_cols


# ─────────────────────────────────────────────
# STEP 2: NAIVE BASELINE (lag_24 persistence)
# ─────────────────────────────────────────────
def naive_baseline_oof(X, y, feat_cols):
    """
    Persistence model — same hour yesterday as forecast.
    Uses lag_24 feature (load_norm 24h ago) from input window.
    This is already normalized so comparison with LightGBM is fair.
    """
    if "lag_24" in feat_cols:
        idx = feat_cols.index("lag_24")
        # Last value of lag_24 in input window = load_norm from 24h ago
        last_lag24 = X[:, -1, idx]  # (N,)
    elif "lag_1" in feat_cols:
        idx = feat_cols.index("lag_1")
        last_lag24 = X[:, -1, idx]
        print("  ⚠ lag_24 not found, using lag_1 for naive baseline")
    else:
        print("  ⚠ No lag feature found, using zeros for naive baseline")
        last_lag24 = np.zeros(len(y))

    # Tile: same value for all 24 forecast hours
    preds = np.tile(last_lag24.reshape(-1, 1), (1, y.shape[1]))  # (N, 24)
    mae   = np.mean(np.abs(y - preds))
    return mae, preds


# ─────────────────────────────────────────────
# STEP 3: LIGHTGBM OOF
# ─────────────────────────────────────────────
def run_lgbm_oof(X, y, config, track_name):
    """
    5-fold walk-forward OOF.
    Normalizes per fold on train only — no leakage.
    Saves best fold models to disk.
    Returns: oof_preds (N, 24), valid_mask (N,), scalers dict
    """
    n_splits  = config["n_splits"]
    horizon   = config["horizon"]
    lgbm_cfg  = config["lgbm"]

    tscv       = TimeSeriesSplit(n_splits=n_splits)
    oof_preds  = np.full((len(X), horizon), np.nan, dtype=np.float32)
    valid_mask = np.zeros(len(X), dtype=bool)
    scalers    = {}  # fold → {mean, std}

    best_fold_mae  = float("inf")
    best_fold_idx  = -1
    best_fold_models = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold+1}/{n_splits} | "
              f"train {len(train_idx)} | val {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalize on train fold only
        X_tr_flat = X_train.reshape(-1, X_train.shape[-1])
        mean = X_tr_flat.mean(axis=0)
        std  = X_tr_flat.std(axis=0)
        std  = np.where(std < 1e-8, 1.0, std)

        X_train_n = (X_train - mean) / std
        X_val_n   = (X_val   - mean) / std

        scalers[fold] = {"mean": mean, "std": std}

        # Flatten for LightGBM
        X_tr_2d = np.ascontiguousarray(X_train_n.reshape(len(X_train_n), -1))
        X_vl_2d = np.ascontiguousarray(X_val_n.reshape(len(X_val_n),   -1))

        # Train one model per horizon
        fold_preds  = np.zeros((len(val_idx), horizon))
        fold_models = []

        for h in range(horizon):
            model = lgb.LGBMRegressor(**lgbm_cfg)
            model.fit(
                X_tr_2d, y_train[:, h],
                eval_set=[(X_vl_2d, y_val[:, h])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1)
                ]
            )
            fold_preds[:, h] = model.predict(X_vl_2d)
            fold_models.append(model)

        oof_preds[val_idx]  = fold_preds
        valid_mask[val_idx] = True

        fold_mae = np.mean(np.abs(y_val - fold_preds))
        print(f"    Fold {fold+1} MAE: {fold_mae:.4f}")

        if fold_mae < best_fold_mae:
            best_fold_mae    = fold_mae
            best_fold_idx    = fold + 1
            best_fold_models = fold_models
            best_scaler      = scalers[fold]

    print(f"\n  Best fold: {best_fold_idx} (MAE={best_fold_mae:.4f})")

    # Save best fold models
    track_model_dir = MODELS_DIR / track_name
    track_model_dir.mkdir(parents=True, exist_ok=True)

    for h, model in enumerate(best_fold_models):
        model_path = track_model_dir / f"lgbm_h{h+1:02d}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Save scaler for best fold
    np.save(track_model_dir / "scaler_mean.npy", best_scaler["mean"])
    np.save(track_model_dir / "scaler_std.npy",  best_scaler["std"])
    print(f"  Models saved to {track_model_dir}")

    return oof_preds, valid_mask, scalers


# ─────────────────────────────────────────────
# STEP 4: EVALUATION
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred, naive_pred, mask, track_name):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_n = naive_pred[mask]

    def metrics(a, b):
        mae  = np.mean(np.abs(a - b))
        rmse = np.sqrt(np.mean((a - b) ** 2))
        d    = (np.abs(a) + np.abs(b)) / 2
        m    = d > 1e-8
        smape = np.mean(np.abs(a[m] - b[m]) / d[m]) * 100
        corr  = np.corrcoef(a.flatten(), b.flatten())[0, 1]
        return mae, rmse, smape, corr

    lgbm_mae,  lgbm_rmse,  lgbm_smape,  lgbm_corr  = metrics(y_t, y_p)
    naive_mae, naive_rmse, naive_smape, naive_corr = metrics(y_t, y_n)
    improve = (naive_mae - lgbm_mae) / naive_mae * 100

    print(f"\n{'='*55}")
    print(f"  {track_name.upper()} — EVALUATION")
    print(f"{'='*55}")
    print(f"  {'Metric':<12} {'LightGBM':>12} {'Naive':>12} {'Improvement':>12}")
    print(f"  {'-'*50}")
    print(f"  {'MAE':<12} {lgbm_mae:>12.4f} {naive_mae:>12.4f} {improve:>11.1f}%")
    print(f"  {'RMSE':<12} {lgbm_rmse:>12.4f} {naive_rmse:>12.4f}")
    print(f"  {'sMAPE':<12} {lgbm_smape:>11.1f}% {naive_smape:>11.1f}%")
    print(f"  {'Pearson r':<12} {lgbm_corr:>12.4f} {naive_corr:>12.4f}")
    print(f"\n  Pred  — mean: {y_p.mean():.4f}  std: {y_p.std():.4f}")
    print(f"  True  — mean: {y_t.mean():.4f}  std: {y_t.std():.4f}")
    print(f"  Naive — mean: {y_n.mean():.4f}  std: {y_n.std():.4f}")

    return {
        "track"     : track_name,
        "lgbm_mae"  : lgbm_mae,
        "lgbm_rmse" : lgbm_rmse,
        "lgbm_smape": lgbm_smape,
        "lgbm_corr" : lgbm_corr,
        "naive_mae" : naive_mae,
        "improve"   : improve,
    }


# ─────────────────────────────────────────────
# STEP 5: STATISTICAL SUMMARY
# ─────────────────────────────────────────────
def print_statistical_summary(y_true, y_pred, naive_pred, mask, track_name):
    y_t = y_true[mask].flatten()
    y_p = y_pred[mask].flatten()
    y_n = naive_pred[mask].flatten()

    res_lgbm  = y_t - y_p
    res_naive = y_t - y_n

    print(f"\n{'='*55}")
    print(f"  STATISTICAL SUMMARY — {track_name.upper()}")
    print(f"{'='*55}")
    print(f"  N windows:     {mask.sum()}")
    print(f"  N predictions: {len(y_t)}")

    print(f"\n  TRUE values:")
    print(f"    mean={y_t.mean():.4f}  std={y_t.std():.4f}")
    print(f"    min={y_t.min():.4f}  max={y_t.max():.4f}")
    print(f"    p25={np.percentile(y_t,25):.4f}  "
          f"p50={np.percentile(y_t,50):.4f}  "
          f"p75={np.percentile(y_t,75):.4f}")

    for label, pred, res in [
        ("LightGBM", y_p, res_lgbm),
        ("Naive",    y_n, res_naive),
    ]:
        print(f"\n  {label} predictions:")
        print(f"    mean={pred.mean():.4f}  std={pred.std():.4f}")
        print(f"    bias={res.mean():.4f}  residual std={res.std():.4f}")
        for thresh in [0.05, 0.10, 0.20]:
            pct = np.mean(np.abs(res) < thresh) * 100
            print(f"    % within ±{thresh:.2f}: {pct:.1f}%")


# ─────────────────────────────────────────────
# STEP 6: PLOTS — LightGBM vs Naive vs Actual
# ─────────────────────────────────────────────
def plot_results(y_true, y_pred, naive_pred, mask, track_name, metrics, output_dir):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_n = naive_pred[mask]

    horizon_mae_lgbm  = np.mean(np.abs(y_t - y_p), axis=0)
    horizon_mae_naive = np.mean(np.abs(y_t - y_n), axis=0)
    horizon_rmse_lgbm = np.sqrt(np.mean((y_t - y_p) ** 2, axis=0))
    window_mae_lgbm   = np.mean(np.abs(y_t - y_p), axis=1)
    window_mae_naive  = np.mean(np.abs(y_t - y_n), axis=1)

    sorted_idx = np.argsort(window_mae_lgbm)
    best_idx   = sorted_idx[0]
    med_idx    = sorted_idx[len(sorted_idx) // 2]
    worst_idx  = sorted_idx[-1]
    hours      = np.arange(1, 25)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"LightGBM vs Naive Baseline — {track_name.upper()} Track\n"
        f"LightGBM: MAE={metrics['lgbm_mae']:.4f} | "
        f"RMSE={metrics['lgbm_rmse']:.4f} | "
        f"sMAPE={metrics['lgbm_smape']:.1f}% | "
        f"r={metrics['lgbm_corr']:.4f}\n"
        f"Naive:     MAE={metrics['naive_mae']:.4f} | "
        f"Improvement={metrics['improve']:.1f}%",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # Plot 1: MAE per horizon — LightGBM vs Naive
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hours, horizon_mae_lgbm,  color="#2196F3", linewidth=2,
             marker="o", markersize=3, label="LightGBM")
    ax1.plot(hours, horizon_mae_naive, color="#e74c3c", linewidth=2,
             linestyle="--", marker="x", markersize=3, label="Naive")
    ax1.fill_between(hours, horizon_mae_lgbm, horizon_mae_naive,
                     alpha=0.15, color="#2196F3",
                     label="LightGBM advantage")
    ax1.set_title("MAE per Forecast Hour", fontweight="bold")
    ax1.set_xlabel("H+n")
    ax1.set_ylabel("MAE")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot 2: RMSE per horizon
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(hours, horizon_rmse_lgbm, color="#FF5722", alpha=0.85)
    ax2.set_title("RMSE per Forecast Hour (LightGBM)", fontweight="bold")
    ax2.set_xlabel("H+n")
    ax2.set_ylabel("RMSE")
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: MAE distribution — LightGBM vs Naive
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(window_mae_lgbm,  bins=50, color="#2196F3",
             alpha=0.6, label="LightGBM")
    ax3.hist(window_mae_naive, bins=50, color="#e74c3c",
             alpha=0.6, label="Naive")
    ax3.axvline(window_mae_lgbm.mean(),  color="#2196F3", linewidth=2,
                linestyle="--", label=f"LGB mean={window_mae_lgbm.mean():.3f}")
    ax3.axvline(window_mae_naive.mean(), color="#e74c3c", linewidth=2,
                linestyle="--", label=f"Naive mean={window_mae_naive.mean():.3f}")
    ax3.set_title("MAE Distribution — LightGBM vs Naive", fontweight="bold")
    ax3.set_xlabel("MAE")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # Plot 4: Predicted vs Actual scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = np.random.choice(len(y_t), min(1000, len(y_t)), replace=False)
    ax4.scatter(y_t[sample].flatten(), y_p[sample].flatten(),
                alpha=0.3, s=2, color="#2196F3", label="LightGBM")
    ax4.scatter(y_t[sample].flatten(), y_n[sample].flatten(),
                alpha=0.2, s=2, color="#e74c3c", label="Naive")
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    ax4.plot(lims, lims, "k--", linewidth=1.5, label="Perfect")
    ax4.set_title("Predicted vs Actual", fontweight="bold")
    ax4.set_xlabel("Actual load_norm")
    ax4.set_ylabel("Predicted load_norm")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # Plots 5, 6, 7: Best / Median / Worst window
    for ax, idx, label in [
        (fig.add_subplot(gs[1, 1]), best_idx,  "Best"),
        (fig.add_subplot(gs[1, 2]), med_idx,   "Median"),
        (fig.add_subplot(gs[2, 0]), worst_idx, "Worst"),
    ]:
        ax.plot(hours, y_t[idx], color="black", linewidth=2,
                marker="o", markersize=3, label="Actual")
        ax.plot(hours, y_p[idx], color="#2196F3", linewidth=2,
                linestyle="--", marker="x", markersize=3, label="LightGBM")
        ax.plot(hours, y_n[idx], color="#e74c3c", linewidth=1.5,
                linestyle=":", marker="s", markersize=2, label="Naive")
        mae_lgbm  = np.mean(np.abs(y_t[idx] - y_p[idx]))
        mae_naive = np.mean(np.abs(y_t[idx] - y_n[idx]))
        ax.set_title(
            f"{label} Window\n"
            f"LGB MAE={mae_lgbm:.3f} | Naive MAE={mae_naive:.3f}",
            fontweight="bold", fontsize=9
        )
        ax.set_xlabel("H+n")
        ax.set_ylabel("load_norm")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Plot 8: Residual distributions — LightGBM vs Naive
    ax8 = fig.add_subplot(gs[2, 1])
    res_lgbm  = (y_t - y_p).flatten()
    res_naive = (y_t - y_n).flatten()
    ax8.hist(res_lgbm,  bins=60, color="#2196F3",
             alpha=0.6, label=f"LightGBM (bias={res_lgbm.mean():.4f})")
    ax8.hist(res_naive, bins=60, color="#e74c3c",
             alpha=0.6, label=f"Naive (bias={res_naive.mean():.4f})")
    ax8.axvline(0, color="black", linewidth=2)
    ax8.set_title("Residual Distribution", fontweight="bold")
    ax8.set_xlabel("Actual - Predicted")
    ax8.set_ylabel("Count")
    ax8.legend(fontsize=7)
    ax8.grid(alpha=0.3)

    # Plot 9: Cumulative MAE — LightGBM vs Naive
    ax9 = fig.add_subplot(gs[2, 2])
    sorted_lgbm  = np.sort(window_mae_lgbm)
    sorted_naive = np.sort(window_mae_naive)
    cum = np.arange(1, len(sorted_lgbm) + 1) / len(sorted_lgbm)
    ax9.plot(sorted_lgbm,  cum, color="#2196F3", linewidth=2,
             label=f"LightGBM (mean={window_mae_lgbm.mean():.3f})")
    ax9.plot(sorted_naive, cum, color="#e74c3c", linewidth=2,
             linestyle="--",
             label=f"Naive (mean={window_mae_naive.mean():.3f})")
    ax9.set_title("Cumulative MAE Distribution", fontweight="bold")
    ax9.set_xlabel("MAE")
    ax9.set_ylabel("Fraction of Windows")
    ax9.legend(fontsize=8)
    ax9.grid(alpha=0.3)

    save_path = output_dir / f"lgbm_{track_name}_oof_results.png"
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

        # Load data
        df = pd.read_parquet(SPLITS_DIR / fname)
        print(f"  Loaded: {df.shape} | "
              f"{df['timestamp'].min().date()} → "
              f"{df['timestamp'].max().date()}")
        print(f"  Assets: {df['asset_id'].nunique()}")

        # Build sequences
        print(f"\n  Building sequences...")
        X, y, assets, feat_cols = build_sequences(
            df,
            input_window=CONFIG["input_window"],
            horizon=CONFIG["horizon"],
            stride=CONFIG["sequence_stride"],
        )

        # Naive baseline
        naive_mae, naive_preds = naive_baseline_oof(X, y, feat_cols)
        print(f"  Naive baseline MAE: {naive_mae:.4f}")

        # LightGBM OOF
        print(f"\n  Running LightGBM OOF...")
        oof_preds, valid_mask, scalers = run_lgbm_oof(X, y, CONFIG, track_name)

        # Evaluate — LightGBM vs Naive
        metrics = evaluate(y, oof_preds, naive_preds, valid_mask, track_name)
        all_metrics.append(metrics)

        # Statistical summary
        print_statistical_summary(y, oof_preds, naive_preds, valid_mask, track_name)

        # Plot — Actual vs LightGBM vs Naive
        print(f"\n  Plotting...")
        plot_results(y, oof_preds, naive_preds, valid_mask,
                     track_name, metrics, OUTPUT_DIR)

        # Save OOF arrays
        track_out = OUTPUT_DIR / track_name
        track_out.mkdir(parents=True, exist_ok=True)
        np.save(track_out / "lgbm_oof.npy",   oof_preds)
        np.save(track_out / "lgbm_mask.npy",  valid_mask)
        np.save(track_out / "naive_oof.npy",  naive_preds)
        np.save(track_out / "y_true.npy",     y)
        np.save(track_out / "assets.npy",     assets)
        print(f"  OOF saved to {track_out}")

    # Final summary table
    print(f"\n{'='*65}")
    print(f"FINAL SUMMARY — LightGBM vs Naive Baseline")
    print(f"{'='*65}")
    print(f"{'Track':<8} {'LGB MAE':>9} {'Naive MAE':>10} "
          f"{'RMSE':>8} {'sMAPE':>8} {'Improve':>10}")
    print("-" * 65)
    for m in all_metrics:
        print(f"{m['track']:<8} {m['lgbm_mae']:>9.4f} "
              f"{m['naive_mae']:>10.4f} {m['lgbm_rmse']:>8.4f} "
              f"{m['lgbm_smape']:>7.1f}% {m['improve']:>9.1f}%")


if __name__ == "__main__":
    main()