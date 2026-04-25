"""
Meta-Model (LightGBM Stacker) — All 3 Tracks
- Combines LightGBM + TFT + N-HiTS OOF predictions
- Learns optimal blending weights per horizon
- Context features: hour, month, is_weekend
- 5-fold walk-forward CV on stacked OOF
- Saves final models for holdout inference
"""
import torch
import pickle
import warnings
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
ROOT       = Path(__file__).resolve().parents[2]
OOF_DIR    = ROOT / "data" / "processed" / "oof"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "data" / "processed" / "models" / "meta"
OUTPUT_DIR = ROOT / "data" / "processed" / "oof" / "meta"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "n_splits"   : 5,
    "horizon"    : 24,
    "input_window": 168,
    "sequence_stride": 24,
    "lgbm": {
        "n_estimators"     : 300,
        "learning_rate"    : 0.05,
        "max_depth"        : 4,
        "num_leaves"       : 15,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_samples": 10,
        "verbose"          : -1,
    }
}

TRACKS = ["load", "solar", "wind"]


# ─────────────────────────────────────────────
# STEP 1: LOAD OOF
# ─────────────────────────────────────────────
def load_oof(track_name):
    base = OOF_DIR

    lgbm_oof  = np.load(base / "lightgbm" / track_name / "lgbm_oof.npy")
    tft_oof   = np.load(base / "tft"       / track_name / "tft_oof.npy")
    nhits_oof = np.load(base / "nhits"     / track_name / "nhits_oof.npy")
    y_true    = np.load(base / "lightgbm"  / track_name / "y_true.npy")
    naive_oof = np.load(base / "lightgbm"  / track_name / "naive_oof.npy")
    assets    = np.load(base / "lightgbm"  / track_name / "assets.npy")

    lgbm_mask  = np.load(base / "lightgbm" / track_name / "lgbm_mask.npy")
    tft_mask   = np.load(base / "tft"       / track_name / "tft_mask.npy")
    nhits_mask = np.load(base / "nhits"     / track_name / "nhits_mask.npy")

    combined_mask = lgbm_mask & tft_mask & nhits_mask

    print(f"  Windows: total={len(y_true)} | "
          f"lgbm={lgbm_mask.sum()} | "
          f"tft={tft_mask.sum()} | "
          f"nhits={nhits_mask.sum()} | "
          f"all valid={combined_mask.sum()}")

    return {
        "lgbm_oof"     : lgbm_oof,
        "tft_oof"      : tft_oof,
        "nhits_oof"    : nhits_oof,
        "y_true"       : y_true,
        "naive_oof"    : naive_oof,
        "combined_mask": combined_mask,
        "assets"       : assets,
    }


# ─────────────────────────────────────────────
# STEP 2: LOAD CONTEXT FEATURES
# ─────────────────────────────────────────────
def load_context_features(track_name, n_windows, combined_mask):
    """
    Load time context features aligned to sequence windows.
    These help the meta-model know WHEN to trust each base model.
    Features: hour, month, is_weekend, day_of_week
    """
    df = pd.read_parquet(
        SPLITS_DIR / f"{track_name}_track_train.parquet"
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    input_window = CONFIG["input_window"]
    horizon      = CONFIG["horizon"]
    stride       = CONFIG["sequence_stride"]

    context_rows = []
    for asset_id, asset_df in df.groupby("asset_id"):
        asset_df = asset_df.sort_values("timestamp").reset_index(drop=True)
        max_start = len(asset_df) - input_window - horizon + 1
        if max_start <= 0:
            continue
        for s in range(0, max_start, stride):
            # Use timestamp at start of forecast horizon
            ts = asset_df.iloc[s + input_window]["timestamp"]
            context_rows.append({
                "hour"       : ts.hour,
                "month"      : ts.month,
                "day_of_week": ts.dayofweek,
                "is_weekend" : int(ts.dayofweek >= 5),
                "hour_sin"   : np.sin(2 * np.pi * ts.hour / 24),
                "hour_cos"   : np.cos(2 * np.pi * ts.hour / 24),
                "month_sin"  : np.sin(2 * np.pi * ts.month / 12),
                "month_cos"  : np.cos(2 * np.pi * ts.month / 12),
            })

    context_df = pd.DataFrame(context_rows)
    assert len(context_df) == n_windows, \
        f"Context rows {len(context_df)} != windows {n_windows}"

    # Apply combined mask
    context_valid = context_df[combined_mask].values.astype(np.float32)
    print(f"  Context features: {context_df.columns.tolist()}")
    return context_valid


# ─────────────────────────────────────────────
# STEP 3: BUILD META FEATURES
# ─────────────────────────────────────────────
def build_meta_features(data, mask, context):
    lgbm  = data["lgbm_oof"][mask]
    tft   = data["tft_oof"][mask]
    nhits = data["nhits_oof"][mask]

    y     = data["y_true"][mask]

    # Pairwise differences — all 6 pairs now
    lgbm_tft   = lgbm - tft
    lgbm_nhits = lgbm - nhits

    tft_nhits  = tft  - nhits


    mean_pred  = (lgbm + tft + nhits ) / 3.0   # NEW: 4-model mean

    X_meta = np.concatenate([
        lgbm, tft, nhits,         # 96 features
        lgbm_tft, lgbm_nhits,# 72 features
        tft_nhits,  # 72 features
        mean_pred,                        # 24 features
        context                           # 8 features
    ], axis=1)                            # Total: 272 features

    return X_meta, y


# ─────────────────────────────────────────────
# STEP 4: META-MODEL OOF
# ─────────────────────────────────────────────
def run_meta_oof(data, config, track_name, context):
    mask     = data["combined_mask"]
    horizon  = config["horizon"]

    lgbm_cfg_short = {
        "n_estimators": 200, "learning_rate": 0.05,
        "max_depth": 3, "num_leaves": 10,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_samples": 10, "verbose": -1,
    }
    lgbm_cfg_long = {
        "n_estimators": 400, "learning_rate": 0.03,
        "max_depth": 5, "num_leaves": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_samples": 10, "verbose": -1,
    }

    X_meta, y = build_meta_features(data, mask, context)
    print(f"  Meta features shape: {X_meta.shape}")
    print(f"  Meta target shape:   {y.shape}")

    tscv       = TimeSeriesSplit(n_splits=config["n_splits"])
    meta_preds = np.full((len(y), horizon), np.nan, dtype=np.float32)
    valid_mask = np.zeros(len(y), dtype=bool)

    feat_names = (
        [f"lgbm_h{h+1}"        for h in range(24)] +
        [f"tft_h{h+1}"         for h in range(24)] +
        [f"nhits_h{h+1}"       for h in range(24)] +
        [f"diff_lgbm_tft_h{h+1}"   for h in range(24)] +
        [f"diff_lgbm_nhits_h{h+1}" for h in range(24)] +
        [f"diff_tft_nhits_h{h+1}"  for h in range(24)] +
        [f"mean_h{h+1}"        for h in range(24)] +
        ["hour", "month", "day_of_week", "is_weekend",
         "hour_sin", "hour_cos", "month_sin", "month_cos"]
    )

    all_importances = np.zeros((horizon, X_meta.shape[1]))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_meta)):
        X_tr, X_vl = X_meta[train_idx], X_meta[val_idx]
        y_tr, y_vl = y[train_idx],      y[val_idx]

        fold_preds = np.zeros((len(val_idx), horizon))

        for h in range(horizon):
            lgbm_cfg = lgbm_cfg_short if h < 8 else lgbm_cfg_long
            model = lgb.LGBMRegressor(**lgbm_cfg)
            model.fit(
                X_tr, y_tr[:, h],
                eval_set=[(X_vl, y_vl[:, h])],
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(period=-1)
                ]
            )
            fold_preds[:, h] = model.predict(X_vl)
            if fold == config["n_splits"] - 1:
                all_importances[h] = model.feature_importances_

        meta_preds[val_idx] = fold_preds
        valid_mask[val_idx] = True

        fold_mae = np.mean(np.abs(y_vl - fold_preds))
        print(f"  Fold {fold+1}/{config['n_splits']} MAE: {fold_mae:.4f}")

    # Train final models on ALL data for holdout inference
    print(f"\n  Training final meta models on full training data...")
    track_model_dir = MODELS_DIR / track_name
    track_model_dir.mkdir(parents=True, exist_ok=True)

    final_models = []
    for h in range(horizon):
        lgbm_cfg = lgbm_cfg_short if h < 8 else lgbm_cfg_long
        model = lgb.LGBMRegressor(**lgbm_cfg)
        model.fit(X_meta, y[:, h])
        final_models.append(model)
        with open(track_model_dir / f"meta_h{h+1:02d}.pkl", "wb") as f:
            pickle.dump(model, f)

    np.save(track_model_dir / "X_meta_mean.npy", X_meta.mean(axis=0))
    np.save(track_model_dir / "X_meta_std.npy",  X_meta.std(axis=0))
    print(f"  Final meta models saved to {track_model_dir}")

    return meta_preds, valid_mask, all_importances, feat_names


# ─────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────
def evaluate_all(data, meta_preds, meta_mask, track_name):
    mask  = data["combined_mask"]
    y_t   = data["y_true"][mask]

    meta_aligned = np.full((mask.sum(), 24), np.nan, dtype=np.float32)
    meta_aligned[meta_mask] = meta_preds[meta_mask]

    def metrics(pred):
        valid = ~np.isnan(pred[:, 0])
        a = y_t[valid].flatten()
        b = pred[valid].flatten()
        mae  = np.mean(np.abs(a - b))
        rmse = np.sqrt(np.mean((a - b) ** 2))
        ss_r = np.sum((a - b) ** 2)
        ss_t = np.sum((a - a.mean()) ** 2)
        r2   = 1 - ss_r / ss_t
        corr = np.corrcoef(a, b)[0, 1]
        return mae, rmse, r2, corr

    results = {}
    preds_dict = {
        "Meta"    : meta_aligned,
        "LightGBM": data["lgbm_oof"][mask],
        "TFT"     : data["tft_oof"][mask],
        "N-HiTS"  : data["nhits_oof"][mask],
        "Naive"   : data["naive_oof"][mask],
    }

    print(f"\n{'='*65}")
    print(f"  {track_name.upper()} — ENSEMBLE EVALUATION")
    print(f"{'='*65}")
    print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'r':>8}")
    print(f"  {'-'*50}")
    for name, pred in preds_dict.items():
        mae, rmse, r2, corr = metrics(pred)
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}
        marker = " ←best" if name == "Meta" else ""
        print(f"  {name:<12} {mae:>8.4f} {rmse:>8.4f} "
              f"{r2:>8.4f} {corr:>8.4f}{marker}")

    valid = ~np.isnan(meta_aligned[:, 0])
    print(f"\n  Meta-Model R² per forecast hour:")
    for h in range(24):
        y_h  = y_t[valid, h]
        p_h  = meta_aligned[valid, h]
        ss_r = np.sum((y_h - p_h) ** 2)
        ss_t = np.sum((y_h - y_h.mean()) ** 2)
        r2_h = 1 - ss_r / ss_t
        bar  = "█" * max(0, int(r2_h * 20))
        print(f"    H+{h+1:02d}: {r2_h:.4f}  {bar}")

    return results, meta_aligned


# ─────────────────────────────────────────────
# STEP 6: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def print_feature_importance(importances, feat_names, track_name):
    lgbm_imp  = importances[:, :24].mean(axis=1)
    tft_imp   = importances[:, 24:48].mean(axis=1)
    nhits_imp = importances[:, 48:72].mean(axis=1)
    ctx_imp   = importances[:, 168:].mean(axis=1)
    total     = lgbm_imp + tft_imp + nhits_imp + ctx_imp + 1e-8

    print(f"\n  Feature importance by model (avg across horizons):")
    print(f"    LightGBM: {lgbm_imp.mean()/total.mean()*100:.1f}%")
    print(f"    TFT:      {tft_imp.mean()/total.mean()*100:.1f}%")
    print(f"    N-HiTS:   {nhits_imp.mean()/total.mean()*100:.1f}%")
    print(f"    Context:  {ctx_imp.mean()/total.mean()*100:.1f}%")


# ─────────────────────────────────────────────
# STEP 7: PLOTS
# ─────────────────────────────────────────────
def plot_results(data, meta_aligned, results, importances,
                 track_name, output_dir):
    mask  = data["combined_mask"]
    y_t   = data["y_true"][mask]
    valid = ~np.isnan(meta_aligned[:, 0])
    meta  = meta_aligned[valid]
    y_t   = y_t[valid]
    lgbm  = data["lgbm_oof"][mask][valid]
    tft   = data["tft_oof"][mask][valid]
    nhits = data["nhits_oof"][mask][valid]
    naive = data["naive_oof"][mask][valid]
    hours = np.arange(1, 25)

    def h_mae(pred):
        return np.mean(np.abs(y_t - pred), axis=0)

    def h_r2(pred):
        r2s = []
        for h in range(24):
            ss_r = np.sum((y_t[:, h] - pred[:, h]) ** 2)
            ss_t = np.sum((y_t[:, h] - y_t[:, h].mean()) ** 2)
            r2s.append(max(0, 1 - ss_r / ss_t))
        return np.array(r2s)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"Meta-Model Ensemble — {track_name.upper()} Track\n"
        f"Meta: MAE={results['Meta']['mae']:.4f} | "
        f"R²={results['Meta']['r2']:.4f} | "
        f"r={results['Meta']['corr']:.4f}\n"
        f"LightGBM: MAE={results['LightGBM']['mae']:.4f} | "
        f"TFT: MAE={results['TFT']['mae']:.4f} | "
        f"N-HiTS: MAE={results['N-HiTS']['mae']:.4f}",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hours, h_mae(meta),  color="black",   linewidth=2.5,
             label=f"Meta ({results['Meta']['mae']:.3f})")
    ax1.plot(hours, h_mae(lgbm),  color="#2196F3", linewidth=1.5,
             linestyle="--", label=f"LightGBM ({results['LightGBM']['mae']:.3f})")
    ax1.plot(hours, h_mae(tft),   color="#9b59b6", linewidth=1.5,
             linestyle="--", label=f"TFT ({results['TFT']['mae']:.3f})")
    ax1.plot(hours, h_mae(nhits), color="#e74c3c", linewidth=1.5,
             linestyle="--", label=f"N-HiTS ({results['N-HiTS']['mae']:.3f})")
    ax1.plot(hours, h_mae(naive), color="gray",    linewidth=1,
             linestyle=":", label=f"Naive ({results['Naive']['mae']:.3f})")
    ax1.set_title("MAE per Forecast Hour", fontweight="bold")
    ax1.set_xlabel("H+n")
    ax1.set_ylabel("MAE")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    meta_r2 = h_r2(meta)
    lgbm_r2 = h_r2(lgbm)
    ax2.bar(hours, meta_r2, color="black",   alpha=0.7, label="Meta")
    ax2.plot(hours, lgbm_r2, color="#2196F3", linewidth=2,
             marker="o", markersize=3, label="LightGBM")
    ax2.axhline(0.85, color="red", linestyle="--",
                linewidth=1.5, label="Target R²=0.85")
    ax2.set_title("R² per Forecast Hour", fontweight="bold")
    ax2.set_xlabel("H+n")
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    lgbm_imp  = importances[:, :24].sum(axis=1)
    tft_imp   = importances[:, 24:48].sum(axis=1)
    nhits_imp = importances[:, 48:72].sum(axis=1)
    diff_imp  = importances[:, 72:168].sum(axis=1)
    ctx_imp   = importances[:, 168:].sum(axis=1)
    total     = lgbm_imp + tft_imp + nhits_imp + diff_imp + ctx_imp + 1e-8
    ax3.stackplot(
        hours,
        lgbm_imp/total*100,
        tft_imp/total*100,
        nhits_imp/total*100,
        diff_imp/total*100,
        ctx_imp/total*100,
        labels=["LightGBM", "TFT", "N-HiTS", "Differences", "Context"],
        colors=["#2196F3", "#9b59b6", "#e74c3c", "#95a5a6", "#f39c12"],
        alpha=0.8
    )
    ax3.set_title("Feature Importance by Model per Horizon",
                  fontweight="bold")
    ax3.set_xlabel("H+n")
    ax3.set_ylabel("Importance %")
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=6, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    sample = np.random.choice(len(y_t), min(2000, len(y_t)), replace=False)
    ax4.scatter(y_t[sample].flatten(), meta[sample].flatten(),
                alpha=0.2, s=1, color="black", label="Meta")
    ax4.scatter(y_t[sample].flatten(), lgbm[sample].flatten(),
                alpha=0.1, s=1, color="#2196F3", label="LightGBM")
    lims = [y_t.min(), y_t.max()]
    ax4.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
    ax4.set_title("Predicted vs Actual", fontweight="bold")
    ax4.set_xlabel("Actual load_norm")
    ax4.set_ylabel("Predicted load_norm")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    window_mae = np.mean(np.abs(y_t - meta), axis=1)
    sorted_idx = np.argsort(window_mae)
    for ax, idx, label in [
        (fig.add_subplot(gs[1, 1]), sorted_idx[0],                 "Best"),
        (fig.add_subplot(gs[1, 2]), sorted_idx[len(sorted_idx)//2],"Median"),
        (fig.add_subplot(gs[2, 0]), sorted_idx[-1],                "Worst"),
    ]:
        ax.plot(hours, y_t[idx],  color="black",   linewidth=2,
                marker="o", markersize=3, label="Actual")
        ax.plot(hours, meta[idx], color="black",   linewidth=2,
                linestyle="--", marker="x", markersize=3, label="Meta")
        ax.plot(hours, lgbm[idx], color="#2196F3", linewidth=1,
                linestyle=":", label="LightGBM")
        ax.plot(hours, tft[idx],  color="#9b59b6", linewidth=1,
                linestyle=":", label="TFT")
        mae_meta = np.mean(np.abs(y_t[idx] - meta[idx]))
        ax.set_title(
            f"{label} Window (Meta MAE={mae_meta:.3f})",
            fontweight="bold", fontsize=9
        )
        ax.set_xlabel("H+n")
        ax.set_ylabel("load_norm")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    improvement = h_mae(lgbm) - h_mae(meta)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in improvement]
    ax8.bar(hours, improvement * 100, color=colors, alpha=0.8)
    ax8.axhline(0, color="black", linewidth=1)
    ax8.set_title("Meta MAE Improvement vs LightGBM (%×100)",
                  fontweight="bold")
    ax8.set_xlabel("H+n")
    ax8.set_ylabel("Improvement (green=better)")
    ax8.grid(axis="y", alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    r2_improvement = meta_r2 - lgbm_r2
    colors9 = ["#2ecc71" if v > 0 else "#e74c3c" for v in r2_improvement]
    ax9.bar(hours, r2_improvement, color=colors9, alpha=0.8)
    ax9.axhline(0, color="black", linewidth=1)
    ax9.set_title("Meta R² Improvement vs LightGBM",
                  fontweight="bold")
    ax9.set_xlabel("H+n")
    ax9.set_ylabel("R² improvement (green=better)")
    ax9.grid(axis="y", alpha=0.3)

    save_path = output_dir / f"meta_{track_name}_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    all_results = {}

    for track_name in TRACKS:
        print(f"\n{'='*60}")
        print(f"TRACK: {track_name.upper()}")
        print(f"{'='*60}")

        print(f"\n  Loading OOF predictions...")
        data = load_oof(track_name)

        # Load context features aligned to windows
        print(f"\n  Loading context features...")
        context = load_context_features(
            track_name,
            n_windows=len(data["y_true"]),
            combined_mask=data["combined_mask"]
        )

        print(f"\n  Training meta-model...")
        meta_preds, meta_mask, importances, feat_names = run_meta_oof(
            data, CONFIG, track_name, context
        )

        results, meta_aligned = evaluate_all(
            data, meta_preds, meta_mask, track_name
        )
        all_results[track_name] = results

        print_feature_importance(importances, feat_names, track_name)

        print(f"\n  Plotting...")
        plot_results(
            data, meta_aligned, results, importances,
            track_name, OUTPUT_DIR
        )

        track_out = OUTPUT_DIR / track_name
        track_out.mkdir(parents=True, exist_ok=True)
        np.save(track_out / f"meta_{track_name}_oof.npy",  meta_aligned)
        np.save(track_out / f"meta_{track_name}_mask.npy", data["combined_mask"])
        print(f"  Saved OOF to {track_out}")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — META-MODEL vs BASE MODELS")
    print(f"{'='*70}")
    print(f"{'Track':<8} {'Model':<12} {'MAE':>8} {'R²':>8} {'r':>8}")
    print("-" * 70)
    for track, results in all_results.items():
        for model, m in results.items():
            marker = " ★" if model == "Meta" else ""
            print(f"{track:<8} {model:<12} "
                  f"{m['mae']:>8.4f} {m['r2']:>8.4f} "
                  f"{m['corr']:>8.4f}{marker}")
        print()


if __name__ == "__main__":
    main()