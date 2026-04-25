import pandas as pd
import os

# ── PATHS ─────────────────────────────────────────────────────────────────────
FEATURES_DIR = 'data/processed/features'
SPLITS_DIR   = 'data/processed/splits'
os.makedirs(SPLITS_DIR, exist_ok=True)

# ── DATES — ignore YAML competition split, define your own ───────────────────
# Full data: Jan 2024 → Dec 2024
# Train:     Jan 2024 → Oct 2024  (10 months, ~304 days)
# Val:       Nov 2024 → Dec 2024  (2 months,  ~61 days)
TRAIN_START = pd.Timestamp('2024-01-01', tz='UTC')
TRAIN_END   = pd.Timestamp('2024-10-31 23:59:59', tz='UTC')
VAL_START   = pd.Timestamp('2024-11-01', tz='UTC')
VAL_END     = pd.Timestamp('2024-12-31 23:59:59', tz='UTC')

print("Split dates:")
print(f"  Train: {TRAIN_START.date()} → {TRAIN_END.date()}")
print(f"  Val:   {VAL_START.date()}   → {VAL_END.date()}")
print()

TRACKS = ['load_track', 'solar_track', 'wind_track']

# ── SPLIT ─────────────────────────────────────────────────────────────────────
for track in TRACKS:
    df = pd.read_parquet(f'{FEATURES_DIR}/{track}.parquet')

    # Ensure tz-aware UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    # Sort
    df = df.sort_values(['asset_id', 'timestamp']).reset_index(drop=True)

    train = df[
        (df['timestamp'] >= TRAIN_START) &
        (df['timestamp'] <= TRAIN_END)
    ].copy()

    val = df[
        (df['timestamp'] >= VAL_START) &
        (df['timestamp'] <= VAL_END)
    ].copy()

    # Verify no overlap
    assert train['timestamp'].max() < val['timestamp'].min(), \
        f"❌ Overlap detected in {track}!"

    # Verify same assets in both splits
    train_assets = set(train['asset_id'].unique())
    val_assets   = set(val['asset_id'].unique())
    missing = train_assets - val_assets
    if missing:
        print(f"  ⚠ {track}: {len(missing)} assets missing from val")

    # Save
    train.to_parquet(f'{SPLITS_DIR}/{track}_train.parquet',     index=False)
    val.to_parquet(  f'{SPLITS_DIR}/{track}_benchmark.parquet', index=False)

    # Stats
    train_days = (train['timestamp'].max() - train['timestamp'].min()).days
    val_days   = (val['timestamp'].max()   - val['timestamp'].min()).days

    print(f"[{track}]")
    print(f"  train : {train.shape[0]:>8} rows | "
          f"{train['asset_id'].nunique()} assets | "
          f"{train_days} days | "
          f"{train['timestamp'].min().date()} → {train['timestamp'].max().date()}")
    print(f"  val   : {val.shape[0]:>8} rows | "
          f"{val['asset_id'].nunique()} assets | "
          f"{val_days} days | "
          f"{val['timestamp'].min().date()} → {val['timestamp'].max().date()}")

    # Check load_norm stats
    print(f"  train load_norm: mean={train['load_norm'].mean():.3f}  "
          f"std={train['load_norm'].std():.3f}  "
          f"min={train['load_norm'].min():.3f}  "
          f"max={train['load_norm'].max():.3f}")
    print(f"  val   load_norm: mean={val['load_norm'].mean():.3f}  "
          f"std={val['load_norm'].std():.3f}  "
          f"min={val['load_norm'].min():.3f}  "
          f"max={val['load_norm'].max():.3f}")
    print()

print("✅ All splits saved to data/processed/splits/")
print(f"Files: {sorted(os.listdir(SPLITS_DIR))}")