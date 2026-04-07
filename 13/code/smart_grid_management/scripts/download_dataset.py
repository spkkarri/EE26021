"""
download_dataset.py
Downloads the Liander 2024 Smart Grid dataset from Hugging Face.
Run from project root: python scripts/download_dataset.py

Requirements:
    pip install huggingface_hub
"""

import sys
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_DATASET_ID  = "nlsfnr/liander2024"
LOCAL_DATA_DIR = Path("data/raw")

EXPECTED = [
    "EPEX.parquet",
    "profiles.parquet",
    "liander2024_targets.yaml",
    "README.md",
    "load_measurements/",
    "weather_forecasts/",
    "weather_measurements/",
]


def check_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download, list_repo_files
        return True
    except ImportError:
        print("huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        return False


def download_dataset():
    if not check_huggingface_hub():
        sys.exit(1)

    from huggingface_hub import snapshot_download, list_repo_files

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Liander 2024 Smart Grid Dataset Downloader")
    print("=" * 60)
    print(f"\n  Source : huggingface.co/datasets/{HF_DATASET_ID}")
    print(f"  Target : {LOCAL_DATA_DIR.resolve()}")

    print("\n  Listing available files on Hugging Face...")
    try:
        files = list(list_repo_files(HF_DATASET_ID, repo_type="dataset"))
        print(f"  Found {len(files)} files:")
        for f in files:
            print(f"    {f}")
    except Exception as e:
        print(f"  ⚠ Could not list files: {e}")

    print(f"\n  Downloading to {LOCAL_DATA_DIR}...")
    print("  (This may take a few minutes — dataset is ~500MB+)")
    try:
        local_path = snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=str(LOCAL_DATA_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.gitattributes", ".git*"],
        )
        print(f"  ✅ Download complete: {local_path}")
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print("\n  Manual download:")
        print(f"  1. Go to: https://huggingface.co/datasets/{HF_DATASET_ID}")
        print(f"  2. Download all files maintaining folder structure")
        print(f"  3. Place everything in: {LOCAL_DATA_DIR.resolve()}/")
        print(f"\n  Expected structure:")
        for f in EXPECTED:
            print(f"    data/raw/{f}")
        sys.exit(1)

    # Verify
    print("\n  Verifying downloaded files...")
    missing = [e for e in EXPECTED if not (LOCAL_DATA_DIR / e.rstrip("/")).exists()]
    if missing:
        print(f"  ⚠ Missing: {missing}")
    else:
        print("  ✅ All expected files present")

    parquets = list(LOCAL_DATA_DIR.rglob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in parquets) / (1024 * 1024)
    print(f"  Parquet files: {len(parquets)}  |  Total: {total_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("  Dataset ready. Run in order:")
    print("=" * 60)
    print("  python src/preprocessing/load_assets.py")
    print("  python src/preprocessing/pre_process.py")
    print("  python src/preprocessing/feature_eng.py")
    print("  python src/preprocessing/SPLITS.py")
    print("  python src/models/lgbm_model.py")
    print("  python src/models/nhits_model.py")
    print("  python src/models/tft_model.py")
    print("  python src/models/meta_model.py")
    print("  python src/models/validate_holdout.py")
    print("  python src/models/decision_engine.py")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        HF_DATASET_ID = sys.argv[1]
    download_dataset()