import pandas as pd
import numpy as np

# Load combined parquet
combined = pd.read_parquet('data/processed/combined/all_assets.parquet')
print(f"After load: {combined.shape}")

# Sort by asset and timestamp before any shift/ffill operation
combined = combined.sort_values(['asset_id', 'group', 'timestamp']).reset_index(drop=True)
print(f"After sort: {combined.shape}")

# Add is_imputed column BEFORE any filling
combined['is_imputed'] = combined['load'].isnull().astype(int)

# Drop rows where null run > 6 (per asset boundary safe)
null_mask = combined['load'].isnull()
null_groups = combined.groupby(['asset_id', 'group'])['load'].transform(
    lambda x: (x.isnull() != x.isnull().shift()).cumsum()
)
run_length = combined.groupby(['asset_id', 'group', null_groups])['load'].transform('size')

drop_rows = null_mask & (run_length > 6)
combined = combined[~drop_rows]
print(f"After drop large runs: {combined.shape}")

# Remove assets with insufficient training data
remove_assets = [
    ('SS Stiens', 'station_installation'),
    ('RS Waskemeer', 'station_installation'),
    ('OS Drachten', 'station_installation'),
    ('RS Anklaar', 'station_installation'),      
    ('SS Noordwolde', 'station_installation')   
]
for asset_id, group in remove_assets:
    mask = (combined['asset_id'] == asset_id) & (combined['group'] == group)
    print(f"Removing {asset_id}: {mask.sum()} rows dropped")
    combined = combined[~mask]
print(f"After asset removal: {combined.shape}")
print(f"Remaining assets: {combined.groupby(['asset_id','group']).ngroups}")

# ffill + bfill per asset for remaining small gaps
combined['load'] = combined.groupby(['asset_id', 'group'])['load'].transform(
    lambda x: x.ffill().bfill()
)
print(f"After ffill: missing load = {combined['load'].isnull().sum()}")

# Flag outliers using YAML limits
combined['is_outlier'] = (
    (combined['load'] > combined['upper_limit']) |
    (combined['load'] < combined['lower_limit'])
)
print(f"Outliers flagged: {combined['is_outlier'].sum()}")

# Normalize load per asset using YAML limits
den = combined['upper_limit'] - combined['lower_limit']
combined['load_norm'] = (combined['load'] - combined['lower_limit']) / den.replace(0, np.nan)
print(f"load_norm range: {combined['load_norm'].min():.3f} to {combined['load_norm'].max():.3f}")

# Drop any remaining NaN rows
combined = combined.dropna(subset=['load'])
print(f"After dropna: {combined.shape}")

# Save clean file
combined.to_parquet('data/processed/combined/all_assets_clean.parquet')
print(f"Final shape: {combined.shape}")
print(f"Saved to data/processed/combined/all_assets_clean.parquet")