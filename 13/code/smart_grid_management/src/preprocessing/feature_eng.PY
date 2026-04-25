import pandas as pd
import numpy as np
import os
os.makedirs('data/processed/features', exist_ok=True)

combined = pd.read_parquet('data/processed/combined/all_assets_clean.parquet')
combined['timestamp'] = pd.to_datetime(combined['timestamp'])
combined = combined.sort_values(['asset_id', 'group', 'timestamp']).reset_index(drop=True)

# Split
load_track  = combined[combined['group'].isin(['mv_feeder', 'transformer', 'station_installation'])].copy()
solar_track = combined[combined['group'] == 'solar_park'].copy()
wind_track  = combined[combined['group'] == 'wind_park'].copy()


# ── LOAD TRACK ───────────────────────────────────────
# Lag features
for lag in [1, 6, 12, 24, 48, 168]:
    load_track[f'lag_{lag}'] = load_track.groupby(['asset_id','group'])['load_norm'].shift(lag)

# Time features
load_track['hour']        = load_track['timestamp'].dt.hour
load_track['day_of_week'] = load_track['timestamp'].dt.dayofweek
load_track['month']       = load_track['timestamp'].dt.month
load_track['is_weekend']  = (load_track['timestamp'].dt.dayofweek >= 5).astype(int)

# Cyclic encoding
load_track['hour_sin'] = np.sin(2 * np.pi * load_track['hour'] / 24)
load_track['hour_cos'] = np.cos(2 * np.pi * load_track['hour'] / 24)
load_track['dow_sin']  = np.sin(2 * np.pi * load_track['day_of_week'] / 7)
load_track['dow_cos']  = np.cos(2 * np.pi * load_track['day_of_week'] / 7)

# Rolling statistics
load_track['rolling_mean_24h'] = load_track.groupby(['asset_id','group'])['load_norm'].transform(
    lambda x: x.rolling(window=96, min_periods=1).mean()
)
load_track['rolling_std_24h'] = load_track.groupby(['asset_id','group'])['load_norm'].transform(
    lambda x: x.rolling(window=96, min_periods=1).std()
)

# Economic lags
load_track['epex_lag_1']  = load_track.groupby(['asset_id','group'])['EPEX_NL'].shift(1)
load_track['epex_lag_96'] = load_track.groupby(['asset_id','group'])['EPEX_NL'].shift(96)

# Interaction features
load_track['temp_x_hour'] = load_track['temperature_2m'] * load_track['hour']
load_track['temp_x_dow']  = load_track['temperature_2m'] * load_track['day_of_week']



# ── SOLAR TRACK ──────────────────────────────────────
# Lag features
for lag in [1, 6, 24, 168]:
    solar_track[f'lag_{lag}'] = solar_track.groupby(['asset_id','group'])['load_norm'].shift(lag)

# Time features
solar_track['hour']       = solar_track['timestamp'].dt.hour
solar_track['month']      = solar_track['timestamp'].dt.month
solar_track['is_daylight'] = ((solar_track['timestamp'].dt.hour >= 6) & 
                               (solar_track['timestamp'].dt.hour <= 18)).astype(int)

# Rolling statistics
solar_track['rolling_mean_6h'] = solar_track.groupby(['asset_id','group'])['load_norm'].transform(
    lambda x: x.rolling(window=24, min_periods=1).mean()
)
solar_track['rolling_std_6h'] = solar_track.groupby(['asset_id','group'])['load_norm'].transform(
    lambda x: x.rolling(window=24, min_periods=1).std()
)



# ── WIND TRACK ───────────────────────────────────────
# Lag features
for lag in [1, 6, 12, 24]:
    wind_track[f'lag_{lag}'] = wind_track.groupby(['asset_id','group'])['load_norm'].shift(lag)

# Time features
wind_track['hour']        = wind_track['timestamp'].dt.hour
wind_track['day_of_week'] = wind_track['timestamp'].dt.dayofweek
wind_track['month']       = wind_track['timestamp'].dt.month

# Cyclic encoding for wind direction
wind_track['wind_dir_sin'] = np.sin(2 * np.pi * wind_track['wind_direction_10m'] / 360)
wind_track['wind_dir_cos'] = np.cos(2 * np.pi * wind_track['wind_direction_10m'] / 360)

# Rolling statistics
wind_track['rolling_mean_3h'] = wind_track.groupby(['asset_id','group'])['load_norm'].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)


# Common columns every track needs
base_cols = ['timestamp', 'asset_id', 'group', 'upper_limit', 'lower_limit',
             'load', 'load_norm', 'is_imputed', 'is_outlier']

# Load track final columns
load_cols = base_cols + [
    # weather
    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
    # economic
    'EPEX_NL', 'epex_lag_1', 'epex_lag_96',
    # behavioral (profile columns)
    'E1A_AZI_A', 'E1A_AMI_A', 'E1B_AZI_A', 'E1B_AMI_A',
    'E1C_AZI_A', 'E1C_AMI_A', 'E2A_AZI_A', 'E2A_AMI_A',
    'E2B_AZI_A', 'E2B_AMI_A', 'E3A_A', 'E3B_A', 'E3C_A', 'E3D_A', 'E4A_A',
    # engineered
    'lag_1', 'lag_6', 'lag_12', 'lag_24', 'lag_48', 'lag_168',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'rolling_mean_24h', 'rolling_std_24h',
    'temp_x_hour', 'temp_x_dow'
]

# Solar track final columns
solar_cols = base_cols + [
    # irradiance
    'shortwave_radiation', 'direct_radiation',
    'diffuse_radiation', 'cloud_cover', 'temperature_2m',
    # engineered
    'lag_1', 'lag_6', 'lag_24', 'lag_168',
    'hour', 'month', 'is_daylight',
    'rolling_mean_6h', 'rolling_std_6h'
]

# Wind track final columns
wind_cols = base_cols + [
    # weather
    'wind_speed_10m', 'wind_speed_80m_forecast',
    'wind_direction_10m', 'surface_pressure', 'temperature_2m',
    # engineered
    'lag_1', 'lag_6', 'lag_12', 'lag_24',
    'hour', 'day_of_week', 'month',
    'wind_dir_sin', 'wind_dir_cos',
    'rolling_mean_3h'
]
load_track  = load_track[load_cols]
solar_track = solar_track[solar_cols]
wind_track  = wind_track[wind_cols]
# REMOVING NAN featured rows for correct training
lag_cols_load  = [f'lag_{l}' for l in [1, 6, 12, 24, 48, 168]]
lag_cols_solar = [f'lag_{l}' for l in [1, 6, 24, 168]]
lag_cols_wind  = [f'lag_{l}' for l in [1, 6, 12, 24]]

load_track  = load_track.dropna(subset=lag_cols_load)
solar_track = solar_track.dropna(subset=lag_cols_solar)
wind_track  = wind_track.dropna(subset=lag_cols_wind)

load_track.to_parquet('data/processed/features/load_track.parquet')
solar_track.to_parquet('data/processed/features/solar_track.parquet')
wind_track.to_parquet('data/processed/features/wind_track.parquet')

print(f"Load track:  {load_track.shape,load_track.columns}")
print(f"Solar track: {solar_track.shape,solar_track.columns}")
print(f"Wind track:  {wind_track.shape,wind_track.columns}")
print("\nDone. All 3 tracks saved separately.")