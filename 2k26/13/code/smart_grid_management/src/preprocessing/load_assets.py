import yaml
import pandas as pd
import os

epex_df= pd.read_parquet(f'data/raw/EPEX.parquet')
epex_df = epex_df.drop(columns=['available_at'])
profile_df= pd.read_parquet(f'data/raw/profiles.parquet')
profile_df = profile_df.drop(columns=['available_at'])
with open("data/raw/liander2024_targets.yaml", "r") as f:
    assets=yaml.safe_load(f)
all_dfs = [] 
for i,asset in enumerate(assets) :
    
    name=asset['name']
    group_name=asset['group_name']
    #LOAD -------------------------------------------------
    load_df = pd.read_parquet(f'data/raw/load_measurements/{group_name}/{name}.parquet')
    #WEATHER ----------------------------------------------
    weather_measuremnts_df= pd.read_parquet(f'data/raw/weather_measurements/{group_name}/{name}.parquet')
    #WEATHER FORECASTS ------------------------------------
    weather_forecasts_df= pd.read_parquet(f'data/raw/weather_forecasts/{group_name}/{name}.parquet')
    weather_forecasts_df.columns = [col + '_forecast' for col in weather_forecasts_df.columns]
    #MERGING ----------------------------------------------
    
    df2=pd.merge(load_df,weather_measuremnts_df,left_on='timestamp',right_index=True,how='left')

    df2=pd.merge(df2,weather_forecasts_df,left_on='timestamp',right_index=True,how='left')
    
    df2=pd.merge(df2,epex_df,on='timestamp',how='left')
    
    df2=pd.merge(df2,profile_df,on='timestamp',how='left')
    #ADDING YAML FEATURES ---------------------------------
    df2['asset_id'] = name
    df2['group'] = group_name
    df2['upper_limit'] = asset['upper_limit']
    df2['lower_limit'] = asset['lower_limit']
    #SAVIING 
    os.makedirs(f'data/processed/{group_name}', exist_ok=True)
    df2.to_parquet(f'data/processed/{group_name}/{name}.parquet')

    print(f"[{i+1}/55] Saved: {group_name}/{name} — shape: {df2.shape}")
    all_dfs.append(df2)

#SAVING COMBINED SETS -------------------------------------
combined = pd.concat(all_dfs, ignore_index=True)
os.makedirs('data/processed/combined', exist_ok=True)
combined.to_parquet('data/processed/combined/all_assets.parquet')
print(f"Combined saved: {combined.shape}")

print(combined.columns)
print(combined.shape)

# 1. Check missing values
print("Missing values:")
print(combined.isnull().sum())

# 2. Check timestamp is correct range
print("\nDate range:")
print(combined['timestamp'].min())
print(combined['timestamp'].max())

# 3. Check no duplicate timestamps
print("\nDuplicate timestamps:")
print(combined.groupby(['asset_id','group']).size())

# 4. Check load stats make sense
print("\nLoad stats:")
print(combined['load'].describe())

