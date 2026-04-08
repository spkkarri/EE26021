"""
smart_grid_env.py  (Advanced Version)
======================================
Upgraded environment using REAL dataset (smart_grid_rl_dataset.xlsx).

Improvements:
  - Real 8760-hour dataset (solar, wind, demand, battery, weather)
  - 9-dimensional state space (adds temperature, sunlight, hour_cos)
  - Time-of-Use (ToU) electricity pricing in reward
  - Realistic capacity (500 MWh battery, 900 MW demand)
  - Train/eval split on real data days

State (9 features, normalised [0,1]):
  [solar, wind, grid_avail, battery_soc, demand,
   hour_sin, hour_cos, temperature, sunlight]

Actions (Discrete 5):
  0-charge battery  1-discharge  2-buy grid
  3-direct renewables  4-idle

Reward:
  + renewable_fraction * W_RENEWABLE
  - grid_cost * TOU_PRICE[hour]
  - unmet_penalty
  - battery_abuse
  + efficiency_bonus (ren > 80%)
"""

import numpy as np
import pandas as pd
import os

TOU_PRICE = np.array([
    0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
    0.15, 0.20, 0.28, 0.28, 0.22, 0.18,
    0.15, 0.15, 0.18, 0.22, 0.28, 0.32,
    0.30, 0.28, 0.24, 0.18, 0.12, 0.08,
], dtype=np.float32)


class SmartGridEnv:
    BATTERY_CAPACITY   = 500.0
    MAX_CHARGE_RATE    = 100.0
    MAX_DISCHARGE_RATE = 100.0
    MAX_DEMAND         = 900.0
    MAX_SOLAR          = 500.0
    MAX_WIND           = 360.0
    MAX_GRID           = 200.0
    MAX_TEMP           = 50.0

    W_RENEWABLE       = 1.2
    W_UNMET           = -3.0
    W_ABUSE           = -0.5
    W_EFFICIENCY_BONUS = 0.3

    def __init__(self, episode_length=24, seed=None, dataset_path=None, mode="train"):
        self.episode_length = episode_length
        self.rng  = np.random.default_rng(seed)
        self.mode = mode
        self.observation_space_size = 9
        self.action_space_size      = 5

        paths = [dataset_path, "smart_grid_rl_dataset.xlsx",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_grid_rl_dataset.xlsx")]
        self._df = None
        for p in paths:
            if p and os.path.exists(p):
                self._df = self._load_dataset(p)
                print(f"[ENV] Real dataset loaded: {p}  ({len(self._df)} hours)")
                break
        if self._df is None:
            print("[ENV] Dataset not found — using synthetic profiles")

        total_days = (len(self._df) // 24) if self._df is not None else 0
        split = int(total_days * 0.8)
        self._train_days = list(range(0, split))
        self._eval_days  = list(range(split, total_days))
        self.state = None
        self._step = 0

    def reset(self):
        self._step        = 0
        self._battery_soc = self.rng.uniform(0.3, 0.7) * self.BATTERY_CAPACITY

        if self._df is not None:
            pool  = self._train_days if self.mode == "train" else self._eval_days
            if not pool: pool = list(range(len(self._df) // 24))
            day   = int(self.rng.choice(pool))
            chunk = self._df.iloc[day*24 : day*24+self.episode_length].reset_index(drop=True)
            self._solar_profile    = chunk["Solar_Generation_MW"].values.astype(np.float32)
            self._wind_profile     = chunk["Wind_Generation_MW"].values.astype(np.float32)
            self._demand_profile   = chunk["Total_Demand_MW"].values.astype(np.float32)
            self._temp_profile     = chunk["Temperature_C"].values.astype(np.float32)
            self._sunlight_profile = chunk["Sunlight_Intensity"].values.astype(np.float32)
            self._hour_profile     = chunk["Hour"].values.astype(np.int32)
        else:
            self._generate_synthetic()

        self.state = self._build_state()
        return self.state.copy()

    def step(self, action):
        assert self.state is not None, "Call reset() first."
        solar  = float(self._solar_profile[self._step])
        wind   = float(self._wind_profile[self._step])
        demand = float(self._demand_profile[self._step])
        hour   = int(self._hour_profile[self._step])
        renew  = solar + wind

        grid_used = unmet = battery_abu = renewable_supplied = 0.0

        if action == 0:
            charge = min(self.MAX_CHARGE_RATE, renew, self.BATTERY_CAPACITY - self._battery_soc)
            self._battery_soc += charge
            leftover = renew - charge
            renewable_supplied = min(leftover, demand)
            unmet = max(0.0, demand - renewable_supplied)
            if unmet > 0: grid_used = min(unmet, self.MAX_GRID); unmet = max(0.0, unmet - grid_used)

        elif action == 1:
            discharge = min(self.MAX_DISCHARGE_RATE, self._battery_soc, demand)
            self._battery_soc -= discharge
            renewable_supplied = min(renew, demand)
            unmet = max(0.0, demand - discharge - renew)
            if unmet > 0: grid_used = min(unmet, self.MAX_GRID); unmet = max(0.0, unmet - grid_used)

        elif action == 2:
            grid_used = min(demand, self.MAX_GRID)
            unmet     = max(0.0, demand - grid_used)

        elif action == 3:
            renewable_supplied = min(renew, demand)
            unmet = max(0.0, demand - renewable_supplied)
            if unmet > 0: grid_used = min(unmet, self.MAX_GRID); unmet = max(0.0, unmet - grid_used)

        else:
            unmet = demand

        if self._battery_soc < 0:
            battery_abu = abs(self._battery_soc); self._battery_soc = 0.0
        elif self._battery_soc > self.BATTERY_CAPACITY:
            battery_abu = self._battery_soc - self.BATTERY_CAPACITY
            self._battery_soc = self.BATTERY_CAPACITY

        ren_fraction = renewable_supplied / (demand + 1e-8)
        tou          = float(TOU_PRICE[hour % 24])
        eff_bonus    = self.W_EFFICIENCY_BONUS if ren_fraction > 0.8 else 0.0
        reward = (
            self.W_RENEWABLE * ren_fraction
            - tou * (grid_used / (self.MAX_GRID + 1e-8))
            + self.W_UNMET   * (unmet       / (demand + 1e-8))
            + self.W_ABUSE   * (battery_abu / self.BATTERY_CAPACITY)
            + eff_bonus
        )

        self._step += 1
        done = self._step >= self.episode_length
        if not done: self.state = self._build_state()

        info = dict(solar=solar, wind=wind, demand=demand, grid_used=grid_used,
                    renewable_supplied=renewable_supplied, unmet=unmet,
                    battery_soc=self._battery_soc, ren_fraction=ren_fraction,
                    tou_price=tou, hour=hour)
        return self.state.copy(), reward, done, info

    def _build_state(self):
        h    = self._step
        hour = int(self._hour_profile[h])
        return np.array([
            self._solar_profile[h]    / self.MAX_SOLAR,
            self._wind_profile[h]     / self.MAX_WIND,
            1.0,
            self._battery_soc         / self.BATTERY_CAPACITY,
            self._demand_profile[h]   / self.MAX_DEMAND,
            np.sin(2*np.pi*hour/24),
            np.cos(2*np.pi*hour/24),
            self._temp_profile[h]     / self.MAX_TEMP,
            float(self._sunlight_profile[h]),
        ], dtype=np.float32)

    def _load_dataset(self, path):
        df = pd.read_excel(path, sheet_name="Hourly_Dataset", header=1)
        df = df[["Hour","Solar_Generation_MW","Wind_Generation_MW",
                 "Total_Demand_MW","Battery_Level_MWh",
                 "Temperature_C","Sunlight_Intensity","Wind_Speed_ms"]].copy()
        df = df.dropna().reset_index(drop=True)
        df["Temperature_C"]      = df["Temperature_C"].clip(0, self.MAX_TEMP)
        df["Sunlight_Intensity"] = df["Sunlight_Intensity"].clip(0, 1)
        df["Wind_Speed_ms"]      = df["Wind_Speed_ms"].clip(0, 12)
        return df

    def _generate_synthetic(self):
        hours = np.arange(self.episode_length)
        peak  = self.rng.uniform(0.7,1.0)*self.MAX_SOLAR
        self._solar_profile    = np.clip(peak*np.exp(-0.5*((hours-12)/4)**2)+self.rng.normal(0,5,len(hours)),0,self.MAX_SOLAR).astype(np.float32)
        base  = self.rng.uniform(0.2,0.6)*self.MAX_WIND
        self._wind_profile     = np.clip(base+self.rng.normal(0,20,len(hours)).cumsum()*0.3,0,self.MAX_WIND).astype(np.float32)
        mor   = 0.7*np.exp(-0.5*((hours-8)/2)**2)
        eve   = 1.0*np.exp(-0.5*((hours-19)/2.5)**2)
        self._demand_profile   = np.clip(self.MAX_DEMAND*(mor+eve)/1.5+self.rng.normal(0,10,len(hours)),50,self.MAX_DEMAND).astype(np.float32)
        self._temp_profile     = (20+10*np.sin(2*np.pi*hours/24)+self.rng.normal(0,2,len(hours))).astype(np.float32)
        self._sunlight_profile = np.clip(np.exp(-0.5*((hours-12)/5)**2),0,1).astype(np.float32)
        self._hour_profile     = (hours%24).astype(np.int32)
