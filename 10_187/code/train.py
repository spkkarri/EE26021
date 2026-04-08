"""
train.py  (Advanced Version)
==============================
Trains THREE agents and compares them:
  1. Dueling Double DQN + PER  (dqn_weights.npz)
  2. PPO Actor-Critic           (ppo_weights.npz)
  3. Rule-Based baseline        (no training needed)

Uses real dataset (smart_grid_rl_dataset.xlsx) via SmartGridEnv.

Outputs:
  dqn_weights.npz   - trained DQN weights
  ppo_weights.npz   - trained PPO weights
  training.png      - reward curves for both agents
  metrics.txt       - full evaluation matching paper Table II
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smart_grid_env import SmartGridEnv
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent

EPISODES   = 1000
EVAL_EPS   = 100
PRINT_FREQ = 100

# ── Environments ──────────────────────────────────────────────────────────────
train_env = SmartGridEnv(episode_length=24, seed=42,  mode="train")
eval_env  = SmartGridEnv(episode_length=24, seed=999, mode="eval")

# ── Agents ────────────────────────────────────────────────────────────────────
dqn = DQNAgent(state_dim=train_env.observation_space_size,
               action_dim=train_env.action_space_size,
               lr=0.01, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05,
               epsilon_decay=0.997, batch_size=64, buffer_size=10_000,
               target_update_freq=100, hidden_size=128, seed=42)

ppo = PPOAgent(state_dim=train_env.observation_space_size,
               action_dim=train_env.action_space_size,
               lr_actor=0.003, lr_critic=0.005, gamma=0.95,
               gae_lambda=0.95, clip_eps=0.2, ppo_epochs=4,
               batch_size=64, seed=0)


# ════════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  TRAINING Dueling Double DQN + PER")
print("=" * 60)

dqn_rewards = []
for ep in range(1, EPISODES + 1):
    state = train_env.reset()
    done  = False
    total = 0.0
    while not done:
        action              = dqn.select_action(state)
        ns, reward, done, _ = train_env.step(action)
        dqn.store(state, action, reward, ns, done)
        dqn.update()
        state = ns
        total += reward
    dqn_rewards.append(total)
    if ep % PRINT_FREQ == 0 or ep == 1:
        avg = np.mean(dqn_rewards[-PRINT_FREQ:])
        print(f"  DQN  ep {ep:>4}/{EPISODES} | reward {total:>8.2f} | avg {avg:>8.2f} | eps {dqn.epsilon:.3f}")

dqn.save("dqn_weights")
print("\nDQN training complete.\n")


print("=" * 60)
print("  TRAINING PPO Actor-Critic")
print("=" * 60)

ppo_rewards = []
for ep in range(1, EPISODES + 1):
    state = train_env.reset()
    done  = False
    total = 0.0
    last_val = 0.0
    while not done:
        action, log_p, value    = ppo.select_action(state)
        ns, reward, done, _     = train_env.step(action)
        ppo.store(state, action, reward, log_p, value, done)
        state = ns
        total += reward
        if done:
            last_val = 0.0
            ppo.update(last_val)
    ppo_rewards.append(total)
    if ep % PRINT_FREQ == 0 or ep == 1:
        avg = np.mean(ppo_rewards[-PRINT_FREQ:])
        print(f"  PPO  ep {ep:>4}/{EPISODES} | reward {total:>8.2f} | avg {avg:>8.2f}")

ppo.save("ppo_weights")
print("\nPPO training complete.\n")


# ════════════════════════════════════════════════════════════════════════════════
#  RULE-BASED BASELINE
# ════════════════════════════════════════════════════════════════════════════════
def rule_based_action(state, hour):
    """
    Simple heuristic:
      - Dawn/dusk & battery high → discharge
      - Solar peak (10-14h) → charge battery
      - Peak price hours (17-21h) & battery available → discharge
      - Otherwise → direct renewables
    """
    battery_soc  = state[3]
    solar        = state[0]
    if 10 <= hour <= 14 and solar > 0.3 and battery_soc < 0.9:
        return 0   # charge
    if (17 <= hour <= 21) and battery_soc > 0.3:
        return 1   # discharge during peak price
    if solar > 0.2 or state[1] > 0.3:
        return 3   # direct renewables
    if battery_soc > 0.2:
        return 1   # discharge
    return 2       # grid


# ════════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════════
def evaluate_agent(name, get_action_fn, n_eps=EVAL_EPS):
    ren_list, grid_list, unmet_list, reward_list = [], [], [], []
    eval_env.mode = "eval"
    for _ in range(n_eps):
        state  = eval_env.reset()
        done   = False
        ep_ren = ep_grid = ep_unmet = ep_dem = ep_r = 0.0
        while not done:
            action             = get_action_fn(state)
            state, r, done, info = eval_env.step(action)
            ep_ren   += info["renewable_supplied"]
            ep_grid  += info["grid_used"]
            ep_unmet += info["unmet"]
            ep_dem   += info["demand"]
            ep_r     += r
        ren_list.append(ep_ren  / (ep_dem + 1e-8))
        grid_list.append(ep_grid / (ep_dem + 1e-8))
        unmet_list.append(ep_unmet / (ep_dem + 1e-8))
        reward_list.append(ep_r)

    eff  = np.mean(ren_list)  * 100
    stab = 1.0 - np.mean(unmet_list)
    cost = (1.0 - np.mean(grid_list)) * 100
    mae  = np.mean(unmet_list) * 100
    mse  = float(np.mean(np.array(unmet_list)**2))
    avg_r= np.mean(reward_list)
    print(f"  {name:<22} | Eff:{eff:5.1f}%  Stab:{stab:.3f}  Cost:{cost:5.1f}%  AvgR:{avg_r:7.2f}")
    return eff, stab, cost, mae, mse, avg_r


print("=" * 60)
print("  EVALUATION")
print("=" * 60)

# DQN greedy
def dqn_action(state):
    q = dqn.online.forward(state[np.newaxis])[0]
    return int(np.argmax(q))

# PPO greedy
def ppo_action(state):
    from ppo_agent import softmax
    logits = ppo.actor.forward(state[np.newaxis])[0]
    return int(np.argmax(softmax(logits)))

# Rule-based
_eval_step_counter = [0]
def rb_action(state):
    hour = int(state[5] * 12 + 12) % 24  # approximate from sin encoding
    return rule_based_action(state, hour)

rb_eff,  rb_stab,  rb_cost,  rb_mae,  rb_mse,  rb_r  = evaluate_agent("Rule-Based",      rb_action)
dqn_eff, dqn_stab, dqn_cost, dqn_mae, dqn_mse, dqn_r = evaluate_agent("Dueling DDQN+PER",dqn_action)
ppo_eff, ppo_stab, ppo_cost, ppo_mae, ppo_mse, ppo_r = evaluate_agent("PPO Actor-Critic", ppo_action)


# ════════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════════
W = 50
def smooth(r): return np.convolve(r, np.ones(W)/W, mode='valid')
cum_dqn = np.cumsum(dqn_rewards)
cum_ppo = np.cumsum(ppo_rewards)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Advanced Smart Grid RL — Training & Evaluation", fontsize=14, fontweight='bold')

# Per-episode rewards
ax = axes[0, 0]
ax.plot(dqn_rewards, alpha=0.2, color='steelblue')
ax.plot(range(W-1, EPISODES), smooth(dqn_rewards), color='steelblue', lw=2, label='Dueling DDQN+PER')
ax.plot(ppo_rewards, alpha=0.2, color='darkorange')
ax.plot(range(W-1, EPISODES), smooth(ppo_rewards), color='darkorange', lw=2, label='PPO Actor-Critic')
ax.set_title("Per-Episode Reward"); ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
ax.legend(); ax.grid(alpha=0.3)

# Cumulative rewards (matches paper Fig. 3)
ax = axes[0, 1]
ax.plot(cum_dqn, color='steelblue', lw=1.5, label='Dueling DDQN+PER')
ax.plot(cum_ppo, color='darkorange', lw=1.5, label='PPO Actor-Critic')
ax.set_title("Cumulative Reward (Paper Fig. 3 style)"); ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
ax.legend(); ax.grid(alpha=0.3)

# Method comparison bar chart
ax = axes[1, 0]
methods  = ['Rule-Based', 'Dueling DDQN+PER', 'PPO']
eff_vals = [rb_eff, dqn_eff, ppo_eff]
colors   = ['#ef4444','#3b82f6','#f97316']
bars = ax.bar(methods, eff_vals, color=colors, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, eff_vals): ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_title("Energy Efficiency Comparison"); ax.set_ylabel("Energy Efficiency (%)"); ax.set_ylim(0, 110); ax.grid(axis='y', alpha=0.3)

# Stability comparison
ax = axes[1, 1]
stab_vals = [rb_stab, dqn_stab, ppo_stab]
bars = ax.bar(methods, stab_vals, color=colors, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, stab_vals): ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_title("Grid Stability Index"); ax.set_ylabel("Stability (0-1)"); ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("training.png", dpi=150)
plt.close()
print("\nPlot saved -> training.png")


# ════════════════════════════════════════════════════════════════════════════════
#  METRICS REPORT
# ════════════════════════════════════════════════════════════════════════════════
metrics_text = f"""
================================================================
  ADVANCED SMART GRID RL -- FULL EVALUATION REPORT
  Paper: Energy Optimization for Smart Grids (INCET 2025)
  Implementation: Dueling DDQN+PER vs PPO vs Rule-Based
================================================================

Dataset: smart_grid_rl_dataset.xlsx (8760 real hours, 2023)
State:   9-dim (solar, wind, grid, soc, demand, sin/cos, temp, sunlight)
Reward:  Renewable fraction + Time-of-Use pricing penalty

---- Table II: Method Comparison (paper format) ----

  Method               Energy Eff   Grid Stability  Cost Reduction
  Rule-Based           {rb_eff:5.1f}%        {rb_stab:.3f}           {rb_cost:5.1f}%
  Dueling DDQN+PER     {dqn_eff:5.1f}%        {dqn_stab:.3f}           {dqn_cost:5.1f}%
  PPO Actor-Critic     {ppo_eff:5.1f}%        {ppo_stab:.3f}           {ppo_cost:5.1f}%

  Paper baseline (original DQN): 92% / 0.89 / 35%

---- Accuracy Metrics (Section IV.D format) ----

  DQN MAE : {dqn_mae:.2f}%   (paper: 4.2%)
  DQN MSE : {dqn_mse:.4f}   (paper: 0.015)
  PPO MAE : {ppo_mae:.2f}%
  PPO MSE : {ppo_mse:.4f}

---- Average Reward per Episode ----

  Rule-Based     : {rb_r:.3f}
  DDQN+PER       : {dqn_r:.3f}
  PPO            : {ppo_r:.3f}

================================================================
Upgrades over basic paper implementation:
  [x] Dueling network architecture
  [x] Double DQN (reduces overestimation)
  [x] Prioritized Experience Replay (PER)
  [x] PPO Actor-Critic (on-policy alternative)
  [x] Real dataset (8760h) instead of synthetic only
  [x] 9-dim state with weather features
  [x] Time-of-Use electricity pricing in reward
  [x] Train/eval split on real data
================================================================
"""

print(metrics_text)
with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write(metrics_text)
print("Metrics saved -> metrics.txt")
print("\nAll training complete!")
