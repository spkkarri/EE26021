"""
app.py  (Advanced Version)
============================
Flask API serving both DQN and PPO agents.
Endpoints:
  GET /step?agent=dqn|ppo   - one RL step
  GET /reset                 - reset episode
  GET /switch?agent=dqn|ppo - switch active agent
  GET /health                - status

Run:
  python app.py
Then open smart_grid_rl.html
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

from smart_grid_env import SmartGridEnv
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent, softmax

app  = Flask(__name__)
CORS(app)

# ── Init env ──────────────────────────────────────────────────────────────────
env = SmartGridEnv(episode_length=24, mode="eval")

# ── Load DQN ─────────────────────────────────────────────────────────────────
dqn = DQNAgent(
    state_dim  = env.observation_space_size,
    action_dim = env.action_space_size,
    hidden_size = 128,
)
try:
    dqn.load("dqn_weights.npz")
except FileNotFoundError:
    print("[APP] dqn_weights.npz not found — run train.py first")

# ── Load PPO ─────────────────────────────────────────────────────────────────
ppo = PPOAgent(
    state_dim  = env.observation_space_size,
    action_dim = env.action_space_size,
)
try:
    ppo.load("ppo_weights.npz")
except FileNotFoundError:
    print("[APP] ppo_weights.npz not found — run train.py first")

# ── Episode state ─────────────────────────────────────────────────────────────
state             = env.reset()
cumulative_reward = 0.0
episode_step      = 0
active_agent      = "dqn"

ACTION_NAMES = {
    0: "Charge Battery",
    1: "Discharge Battery",
    2: "Buy from Grid",
    3: "Direct Renewables",
    4: "Idle / Curtail",
}

def get_action(s, agent_name):
    if agent_name == "ppo":
        logits = ppo.actor.forward(s[np.newaxis])[0]
        probs  = softmax(logits)
        return int(np.argmax(probs)), float(probs.max())
    else:
        q = dqn.online.forward(s[np.newaxis])[0]
        probs = np.exp(q - q.max()); probs /= probs.sum()
        return int(np.argmax(q)), float(probs.max())


@app.route("/step")
def step():
    global state, cumulative_reward, episode_step, active_agent

    agent_name = request.args.get("agent", active_agent)
    action, confidence = get_action(state, agent_name)

    next_state, reward, done, info = env.step(action)
    cumulative_reward += reward
    episode_step      += 1

    if done:
        state        = env.reset()
        episode_step = 0
    else:
        state = next_state

    return jsonify({
        "solar":             round(info["solar"],              2),
        "wind":              round(info["wind"],               2),
        "demand":            round(info["demand"],             2),
        "battery":           round(info["battery_soc"],        2),
        "battery_pct":       round(info["battery_soc"] / env.BATTERY_CAPACITY * 100, 1),
        "grid":              round(info["grid_used"],          2),
        "renewable":         round(info["renewable_supplied"], 2),
        "ren_fraction":      round(info["ren_fraction"] * 100, 1),
        "unmet":             round(info["unmet"],              2),
        "tou_price":         round(info["tou_price"],          3),
        "hour":              int(info["hour"]),
        "reward":            round(reward,                     4),
        "cumulative_reward": round(cumulative_reward,          3),
        "action":            action,
        "action_name":       ACTION_NAMES[action],
        "confidence":        round(confidence * 100,           1),
        "step":              episode_step,
        "done":              done,
        "agent":             agent_name,
    })


@app.route("/reset")
def reset_ep():
    global state, cumulative_reward, episode_step
    state             = env.reset()
    cumulative_reward = 0.0
    episode_step      = 0
    return jsonify({"status": "reset"})


@app.route("/switch")
def switch_agent():
    global active_agent
    active_agent = request.args.get("agent", "dqn")
    return jsonify({"active_agent": active_agent})


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "active_agent": active_agent,
        "dqn_epsilon":  round(dqn.epsilon, 4),
        "dataset":      "real" if env._df is not None else "synthetic",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
