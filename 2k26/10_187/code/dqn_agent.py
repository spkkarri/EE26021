"""
dqn_agent.py  (Advanced Version)
==================================
Dueling Double DQN with Prioritized Experience Replay (PER).

Upgrades over basic DQN:
  - Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
  - Double DQN: online net selects action, target net scores it
  - Prioritized Experience Replay (PER): important transitions sampled more
  - Larger network (9→128→128→5) for 9-dim state
  - Noisy epsilon schedule with warmup

Paper hyperparameters (Table I) preserved:
  alpha=0.01, gamma=0.95, epsilon decay, batch=64, buffer=10000
"""

import numpy as np
from collections import deque
import random


# ── Activations ───────────────────────────────────────────────────────────────
def relu(x):      return np.maximum(0.0, x)
def relu_grad(x): return (x > 0).astype(np.float32)


# ── Dueling MLP ───────────────────────────────────────────────────────────────
class DuelingMLP:
    """
    Dueling network:
      shared_body: input → 128 → 128
      value_head:  128 → 1
      adv_head:    128 → action_dim
      Q = V + (A - mean(A))
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.action_dim = action_dim

        def he(fan_in, fan_out):
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)

        # Shared body (2 hidden layers)
        self.W1 = he(state_dim, hidden);  self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = he(hidden, hidden);     self.b2 = np.zeros(hidden, dtype=np.float32)
        # Value head
        self.Wv = he(hidden, 1);          self.bv = np.zeros(1,      dtype=np.float32)
        # Advantage head
        self.Wa = he(hidden, action_dim); self.ba = np.zeros(action_dim, dtype=np.float32)

    def forward(self, x: np.ndarray, store_cache: bool = False):
        # body
        z1 = x  @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = relu(z2)
        # heads
        V  = a2 @ self.Wv + self.bv                        # (B,1)
        A  = a2 @ self.Wa + self.ba                        # (B, act)
        Q  = V + (A - A.mean(axis=1, keepdims=True))       # dueling combine

        if store_cache:
            self._cache = (x, z1, a1, z2, a2, V, A, Q)
        return Q

    def backward(self, dQ: np.ndarray, lr: float):
        x, z1, a1, z2, a2, V, A, Q = self._cache
        B = dQ.shape[0]

        # dQ → dV, dA
        dV = dQ.sum(axis=1, keepdims=True)                  # (B,1)
        dA = dQ - dQ.mean(axis=1, keepdims=True)            # (B,act)

        # value head
        dWv = a2.T @ dV / B;  dbv = dV.mean(axis=0)
        da2_v = dV @ self.Wv.T

        # advantage head
        dWa = a2.T @ dA / B;  dba = dA.mean(axis=0)
        da2_a = dA @ self.Wa.T

        da2 = da2_v + da2_a

        # body layer 2
        da2_act = da2 * relu_grad(z2)
        dW2 = a1.T @ da2_act / B;  db2 = da2_act.mean(axis=0)
        da1 = da2_act @ self.W2.T

        # body layer 1
        da1_act = da1 * relu_grad(z1)
        dW1 = x.T @ da1_act / B;   db1 = da1_act.mean(axis=0)

        # gradient clip (norm=1)
        for dW, db, W, b, attr_w, attr_b in [
            (dW1, db1, self.W1, self.b1, 'W1', 'b1'),
            (dW2, db2, self.W2, self.b2, 'W2', 'b2'),
            (dWv, dbv, self.Wv, self.bv, 'Wv', 'bv'),
            (dWa, dba, self.Wa, self.ba, 'Wa', 'ba'),
        ]:
            gnorm = np.sqrt((dW**2).sum() + (db**2).sum())
            if gnorm > 1.0: dW /= gnorm; db /= gnorm
            setattr(self, attr_w, W - lr * dW)
            setattr(self, attr_b, b - lr * db)

    def copy_weights_from(self, other: "DuelingMLP"):
        for attr in ['W1','b1','W2','b2','Wv','bv','Wa','ba']:
            setattr(self, attr, getattr(other, attr).copy())

    def save_weights(self) -> dict:
        return {a: getattr(self, a).copy() for a in ['W1','b1','W2','b2','Wv','bv','Wa','ba']}

    def load_weights(self, d: dict):
        for a, v in d.items(): setattr(self, a, v.copy())


# ── Prioritized Replay Buffer (PER) ──────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Proportional prioritization: P(i) = p_i^alpha / sum(p_j^alpha)
    Importance-sampling weights correct for bias.
    """
    def __init__(self, capacity: int = 10_000, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity  = capacity
        self.alpha     = alpha
        self.beta      = beta_start
        self.beta_inc  = (1.0 - beta_start) / 50_000   # anneal to 1 over training
        self.buf       = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos       = 0
        self.max_prio  = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        N = len(self.buf)
        p = self.priorities[:N] ** self.alpha
        p = p / p.sum()
        indices = np.random.choice(N, batch_size, replace=False, p=p)
        weights = (N * p[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_inc)

        s, a, r, ns, d = zip(*[self.buf[i] for i in indices])
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int32),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32),
                indices,
                weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[i] = abs(e) + 1e-6
        self.max_prio = float(self.priorities[:len(self.buf)].max())

    def __len__(self): return len(self.buf)


# ── Dueling Double DQN Agent ──────────────────────────────────────────────────
class DQNAgent:
    """
    Dueling Double DQN + Prioritized Experience Replay.
    Hyperparameters match paper Table I; architecture upgraded.
    """
    def __init__(
        self,
        state_dim:          int   = 9,
        action_dim:         int   = 5,
        lr:                 float = 0.01,
        gamma:              float = 0.95,
        epsilon_start:      float = 1.0,
        epsilon_min:        float = 0.05,
        epsilon_decay:      float = 0.997,
        batch_size:         int   = 64,
        buffer_size:        int   = 10_000,
        target_update_freq: int   = 100,
        hidden_size:        int   = 128,
        seed:               int   = 42,
    ):
        random.seed(seed); np.random.seed(seed)
        self.action_dim         = action_dim
        self.lr                 = lr
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self.online = DuelingMLP(state_dim, action_dim, hidden_size, seed)
        self.target = DuelingMLP(state_dim, action_dim, hidden_size, seed+1)
        self.target.copy_weights_from(self.online)

        self.buffer        = PrioritizedReplayBuffer(buffer_size)
        self._update_count = 0
        self.losses        = []

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.online.forward(state[np.newaxis])[0]
        return int(np.argmax(q))

    def store(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d, indices, weights = self.buffer.sample(self.batch_size)

        # Double DQN: online net picks action, target net scores it
        q_online_next = self.online.forward(ns)
        best_actions  = np.argmax(q_online_next, axis=1)
        q_target_next = self.target.forward(ns)
        q_pred        = self.online.forward(s, store_cache=True)
        q_target      = q_pred.copy()

        td_errors = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            target_val = r[i] if d[i] else r[i] + self.gamma * q_target_next[i, best_actions[i]]
            td_errors[i] = target_val - q_pred[i, a[i]]
            q_target[i, a[i]] = target_val

        self.buffer.update_priorities(indices, td_errors)

        # PER-weighted MSE gradient
        w = weights[:, np.newaxis]
        loss_grad = 2.0 * w * (q_pred - q_target) / self.batch_size
        loss = float(np.mean(weights * (q_pred - q_target).mean(axis=1) ** 2))

        self.online.backward(loss_grad, self.lr)

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target.copy_weights_from(self.online)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.losses.append(loss)
        return loss

    def save(self, path: str):
        """path without .npz extension"""
        data = self.online.save_weights()
        data["epsilon"] = np.array([self.epsilon])
        np.savez(path, **data)
        print(f"[DQN] Saved -> {path}.npz")

    def load(self, path: str):
        """path with .npz extension"""
        data = np.load(path)
        w = {k: data[k] for k in ['W1','b1','W2','b2','Wv','bv','Wa','ba']}
        self.online.load_weights(w)
        self.target.copy_weights_from(self.online)
        self.epsilon = float(data["epsilon"][0])
        print(f"[DQN] Loaded <- {path}")
