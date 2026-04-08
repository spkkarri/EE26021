"""
ppo_agent.py
=============
Proximal Policy Optimisation (PPO-Clip) in pure NumPy.
Actor-Critic architecture with separate policy and value networks.

PPO is more stable than DQN for continuous control and is mentioned
in the paper as a future direction (Section II.C).

Architecture:
  Actor  (policy): FC(9→128) → ReLU → FC(128→64) → ReLU → FC(64→5) → Softmax
  Critic (value):  FC(9→128) → ReLU → FC(128→64) → ReLU → FC(64→1)

Hyperparameters:
  clip_eps=0.2, lr_actor=0.003, lr_critic=0.005, gamma=0.95,
  gae_lambda=0.95, epochs=4, batch_size=64
"""

import numpy as np


def relu(x):      return np.maximum(0.0, x)
def relu_grad(x): return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── Simple MLP helper ─────────────────────────────────────────────────────────
class SmallMLP:
    def __init__(self, dims: list[int], seed: int):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fin, fout in zip(dims[:-1], dims[1:]):
            std = np.sqrt(2.0 / fin)
            self.W.append(rng.normal(0, std, (fin, fout)).astype(np.float32))
            self.b.append(np.zeros(fout, dtype=np.float32))

    def forward(self, x, store=False):
        self._cache = []
        out = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = out @ W + b
            is_last = (i == len(self.W) - 1)
            a = z if is_last else relu(z)
            if store: self._cache.append((out, z, a, is_last))
            out = a
        return out

    def backward(self, grad, lr):
        delta = grad
        for i in reversed(range(len(self.W))):
            inp, z, a, is_last = self._cache[i]
            if not is_last: delta = delta * relu_grad(z)
            dW = inp.T @ delta / delta.shape[0]
            db = delta.mean(axis=0)
            gnorm = np.sqrt((dW**2).sum() + (db**2).sum())
            if gnorm > 0.5: dW /= gnorm; db /= gnorm
            self.W[i] -= lr * dW
            self.b[i]  -= lr * db
            if i > 0: delta = delta @ self.W[i].T

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def get_params(self):
        return [(w.copy(), b.copy()) for w, b in zip(self.W, self.b)]

    def set_params(self, params):
        for i, (w, b) in enumerate(params): self.W[i]=w.copy(); self.b[i]=b.copy()


# ── PPO Agent ─────────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(
        self,
        state_dim:   int   = 9,
        action_dim:  int   = 5,
        lr_actor:    float = 0.003,
        lr_critic:   float = 0.005,
        gamma:       float = 0.95,
        gae_lambda:  float = 0.95,
        clip_eps:    float = 0.2,
        ppo_epochs:  int   = 4,
        batch_size:  int   = 64,
        seed:        int   = 0,
    ):
        self.action_dim  = action_dim
        self.lr_actor    = lr_actor
        self.lr_critic   = lr_critic
        self.gamma       = gamma
        self.lam         = gae_lambda
        self.clip_eps    = clip_eps
        self.ppo_epochs  = ppo_epochs
        self.batch_size  = batch_size

        self.actor  = SmallMLP([state_dim, 128, 64, action_dim], seed)
        self.critic = SmallMLP([state_dim, 128, 64, 1],          seed+10)

        # Trajectory buffer
        self._reset_buffer()
        self.losses = []

    def _reset_buffer(self):
        self._states   = []
        self._actions  = []
        self._rewards  = []
        self._values   = []
        self._log_probs= []
        self._dones    = []

    def select_action(self, state: np.ndarray):
        logits = self.actor.forward(state[np.newaxis])[0]
        probs  = softmax(logits)
        action = np.random.choice(self.action_dim, p=probs)
        log_p  = np.log(probs[action] + 1e-8)
        value  = float(self.critic.forward(state[np.newaxis])[0, 0])
        return action, log_p, value

    def store(self, state, action, reward, log_prob, value, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._dones.append(done)

    def update(self, last_value: float = 0.0):
        """Call at end of episode. last_value=0 if done."""
        S  = np.array(self._states,    dtype=np.float32)
        A  = np.array(self._actions,   dtype=np.int32)
        R  = np.array(self._rewards,   dtype=np.float32)
        LP = np.array(self._log_probs, dtype=np.float32)
        V  = np.array(self._values,    dtype=np.float32)
        D  = np.array(self._dones,     dtype=np.float32)
        T  = len(R)

        # GAE advantages
        adv   = np.zeros(T, dtype=np.float32)
        gae   = 0.0
        V_ext = np.append(V, last_value)
        for t in reversed(range(T)):
            delta = R[t] + self.gamma * V_ext[t+1] * (1-D[t]) - V_ext[t]
            gae   = delta + self.gamma * self.lam * (1-D[t]) * gae
            adv[t] = gae

        returns = adv + V
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                mb = idx[start:start+self.batch_size]
                if len(mb) < 2: continue

                s_mb = S[mb]; a_mb = A[mb]; adv_mb = adv[mb]
                ret_mb = returns[mb]; old_lp_mb = LP[mb]

                # Actor update (PPO-clip)
                logits = self.actor.forward(s_mb, store=True)
                probs  = softmax(logits)
                new_lp = np.log(probs[np.arange(len(mb)), a_mb] + 1e-8)
                ratio  = np.exp(new_lp - old_lp_mb)
                clip_r = np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)
                actor_loss = -np.minimum(ratio*adv_mb, clip_r*adv_mb).mean()

                # Policy gradient
                dlog = np.zeros_like(probs)
                for i in range(len(mb)):
                    r_i = ratio[i]
                    c_i = clip_r[i]
                    grad_sign = -adv_mb[i] * (1.0 if abs(r_i-c_i)<1e-8 else 0.0)
                    dlog[i, a_mb[i]] = grad_sign / (probs[i, a_mb[i]] + 1e-8)
                dlogits = probs * (dlog - (dlog * probs).sum(axis=1, keepdims=True))
                self.actor.backward(dlogits / len(mb), self.lr_actor)

                # Critic update (MSE)
                vals    = self.critic.forward(s_mb, store=True)[:, 0]
                c_loss  = ((vals - ret_mb) ** 2).mean()
                dc      = 2.0 * (vals - ret_mb)[:, np.newaxis] / len(mb)
                self.critic.backward(dc, self.lr_critic)

                total_loss += actor_loss + c_loss

        self.losses.append(total_loss / self.ppo_epochs)
        self._reset_buffer()
        return total_loss / self.ppo_epochs

    def save(self, path: str):
        data = {}
        for i, (w, b) in enumerate(zip(self.actor.W,  self.actor.b)):
            data[f"aW{i}"]=w; data[f"ab{i}"]=b
        for i, (w, b) in enumerate(zip(self.critic.W, self.critic.b)):
            data[f"cW{i}"]=w; data[f"cb{i}"]=b
        np.savez(path, **data)
        print(f"[PPO] Saved -> {path}.npz")

    def load(self, path: str):
        data = np.load(path)
        for i in range(len(self.actor.W)):
            self.actor.W[i]  = data[f"aW{i}"]; self.actor.b[i]  = data[f"ab{i}"]
        for i in range(len(self.critic.W)):
            self.critic.W[i] = data[f"cW{i}"]; self.critic.b[i] = data[f"cb{i}"]
        print(f"[PPO] Loaded <- {path}")
