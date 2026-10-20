"""Shared helpers for the SupplyRun offline-RL pipeline.

* ``MDP``              - loads behaviour_policy.npz (the complete tabular MDP + behaviour policy b)
* ``eps_greedy_probs`` - pi_eps(a|s) built from a Q table (works for (S,A) or (n,S,A))
* ``exact_value``      - exact finite-horizon (200-step truncated) value of any stationary policy
* ``greedy_rollout``   - deterministic greedy rollout from s0 (the MDP is deterministic)

Conventions
-----------
* ``done[s,a]`` marks a *terminating* transition; ``next_idx`` is -1 there, so we never follow it
  (we use ``next_safe`` which maps those entries to 0 and always multiply by ``~done``).
* Truncation (200 steps) is NOT termination: a truncated trajectory's last transition has
  done=False and we bootstrap from the next state.  In the exact-value DP the horizon-end value is 0
  because the quantity we estimate by IS is the return of the *truncated* trajectory (<=200 rewards).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "behaviour_policy.npz"
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")


class MDP:
    def __init__(self, path=ARTIFACT):
        d = np.load(path)
        self.states = d["states"]                                  # (S, 4): row, col, fuel, cargo
        self.next_idx = d["next_idx"].astype(np.int64)             # (S, A), -1 on terminating transition
        self.reward = d["reward"].astype(np.float64)               # (S, A)
        self.b = d["b_probs"].astype(np.float64)                   # (S, A) behaviour policy
        self.start = int(d["start_idx"])
        self.gamma = float(d["gamma"])
        self.T = int(d["trunc_limit"])
        self.S, self.A = self.reward.shape
        # terminating if flagged done OR next_idx == -1  (belt and braces)
        self.done = d["done"].astype(bool) | (self.next_idx < 0)
        self.next_safe = np.where(self.done, 0, self.next_idx)     # safe to index; mask with ~done
        assert np.allclose(self.b.sum(1), 1.0, atol=1e-6), "b_probs rows must sum to 1"
        assert (self.b >= 0).all()

    def __repr__(self):
        return (f"MDP(S={self.S}, A={self.A}, start={self.start}, gamma={self.gamma}, "
                f"T={self.T}, min b>0 = {self.b[self.b > 0].min():.4g})")


def eps_greedy_probs(Q, eps):
    """pi_eps(a|s) = eps/|A| + (1-eps) * 1[a = argmax Q(s,.)]   (first-index tie-break)."""
    A = Q.shape[-1]
    p = np.full(Q.shape, eps / A)
    greedy = Q.argmax(-1)
    np.put_along_axis(p, greedy[..., None], 1.0 - eps + eps / A, axis=-1)
    return p


def exact_value(mdp, pi, gamma=None):
    """Exact V^pi_0(s) for all s: expected discounted return of at most T=trunc_limit steps.
    Backward induction over the horizon (finite-horizon policy evaluation)."""
    gamma = mdp.gamma if gamma is None else gamma
    V = np.zeros(mdp.S)
    for _ in range(mdp.T):
        Qsa = mdp.reward + gamma * np.where(mdp.done, 0.0, V[mdp.next_safe])
        V = (pi * Qsa).sum(1)
    return V


def greedy_rollout(mdp, Q):
    """Deterministic greedy rollouts from s0 for a batch of Q tables, Q: (n, S, A).
    Returns discounted return, undiscounted return, terminated-flag (False => hit the 200-step cap)."""
    n = Q.shape[0]
    sd = np.arange(n)
    s = np.full(n, mdp.start)
    alive = np.ones(n, bool)
    Gd = np.zeros(n)
    Gu = np.zeros(n)
    disc = 1.0
    for _ in range(mdp.T):
        a = Q[sd, s].argmax(-1)
        r = mdp.reward[s, a]
        Gd += alive * disc * r
        Gu += alive * r
        d = mdp.done[s, a]
        nxt = mdp.next_safe[s, a]
        alive = alive & ~d
        s = np.where(alive, nxt, s)
        disc *= mdp.gamma
        if not alive.any():
            break
    return Gd, Gu, ~alive
