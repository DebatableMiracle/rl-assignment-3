"""Step (iv)-(vii): offline control from the fixed behaviour dataset.

An "episode" of training = replay ONE stored trajectory, in time order, applying one TD update per
transition.  Per seed, a seeded permutation decides which dataset trajectories are replayed (800 of 2000),
so seeds differ in data order/subset.  No new environment interaction ever happens.

Algorithms (all tabular, from scratch, vectorised ACROSS seeds - each seed has its own Q table):

  qlearning        Q(s,a) += a [ r + g (1-done) max_a' Q(s',a') - Q(s,a) ]
  expected_sarsa   Q(s,a) += a [ r + g (1-done) sum_a' pi_eps(a'|s') Q(s',a') - Q(s,a) ]
                   pi_eps = eps_pi-greedy wrt current Q.  Off-policy by construction (no IS ratio needed,
                   because the expectation over a' is taken under the target policy, not sampled from b).
  is_sarsa         1-step off-policy SARSA.  The sampled next action a' ~ b is re-weighted:
                   Q(s,a) += a [ r + g (1-done) rho' Q(s',a') - Q(s,a) ],  rho' = pi_eps(a'|s') / b(a'|s')
                   (optionally ratio-clipped).  Same fixed point as Expected SARSA, but higher variance.
                   On a *truncated* last transition no next action was recorded, so we fall back to the
                   expectation (same expected value).

Truncation: the final transition of a truncated trajectory has done=False, so we bootstrap from Q(s_T).
Greedy evaluation: the MDP is deterministic, so one greedy rollout from s0 gives the exact greedy return.
"""
import numpy as np

from mdp_utils import greedy_rollout


def train_offline(mdp, ds, algo, eps_pi=0.1, seeds=range(10), n_episodes=800, alpha=0.1,
                  eval_every=20, rho_clip=None):
    seeds = list(seeds)
    nS_, A, T, gamma = len(seeds), mdp.A, mdp.T, mdp.gamma
    N = len(ds["length"])
    sd_all = np.arange(nS_)

    # per-seed replay order (separate RNG stream per seed)
    reps = int(np.ceil(n_episodes / N))
    order = np.stack([np.concatenate([np.random.default_rng(1000 + s).permutation(N) for _ in range(reps)])[:n_episodes]
                      for s in seeds])                                    # (n_seeds, n_episodes)

    Q = np.zeros((nS_, mdp.S, A))
    ep_log, Gd_log, Gu_log, term_log = [], [], [], []

    def evaluate(k):
        gd, gu, term = greedy_rollout(mdp, Q)
        ep_log.append(k); Gd_log.append(gd); Gu_log.append(gu); term_log.append(term)

    evaluate(0)
    for k in range(n_episodes):
        traj = order[:, k]
        L = ds["length"][traj]
        for t in range(int(L.max())):
            act = t < L
            i, tj = sd_all[act], traj[act]
            s, a = ds["S"][tj, t], ds["A"][tj, t]
            r, s2, d = ds["R"][tj, t], ds["S2"][tj, t], ds["D"][tj, t]
            q2 = Q[i, s2]                                                 # (m, A)
            if algo == "qlearning":
                nxt = q2.max(1)
            else:
                expct = (1 - eps_pi) * q2.max(1) + eps_pi * q2.mean(1)    # sum_a' pi_eps(a'|s') Q(s',a')
                if algo == "expected_sarsa":
                    nxt = expct
                elif algo == "is_sarsa":
                    tn = min(t + 1, T - 1)
                    a2, bp2 = ds["A"][tj, tn], ds["BP"][tj, tn]
                    pi2 = eps_pi / A + (1 - eps_pi) * (q2.argmax(1) == a2)
                    rho = pi2 / bp2
                    if rho_clip is not None:
                        rho = np.minimum(rho, rho_clip)
                    sampled = rho * q2[np.arange(len(i)), a2]
                    nxt = np.where(t + 1 < L[act], sampled, expct)        # last step: no next action stored
                else:
                    raise ValueError(algo)
            target = r + gamma * np.where(d, 0.0, nxt)
            Q[i, s, a] += alpha * (target - Q[i, s, a])
        if (k + 1) % eval_every == 0 or k + 1 == n_episodes:
            evaluate(k + 1)

    return Q, dict(episodes=np.array(ep_log), Gd=np.array(Gd_log), Gu=np.array(Gu_log), term=np.array(term_log))


def build_configs(eps_grid):
    """(name, algo, eps_pi, extra_kwargs) for every learner.  Q-Learning has no eps_pi in its update.
    IS-SARSA is run both unclipped and with the ratio clipped at 1 (truncated IS) - see checks.py for why."""
    cfgs = [("QL", "qlearning", None, {})]
    cfgs += [(f"ExpSARSA(eps={e})", "expected_sarsa", e, {}) for e in eps_grid]
    cfgs += [(f"IS-SARSA(eps={e})", "is_sarsa", e, {}) for e in eps_grid]
    cfgs += [(f"IS-SARSA-clip1(eps={e})", "is_sarsa", e, {"rho_clip": 1.0}) for e in eps_grid]
    return cfgs
