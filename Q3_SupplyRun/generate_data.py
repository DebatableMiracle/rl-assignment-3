"""Step (i) and (iii): generate N trajectories using ONLY the supplied behaviour policy b.

Stored per trajectory (padded to trunc_limit, with a mask):
    S   state index            A   action                 R   reward
    BP  b(A_t | S_t)  <- needed for the importance ratio
    S2  next state index (safe)  D  done flag of the transition   M  validity mask
plus length, terminated/truncated flags and the (discounted / undiscounted) returns G_i.
"""
import argparse

import numpy as np

from mdp_utils import MDP


def generate_trajectories(mdp, n_traj=2000, seed=0):
    rng = np.random.default_rng(seed)             # dedicated RNG -> reproducible dataset
    N, T, A = n_traj, mdp.T, mdp.A
    S = np.zeros((N, T), np.int64)
    Act = np.zeros((N, T), np.int64)
    R = np.zeros((N, T))
    BP = np.ones((N, T))                          # 1.0 on padding => log = 0
    S2 = np.zeros((N, T), np.int64)
    D = np.zeros((N, T), bool)
    M = np.zeros((N, T), bool)

    cum = np.cumsum(mdp.b, axis=1)
    s = np.full(N, mdp.start)
    alive = np.ones(N, bool)
    for t in range(T):                            # all N trajectories advance together
        u = rng.random(N)
        a = np.minimum((u[:, None] >= cum[s]).sum(1), A - 1)   # inverse-CDF sampling from b(.|s)
        r, d, s2 = mdp.reward[s, a], mdp.done[s, a], mdp.next_safe[s, a]
        S[:, t] = np.where(alive, s, 0)
        Act[:, t] = np.where(alive, a, 0)
        R[:, t] = np.where(alive, r, 0.0)
        BP[:, t] = np.where(alive, mdp.b[s, a], 1.0)
        S2[:, t] = np.where(alive, s2, 0)
        D[:, t] = alive & d
        M[:, t] = alive
        alive = alive & ~d                        # truncation (t == T-1) leaves alive=True
        s = np.where(alive, s2, s)

    length = M.sum(1)
    gam = mdp.gamma ** np.arange(T)
    ds = dict(S=S, A=Act, R=R, BP=BP, S2=S2, D=D, M=M, length=length,
              terminated=D.any(1), truncated=~D.any(1),
              G_disc=(R * gam).sum(1), G_undisc=R.sum(1), seed=np.array(seed))
    return ds


def save_dataset(ds, path):
    np.savez_compressed(path, **ds)


def load_dataset(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def return_statistics(ds):
    """Step (iii): mean and std of the trajectory returns G_i."""
    def st(x):
        return dict(mean=float(x.mean()), std=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()))
    return dict(n_trajectories=int(len(ds["G_disc"])),
                discounted_return=st(ds["G_disc"]),
                undiscounted_return=st(ds["G_undisc"]),
                mean_length=float(ds["length"].mean()),
                frac_terminated=float(ds["terminated"].mean()),
                frac_truncated=float(ds["truncated"].mean()),
                frac_positive_return=float((ds["G_undisc"] > 0).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dataset.npz")
    a = ap.parse_args()
    mdp = MDP()
    ds = generate_trajectories(mdp, a.n, a.seed)
    save_dataset(ds, a.out)
    print(return_statistics(ds))
