"""Step (ii) and (viii): importance-sampling estimators of V^{pi_eps}(s0) from behaviour data.

Trajectory-level ratio   rho_i = prod_t pi(A_t|S_t) / b(A_t|S_t)
Everything is computed in log-space (a 200-step product of ratios easily over/underflows).

Estimators of V^pi(s0)  (G_i = discounted return of trajectory i, gamma = 0.99):
    Ordinary IS  (OIS)  : (1/N) sum_i rho_i G_i               unbiased,   high / unbounded variance
    Weighted IS  (WIS)  : sum_i rho_i G_i / sum_i rho_i       biased (O(1/N)), consistent, far lower variance
    Per-decision (PDIS) : (1/N) sum_i sum_t gamma^t rho_{i,0:t} r_{i,t}   unbiased, usually lower variance than OIS

Note: the dataset has a 200-step cap, so the estimand is the value of the *truncated* problem; the exact
reference value is computed with finite-horizon DP in mdp_utils.exact_value (same truncation).
"""
import numpy as np


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -np.inf
    m = x.max()
    if not np.isfinite(m):
        return m
    return m + np.log(np.exp(x - m).sum())


def log_ratio_steps(ds, pi):
    """Per-step log( pi(A_t|S_t) / b(A_t|S_t) ), zero on padding.  Requires pi > 0 on taken actions."""
    with np.errstate(divide="ignore"):
        logpi = np.log(pi[ds["S"], ds["A"]])
    step = logpi - np.log(ds["BP"])
    return np.where(ds["M"], step, 0.0)


def ratio_statistics(logrho):
    """mean(rho), Var(rho) (sample, ddof=1), effective sample size, all computed stably in log space."""
    N = len(logrho)
    l1, l2 = logsumexp(logrho), logsumexp(2 * logrho)
    mean = np.exp(l1 - np.log(N))
    with np.errstate(over="ignore", invalid="ignore"):
        second = np.exp(l2 - np.log(N))
        var = (second - mean ** 2) * N / (N - 1)
    ess = np.exp(2 * l1 - l2)                      # (sum rho)^2 / sum rho^2
    return dict(rho_mean=float(mean), rho_var=float(var),
                rho_max=float(np.exp(logrho.max())) if logrho.max() < 700 else float("inf"),
                log10_rho_var=float(np.log10(var)) if np.isfinite(var) and var > 0 else float("nan"),
                ess=float(ess), ess_frac=float(ess / N))


def ordinary_is(logrho, G):
    N = len(G)
    pos, neg = G > 0, G < 0
    lp = logsumexp(logrho[pos] + np.log(G[pos])) if pos.any() else -np.inf
    ln = logsumexp(logrho[neg] + np.log(-G[neg])) if neg.any() else -np.inf
    m = max(lp, ln)
    if not np.isfinite(m):
        return 0.0
    with np.errstate(over="ignore"):
        return float(np.exp(m) * (np.exp(lp - m) - np.exp(ln - m)) / N)


def weighted_is(logrho, G):
    w = np.exp(logrho - logrho.max())
    return float((w * G).sum() / w.sum())


def per_decision_is(ds, step, gamma):
    cl = np.cumsum(step, axis=1)                   # log rho_{0:t}
    gam = gamma ** np.arange(step.shape[1])
    with np.errstate(over="ignore"):
        term = np.where(ds["M"] & (ds["R"] != 0), np.exp(cl) * gam * ds["R"], 0.0)
    return float(term.sum(1).mean())


def estimate_target(ds, pi, gamma, truth=None):
    """All estimators + ratio statistics for one target policy pi (S,A)."""
    step = log_ratio_steps(ds, pi)
    logrho = step.sum(1)
    out = ratio_statistics(logrho)
    out["ois"] = ordinary_is(logrho, ds["G_disc"])
    out["wis"] = weighted_is(logrho, ds["G_disc"])
    out["pdis"] = per_decision_is(ds, step, gamma)
    if truth is not None:
        out["truth"] = float(truth)
        for k in ("ois", "wis", "pdis"):
            out[f"{k}_err"] = out[k] - truth
    return out


def bootstrap_comparison(ds, pi, truth, sizes=(50, 100, 200, 500, 1000, 2000), reps=300, seed=0):
    """Bias / std / RMSE of OIS vs WIS as a function of the number of trajectories n.
    Resamples n trajectories (with replacement) from the dataset `reps` times.
    This makes the bias-variance trade-off between OIS and WIS visible."""
    rng = np.random.default_rng(seed)
    logrho = log_ratio_steps(ds, pi).sum(1)
    G = ds["G_disc"]
    N = len(G)
    out = {}
    for n in sizes:
        idx = rng.integers(0, N, size=(reps, n))
        lr, g = logrho[idx], G[idx]
        m = lr.max(1, keepdims=True)
        w = np.exp(lr - m)
        wis = (w * g).sum(1) / w.sum(1)
        with np.errstate(over="ignore", invalid="ignore"):
            ois = np.exp(m[:, 0]) * (w * g).sum(1) / n
        res = {}
        for name, est in (("ois", ois), ("wis", wis)):
            fin = np.isfinite(est)
            e = est[fin]
            res[name] = dict(bias=float(e.mean() - truth), std=float(e.std()),
                             rmse=float(np.sqrt(((e - truth) ** 2).mean())), frac_finite=float(fin.mean()))
        out[int(n)] = res
    return out
