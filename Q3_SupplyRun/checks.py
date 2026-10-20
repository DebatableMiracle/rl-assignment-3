"""Diagnostic checks (called from main.py, or run alone:  python checks.py).

A. Why are results identical across eps_pi?   -> compare greedy policies across eps (all states / path from s0)
B. Why is seed variance high at 800 episodes? -> successes replayed per seed; sensitivity to #episodes and alpha
C. Why does IS-SARSA misbehave?               -> max|Q| vs training length for unclipped / clipped / small alpha

Findings these checks document (real artifact, see report):
  * identical returns across eps_pi: the greedy action agrees on ~95-99% of states but NOT all, and the greedy path
    from s0 is the same in only ~50-100% of seeds. The final returns coincide because each seed's outcome is
    essentially binary (delivered ~80.2 / failed -17.4), so different paths can land in the same outcome class.
  * 800 replayed trajectories = ~12 successful (delivery) trajectories per seed out of 34 in the dataset. With one full
    pass (2000) Q-Learning and Expected SARSA solve the task on 10/10 seeds.
  * unclipped 1-step IS-SARSA diverges at alpha=0.1 with long training: unbiased mean update, but the update VARIANCE
    blows up (gamma^2 * E_b[rho^2] > 1 on cycles), Q reaches ~1e14 and the greedy policy loops between refuel cells.
    Clipping rho at 1 (truncated IS) or a small alpha restores stability.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mdp_utils import greedy_rollout
from offline_control import train_offline

SEEDS = list(range(10))
EPISODES = (800, 2000, 8000)
GRID = [  # label, algo, eps_pi, alpha, kwargs
    ("Q-Learning  a=0.1", "qlearning", None, 0.1, {}),
    ("Q-Learning  a=0.5", "qlearning", None, 0.5, {}),
    ("ExpSARSA(.1)  a=0.1", "expected_sarsa", 0.1, 0.1, {}),
    ("ExpSARSA(.1)  a=0.5", "expected_sarsa", 0.1, 0.5, {}),
    ("IS-SARSA(.1)  a=0.1", "is_sarsa", 0.1, 0.1, {}),
    ("IS-SARSA(.1)  a=0.02", "is_sarsa", 0.1, 0.02, {}),
    ("IS-SARSA-clip1(.1)  a=0.1", "is_sarsa", 0.1, 0.1, {"rho_clip": 1.0}),
    ("IS-SARSA-clip1(.1)  a=0.02", "is_sarsa", 0.1, 0.02, {"rho_clip": 1.0}),
]


def _path_actions(mdp, q):
    """Greedy action sequence from s0 (until termination / cap)."""
    s, acts = mdp.start, []
    for _ in range(mdp.T):
        a = int(q[s].argmax()); acts.append(a)
        if mdp.done[s, a]:
            break
        s = int(mdp.next_idx[s, a])
    return acts


def check_greedy_agreement(mdp, Qs, eps_grid):
    print("\n== Check A: are the greedy policies really the same across eps_pi? (800-episode Q tables) ==")
    out = {}
    for fam in ("ExpSARSA", "IS-SARSA"):
        names = [f"{fam}(eps={e})" for e in eps_grid]
        ref = Qs[names[0]]
        for n in names[1:]:
            all_states = np.mean([(ref[i].argmax(-1) == Qs[n][i].argmax(-1)).mean() for i in range(len(SEEDS))])
            same_path = np.mean([_path_actions(mdp, ref[i]) == _path_actions(mdp, Qs[n][i]) for i in range(len(SEEDS))])
            max_dq = float(np.abs(ref - Qs[n]).max())
            out[f"{names[0]} vs {n}"] = dict(greedy_agreement_all_states=float(all_states),
                                             frac_seeds_same_greedy_path_from_s0=float(same_path), max_abs_Q_diff=max_dq)
            print(f"  {names[0]:20s} vs {n:20s}: action agreement over all states = {all_states:.3f}, "
                  f"same greedy path from s0 in {same_path*100:.0f}% of seeds, max|dQ| = {max_dq:.3f}")
    return out


def check_data_budget(mdp, ds, Qs, n_episodes):
    print("\n== Check B1: successful (delivery) trajectories replayed per seed vs final return ==")
    succ = ds["G_undisc"] > 0
    N = len(succ)
    order = np.stack([np.random.default_rng(1000 + s).permutation(N)[:n_episodes] for s in SEEDS])
    n_succ = succ[order].sum(1)
    gd, _, _ = greedy_rollout(mdp, Qs["QL"])
    print(f"  dataset has {int(succ.sum())} successful trajectories of {N}; replayed per seed: {n_succ.tolist()}")
    print(f"  QL final greedy return per seed: {np.round(gd, 1).tolist()}")
    return dict(n_success_dataset=int(succ.sum()), n_success_per_seed=n_succ.tolist(), ql_final=gd.tolist())


def run_grid(mdp, ds):
    print("\n== Check B2/C: sensitivity to #episodes and alpha (10 seeds, final greedy discounted return) ==")
    print(f"  {'variant':28s}" + "".join(f"| {e:>5d} episodes: mean±std (#solved) max|Q| " for e in EPISODES))
    res = {}
    for label, algo, eps, alpha, kw in GRID:
        row, cells = {}, []
        for ep in EPISODES:
            Q, c = train_offline(mdp, ds, algo, eps_pi=eps, seeds=SEEDS, n_episodes=ep, alpha=alpha, eval_every=ep, **kw)
            g = c["Gd"][-1]
            row[ep] = dict(mean=float(g.mean()), std=float(g.std(ddof=1)), n_solved=int((g > 70).sum()),
                           max_abs_Q=float(np.abs(Q).max()))
            cells.append(f"| {g.mean():7.1f}±{g.std(ddof=1):5.1f} ({int((g > 70).sum()):2d}/10) {np.abs(Q).max():9.3g} ")
        res[label] = row
        print(f"  {label:28s}" + "".join(cells), flush=True)
    return res


def plot_grid(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for label, row in res.items():
        eps_ = sorted(row)
        ax[0].errorbar(eps_, [row[e]["mean"] for e in eps_], [row[e]["std"] for e in eps_], marker="o", capsize=3,
                       label=label, ls="--" if "IS-SARSA" in label else "-")
        ax[1].plot(eps_, [max(row[e]["max_abs_Q"], 1e-3) for e in eps_], "o-", label=label)
    ax[0].set_xscale("log"); ax[0].set_xlabel("training episodes (replayed trajectories)")
    ax[0].set_ylabel("final greedy return (mean ± std, 10 seeds)"); ax[0].set_title("Sensitivity to training length and alpha")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=6)
    ax[1].axhline(100, color="k", ls=":", label="true |Q| <= ~100")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("training episodes")
    ax[1].set_ylabel("max |Q| over all seeds/states"); ax[1].set_title("Stability: unclipped IS-SARSA (alpha=0.1) diverges")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "07_sensitivity_and_stability.png"), dpi=150); plt.close(fig)


def run_checks(mdp, ds, Qs, eps_grid, plot_dir, alpha=0.1, n_episodes=800):
    out = dict(greedy_agreement=check_greedy_agreement(mdp, Qs, eps_grid),
               data_budget=check_data_budget(mdp, ds, Qs, n_episodes))
    out["grid"] = {k: {str(e): v for e, v in row.items()} for k, row in run_grid(mdp, ds).items()}
    plot_grid({k: {int(e): v for e, v in row.items()} for k, row in out["grid"].items()}, plot_dir)
    return out


if __name__ == "__main__":
    from generate_data import generate_trajectories
    from mdp_utils import MDP
    from offline_control import build_configs
    mdp = MDP(); ds = generate_trajectories(mdp, 2000, 0)
    eps_grid = [0.05, 0.1, 0.2, 0.3]
    Qs = {n: train_offline(mdp, ds, a, eps_pi=e, seeds=SEEDS, **kw)[0] for n, a, e, kw in build_configs(eps_grid)}
    res = run_checks(mdp, ds, Qs, eps_grid, "plots")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/checks.json", "w"), indent=1)
