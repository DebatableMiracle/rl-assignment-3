"""All required plots.  Everything is saved to plots/ (Agg backend, no display needed)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)


def plot_return_distribution(ds, stats, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, key, lab in zip(ax, ("G_disc", "G_undisc"), ("discounted (gamma=0.99)", "undiscounted")):
        a.hist(ds[key], bins=40, color="steelblue", edgecolor="k", alpha=.8)
        a.axvline(ds[key].mean(), color="r", ls="--", label=f"mean={ds[key].mean():.2f}")
        a.set_xlabel(f"trajectory return G_i, {lab}"); a.set_ylabel("count"); a.legend()
    fig.suptitle(f"Behaviour-policy returns, N={stats['n_trajectories']}")
    _save(fig, outdir, "01_behaviour_return_distribution.png")


def plot_learning_curves(curves, outdir, key="Gd"):
    """curves: {name: {episodes, Gd (n_eval, n_seeds), ...}} -> mean +- std over seeds."""
    fams = [("ExpSARSA", "Expected SARSA vs Q-Learning"), ("IS-SARSA(", "IS-SARSA (unclipped) vs Q-Learning"),
            ("IS-SARSA-clip1", "IS-SARSA (rho clipped at 1) vs Q-Learning")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.4), sharey=True)

    def draw(ax, names, title):
        cmap = plt.get_cmap("tab10")
        for j, n in enumerate(names):
            c = curves[n]; m, s = c[key].mean(1), c[key].std(1, ddof=1)
            col = "k" if n == "QL" else cmap(j)
            ax.plot(c["episodes"], m, label=n, color=col, lw=2.2 if n == "QL" else 1.6)
            ax.fill_between(c["episodes"], m - s, m + s, color=col, alpha=.12)
        ax.set_title(title); ax.set_xlabel("training episodes (replayed trajectories)"); ax.grid(alpha=.3)
        ax.legend(fontsize=7)

    for ax, (pref, title) in zip(axes[:2], fams):
        draw(ax, ["QL"] + [n for n in curves if n.startswith(pref)], title)
    draw(axes[3], list(curves), "All learners")
    axes[0].set_ylabel("greedy return from s0 (discounted), mean +- std over seeds")
    _save(fig, outdir, "02_learning_curves.png")


def plot_final_returns(summary, outdir):
    names = list(summary)
    m = [summary[n]["final_greedy_discounted"]["mean"] for n in names]
    s = [summary[n]["final_greedy_discounted"]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = ["k" if n == "QL" else ("tab:blue" if n.startswith("Exp") else ("tab:green" if "clip" in n else "tab:orange")) for n in names]
    ax.bar(range(len(names)), m, yerr=s, capsize=4, color=cols, alpha=.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("final greedy return (discounted)"); ax.set_title("Final greedy return, mean +- std over 10 seeds")
    ax.grid(axis="y", alpha=.3)
    _save(fig, outdir, "03_final_greedy_returns.png")


def plot_is_estimates(targets, outdir):
    """OIS / WIS / PDIS (mean +- std over seeds) against the exact value, per target policy."""
    keys = list(targets)
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for j, (k, lab) in enumerate((("ois", "OIS"), ("wis", "WIS"), ("pdis", "PDIS"))):
        m = [targets[t][k]["mean"] for t in keys]; s = [targets[t][k]["std"] for t in keys]
        ax.bar(x + (j - 1) * w, m, w, yerr=s, capsize=2, label=lab, alpha=.85)
    ax.plot(x, [targets[t]["truth"]["mean"] for t in keys], "k_", ms=22, mew=2.5, label="exact V(s0)")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("estimate of V^pi(s0)"); ax.set_title("IS estimates of target-policy value (mean +- std over seeds)")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    _save(fig, outdir, "04_is_estimates_vs_truth.png")


def plot_ratio_stats(targets, outdir):
    keys = list(targets); x = np.arange(len(keys))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    ax[0].bar(x, [targets[t]["rho_mean"]["mean"] for t in keys], yerr=[targets[t]["rho_mean"]["std"] for t in keys],
              capsize=2, color="teal"); ax[0].axhline(1, color="r", ls="--", label="E_b[rho]=1")
    ax[0].set_ylabel("mean(rho_i)"); ax[0].legend()
    ax[1].bar(x, [targets[t]["log10_rho_var"]["mean"] for t in keys], color="firebrick")
    ax[1].set_ylabel("log10 Var(rho_i)")
    ax[2].bar(x, [targets[t]["ess_frac"]["mean"] for t in keys], color="purple")
    ax[2].set_yscale("log"); ax[2].set_ylabel("effective sample size / N")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(keys, rotation=60, ha="right", fontsize=6); a.grid(alpha=.3)
    fig.suptitle("Importance-ratio statistics per target policy (mean over seeds)")
    _save(fig, outdir, "05_importance_ratio_statistics.png")


def plot_bootstrap(boots, outdir):
    """boots: {label: {n: {ois:{bias,std,rmse}, wis:{...}}}}"""
    fig, axes = plt.subplots(len(boots), 3, figsize=(15, 4.2 * len(boots)), squeeze=False)
    for r, (lab, b) in enumerate(boots.items()):
        ns = sorted(b)
        for c, met in enumerate(("bias", "std", "rmse")):
            ax = axes[r, c]
            for est, col in (("ois", "tab:red"), ("wis", "tab:green")):
                v = [abs(b[n][est][met]) if met == "bias" else b[n][est][met] for n in ns]
                ax.plot(ns, v, "o-", color=col, label=est.upper())
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.3)
            ax.set_xlabel("number of trajectories n"); ax.set_ylabel(("|bias|" if met == "bias" else met))
            ax.set_title(f"{lab}: {met}"); ax.legend()
    _save(fig, outdir, "06_ois_vs_wis_bias_variance.png")
