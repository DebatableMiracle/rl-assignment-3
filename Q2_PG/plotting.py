"""--plot : reads saved runs only (never retrains).  One plot per (method, metric): 5 light seed curves, a dark mean
curve and a +-1 std band, all versus completed training episodes.  Plus overlay comparison plots across methods."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_ORDER = ["M1", "M2", "M3", "M4", "M5", "M6"]
WINDOW = 50


def load_runs(runs_dir, methods=METHOD_ORDER):
    runs = {}
    for m in methods:
        files = sorted(glob.glob(os.path.join(runs_dir, f"{m}_seed*.npz")))
        rs = []
        for f in files:
            d = np.load(f)
            r = {k: d[k] for k in d.files if k != "meta"}
            r["meta"] = json.loads(str(d["meta"]))
            rs.append(r)
        if rs:
            runs[m] = sorted(rs, key=lambda r: r["meta"]["seed"])
    return runs


def trailing_mean(x, w=WINDOW):
    c = np.cumsum(np.insert(x, 0, 0.0))
    k = np.arange(1, len(x) + 1)
    lo = np.maximum(0, k - w)
    return (c[k] - c[lo]) / (k - lo)          # expanding mean for the first 49 episodes, then trailing-50


METRICS = {   # name -> (title, y-label, extractor(run) -> (x, y))
    "raw_return": ("Raw training return", "episode return",
                   lambda r: (np.arange(1, len(r["train_returns"]) + 1), r["train_returns"])),
    "trailing50_return": ("Trailing-50 mean training return", "mean return (last 50 episodes)",
                          lambda r: (np.arange(1, len(r["train_returns"]) + 1), trailing_mean(r["train_returns"]))),
    "param_norm": ("Parameter norm", "||theta||_2", lambda r: (r["update_episodes"], r["param_norm"])),
    "grad_norm": ("Actor-gradient norm (before Adam step)", "||g||_2", lambda r: (r["update_episodes"], r["grad_norm"])),
    "eval_return": ("Evaluation return (stochastic policy, 20 episodes)", "mean eval return",
                    lambda r: (r["eval_episodes"], r["eval_returns"].mean(1))),
}


def _set_scale(ax, metric):
    """param norm is always > 0 -> log.  The actor-gradient norm can be EXACTLY 0 (e.g. M3 once all 16 episodes hit the
    1000-step cap, so every leave-one-out weight is 0); a log axis cannot show 0, so use symlog (linear below 0.1)."""
    if metric == "param_norm":
        ax.set_yscale("log")
    elif metric == "grad_norm":
        ax.set_yscale("symlog", linthresh=0.1)


def stack(rs, metric):
    xs, ys = zip(*[METRICS[metric][2](r) for r in rs])
    n = min(len(x) for x in xs)
    return xs[0][:n], np.stack([y[:n] for y in ys])


def plot_metric(method, rs, metric, outdir, desc=""):
    x, Y = stack(rs, metric)
    m, s = Y.mean(0), Y.std(0, ddof=1) if len(Y) > 1 else np.zeros(Y.shape[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for r, y in zip(rs, Y):
        ax.plot(x, y, color="tab:blue", alpha=0.25, lw=0.8, label="seed curves" if r is rs[0] else None)
    ax.plot(x, m, color="navy", lw=2.2, label=f"mean ({len(rs)} seeds)")
    ax.fill_between(x, m - s, m + s, color="navy", alpha=0.18, label="±1 std")
    ax.set_xlabel("completed training episodes"); ax.set_ylabel(METRICS[metric][1])
    ax.set_title(f"{method}: {METRICS[metric][0]}" + (f"\n{desc}" if desc else ""), fontsize=10)
    _set_scale(ax, metric)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{method}_{metric}.png"), dpi=140); plt.close(fig)


def plot_compare(runs, metric, outdir):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for m, rs in runs.items():
        x, Y = stack(rs, metric)
        mu, sd = Y.mean(0), (Y.std(0, ddof=1) if len(Y) > 1 else 0 * Y[0])
        ax.plot(x, mu, lw=1.8, label=m); ax.fill_between(x, mu - sd, mu + sd, alpha=0.12)
    ax.set_xlabel("completed training episodes"); ax.set_ylabel(METRICS[metric][1])
    ax.set_title(f"All methods: {METRICS[metric][0]} (mean ± 1 std over seeds)", fontsize=10)
    _set_scale(ax, metric)
    ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"compare_{metric}.png"), dpi=140); plt.close(fig)


def make_plots(runs_dir, outdir):
    from train import METHODS
    os.makedirs(outdir, exist_ok=True)
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no saved runs found in {runs_dir}")
    for m, rs in runs.items():
        for metric in METRICS:
            plot_metric(m, rs, metric, outdir, METHODS[m]["desc"])
    for metric in METRICS:
        plot_compare(runs, metric, outdir)
    print(f"saved {len(runs) * len(METRICS)} per-method plots + {len(METRICS)} comparison plots to {outdir}")