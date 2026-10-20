"""--results : numerical results from saved runs only.

(a) F_{m,s}  = mean of the final 20 evaluation episodes (the last checkpoint) of method m, seed s
(b) mean and std (sample std, ddof=1) of the seed-level F over the seeds, per method
(c)(d) shared threshold  H = 0.9 * max_m mean_s F_{m,s};  per (m,s) the number of COMPLETED training episodes at the
       first evaluation checkpoint (incl. the pre-training one) whose 20-episode mean return is >= H, else 'not reached';
       per method: #reached, #not reached, mean/std of the threshold-reaching episode over successful seeds only.
"""
import json
import os

import numpy as np

from plotting import load_runs


def sstd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else None


def make_results(runs_dir, outdir):
    from train import METHODS
    os.makedirs(outdir, exist_ok=True)
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no saved runs found in {runs_dir}")
    F = {m: {r["meta"]["seed"]: float(r["eval_returns"][-1].mean()) for r in rs} for m, rs in runs.items()}
    means = {m: float(np.mean(list(v.values()))) for m, v in F.items()}
    H = 0.9 * max(means.values())
    if len(runs) < len(METHODS):
        print(f"WARNING: only methods {list(runs)} found; H is computed over these only")

    out = dict(threshold_H=H, final_eval_definition="mean of the final 20 evaluation episodes", methods={})
    lines = [f"Shared threshold H = 0.9 * max_m mean_s F = {H:.2f}  (best method mean = {max(means.values()):.2f})", "",
             "| Method | Description | final eval mean ± std (seeds) | reached | not reached | threshold episode mean ± std (successful) |",
             "|---|---|---|---|---|---|"]
    per_seed_lines = ["", "| Method | seed | F (final-20 eval mean) | episodes to reach H |", "|---|---|---|---|"]
    for m, rs in runs.items():
        seeds = [r["meta"]["seed"] for r in rs]
        f = np.array([F[m][s] for s in seeds])
        reach = {}
        for r in rs:
            hit = np.where(r["eval_returns"].mean(1) >= H)[0]
            reach[r["meta"]["seed"]] = int(r["eval_episodes"][hit[0]]) if len(hit) else None
            per_seed_lines.append(f"| {m} | {r['meta']['seed']} | {F[m][r['meta']['seed']]:.2f} | "
                                  f"{reach[r['meta']['seed']] if reach[r['meta']['seed']] is not None else 'not reached'} |")
        ok = [v for v in reach.values() if v is not None]
        out["methods"][m] = dict(desc=METHODS[m]["desc"], seeds=seeds, final_eval_per_seed=F[m],
                                 final_eval_mean=float(f.mean()), final_eval_std=sstd(f),
                                 threshold_episode_per_seed={str(k): v for k, v in reach.items()},
                                 n_reached=len(ok), n_not_reached=len(reach) - len(ok),
                                 threshold_episode_mean=float(np.mean(ok)) if ok else None,
                                 threshold_episode_std=sstd(ok) if ok else None)
        t_mean = f"{np.mean(ok):.0f} ± {sstd(ok):.0f}" if len(ok) > 1 else (f"{ok[0]} (1 seed, std undefined)" if ok else "undefined")
        lines.append(f"| {m} | {METHODS[m]['desc']} | {f.mean():.2f} ± {sstd(f):.2f} | {len(ok)} | {len(reach) - len(ok)} | {t_mean} |")

    # copy the env inspection + M3 gradient-equivalence check if they exist
    for name in ("env_info.json", "loo_gradient_check.json"):
        p = os.path.join(runs_dir, name)
        if os.path.exists(p):
            out[name.split(".")[0]] = json.load(open(p))
    json.dump(out, open(os.path.join(outdir, "results.json"), "w"), indent=1)
    text = "\n".join(lines + per_seed_lines)
    open(os.path.join(outdir, "results.md"), "w").write(text + "\n")
    print(text)
    if "loo_gradient_check" in out:
        print("\nM3 leave-one-out vs batch-mean gradient check:", out["loo_gradient_check"])
