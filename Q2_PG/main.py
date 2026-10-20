"""Single interface for Section 2 (REINFORCE on InvertedPendulum-v5).

    python main.py --run all --seeds 0 1 2 3 4 --output runs/            # training (all 6 methods)
    python main.py --plot    --runs-dir runs/ --output plots/            # plots from saved runs
    python main.py --results --runs-dir runs/ --output results/          # numerical results from saved runs
    python main.py --videos  --runs-dir runs/ --output videos/           # eval videos (seed 0) from saved runs

Extras: --workers N (parallel processes, default = min(#cores, #jobs)), --episodes N (default 7000; use a small number
for a smoke test), --lr-table / --lr-neural, --overwrite (otherwise finished runs are skipped, so training is resumable).
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import train as T


def _job(args):
    return T.train_run(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", metavar="M", help="'all' or any of M1..M6")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--output", default="runs/")
    ap.add_argument("--runs-dir", default="runs/")
    ap.add_argument("--episodes", type=int, default=T.EPISODES)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr-table", type=float, default=T.LR_TABLE)
    ap.add_argument("--lr-neural", type=float, default=T.LR_NEURAL)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.run:
        methods = list(T.METHODS) if "all" in a.run else a.run
        os.makedirs(a.output, exist_ok=True)
        info = T.inspect_env()
        json.dump(info, open(os.path.join(a.output, "env_info.json"), "w"), indent=1)
        print("Environment:", info)
        print("M3 leave-one-out vs batch-mean gradient check:", T.check_loo_equivalence(a.output))
        jobs = [(m, s, a.output, a.episodes, a.lr_table, a.lr_neural)
                for s in a.seeds for m in methods
                if a.overwrite or not os.path.exists(os.path.join(a.output, f"{m}_seed{s}.npz"))]
        print(f"{len(jobs)} training jobs ({len(methods)} methods x {len(a.seeds)} seeds, finished runs skipped)")
        t0 = time.time()
        workers = a.workers or min(os.cpu_count() or 1, max(len(jobs), 1))
        if workers > 1 and len(jobs) > 1:
            with mp.get_context("spawn").Pool(workers) as pool:
                for m, s in pool.imap_unordered(_job, jobs):
                    print(f"== finished {m} seed {s}  ({time.time() - t0:.0f}s elapsed)", flush=True)
        else:
            for j in jobs:
                _job(j)
        print(f"training done in {time.time() - t0:.0f}s")
    if a.plot:
        import plotting
        plotting.make_plots(a.runs_dir, a.output if not a.run else "plots/")
    if a.results:
        import results
        results.make_results(a.runs_dir, a.output if not a.run else "results/")
    if a.videos:
        import videos
        videos.make_videos(a.runs_dir, a.output if not a.run else "videos/")
    if not (a.run or a.plot or a.results or a.videos):
        ap.print_help()


if __name__ == "__main__":
    main()
