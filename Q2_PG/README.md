# Q2 PG — REINFORCE on InvertedPendulum-v5 (M1–M6)

## Reproduce
```bash
cd Q2_PG
python -m venv .venv && source .venv/bin/activate
pip install -r installations.txt
python main.py --run all --seeds 0 1 2 3 4 --output runs/            # training, 6 methods x 5 seeds
python main.py --plot    --runs-dir runs/ --output plots/
python main.py --results --runs-dir runs/ --output results/
python main.py --videos  --runs-dir runs/ --output videos/          # headless box: MUJOCO_GL=egl python main.py ...
```
Training runs `--workers` processes in parallel (default = min(#cores, #jobs)); each (method, seed) is independent.
Finished runs are skipped, so an interrupted `--run` can simply be restarted (`--overwrite` to redo).
Rough cost: one 7000-episode run is a few CPU-minutes (M3/M4/M5 are the slowest because their episodes reach 1000 steps);
all 30 runs ≈ 1.5–2 CPU-hours, so use a multi-core machine. Smoke test: `python main.py --run all --seeds 0 --episodes 192 --output /tmp/runs`.

## Files
`main.py` CLI · `train.py` rollouts, REINFORCE M1–M6, evaluation, M3 gradient check · `policies.py` table + neural policy ·
`plotting.py` `results.py` `videos.py` (read saved runs only) · `installations.txt`.

## Settings (all fixed by the assignment unless marked)
Adam; batch 16 complete episodes; 7000 training episodes (437 batches of 16 + a final batch of 8, averaged over its actual 8
episodes); gamma = 1; sigma = 0.3; hidden 64-64 ReLU; evaluation = 20 stochastic episodes before training, after every
10 actor updates and after the final update; smoothing window 50. **Learning rates (my choice, one per rule): M1 = 0.02, M2–M6 = 1e-3.**

### Learning-rate selection (be transparent about this in the report)
Small pilot on seed 0 only (so it overlaps a reporting seed): neural, 7000 episodes — lr 1e-3 / 3e-3 solved M3 (return 1000), while M2 and
M6 were unstable at both (peak then collapse); lr 1e-2 failed at 2000 episodes. Table — lr 0.02 > 0.01 > 0.005 and 0.05/0.2 were worse
at 2000 episodes. 1e-3 was kept for the neural methods because it is the smaller (safer) of the two working rates.

### Binning of the discretized policy (M1)
cart pos [-0.5, 0.5], pole angle [-0.2, 0.2], cart vel [-1, 1], pole ang. vel [-2, 2]; 8 equal bins each via `np.digitize` on the 7
interior edges. Out-of-range values go to the first/last bin of that dimension. Index = base-8 number of the 4 bin indices →
8^4 = 4096 unique rows, `theta` in R^{4096x1}, zero init.

### Seeding procedure
Policy init `torch.manual_seed(seed)` (all neural methods of a seed share identical initial parameters; final layer zero).
Training: `torch.Generator(10000+seed)` for action noise, episode k reset with seed `1_000_000*seed + k`.
Evaluation: separate `torch.Generator(20000+seed)`, c-th evaluation episode reset with seed `100_000_000 + 1_000_000*seed + c`.

### Other implementation notes
* log pi uses the UNCLIPPED sample u; only the env receives clip(u, -3, 3); no gradient flows through the sampled action.
* M3: LOO weights vs equivalent batch-mean form n/(n-1)(G_i − mean G) are compared numerically at the start of `--run`
  (`runs/loo_gradient_check.json`, copied into `results/results.json`).
* M4/M5: the baseline starts at 0, is used for the current update and updated afterwards (current batch never affects its own weights).
* "Final 20 evaluation episodes" = the last evaluation checkpoint. Std over seeds is the sample std (ddof=1). Threshold episode =
  completed training episodes at the first checkpoint (incl. the pre-training one) with 20-episode mean >= H.
* Saved per run (`runs/<M>_seed<k>.npz`): raw training returns, grad/param norms per update, all evaluation returns; final policy in `*_params.pt`.
