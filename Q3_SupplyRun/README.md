# Q3 SupplyRun — Off-Policy Evaluation and Offline Learning

## Reproduce everything

```bash
cd Q3_SupplyRun
pip install numpy matplotlib        # only dependencies
python main.py                      # defaults: N=2000, 800 episodes, 10 seeds, alpha=0.1
```

`behaviour_policy.npz` (supplied, unmodified) must sit in this folder. Runtime is about 2 minutes on a laptop CPU (about 15 s with `--skip-checks`).

Equivalent explicit command (every default written out):

```bash
python main.py --n-traj 2000 --data-seed 0 --episodes 800 --n-seeds 10 --alpha 0.1 \
               --eps 0.05 0.1 0.2 0.3 --eval-every 20
```

Outputs: `results/results.json` (all numbers), `results/dataset.npz` (the 2000 trajectories), `plots/*.png`.

## Files

| file | role |
|---|---|
| `mdp_utils.py` | artifact loader, eps-greedy, exact finite-horizon policy evaluation (ground truth), greedy rollout |
| `generate_data.py` | simulate N trajectories from `b`; store S, A, R, b(A\|S), next state, done, mask; return statistics |
| `is_estimators.py` | log-space importance ratios, OIS / WIS / per-decision IS, mean/Var(rho), ESS, OIS-vs-WIS bootstrap |
| `offline_control.py` | offline Q-Learning, Expected SARSA, IS-SARSA (replay of stored trajectories) |
| `visualize.py` | all plots |
| `checks.py` | diagnostics: greedy-policy agreement across eps, data-budget, episodes/alpha sensitivity, IS-SARSA stability (also runnable alone) |
| `main.py` | single entry point |

## Seeding / reproducibility

* Dataset: `numpy.random.default_rng(data_seed=0)`; all trajectories advance in lock-step, so the same seed gives the same data.
* Learning seed `k` in `0..9`: replay order = `default_rng(1000+k).permutation(N)[:800]`.
* The MDP is deterministic, so greedy evaluation is one exact rollout from `s0` (no evaluation randomness).
* Bootstrap comparison: `default_rng(0)`.

## Design decisions (details for the report)

* **Offline "episode"** = replaying one stored trajectory in time order, one TD update per transition. No env interaction.
* **Truncation**: the last transition of a truncated trajectory has `done=False`, so Q-Learning / Expected SARSA bootstrap from the next state. `next_idx == -1` is never followed (`done` mask).
* **SARSA-family methods**: Expected SARSA (target = expectation under `pi_eps`, no ratio needed), 1-step IS-SARSA (sampled next action re-weighted by `pi_eps/b`), and IS-SARSA-clip1 (ratio clipped at 1, added because the unclipped version diverges, see `checks.py`). `eps_pi` in {0.05, 0.1, 0.2, 0.3}.
* **Target policies for IS**: `pi_eps` built from each learned Q. SARSA variants use their own training `eps_pi`; Q-Learning (which has no `eps_pi`) is evaluated at every `eps_pi` in the grid. `eps_pi > 0` always.
* **Ground truth**: exact 200-step-truncated discounted value by backward induction, to measure the error of OIS / WIS / PDIS.
