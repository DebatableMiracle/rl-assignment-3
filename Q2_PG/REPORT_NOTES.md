# Report skeleton — Section 2 (Policy gradients), 5 marks
Fill `<…>` from `results/results.md`, `results/results.json` and the plots.

## 1. Environment (report at the start)
obs dim ds = 4 (cart pos, pole angle, cart vel, pole ang. vel); action dim da = 1; action bounds [-3, 3]; max episode length 1000; reward +1 per step until |angle| > 0.2 (so return = episode length); gamma = 1.

## 2. Implementation (short)
Policies (table 8^4 x 1, zero init; MLP 4-64-64-1, zero last layer), Gaussian sigma = 0.3, clip for the env only, log-prob on unclipped u; REINFORCE with one Adam update per batch of 16; M1–M6 differ only in the weight w_{i,t}. Include the binning ranges, out-of-range rule, seeding procedure, LRs and how they were chosen (see README), and the M3 gradient check result.

## 3. Results to include
Per-method plots (5 metrics x 6 methods) + `compare_*.png`; table of final eval mean ± std (b); H and per-method threshold stats (c, d).

## 4. Discussion drafts (check each claim against YOUR numbers; keep to the stated length)
**D1 Table policy.** An entry is only updated when the agent visits its bin; with zero init a rarely-visited entry stays near 0 (mean action 0 = "do nothing") and its Adam moments are stale, so it contributes little and learns slowly. Adjacent observations in different bins share NO parameters (each tuple has its own row), so there is no generalisation across a bin edge; observations in the same bin share one parameter (piecewise-constant policy). Quote the visited-bin coverage if you compute it.

**D2 Table vs neural.** Report which has the higher final eval mean and which reaches H in fewer episodes (from your table). What changed: ONLY the parameterisation (table vs MLP) and the LR; what stayed: the full-return gradient estimator w = G_i, sigma, batch size, optimizer, budget, seeds. The neural net generalises across states (a smooth function of the 4-D state) while the table cannot. Log-prob uses u (not a_t = clip(u)) because u is the sample actually drawn from N(mu, sigma^2): the density is defined for it, and the score function grad log pi(u|s) is the correct one. For a clipped action the density has point masses at the bounds and the score estimator would be biased.

**D3 Baselines.** Subtracting the plain current-batch mean G-bar from every G_i uses G_i itself in its own baseline, so the baseline is correlated with the sample's action and the estimator is biased by a factor (n-1)/n; the leave-one-out baseline b_{-i} excludes episode i, so it is independent of episode i's actions and the estimator stays unbiased (equivalently, scale the batch-mean form by n/(n-1) — verified numerically). The moving-average baseline is updated after the actor update so that the current batch does not influence its own weights (otherwise the baseline depends on the sample and biases the gradient). Tracking speed: with beta the baseline is an exponential average with time constant ~1/beta batches: beta = 0.01 ≈ 100 batches (≈ 1600 episodes, i.e. a quarter of the budget — it lags far behind rapidly increasing returns, starting from b0 = 0), beta = 0.1 ≈ 10 batches (≈ 160 episodes). Use your M4 vs M5 curves to show the consequence.

**D4 Return-to-go.** Causality: the reward r_k at k < t is independent of the action a_t, so E[ grad log pi(a_t|s_t) * r_k ] = 0 for k < t; removing those terms keeps the expected gradient but deletes pure-noise terms, lowering variance. Difference from M3–M5: scalar baselines subtract a quantity (independent of the action) from the SAME weight for every timestep of an episode, whereas return-to-go changes the weight itself per timestep by using only the future rewards; it needs no baseline and is time-dependent.

**D5 Estimator comparison.** (i) A factor other than gradient variance: bias of the estimator (e.g. M4's lagging baseline), the effective step size / Adam's normalisation, the exploration level (sigma), or how informative the signal is per sample (number of timesteps per update) — name one and support it with a measured quantity, e.g. grad-norm or parameter-norm plots. (ii) Matching architecture, optimizer, LR, sigma, budget and initialisation isolates the effect of the gradient estimator: otherwise a difference in final return could come from any of those instead of the estimator (a confound).

## 5. Likely viva questions
Why is REINFORCE high variance? Why does a baseline not bias the gradient (needs independence from the action)? Why is the discounted-vs-undiscounted choice irrelevant here (gamma = 1, return = episode length)? What does the final-layer zero init do (initial policy = N(0, 0.3^2), identical for all neural methods)? Why do M2/M6 collapse after peaking (large noisy updates with unnormalised returns up to 1000, constant LR)?
