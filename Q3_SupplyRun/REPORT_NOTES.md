# Report skeleton — Section 1 (SupplyRun offline), 5 marks

Fill the `<…>` from `results/results.json` after running `main.py` on the real artifact. Discussion text below is written to be correct regardless of the exact numbers, but check each claim against your plots.

## 1. Approach (short)
* Data: 2000 trajectories from `b` (seed 0), stored with `b(A_t|S_t)`, next state, done, mask. Cap 200 steps.
* Return stats (step iii): discounted G: mean `<…>` ± std `<…>`; undiscounted: `<…>` ± `<…>`; mean length `<…>`; fraction truncated `<…>`; fraction with positive return `<…>`. Sanity: mean(G_i) matches exact V^b(s0) `<…>` within s.e.
* Offline learners (step iv): Q-Learning and **Expected SARSA** (+ **IS-SARSA** variant); 800 replayed trajectories, alpha `<0.1>`, gamma 0.99, Q0 = 0, 10 seeds. `eps_pi` investigated: {0.05, 0.1, 0.2, 0.3} (step v).

## 2. Importance-sampling estimator (step ii)
Estimand: V^{pi_eps}(s0) (discounted, 200-step truncated). Estimators:

* OIS = (1/N) Σ ρ_i G_i, WIS = Σ ρ_i G_i / Σ ρ_i, PDIS = (1/N) Σ_i Σ_t γ^t ρ_{i,0:t} r_{i,t}; ρ_i = Π_t π(A_t|S_t)/b(A_t|S_t), computed in log space.
* **OIS vs WIS trade-off** (use plots 04, 05, 06):
  * OIS is unbiased (E_b[ρ]=1, verified: mean(ρ) ≈ 1 when π = b) but its variance is driven by Var(ρ), which grows roughly exponentially with horizon and with the divergence between π and b. Estimates can be wildly outside the return range [−20, 100]·(discount) and are dominated by a few trajectories.
  * WIS is biased for finite N (ratio of random variables; bias O(1/N)) but consistent, always lies inside the range of observed G_i, and its variance is far lower. Effective sample size ESS = (Σρ)²/Σρ² quantifies how few trajectories carry the weight.
  * When π is close to b (large eps_pi), OIS and WIS agree; as eps_pi → 0 the ratios become degenerate (mean(ρ) → 0 in sample, ESS/N → tiny) and OIS can collapse to ≈ 0 while WIS stays bounded but biased toward the few matching trajectories.
  * Bootstrap plot 06: |bias| and std vs n for both estimators against the exact value.
* PDIS: unbiased like OIS, usually lower variance because only the ratio up to time t multiplies r_t.
* Caveat to state: the estimand is the *truncated* value; ground truth computed by finite-horizon DP.

## 3. Results to tabulate
| item | where |
|---|---|
| learning curves, all variants (step vi) | `plots/02_learning_curves.png` |
| final greedy return mean ± std, 10 seeds, Q-Learning and each SARSA variant/eps (step vii) | `results.json → training`, `plots/03` |
| per target policy: mean(ρ), Var(ρ), ESS, OIS, WIS, PDIS, truth (step viii) | `results.json → targets`, `plots/04, 05` |
| non-terminating greedy seeds (greedy policy hits 200 cap) | `results.json → training[*].n_greedy_nonterminating` |

## 4. Discussion prompts (answer with your measured numbers)
* **Q-Learning vs SARSA family offline.** Q-Learning's max-target is off-policy and learns Q* *if the dataset covers the relevant state-actions*; Expected SARSA with `pi_eps` learns Q^{pi_eps}, a more conservative value (it accounts for the eps-random actions, which waste fuel here), so its greedy policy is typically more cautious but not necessarily better. Compare against your measured final returns.
* **Effect of eps_pi.** Larger eps_pi → target closer to b → smaller Var(ρ), larger ESS (plot 05), better-behaved IS estimates; but a more pessimistic Q^{pi_eps} for Expected SARSA. Smaller eps_pi → nearly deterministic target, ρ degenerate (this is why eps_pi > 0 is required: a deterministic target gives ρ_i = 0 for any trajectory that deviates).
* **IS-SARSA vs Expected SARSA (measured).** Same fixed point in expectation, but unclipped IS-SARSA at alpha=0.1 *diverged* with long training (max|Q| about 1e14 at 8000 episodes; true |Q| is at most about 100). The greedy policy then cycles between the two refuel cells (return exactly 0, hitting the 200-step cap). Reason: the update is unbiased but its variance is multiplied by gamma^2 E_b[rho'^2] > 1 around cycles (rho' up to about 6.4 here). Fixes that worked: clipping rho at 1 (truncated IS, biased) or alpha=0.02. Expected SARSA needs no ratio and is stable. See `plots/07_sensitivity_and_stability.png`.
* **Identical final returns across eps_pi (measured).** The greedy action agrees on about 95-99% of states, and the greedy path from s0 is identical in only 50-100% of seeds (Check A). The mean returns coincide because each seed's outcome is essentially binary (delivered ~80.2 or failed -17.4). Do not claim the policies are identical.
* **800 vs 2000+ episodes (measured).** At 800 replays each seed sees only ~10-16 of the 34 successful trajectories, so results depend on which seeds saw enough successes (Expected SARSA 70 ± 31 at 800). With 2000 (one full pass) Q-Learning and Expected SARSA solve 10/10 seeds. Report 800 as the specified setting and the sensitivity table as supporting evidence.
* **Coverage / distribution shift.** Greedy actions in rarely visited states are essentially untrained (Q stays at init 0, argmax tie-breaks to UP); offline methods cannot fix this without new data.
* **Truncation handling.** Last transition of a truncated trajectory bootstraps from Q(s_T); `done` (true termination) uses target = r only.

## 5. Likely viva questions
* Why is E_b[ρ] = 1, and why does the sample mean(ρ) usually come out < 1 for a far-away target?
* Why is WIS biased but consistent? Why can OIS have infinite variance in principle?
* Why does Expected SARSA need no importance ratio but 1-step SARSA with a sampled a' does?
* Why do we compute everything in log space?
* What does ESS tell you? Why `next_idx = −1` must be masked with `done`?
