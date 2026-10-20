"""Single entry point:  cd Q3_SupplyRun && python main.py

Pipeline
  (i)   generate N=2000 trajectories with the supplied behaviour policy b
  (iii) return statistics of G_i
  (iv)  offline Q-Learning + Expected SARSA + IS-SARSA (several eps_pi), 800 episodes, 10 seeds
  (vi)  learning curves,  (vii) final greedy return mean +- std over seeds
  (ii,viii) OIS / WIS / PDIS estimates of V^{pi_eps}(s0) for every learned target policy,
            mean(rho), Var(rho), ESS, and an OIS-vs-WIS bias/variance bootstrap
All numbers -> results/results.json, all figures -> plots/
"""
import argparse
import json
import os
import time

import numpy as np

from generate_data import generate_trajectories, return_statistics, save_dataset
from is_estimators import bootstrap_comparison, estimate_target
from mdp_utils import MDP, eps_greedy_probs, exact_value
from offline_control import build_configs, train_offline
import visualize as viz


def ms(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(np.nanmean(x)), std=float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0,
                per_seed=[float(v) for v in x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=2000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--skip-checks", action="store_true", help="skip the (slower) sensitivity / stability checks")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    res_dir, plot_dir = os.path.join(args.out, "results"), os.path.join(args.out, "plots")
    os.makedirs(res_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
    t0 = time.time()

    mdp = MDP()
    print(mdp)
    seeds = list(range(args.n_seeds))

    # ---------------- (i) + (iii) data ----------------
    ds = generate_trajectories(mdp, args.n_traj, args.data_seed)
    save_dataset(ds, os.path.join(res_dir, "dataset.npz"))
    stats = return_statistics(ds)
    V_b = float(exact_value(mdp, mdp.b)[mdp.start])
    print("\n== Behaviour-policy dataset ==")
    print(json.dumps(stats, indent=1))
    se = stats["discounted_return"]["std"] / np.sqrt(args.n_traj)
    print(f"exact V^b(s0) = {V_b:.4f}   vs  mean(G_i) = {stats['discounted_return']['mean']:.4f} (+-{se:.3f} s.e.)")

    # sanity: target == behaviour => rho == 1, OIS == WIS == mean(G)
    chk = estimate_target(ds, mdp.b, mdp.gamma, V_b)
    assert abs(chk["rho_mean"] - 1) < 1e-9 and abs(chk["ois"] - chk["wis"]) < 1e-6, "IS sanity check failed"

    # ---------------- (iv) offline control ----------------
    cfgs = build_configs(args.eps)
    Qs, curves, summary = {}, {}, {}
    print("\n== Offline learning (800 episodes, %d seeds, alpha=%g) ==" % (args.n_seeds, args.alpha))
    for name, algo, e, kw in cfgs:
        Q, cur = train_offline(mdp, ds, algo, eps_pi=e, seeds=seeds, n_episodes=args.episodes,
                               alpha=args.alpha, eval_every=args.eval_every, **kw)
        if not np.isfinite(Q).all():
            print(f"  WARNING: {name} diverged (non-finite Q)")
        Qs[name], curves[name] = Q, cur
        summary[name] = dict(algo=algo, eps_pi=e, extra=kw,
                             final_greedy_discounted=ms(cur["Gd"][-1]),
                             final_greedy_undiscounted=ms(cur["Gu"][-1]),
                             n_greedy_nonterminating=int((~cur["term"][-1]).sum()),
                             episodes=cur["episodes"].tolist(),
                             curve_mean=cur["Gd"].mean(1).tolist(), curve_std=cur["Gd"].std(1, ddof=1).tolist())
        f = summary[name]["final_greedy_discounted"]
        print(f"  {name:20s} final greedy return = {f['mean']:8.3f} +- {f['std']:.3f}   "
              f"(non-terminating seeds: {summary[name]['n_greedy_nonterminating']})")

    # ---------------- (ii) + (viii) target-policy evaluation ----------------
    print("\n== Target policies: IS estimates and ratio statistics ==")
    per_policy = {}
    for name, algo, e, kw in cfgs:
        for te in ([e] if e is not None else args.eps):          # QL: evaluate every eps_pi in the grid
            key = f"{name}|target_eps={te}" if e is None else name
            rows = []
            for si in range(len(seeds)):
                pi = eps_greedy_probs(Qs[name][si], te)
                truth = exact_value(mdp, pi)[mdp.start]
                rows.append(estimate_target(ds, pi, mdp.gamma, truth))
            per_policy[key] = {k: ms([r[k] for r in rows]) for k in rows[0]}
            per_policy[key]["eps_pi"] = te
            p = per_policy[key]
            print(f"  {key:32s} truth={p['truth']['mean']:7.3f} OIS={p['ois']['mean']:9.3f} WIS={p['wis']['mean']:7.3f} "
                  f"PDIS={p['pdis']['mean']:9.3f} | mean(rho)={p['rho_mean']['mean']:.3g} Var(rho)={p['rho_var']['mean']:.3g} "
                  f"ESS/N={p['ess_frac']['mean']:.3g}")

    # OIS vs WIS bias/variance resampling for a low- and a high-eps target (QL, seed 0)
    boots = {}
    for te in (min(args.eps), max(args.eps)):
        pi = eps_greedy_probs(Qs["QL"][0], te)
        boots[f"QL seed0, eps_pi={te}"] = bootstrap_comparison(ds, pi, exact_value(mdp, pi)[mdp.start])

    # ---------------- plots ----------------
    viz.plot_return_distribution(ds, stats, plot_dir)
    viz.plot_learning_curves(curves, plot_dir)
    viz.plot_final_returns(summary, plot_dir)
    viz.plot_is_estimates(per_policy, plot_dir)
    viz.plot_ratio_stats(per_policy, plot_dir)
    viz.plot_bootstrap(boots, plot_dir)
    checks_out = None
    if not args.skip_checks:
        import checks
        checks_out = checks.run_checks(mdp, ds, Qs, args.eps, plot_dir, args.alpha)

    # ---------------- results.json ----------------
    out = dict(config=vars(args), mdp=dict(S=mdp.S, A=mdp.A, gamma=mdp.gamma, trunc_limit=mdp.T),
               behaviour=dict(dataset_stats=stats, exact_V_b_s0=V_b),
               training=summary, targets=per_policy, bootstrap=boots, checks=checks_out, runtime_s=time.time() - t0)
    with open(os.path.join(res_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nDone in {time.time() - t0:.1f}s -> results/results.json, plots/")


if __name__ == "__main__":
    main()
