"""REINFORCE on InvertedPendulum-v5: six configurations (M1-M6), batch of n=16 complete episodes per actor update.

Seeding procedure (recorded as required)
----------------------------------------
* Policy init     : torch.manual_seed(seed) right before building the policy -> all neural methods of a seed start
                    from IDENTICAL parameters.
* Training stream : action noise from torch.Generator seeded 10_000+seed; episode k of the run is reset with
                    env.reset(seed = 1_000_000*seed + k)           (k = 0..6999).
* Evaluation stream: SEPARATE torch.Generator seeded 20_000+seed; the c-th evaluation episode is reset with
                    env.reset(seed = 100_000_000 + 1_000_000*seed + c).   -> evaluation never touches training RNG.
Training / evaluation streams are identical across methods for a given seed, so methods differ only by the estimator.

Implementation rules honoured: no return normalisation, entropy bonus, gradient clipping, observation normalisation or
reward scaling; actor frozen while a batch is collected, ONE Adam update per batch; last batch averaged over its actual
number of episodes; stochastic policy evaluated (not the mean action); evaluation before training, after every 10
actor updates and after the final update; ||g_k||_2 recorded right before every Adam update; ||theta_k||_2 after it.
"""
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from policies import make_policy, param_norm

ENV_ID = "InvertedPendulum-v5"
SIGMA = 0.3
BATCH = 16
EPISODES = 7000
EVAL_EPISODES = 20
EVAL_EVERY = 10          # actor updates
GAMMA = 1.0
LR_TABLE = 0.02          # one fixed LR for M1   (chosen by a pilot, see README)
LR_NEURAL = 1e-3         # one fixed LR for M2-M6

METHODS = {
    "M1": dict(policy="table", weight="full", desc="Discretized policy + full-return REINFORCE"),
    "M2": dict(policy="neural", weight="full", desc="Neural policy + full-return REINFORCE"),
    "M3": dict(policy="neural", weight="loo", desc="Neural policy + leave-one-out batch baseline"),
    "M4": dict(policy="neural", weight="ma", beta=0.01, desc="Neural policy + moving-average baseline, beta=0.01"),
    "M5": dict(policy="neural", weight="ma", beta=0.1, desc="Neural policy + moving-average baseline, beta=0.1"),
    "M6": dict(policy="neural", weight="rtg", desc="Neural policy + return-to-go"),
}


def inspect_env():
    env = gym.make(ENV_ID)
    info = dict(env_id=ENV_ID, obs_dim=int(env.observation_space.shape[0]), act_dim=int(env.action_space.shape[0]),
                action_low=env.action_space.low.tolist(), action_high=env.action_space.high.tolist(),
                max_episode_steps=int(env.spec.max_episode_steps), gamma=GAMMA, sigma=SIGMA)
    env.close()
    return info


def batch_sizes(total, n=BATCH):
    """16,16,...  A final remainder of ONE episode is merged into the preceding batch (n>1 is required for M3);
    otherwise the final batch simply has its actual size (7000 = 437*16 + 8)."""
    sizes = [n] * (total // n)
    r = total % n
    if r == 1 and sizes:
        sizes[-1] += 1
    elif r > 1:
        sizes.append(r)
    return sizes


def train_seed(seed, k):
    return 1_000_000 * seed + k


def eval_seed(seed, c):
    return 100_000_000 + 1_000_000 * seed + c


def rollout(envs, policy, seeds, gen, low, high, record):
    """Run len(seeds) complete episodes in lock-step (one batched policy forward per time step).
    Sampling: u ~ N(mu, sigma^2); the env receives clip(u); the log-prob is later computed on the UNCLIPPED u."""
    n = len(seeds)
    obs = np.stack([envs[i].reset(seed=seeds[i])[0] for i in range(n)])
    ep_obs, ep_u, ep_r = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    alive = list(range(n))
    while alive:
        o = obs[alive].copy()
        with torch.no_grad():
            mu = policy.mean(o)
            u = mu + SIGMA * torch.randn(mu.shape, generator=gen)
            a = u.clamp(low, high).numpy()
        u_np = u.numpy()
        nxt = []
        for j, i in enumerate(alive):
            if record:
                ep_obs[i].append(o[j]); ep_u[i].append(u_np[j])
            o2, r, term, trunc, _ = envs[i].step(a[j])
            ep_r[i].append(r)
            obs[i] = o2
            if not (term or trunc):
                nxt.append(i)
        alive = nxt
    eps = []
    for i in range(n):
        eps.append(dict(rew=np.array(ep_r[i], dtype=np.float64),
                        obs=np.array(ep_obs[i], dtype=np.float64) if record else None,
                        u=np.array(ep_u[i], dtype=np.float32) if record else None))
    return eps


def episode_weights(weight, eps, G, baseline=0.0):
    """Per-timestep weights w_{i,t} for every episode of the batch."""
    n = len(eps)
    if weight == "full":
        return [np.full(len(e["rew"]), G[i]) for i, e in enumerate(eps)]
    if weight == "loo":                                    # b_{-i} = (sum_j G_j - G_i)/(n-1)
        b = (G.sum() - G) / (n - 1)
        return [np.full(len(e["rew"]), G[i] - b[i]) for i, e in enumerate(eps)]
    if weight == "ma":                                     # baseline from PREVIOUS batches only
        return [np.full(len(e["rew"]), G[i] - baseline) for i, e in enumerate(eps)]
    if weight == "rtg":                                    # G_{i,t} via backward pass
        out = []
        for e in eps:
            g, acc = np.zeros(len(e["rew"])), 0.0
            for t in range(len(e["rew"]) - 1, -1, -1):
                acc = e["rew"][t] + GAMMA * acc
                g[t] = acc
            out.append(g)
        return out
    raise ValueError(weight)


def actor_loss(policy, eps, weights):
    """L = -(1/n) sum_i sum_t w_{i,t} log pi(u_{i,t}|s_{i,t}); w treated as a constant; log-prob on unclipped u."""
    obs = np.concatenate([e["obs"] for e in eps])
    u = torch.as_tensor(np.concatenate([e["u"] for e in eps]))
    w = torch.as_tensor(np.concatenate(weights), dtype=torch.float32)
    mu = policy.mean(obs)
    logp = torch.distributions.Normal(mu, SIGMA).log_prob(u).sum(-1)
    return -(w * logp).sum() / len(eps)


def grad_vector(policy):
    return torch.cat([p.grad.reshape(-1) for p in policy.parameters()])


def evaluate(policy, envs, seed, counter, gen, low, high):
    seeds = [eval_seed(seed, counter + i) for i in range(len(envs))]
    eps = rollout(envs, policy, seeds, gen, low, high, record=False)
    return np.array([e["rew"].sum() for e in eps]), counter + len(envs)


def train_run(method, seed, out_dir, episodes=EPISODES, lr_table=LR_TABLE, lr_neural=LR_NEURAL, verbose=True):
    torch.set_num_threads(1)
    cfg = METHODS[method]
    info = inspect_env()
    lr = lr_table if cfg["policy"] == "table" else lr_neural
    policy = make_policy(cfg["policy"], seed, info["obs_dim"], info["act_dim"])
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    gen_tr = torch.Generator().manual_seed(10_000 + seed)
    gen_ev = torch.Generator().manual_seed(20_000 + seed)
    low, high = info["action_low"][0], info["action_high"][0]
    sizes = batch_sizes(episodes)
    tr_envs = [gym.make(ENV_ID) for _ in range(max(sizes))]
    ev_envs = [gym.make(ENV_ID) for _ in range(EVAL_EPISODES)]

    train_returns, upd_eps, gnorm, pnorm = [], [], [], []
    ev_eps, ev_ret, ev_counter = [], [], 0
    t0 = time.time()

    def do_eval(done):
        nonlocal ev_counter
        r, ev_counter = evaluate(policy, ev_envs, seed, ev_counter, gen_ev, low, high)
        ev_eps.append(done); ev_ret.append(r)

    do_eval(0)
    baseline, done = 0.0, 0
    for k, n in enumerate(sizes):
        seeds = [train_seed(seed, done + i) for i in range(n)]
        eps = rollout(tr_envs, policy, seeds, gen_tr, low, high, record=True)
        G = np.array([e["rew"].sum() for e in eps])
        train_returns.extend(G.tolist())
        w = episode_weights(cfg["weight"], eps, G, baseline)
        opt.zero_grad()
        actor_loss(policy, eps, w).backward()
        gnorm.append(float(torch.linalg.norm(grad_vector(policy))))     # ||g_k||_2 right before the Adam step
        opt.step()
        if cfg["weight"] == "ma":                                         # update baseline AFTER the actor update
            baseline = (1 - cfg["beta"]) * baseline + cfg["beta"] * G.mean()
        done += n
        upd_eps.append(done); pnorm.append(param_norm(policy))
        if (k + 1) % EVAL_EVERY == 0 or k == len(sizes) - 1:
            do_eval(done)
            if verbose:
                print(f"[{method} seed {seed}] episodes {done:5d}/{episodes}  train(last batch) {G.mean():7.1f}  "
                      f"eval {ev_ret[-1].mean():7.1f}  |g| {gnorm[-1]:.3g}  |theta| {pnorm[-1]:.3g}  "
                      f"{time.time() - t0:.0f}s", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{method}_seed{seed}.npz"),
             train_returns=np.array(train_returns), update_episodes=np.array(upd_eps), grad_norm=np.array(gnorm),
             param_norm=np.array(pnorm), eval_episodes=np.array(ev_eps), eval_returns=np.array(ev_ret),
             meta=json.dumps(dict(method=method, seed=seed, lr=lr, episodes=episodes, batch=BATCH, sigma=SIGMA,
                                  policy=cfg["policy"], weight=cfg["weight"], beta=cfg.get("beta"),
                                  runtime_s=time.time() - t0)))
    torch.save(dict(kind=cfg["policy"], state_dict=policy.state_dict()), os.path.join(out_dir, f"{method}_seed{seed}_params.pt"))
    for e in tr_envs + ev_envs:
        e.close()
    return method, seed


def check_loo_equivalence(out_dir, seed=0):
    """M3 requirement: leave-one-out weights  w_i = G_i - b_{-i}  must give the SAME gradient as the equivalent
    batch-mean form  w_i = n/(n-1) * (G_i - mean(G)).  Verified numerically on a real 16-episode batch."""
    torch.set_num_threads(1)
    info = inspect_env()
    torch.manual_seed(seed)
    policy = make_policy("neural", seed, info["obs_dim"], info["act_dim"])
    with torch.no_grad():                                   # non-zero output layer so every layer gets a gradient
        policy.net[-1].weight.normal_(0, 0.3)
    gen = torch.Generator().manual_seed(123)
    envs = [gym.make(ENV_ID) for _ in range(BATCH)]
    eps = rollout(envs, policy, list(range(BATCH)), gen, info["action_low"][0], info["action_high"][0], record=True)
    G = np.array([e["rew"].sum() for e in eps]); n = len(eps)
    grads = []
    for weights in (episode_weights("loo", eps, G),
                    [np.full(len(e["rew"]), n / (n - 1) * (G[i] - G.mean())) for i, e in enumerate(eps)]):
        policy.zero_grad(); actor_loss(policy, eps, weights).backward()
        grads.append(grad_vector(policy).clone())
    diff = float((grads[0] - grads[1]).abs().max()); scale = float(grads[0].abs().max())
    res = dict(n=n, returns=G.tolist(), max_abs_grad_diff=diff, max_abs_grad=scale, relative_diff=diff / max(scale, 1e-30),
               allclose=bool(torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-5 * scale)))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "loo_gradient_check.json"), "w"), indent=1)
    for e in envs:
        e.close()
    return res
