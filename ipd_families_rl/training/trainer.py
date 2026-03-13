import os
import json
import time
import random
import numpy as np
import torch
import axelrod as axl

from replay import ReplayBuffer, Transition
from losses import ddqn_loss

Action = axl.Action


def int_to_action(i: int) -> Action:
    return axl.Action.C if i == 1 else axl.Action.D


def linear_epsilon(t: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    frac = min(1.0, t / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def _single_obs_to_torch(s, device):
    # Dict obs (Layer 5): {"tokens": np.ndarray, "aux": np.ndarray}
    if isinstance(s, dict):
        return {
            "tokens": torch.as_tensor(s["tokens"], dtype=torch.long, device=device).unsqueeze(0),
            "aux": torch.as_tensor(s["aux"], dtype=torch.float32, device=device).unsqueeze(0),
        }

    # Already a torch tensor
    if torch.is_tensor(s):
        return s.unsqueeze(0) if s.dim() == 1 else s  # ensure batch dim

    # Numpy array obs
    if isinstance(s, np.ndarray) and np.issubdtype(s.dtype, np.integer):
        return torch.as_tensor(s, dtype=torch.long, device=device).unsqueeze(0)

    return torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)


def evaluate_policy(
    env,
    model,
    opponents,          # list of (name, opponent_factory)
    *,
    seeds_per_opp=20,
    base_seed=0,
    device="cpu",
):
    """
    Quick sanity evaluation: average reward per step against a small opponent set.
    Assumes env.reset(opponent_factory, seed) and env.step(Action) -> (st2, r, done, info)
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for (name, opp_factory) in opponents:
            total_r = 0.0
            total_steps = 0

            for k in range(seeds_per_opp):
                st = env.reset(opp_factory, seed=base_seed + 1000 * k)
                s = st["obs"]
                done = False

                while not done:
                    # obs_t = _obs_tensor(s, device).unsqueeze(0)
                    # q = model.forward_obs(obs_t)
                    obs_t = _single_obs_to_torch(s, device)
                    q = model.forward_obs(obs_t)
                    a_int = int(torch.argmax(q, dim=1).item())

                    st2, r, done, info = env.step(int_to_action(a_int))
                    total_r += float(r)
                    total_steps += 1
                    s = st2["obs"]

            results[name] = {
                "reward_per_step": total_r / max(1, total_steps),
                "steps": int(total_steps),
            }

    model.train()
    return results


def train_ddqn(
    env,
    sampler,
    model_online,
    model_target,
    *,
    steps: int = 200_000,
    start_learning: int = 5_000,
    batch_size: int = 128,
    buffer_size: int = 200_000,
    gamma: float = 0.96,
    lr: float = 3e-4,
    target_update: int = 2000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 100_000,
    grad_clip: float = 10.0,
    seed: int = 0,
    device: str = "cpu",
    # logging/eval/checkpoint
    log_every: int = 5000,
    eval_every: int = 20000,
    opponents_for_eval=None,   # list[(name, factory)] or None
    eval_seeds_per_opp: int = 10,
    ckpt_dir: str = "checkpoints",
    run_name: str = "ddqn_run",
):
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(ckpt_dir, exist_ok=True)

    model_online.to(device)
    model_target.to(device)
    model_target.load_state_dict(model_online.state_dict())
    model_target.eval()

    opt = torch.optim.Adam(model_online.parameters(), lr=lr)
    rb = ReplayBuffer(buffer_size, seed=seed)

    # Episode init
    fam, opp = sampler.sample()
    st = env.reset(opp, seed=seed)
    s = st["obs"]

    # Diagnostics
    ema_beta = 0.99
    ema_r = 0.0
    ema_init = False

    window = 2000
    a_hist = np.zeros(window, dtype=np.int64)
    r_hist = np.zeros(window, dtype=np.float32)
    wi = 0

    ep_return = 0.0
    ep_len = 0
    ep_count = 0

    t0 = time.time()

    # Save config snapshot
    cfg = dict(
        steps=steps,
        start_learning=start_learning,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        lr=lr,
        target_update=target_update,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_steps=eps_decay_steps,
        grad_clip=grad_clip,
        seed=seed,
        device=device,
        run_name=run_name,
    )
    with open(os.path.join(ckpt_dir, f"{run_name}_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    for t in range(1, steps + 1):
        eps = linear_epsilon(t, eps_start, eps_end, eps_decay_steps)

        # choose action
        if rng.random() < eps:
            a_int = rng.choice([0, 1])
        else:
            with torch.no_grad():
                # obs_t = _obs_tensor(s, device).unsqueeze(0)
                # q = model_online.forward_obs(obs_t)
                obs_t = _single_obs_to_torch(s, device)
                q  = model_online.forward_obs(obs_t)
                a_int = int(torch.argmax(q, dim=1).item())

        a = int_to_action(a_int)
        st2, r, done, info = env.step(a)
        s2 = st2["obs"]

        # store transition
        rb.add(Transition(s=s, a=a_int, r=float(r), s2=s2, done=bool(done)))
        s = s2

        # diagnostics update
        a_hist[wi % window] = a_int
        r_hist[wi % window] = float(r)
        wi += 1

        if not ema_init:
            ema_r = float(r)
            ema_init = True
        else:
            ema_r = ema_beta * ema_r + (1.0 - ema_beta) * float(r)

        ep_return += float(r)
        ep_len += 1

        # reset episode
        if done:
            ep_count += 1
            fam, opp = sampler.sample()
            st = env.reset(opp, seed=seed + t)
            s = st["obs"]

            ep_return = 0.0
            ep_len = 0

        # learn
        if len(rb) >= start_learning and len(rb) >= batch_size:
            batch = rb.sample(batch_size)
            loss = ddqn_loss(model_online, model_target, batch, gamma, device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_online.parameters(), grad_clip)
            opt.step()

        # target update
        if t % target_update == 0:
            model_target.load_state_dict(model_online.state_dict())

        # logging
        if t % log_every == 0:
            # p(C) where action 1 is C by your convention
            n = min(window, wi)
            pC = float(a_hist[:n].mean()) if n > 0 else 0.0
            rmean = float(r_hist[:n].mean()) if n > 0 else 0.0
            dt = time.time() - t0
            print(
                f"t={t} eps={eps:.3f} buffer={len(rb)} "
                f"ema_r={ema_r:.3f} r_mean={rmean:.3f} pC={pC:.3f} "
                f"episodes={ep_count} time={dt:.1f}s"
            )

        # evaluation + checkpoint
        if opponents_for_eval is not None and (t % eval_every == 0):
            eval_res = evaluate_policy(
                env=env,
                model=model_online,
                opponents=opponents_for_eval,
                seeds_per_opp=eval_seeds_per_opp,
                base_seed=seed + 10_000_000 + t,
                device=device,
            )

            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_t{t}.pt")
            torch.save(
                {
                    "t": t,
                    "model_state": model_online.state_dict(),
                    "opt_state": opt.state_dict(),
                    "cfg": cfg,
                    "eval": eval_res,
                },
                ckpt_path,
            )

            with open(os.path.join(ckpt_dir, f"{run_name}_t{t}_eval.json"), "w", encoding="utf-8") as f:
                json.dump(eval_res, f, indent=2)

            print("saved:", ckpt_path)
            for k, v in eval_res.items():
                print(f"  eval {k}: {v['reward_per_step']:.3f} (steps={v['steps']})")

    return model_online
