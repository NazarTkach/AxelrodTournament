# ipd_families_rl/experiments/E5_train_attn_serious.py
"""
Layer 5 serious run: Transformer + multiscale aux, dozens of opponents, hours-long training.

Usage:
  python experiments/E5_train_attn_serious.py --seed 0 --device cuda \
      --train_list experiments/sets/train_60.json \
      --test_list  experiments/sets/test_20.json \
      --out_dir checkpoints

Minimal assumptions:
- training.ipd_rl_env.IPDRLEnv supports obs_mode="tokens_aux"
- agents.ddqn_atten.QNetMultiScale exists and expects dict obs {"tokens","aux"}
- training.trainer.train_ddqn exists (we won't rely on its built-in eval; we do our own)
- training.replay.Transition / ReplayBuffer exists
- training.losses.ddqn_loss exists and supports dict obs (your patched version)
- strategies.base.resolve_strategy can resolve qualname to factory (or you can plug your resolver)
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import axelrod as axl

from agents.ddqn_atten import QNetMultiScale
from training.replay import ReplayBuffer, Transition
from training.losses import ddqn_loss
from training.ipd_rl_env import IPDRLEnv

# Adjust these imports to your project:
try:
    from strategies.base import resolve_strategy  # qualname -> factory
except Exception:
    resolve_strategy = None

# If you have config dataclasses, import them; otherwise keep simple dicts.
# Example placeholders:
try:
    from ipd.environment import EnvConfig
except Exception:
    EnvConfig = None

try:
    from training.ipd_rl_env import RLEnvConfig
except Exception:
    RLEnvConfig = None


Action = axl.Action


def int_to_action(i: int) -> Action:
    return axl.Action.C if i == 1 else axl.Action.D


def linear_epsilon(t: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    frac = min(1.0, t / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def load_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Expected a JSON list"
    # each item minimally: {"qualname": "..."}; optional: name, family, type
    for it in data:
        assert "qualname" in it, f"Missing qualname in {it}"
    return data


def resolve_factory(qualname: str):
    if resolve_strategy is None:
        raise RuntimeError(
            "resolve_strategy not available. Implement strategies.base.resolve_strategy(qualname)->factory "
            "or change resolve_factory() to your resolver."
        )
    return resolve_strategy(qualname)


@dataclass
class SamplerCfg:
    mode: str = "uniform"  # "uniform" or "balanced_type"
    curriculum_switch_step: int = 50_000
    seed: int = 0


class ListSampler:
    """
    Draw opponents from a provided list of strategy qualnames.
    Optional 'type' field can be used for balancing (deterministic/stochastic/automata).
    """

    def __init__(self, items: List[Dict[str, Any]], cfg: SamplerCfg):
        self.items = items
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Build groups for simple balancing if "type" exists
        self.by_type: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            t = it.get("type", "unknown")
            self.by_type.setdefault(t, []).append(it)

        self.types = sorted(self.by_type.keys())

    def sample(self, step: int) -> Tuple[str, Any]:
        # curriculum: uniform first, then optionally balanced over types
        if self.cfg.mode == "uniform" or step < self.cfg.curriculum_switch_step:
            it = self.rng.choice(self.items)
            return it.get("family", "NA"), resolve_factory(it["qualname"])

        if self.cfg.mode == "balanced_type":
            t = self.rng.choice(self.types)
            it = self.rng.choice(self.by_type[t])
            return it.get("family", "NA"), resolve_factory(it["qualname"])

        raise ValueError(self.cfg.mode)


def _single_obs_to_torch(s, device):
    # dict obs: {"tokens": np.int64[L], "aux": np.float32[d]}
    if isinstance(s, dict):
        return {
            "tokens": torch.as_tensor(s["tokens"], dtype=torch.long, device=device).unsqueeze(0),
            "aux": torch.as_tensor(s["aux"], dtype=torch.float32, device=device).unsqueeze(0),
        }
    # ndarray obs
    if isinstance(s, np.ndarray) and np.issubdtype(s.dtype, np.integer):
        return torch.as_tensor(s, dtype=torch.long, device=device).unsqueeze(0)
    return torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)


@torch.no_grad()
def evaluate_against_list(
    env: IPDRLEnv,
    model: torch.nn.Module,
    items: List[Dict[str, Any]],
    *,
    seeds_per_opp: int,
    base_seed: int,
    device: str,
) -> Dict[str, Any]:
    """
    Returns a summary dict with per-opponent stats + aggregates.
    We measure:
      - reward_per_step
      - cooperation_rate (agent C frequency)
      - exploitation_rate (agent D when opp C)
    """
    model.eval()

    per = []
    for idx, it in enumerate(items):
        opp_factory = resolve_factory(it["qualname"])

        total_r = 0.0
        total_steps = 0
        total_C = 0
        total_exploit = 0
        total_oppC = 0

        for k in range(seeds_per_opp):
            seed = base_seed + 100_000 * idx + k
            st = env.reset(opp_factory, seed=seed)
            s = st["obs"]
            done = False

            while not done:
                obs_t = _single_obs_to_torch(s, device)
                q = model.forward_obs(obs_t)
                a_int = int(torch.argmax(q, dim=1).item())
                a = int_to_action(a_int)

                st2, r, done, info = env.step(a)
                # info: {"opp": b, "agent": a, ...} per your env
                b = info.get("opp", None)
                a_real = info.get("agent", a)

                total_r += float(r)
                total_steps += 1
                total_C += 1 if a_real == axl.Action.C else 0

                if b == axl.Action.C:
                    total_oppC += 1
                    if a_real == axl.Action.D:
                        total_exploit += 1

                s = st2["obs"]

        per.append(
            dict(
                name=it.get("name", it["qualname"]),
                qualname=it["qualname"],
                family=it.get("family", "NA"),
                type=it.get("type", "NA"),
                reward_per_step=total_r / max(1, total_steps),
                coop_rate=total_C / max(1, total_steps),
                exploit_rate=(total_exploit / max(1, total_oppC)) if total_oppC > 0 else 0.0,
                steps=int(total_steps),
            )
        )

    # aggregates
    rewards = np.array([x["reward_per_step"] for x in per], dtype=np.float32)
    summary = dict(
        mean=float(rewards.mean()) if len(rewards) else 0.0,
        p10=float(np.percentile(rewards, 10)) if len(rewards) else 0.0,
        median=float(np.median(rewards)) if len(rewards) else 0.0,
        n=int(len(per)),
    )

    model.train()
    return {"summary": summary, "per_opponent": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train_list", type=str, default="train_resolved.json")
    ap.add_argument("--test_list", type=str, default="test_resolved.json")
    ap.add_argument("--out_dir", type=str, default="checkpoints")

    # env
    ap.add_argument("--max_rounds", type=int, default=200)
    ap.add_argument("--p_end", type=float, default=0.02)
    ap.add_argument("--tremble_eps", type=float, default=0.0)

    # obs/model
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--aux_windows", type=str, default="5,10,20,50")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--ff_dim", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # training
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--start_learning", type=int, default=10_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--buffer_size", type=int, default=300_000)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target_update", type=int, default=5_000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.02)
    ap.add_argument("--eps_decay_steps", type=int, default=150_000)
    ap.add_argument("--grad_clip", type=float, default=10.0)

    # logging/eval
    ap.add_argument("--log_every", type=int, default=5_000)
    ap.add_argument("--eval_every", type=int, default=20_000)
    ap.add_argument("--eval_seeds_train", type=int, default=10)
    ap.add_argument("--eval_seeds_test", type=int, default=20)

    # sampling curriculum
    ap.add_argument("--sampler_mode", type=str, default="balanced_type")  # or "uniform"
    ap.add_argument("--curriculum_switch", type=int, default=50_000)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # reproducibility
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_items = load_list(args.train_list)
    test_items = load_list(args.test_list)

    aux_windows = tuple(int(x) for x in args.aux_windows.split(",") if x.strip())
    aux_dim = 4 * len(aux_windows) + 4  # rates only

    # env configs
    if EnvConfig is not None and RLEnvConfig is not None:
        env_cfg = EnvConfig(p_end=args.p_end, max_rounds=args.max_rounds, tremble_eps=args.tremble_eps, seed=args.seed)
        rl_cfg = RLEnvConfig(max_len=args.max_len)
        env = IPDRLEnv(env_cfg, rl_cfg, obs_mode="tokens_aux", aux_windows=aux_windows)
    else:
        # If your project uses plain dicts, adapt here.
        # This block is intentionally strict so you don't silently run wrong configs.
        raise RuntimeError("EnvConfig/RLEnvConfig not found. Adapt env construction to your project.")

    sampler = ListSampler(
        train_items,
        SamplerCfg(
            mode=args.sampler_mode,
            curriculum_switch_step=args.curriculum_switch,
            seed=args.seed,
        ),
    )

    # model
    model_online = QNetMultiScale(
        max_len=args.max_len,
        aux_dim=aux_dim,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    model_target = QNetMultiScale(
        max_len=args.max_len,
        aux_dim=aux_dim,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    model_target.load_state_dict(model_online.state_dict())
    model_target.eval()

    opt = torch.optim.Adam(model_online.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size, seed=args.seed)

    # run metadata
    run_name = f"E5_attn_serious_seed{args.seed}"
    meta_path = os.path.join(args.out_dir, f"{run_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # init episode
    fam, opp_factory = sampler.sample(step=0)
    st = env.reset(opp_factory, seed=args.seed)
    s = st["obs"]

    # diagnostics
    window = 4000
    a_hist = np.zeros(window, dtype=np.int64)
    r_hist = np.zeros(window, dtype=np.float32)
    wi = 0
    ema_beta = 0.99
    ema_r = 0.0
    ema_init = False
    t0 = time.time()

    for t in range(1, args.steps + 1):
        eps = linear_epsilon(t, args.eps_start, args.eps_end, args.eps_decay_steps)

        # choose action
        if rng.random() < eps:
            a_int = rng.choice([0, 1])
        else:
            with torch.no_grad():
                obs_t = _single_obs_to_torch(s, device)
                q = model_online.forward_obs(obs_t)
                a_int = int(torch.argmax(q, dim=1).item())

        a = int_to_action(a_int)
        st2, r, done, info = env.step(a)
        s2 = st2["obs"]

        rb.add(Transition(s=s, a=a_int, r=float(r), s2=s2, done=bool(done)))
        s = s2

        # diagnostics
        a_hist[wi % window] = a_int
        r_hist[wi % window] = float(r)
        wi += 1
        if not ema_init:
            ema_r = float(r)
            ema_init = True
        else:
            ema_r = ema_beta * ema_r + (1.0 - ema_beta) * float(r)

        # reset episode
        if done:
            fam, opp_factory = sampler.sample(step=t)
            st = env.reset(opp_factory, seed=args.seed + t)
            s = st["obs"]

        # learn
        if len(rb) >= args.start_learning and len(rb) >= args.batch_size:
            batch = rb.sample(args.batch_size)
            loss = ddqn_loss(model_online, model_target, batch, args.gamma, device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_online.parameters(), args.grad_clip)
            opt.step()

        # target update
        if t % args.target_update == 0:
            model_target.load_state_dict(model_online.state_dict())

        # logs
        if t % args.log_every == 0:
            n = min(window, wi)
            pC = float(a_hist[:n].mean()) if n > 0 else 0.0
            rmean = float(r_hist[:n].mean()) if n > 0 else 0.0
            dt = time.time() - t0
            print(
                f"t={t} eps={eps:.3f} buffer={len(rb)} "
                f"ema_r={ema_r:.3f} r_mean={rmean:.3f} pC={pC:.3f} "
                f"time={dt/60:.1f}min"
            )

        # eval + checkpoint
        if t % args.eval_every == 0:
            base_seed = args.seed + 10_000_000 + t
            eval_train = evaluate_against_list(
                env, model_online, train_items,
                seeds_per_opp=args.eval_seeds_train,
                base_seed=base_seed,
                device=device,
            )
            eval_test = evaluate_against_list(
                env, model_online, test_items,
                seeds_per_opp=args.eval_seeds_test,
                base_seed=base_seed + 777_777,
                device=device,
            )

            ckpt_path = os.path.join(args.out_dir, f"{run_name}_t{t}.pt")
            torch.save(
                {
                    "t": t,
                    "model_state": model_online.state_dict(),
                    "target_state": model_target.state_dict(),
                    "opt_state": opt.state_dict(),
                    "args": vars(args),
                    "eval_train": eval_train["summary"],
                    "eval_test": eval_test["summary"],
                },
                ckpt_path,
            )

            with open(os.path.join(args.out_dir, f"{run_name}_t{t}_eval_train.json"), "w", encoding="utf-8") as f:
                json.dump(eval_train, f, indent=2)
            with open(os.path.join(args.out_dir, f"{run_name}_t{t}_eval_test.json"), "w", encoding="utf-8") as f:
                json.dump(eval_test, f, indent=2)

            print("saved:", ckpt_path)
            print("  train:", eval_train["summary"])
            print("  test :", eval_test["summary"])

    print("done.")


if __name__ == "__main__":
    main()
