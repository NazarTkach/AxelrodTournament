import torch
import axelrod as axl

from agents.ddqn_atten import QNetMultiScale
from training.trainer import train_ddqn
from training.ipd_rl_env import IPDRLEnv
from training.losses import ddqn_loss

# use your existing configs + sampler
from ipd.environment import EnvConfig
from training.ipd_rl_env import RLEnvConfig
from training.samplers import OpponentSampler, OpponentSamplerConfig


def main():
    env_cfg = EnvConfig(p_end=0.1, max_rounds=50, tremble_eps=0.0, seed=0)
    rl_cfg = RLEnvConfig(max_len=200)

    env = IPDRLEnv(env_cfg, rl_cfg, obs_mode="tokens_aux", aux_windows=(5, 10, 20, 50))

    sampler = OpponentSampler(
        catalog_path="axelrod_catalog_100.json",
        labels_path="families/labels.json",
        cfg=OpponentSamplerConfig(seed=0, mode="uniform_family"),
    )

    aux_dim = 4 * 4 + 4  # 4 windows * 4 rates + 4 global = 20
    online = QNetMultiScale(max_len=rl_cfg.max_len, aux_dim=aux_dim, embed_dim=64, n_heads=4, n_layers=2)
    target = QNetMultiScale(max_len=rl_cfg.max_len, aux_dim=aux_dim, embed_dim=64, n_heads=4, n_layers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # run a short training to validate end-to-end
    train_ddqn(
        env=env,
        sampler=sampler,
        model_online=online,
        model_target=target,
        steps=2000,
        start_learning=200,
        batch_size=64,
        buffer_size=5000,
        target_update=500,
        eps_decay_steps=1500,
        seed=0,
        device=device,
        log_every=500,
        eval_every=1000,
        opponents_for_eval=[
            ("TFT", axl.TitForTat),
            ("Defector", axl.Defector),
            ("WSLS", axl.WinStayLoseShift),
        ],
        eval_seeds_per_opp=5,
        run_name="attn_smoketest",
    )

    print("OK: transformer smoketest completed")


if __name__ == "__main__":
    main()
