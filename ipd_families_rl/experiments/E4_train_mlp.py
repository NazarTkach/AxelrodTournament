from environment import EnvConfig
from ipd_rl_env import IPDRLEnv, RLEnvConfig
from samplers import OpponentSampler, OpponentSamplerConfig

from q_models import QNetMLP
from trainer import train_ddqn


def main():
    env_cfg = EnvConfig(p_end=0.1, max_rounds=30, tremble_eps=0.0, seed=0)
    rl_cfg = RLEnvConfig(max_len=200)
    env = IPDRLEnv(env_cfg, rl_cfg)

    sampler = OpponentSampler(
        catalog_path="axelrod_catalog_100.json",
        labels_path="families/labels.json",
        cfg=OpponentSamplerConfig(seed=0, mode="uniform_family"),
    )

    # obs_dim = 2*max_len + 2*len(windows)
    max_len = rl_cfg.max_len
    windows = (1, 5, 10, 20, 50)
    obs_dim = 2 * max_len + 2 * len(windows)

    online = QNetMLP(obs_dim=obs_dim, hidden=256)
    target = QNetMLP(obs_dim=obs_dim, hidden=256)

    model = train_ddqn(
        env=env,
        sampler=sampler,
        model_online=online,
        model_target=target,
        steps=50_000,
        seed=0,
        device="cpu",  # set to "cuda" if available
    )

    # Save checkpoint (simple)
    import torch
    torch.save(model.state_dict(), "checkpoints/ddqn_mlp_seed0.pt")
    print("saved checkpoints/ddqn_mlp_seed0.pt")


if __name__ == "__main__":
    main()
