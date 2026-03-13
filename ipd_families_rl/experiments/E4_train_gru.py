from environment import EnvConfig
from ipd_rl_env import IPDRLEnv, RLEnvConfig
from samplers import OpponentSampler, OpponentSamplerConfig
from q_models_gru import QNetGRU
from trainer import train_ddqn
import torch
import axelrod as axl
def main():
    env_cfg = EnvConfig(p_end=0.1, max_rounds=30, tremble_eps=0.0, seed=0)
    rl_cfg = RLEnvConfig(max_len=200)
    opponents_for_eval = [
        ("TFT", axl.TitForTat),
        ("Defector", axl.Defector),
        ("WSLS", axl.WinStayLoseShift),
        ("Grudger", axl.Grudger),
        ("Random", axl.Random),
    ]

    env = IPDRLEnv(env_cfg, rl_cfg, obs_mode="sequence")

    sampler = OpponentSampler(
        catalog_path="axelrod_catalog_100.json",
        labels_path="families/labels.json",
        cfg=OpponentSamplerConfig(seed=0, mode="uniform_family"),
    )

    online = QNetGRU(hidden_dim=128)
    target = QNetGRU(hidden_dim=128)

    model = train_ddqn(
        env=env,
        sampler=sampler,
        model_online=online,
        model_target=target,
        steps=20000,
        device="cuda",
        opponents_for_eval=opponents_for_eval,
        eval_every=5000,
        eval_seeds_per_opp=20,
        run_name="gru_ddqn_seed0",
    )
    torch.save(model.state_dict(), "checkpoints/ddqn_gru_seed0.pt")

if __name__ == "__main__":
    main()
