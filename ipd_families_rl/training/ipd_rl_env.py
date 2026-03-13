from dataclasses import dataclass
from typing import Dict, Tuple, Callable, List
import axelrod as axl

from environment import EnvConfig, IPDEnv
from base import StrategyAdapter
from obs import make_multiscale_obs

from ipd.observations import make_tokens_aux

Action = axl.Action

@dataclass
class RLEnvConfig:
    max_len: int = 200


class IPDRLEnv:
    """
    RL wrapper around IPDEnv (Phase 2).
    Agent plays as Player A; opponent is an Axelrod strategy factory.
    """
    def __init__(self, env_cfg: EnvConfig, rl_cfg: RLEnvConfig,
                  obs_mode: str = "vector",
                 aux_windows=(5, 10, 20, 50)
                 ):
        self.env_cfg = env_cfg
        self.rl_cfg = rl_cfg
        self.env = IPDEnv(env_cfg)
        self.history: List[Tuple[Action, Action]] = []
        self.opp = None
        self.done = True
        self.obs_mode = obs_mode
        self.aux_windows = aux_windows

    def _make_obs(self):
        if self.obs_mode == "vector":
            return make_multiscale_obs(self.history, max_len=self.rl_cfg.max_len)
        elif self.obs_mode == "sequence":
            from obs_seq import make_sequence_obs
            return make_sequence_obs(self.history, max_len=self.rl_cfg.max_len)
        elif self.obs_mode == "tokens_aux":
            return make_tokens_aux(self.history, max_len=self.rl_cfg.max_len, windows=self.aux_windows)
        elif self.obs_mode == "tokens":
            d = make_tokens_aux(self.history, max_len=self.rl_cfg.max_len, windows=self.aux_windows)
            return d["tokens"]
        else:
            raise ValueError(self.obs_mode)

    def reset(self, opponent_factory: Callable, seed: int = 0) -> Dict:
        self.env.reset(seed=seed)
        self.history = []
        self.done = False

        self.opp = StrategyAdapter(opponent_factory, seed=seed + 999)

        obs = make_multiscale_obs(self.history, max_len=self.rl_cfg.max_len)
        return {"obs": self._make_obs()}

    def step(self, agent_action: Action):
        if self.done:
            return {"obs": self._make_obs()}, 0.0, True, {"error": "step_called_after_done"}

        b_raw = self.opp.act()
        obs2, (r_a, r_b), done, info = self.env.step(agent_action, b_raw)

        a = info["a"]
        b = info["b"]
        self.opp.record(b, a)

        self.history.append((a, b))
        self.done = bool(done)

        return {"obs": self._make_obs()}, float(r_a), self.done, {"opp_raw": b_raw, "opp": b, "agent": a}
