import axelrod as axl
from environment import EnvConfig
from match import play_match

cfg = EnvConfig(p_end=0.1, max_rounds=30, tremble_eps=0.01, seed=0)

res = play_match(axl.TitForTat, axl.Defector, cfg=cfg, seed=123)
print("Rounds:", res.rounds)
print("Total:", res.total)
print("Score/move:", res.score_per_move())
print("First 10 actions:", res.actions[:10])
