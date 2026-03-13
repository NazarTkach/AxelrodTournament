from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Type
import axelrod as axl
import random

Action = axl.Action



def _ensure_rng_recursive(p: axl.Player, seen=None) -> None:
    """
    Ensure `_random` exists on p and on any nested sub-players used by meta strategies.

    We walk common container attributes used by Axelrod meta strategies:
      - players / team / team_members
      - player / opponent_model / classifier_player / etc.
      - any attribute that is an axl.Player
      - any list/tuple of axl.Player
    """
    if seen is None:
        seen = set()

    pid = id(p)
    if pid in seen:
        return
    seen.add(pid)

    # Ensure RNG on this player
    if not hasattr(p, "_random") or getattr(p, "_random") is None:
        try:
            p._random = axl.RandomGenerator()
        except Exception:
            p._random = random.Random()

    # Common attribute names used by meta strategies
    candidate_attrs = [
        "players", "team", "team_members", "strategies",
        "player", "inner_player", "wrapped_player",
        "model", "opponent_model", "classifier_player",
    ]

    for attr in candidate_attrs:
        if hasattr(p, attr):
            obj = getattr(p, attr)
            _walk_nested(obj, seen)

    # Generic fallback: inspect __dict__ but keep it lightweight
    # (avoids missing weird internal names)
    try:
        for _, obj in vars(p).items():
            _walk_nested(obj, seen)
    except Exception:
        pass


def _walk_nested(obj, seen) -> None:
    import axelrod as axl
    if isinstance(obj, axl.Player):
        _ensure_rng_recursive(obj, seen)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            if isinstance(x, axl.Player):
                _ensure_rng_recursive(x, seen)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    qualname: str
    is_stochastic: Optional[bool]
    memory_depth: Optional[int]
    classifier: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def resolve_strategy(qualname: str):
    """
    Resolve a strategy from qualname only.
    Returns a FACTORY (PostInitCaller or callable) that produces axl.Player.
    """
    for obj in axl.all_strategies:
        try:
            player = obj() if callable(obj) else None
        except Exception:
            continue

        if player is None:
            continue

        cls = type(player)
        if cls.__qualname__ == qualname:
            return obj  # IMPORTANT: return the factory, not the instance

    raise KeyError(f"Strategy with qualname '{qualname}' not found in axl.all_strategies")



def build_spec(obj) -> Optional[StrategySpec]:
    """
    Axelrod version note:
    - axl.all_strategies may contain PostInitCaller factories.
    - This function accepts:
        * Player instance
        * Player class
        * callable factory (e.g., PostInitCaller) that returns a Player
    Returns None for unusable objects.
    """
    player = None

    # Case 1: already an instance
    if isinstance(obj, axl.Player):
        player = obj

    # Case 2: class
    elif isinstance(obj, type) and issubclass(obj, axl.Player):
        try:
            player = obj()
        except Exception:
            return None

    # Case 3: factory / PostInitCaller
    elif callable(obj):
        try:
            player = obj()  # <-- critical for PostInitCaller
        except Exception:
            return None
        if not isinstance(player, axl.Player):
            return None
    else:
        return None

    cls: Type[axl.Player] = type(player)

    name = getattr(player, "name", None)
    if name is None:
        return None

    classifier = None
    is_stochastic = None
    memory_depth = None

    classifier_obj = getattr(cls, "classifier", None)
    if classifier_obj is not None:
        try:
            classifier = dict(classifier_obj)
            is_stochastic = classifier.get("stochastic")
            memory_depth = classifier.get("memory_depth")
        except Exception:
            classifier = None

    return StrategySpec(
        name=str(name),
        qualname=str(cls.__qualname__),  # no module path
        is_stochastic=is_stochastic,
        memory_depth=memory_depth,
        classifier=classifier,
    )

class StrategyAdapter:
    def __init__(self, strategy_obj, seed: Optional[int] = None):
        self.seed = seed
        self.player = self._make_player(strategy_obj)
        self.opponent = axl.Player()

        # Reset FIRST: many strategies initialize internal state/RNG here
        self.reset()

        # Then seed / ensure RNG exists
        if seed is not None:

            self._try_seed(self.player, seed)
            self._try_seed(self.opponent, seed + 1)

    @staticmethod
    def _make_player(strategy_obj) -> axl.Player:
        if isinstance(strategy_obj, axl.Player):
            return strategy_obj
        if isinstance(strategy_obj, type) and issubclass(strategy_obj, axl.Player):
            return strategy_obj()
        if callable(strategy_obj):
            p = strategy_obj()  # PostInitCaller factory
            if not isinstance(p, axl.Player):
                raise TypeError(f"Factory did not return axl.Player: {type(p)}")
            return p
        raise TypeError(f"Unsupported strategy type: {type(strategy_obj)}")

    def reset(self) -> None:
        self.player.reset()
        self.opponent.reset()
        # After reset, ensure RNG exists for stochastic strategies
        _ensure_rng_recursive(self.player)
        _ensure_rng_recursive(self.opponent)

    def act(self) -> Action:
        # Some meta strategies create nested players lazily; keep RNG sane.
        _ensure_rng_recursive(self.player)

        try:
            a = self.player.strategy(self.opponent)
        except IndexError:
            # First-move bug / empty history access in some strategies.
            a = self._initial_action_fallback()

        if a in (axl.Action.C, axl.Action.D):
            return a

        s = str(a).upper()
        if s.startswith("C"):
            return axl.Action.C
        if s.startswith("D"):
            return axl.Action.D
        raise ValueError(f"Invalid action from {getattr(self.player, 'name', type(self.player))}: {a}")

    def _initial_action_fallback(self) -> Action:
        """
        Best-effort initial action.
        Prefer strategy's declared initial action if available; otherwise cooperate.
        """
        # Some strategies expose initial_action on instance
        ia = getattr(self.player, "initial_action", None)
        if ia in (axl.Action.C, axl.Action.D):
            return ia

        # Many strategies store it in classifier
        cls = type(self.player)
        classifier = getattr(cls, "classifier", None)
        if classifier and isinstance(classifier, dict):
            ia = classifier.get("initial_action", None)
            if ia in (axl.Action.C, axl.Action.D):
                return ia

        return axl.Action.C

    def record(self, my_action: Action, opp_action: Action) -> None:
        """
        Your Axelrod History requires append(play, coplay).
        """
        h1 = self.player.history
        h2 = self.opponent.history
        try:
            h1.append(my_action, opp_action)
            h2.append(opp_action, my_action)
        except TypeError:
            # fallback for list-like histories
            h1.append(my_action)
            h2.append(opp_action)

    @staticmethod
    def _ensure_rng(p: axl.Player) -> None:
        """
        Ensure `p._random` exists for stochastic strategies like axl.Random.
        """
        if not hasattr(p, "_random") or getattr(p, "_random") is None:
            # Axelrod provides RandomGenerator; if not, fall back to python Random
            try:
                p._random = axl.RandomGenerator()
            except Exception:
                p._random = random.Random()

    @staticmethod
    def _try_seed(p: axl.Player, seed: int) -> None:
        """
        Best-effort seeding: handle seed(), _random, and python Random.
        """
        # Preferred: Player.seed()
        if hasattr(p, "seed"):
            try:
                p.seed(seed)
                return
            except Exception:
                pass

        # Otherwise seed its RNG
        if hasattr(p, "_random") and getattr(p, "_random") is not None:
            r = p._random
            # Axelrod RandomGenerator supports .seed; python Random too
            if hasattr(r, "seed"):
                try:
                    r.seed(seed)
                    return
                except Exception:
                    pass