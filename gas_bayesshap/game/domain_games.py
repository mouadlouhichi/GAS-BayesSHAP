"""Domain game formulations (spec section 3).

Only the games declared in the specification are implemented:

A. Primary membership attribution   ``v(S) in [0, 1]``   ``R_delta_res = 4``
B. Contrastive regime               ``v(S) in [-1, 1]``  ``R_delta_res = 8``
C. Global archetype                 ``v(S) in [0, 1]``   ``R_delta_res = 4``
D. Intrinsic silhouette quality     ``v(S) in [-1, 1]``  ``R_delta_res = 8``
E. Group-lag spatiotemporal game    M=66 -> M_group=11   exact at 2^11

All games use interventional (marginal) background imputation and are
strictly deterministic given their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..utils.hashing import cache_key, input_hash
from .oracle import CoalitionOracle

R_DELTA_MEMBERSHIP = 4.0
R_DELTA_CONTRASTIVE = 8.0
R_DELTA_ARCHETYPE = 4.0
R_DELTA_SILHOUETTE = 8.0

BOUNDS_MEMBERSHIP = (0.0, 1.0)
BOUNDS_CONTRASTIVE = (-1.0, 1.0)
BOUNDS_ARCHETYPE = (0.0, 1.0)
BOUNDS_SILHOUETTE = (-1.0, 1.0)


@dataclass
class GameSpec:
    """Description of a domain game for a run manifest."""

    name: str
    output_bounds: Tuple[float, float]
    r_delta_res: float
    M: int
    description: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# A. Primary membership attribution game
# --------------------------------------------------------------------------- #
def membership_game(
    g_c: Callable[[np.ndarray], float],
    background: np.ndarray,
    cache=None,
    config_hash: Optional[str] = None,
    logger=None,
) -> Tuple[CoalitionOracle, GameSpec]:
    """``v(S) = 1/B sum_b g_c(x_S, z_{bar S}^{(b)})`` with bounds [0, 1], R=4."""
    oracle = CoalitionOracle(
        model_fn=g_c,
        background=background,
        output_bounds=BOUNDS_MEMBERSHIP,
        model_tag=f"membership:{getattr(g_c, '__name__', 'g_c')}",
        cache=cache,
        config_hash=config_hash,
        logger=logger,
    )
    spec = GameSpec(
        name="membership",
        output_bounds=BOUNDS_MEMBERSHIP,
        r_delta_res=R_DELTA_MEMBERSHIP,
        M=oracle.M,
        description="Primary cluster-membership attribution game",
    )
    return oracle, spec


# --------------------------------------------------------------------------- #
# B. Contrastive regime attribution game
# --------------------------------------------------------------------------- #
def contrastive_game(
    g_c: Callable[[np.ndarray], float],
    g_cp: Callable[[np.ndarray], float],
    background: np.ndarray,
    cache=None,
    config_hash: Optional[str] = None,
    logger=None,
) -> Tuple[CoalitionOracle, GameSpec]:
    """``v(S) = 1/B sum_b [g_c(x_S, ...) - g_{c'}(x_S, ...)]`` in [-1, 1], R=8."""
    model_fn = lambda x: float(g_c(x) - g_cp(x))  # noqa: E731
    oracle = CoalitionOracle(
        model_fn=model_fn,
        background=background,
        output_bounds=BOUNDS_CONTRASTIVE,
        model_tag=f"contrastive:{getattr(g_c, '__name__', 'g_c')}:{getattr(g_cp, '__name__', 'g_cp')}",
        cache=cache,
        config_hash=config_hash,
        logger=logger,
    )
    spec = GameSpec(
        name="contrastive",
        output_bounds=BOUNDS_CONTRASTIVE,
        r_delta_res=R_DELTA_CONTRASTIVE,
        M=oracle.M,
        description="Contrastive regime attribution game (why x in c instead of c')",
    )
    return oracle, spec


# --------------------------------------------------------------------------- #
# C. Global archetype game
# --------------------------------------------------------------------------- #
class ArchetypeOracle(CoalitionOracle):
    """Oracle for the global archetype game.

    ``v(S) = 1/(|I_tilde| * B) sum_{x in I_tilde} sum_b g_c(x_S, z_{bar S}^{(b)})``.

    Each coalition evaluation performs ``|I_tilde| * B`` model passes (the
    true cost of the game); a coalition evaluation is counted once.
    """

    def __init__(
        self,
        g_c: Callable[[np.ndarray], float],
        archetypes: np.ndarray,
        background: np.ndarray,
        cache=None,
        config_hash: Optional[str] = None,
        logger=None,
    ):
        self.archetypes = np.array(np.atleast_2d(archetypes), dtype=np.float64)
        self.n_archetypes = self.archetypes.shape[0]
        super().__init__(
            model_fn=g_c,
            background=background,
            output_bounds=BOUNDS_ARCHETYPE,
            model_tag=f"archetype:{getattr(g_c, '__name__', 'g_c')}",
            cache=cache,
            config_hash=config_hash,
            logger=logger,
        )

    def _evaluate_uncached(self, x: np.ndarray, S_mask: np.ndarray) -> float:
        """Evaluates the archetype game.

        .. math::

            v(S) = \\frac{1}{|\\tilde{\\mathcal{I}}_c| \\cdot B}
                \\sum_{x \\in \\tilde{\\mathcal{I}}_c} \\sum_{b=1}^B
                g_c(x_S, z^{(b)}_{\\bar S})

        For ``S = ∅`` no features come from the archetypes, so every
        (archetype, background) pair evaluates ``g_c(z^{(b)})`` and
        :math:`v(\\emptyset) = \\frac{1}{B}\\sum_b g_c(z^{(b)}) = E_{\\text{base}}`
        (baseline shortcut, 0 model passes — consistent with the standard
        oracle's empty-coalition semantics).
        """
        if np.all(S_mask):
            vals = [self.model_fn(a) for a in self.archetypes]
            self.total_model_evals += self.n_archetypes
            self._last_model_evals = self.n_archetypes
            return float(np.mean(vals))
        if not np.any(S_mask):
            self._last_model_evals = 0
            return float(self.E_base)
        total = 0.0
        for a in self.archetypes:
            X_hybrid = np.tile(a, (self.B, 1))
            X_hybrid[:, ~S_mask] = self.background[:, ~S_mask]
            total += sum(self.model_fn(X_hybrid[b]) for b in range(self.B))
        self.total_model_evals += self.n_archetypes * self.B
        self._last_model_evals = self.n_archetypes * self.B
        return float(total / (self.n_archetypes * self.B))


def archetype_game(
    g_c: Callable[[np.ndarray], float],
    archetypes: np.ndarray,
    background: np.ndarray,
    cache=None,
    config_hash: Optional[str] = None,
    logger=None,
) -> Tuple[ArchetypeOracle, GameSpec]:
    oracle = ArchetypeOracle(g_c, archetypes, background, cache=cache, config_hash=config_hash, logger=logger)
    spec = GameSpec(
        name="archetype",
        output_bounds=BOUNDS_ARCHETYPE,
        r_delta_res=R_DELTA_ARCHETYPE,
        M=oracle.M,
        description="Global cluster-level archetype game",
        extra={"n_archetypes": oracle.n_archetypes},
    )
    return oracle, spec


# --------------------------------------------------------------------------- #
# D. Intrinsic silhouette quality game
# --------------------------------------------------------------------------- #
class SilhouetteOracle(CoalitionOracle):
    """``v_sil(S) = Silhouette(Cluster(X_S))`` with ``v_sil(empty) = 0``.

    The clustering is run deterministically (fixed ``random_state``) on the
    dataset columns indexed by ``S``.  The query point ``x`` is unused for
    this dataset-level game; the coalition mask selects feature columns.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_clusters: int = 3,
        clustering=None,
        random_state: int = 0,
        cache=None,
        config_hash: Optional[str] = None,
        logger=None,
    ):
        self.X = np.array(X, dtype=np.float64)
        self.n_clusters = int(n_clusters)
        self._clustering = clustering
        self._random_state = int(random_state)
        n, m = self.X.shape
        super().__init__(
            model_fn=lambda x: 0.0,  # placeholder; _evaluate_uncached overrides
            background=np.zeros((1, m), dtype=np.float64),
            output_bounds=BOUNDS_SILHOUETTE,
            model_tag=f"silhouette:{n}x{m}:k{n_clusters}",
            cache=cache,
            config_hash=config_hash,
            logger=logger,
        )
        self.B = n  # dataset rows act as background only for interface shape

    def evaluate(self, x=None, coalition: np.ndarray = None) -> float:
        """Dataset-level evaluation: ``x`` is unused; ``coalition`` selects
        feature columns of ``X``."""
        if coalition is None:
            raise ValueError("silhouette game requires a coalition mask")
        S_mask = np.asarray(coalition, dtype=bool)
        if S_mask.shape[0] != self.M:
            raise ValueError(f"coalition length {S_mask.shape[0]} != M={self.M}")
        bitmask = int(_bitmask_plain(S_mask))
        if self.cache is not None:
            from ..utils.hashing import input_hash
            key = cache_key(self.oracle_h, input_hash(np.zeros(self.M)), self.background_h,
                            bitmask, self.config_hash)
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return float(hit)
        self.total_coalition_evals += 1
        self.cache_misses += 1
        value = self._evaluate_uncached(np.zeros(self.M), S_mask)
        if self.cache is not None:
            from ..utils.hashing import input_hash
            key = cache_key(self.oracle_h, input_hash(np.zeros(self.M)), self.background_h,
                            bitmask, self.config_hash)
            self.cache.put(key, float(value))
        return float(value)

    def _evaluate_uncached(self, x: np.ndarray, S_mask: np.ndarray) -> float:
        if not np.any(S_mask):
            return 0.0  # convention v_sil(empty) = 0 (no model passes)
        X_S = self.X[:, S_mask]
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        km = self._clustering if self._clustering is not None else KMeans(
            n_clusters=self.n_clusters, random_state=self._random_state, n_init=10
        )
        labels = km.fit_predict(X_S)
        n_labels = len(np.unique(labels))
        # silhouette_score requires 2 <= n_labels <= n_samples - 1
        if n_labels < 2 or n_labels >= len(X_S):
            return 0.0
        val = float(silhouette_score(X_S, labels))
        self._last_model_evals = 1
        self.total_model_evals += 1
        return val


def silhouette_game(
    X: np.ndarray,
    n_clusters: int = 3,
    clustering=None,
    random_state: int = 0,
    cache=None,
    config_hash: Optional[str] = None,
    logger=None,
) -> Tuple[SilhouetteOracle, GameSpec]:
    oracle = SilhouetteOracle(
        X,
        n_clusters=n_clusters,
        clustering=clustering,
        random_state=random_state,
        cache=cache,
        config_hash=config_hash,
        logger=logger,
    )
    spec = GameSpec(
        name="silhouette",
        output_bounds=BOUNDS_SILHOUETTE,
        r_delta_res=R_DELTA_SILHOUETTE,
        M=oracle.M,
        description="Intrinsic clustering-quality game (silhouette of Cluster(X_S))",
    )
    return oracle, spec


# --------------------------------------------------------------------------- #
# E. Group-lag spatiotemporal game
# --------------------------------------------------------------------------- #
def build_group_lags(n_vars: int, lags: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Return macro-player feature groups ``[(var, lag), ...]``.

    Example: ``n_vars=11, lags=(0,1,3,6,12,24)`` -> 11 groups x 6 lags = 66 features.

    Validation: ``n_vars >= 1``, non-empty strictly-increasing positive lags,
    and the resulting groups are disjoint (macro-players must partition the
    feature set for the cooperative game to be well defined).
    """
    if not isinstance(n_vars, int) or n_vars < 1:
        raise ValueError(f"n_vars must be a positive integer, got {n_vars!r}")
    if not isinstance(lags, (tuple, list)) or len(lags) == 0:
        raise ValueError("lags must be a non-empty sequence")
    lags = tuple(int(l) for l in lags)
    if any(l < 0 for l in lags):
        raise ValueError(f"lags must be non-negative, got {lags}")
    if len(set(lags)) != len(lags):
        raise ValueError(f"lags must be distinct, got {lags}")

    groups: List[Tuple[int, ...]] = []
    seen: set = set()
    for var in range(n_vars):
        members = tuple(var * len(lags) + li for li in range(len(lags)))
        if any(m in seen for m in members):
            raise ValueError(f"overlapping group members for var {var}")
        seen.update(members)
        groups.append(members)
    return groups


def group_mask_to_feature_mask(macro_mask: np.ndarray, groups: List[Tuple[int, ...]], M: int) -> np.ndarray:
    """Expand a mask over macro-players to the full feature mask."""
    feat = np.zeros(M, dtype=bool)
    for g, include in enumerate(np.asarray(macro_mask, dtype=bool)):
        if include:
            for f in groups[g]:
                feat[f] = True
    return feat


class GroupLagOracle(CoalitionOracle):
    """Oracle for the group-lag game.

    Evaluates ``v`` at the macro-player level; each macro coalition is
    expanded to its feature groups and evaluated through the underlying
    model with block background imputation (temporal coherence across lags
    of the same variable).
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], float],
        background: np.ndarray,          # (B, M_features)
        groups: List[Tuple[int, ...]],
        output_bounds: Optional[Tuple[float, float]] = None,
        cache=None,
        config_hash: Optional[str] = None,
        logger=None,
    ):
        self.groups = [tuple(sorted(g)) for g in groups]
        M_feat = background.shape[1]
        if any(max(g) >= M_feat for g in self.groups):
            raise ValueError("group members exceed feature dimension")
        self.M_feat = M_feat
        super().__init__(
            model_fn=model_fn,
            background=background,
            output_bounds=output_bounds,
            model_tag=f"grouplag:{len(self.groups)}groups:{M_feat}feat",
            cache=cache,
            config_hash=config_hash,
            logger=logger,
        )
        self.M = len(self.groups)  # macro-player dimension

    def _evaluate_uncached(self, x: np.ndarray, macro_mask: np.ndarray) -> float:
        feat = group_mask_to_feature_mask(macro_mask, self.groups, self.M_feat)
        if np.all(feat):
            self.total_model_evals += 1
            self._last_model_evals = 1
            return float(self.model_fn(x))
        if not np.any(feat):
            self._last_model_evals = 0
            return float(self.E_base)
        X_hybrid = np.tile(x, (self.B, 1))
        X_hybrid[:, ~feat] = self.background[:, ~feat]
        preds = [self.model_fn(X_hybrid[b]) for b in range(self.B)]
        self.total_model_evals += self.B
        self._last_model_evals = self.B
        return float(np.mean(preds))


def group_lag_game(
    model_fn: Callable[[np.ndarray], float],
    background: np.ndarray,
    n_vars: int,
    lags: Tuple[int, ...],
    output_bounds: Optional[Tuple[float, float]] = None,
    cache=None,
    config_hash: Optional[str] = None,
    logger=None,
) -> Tuple[GroupLagOracle, GameSpec]:
    M_feat = n_vars * len(lags)
    if background.shape[1] != M_feat:
        raise ValueError(
            f"background has {background.shape[1]} features but {n_vars} vars x "
            f"{len(lags)} lags = {M_feat}"
        )
    groups = build_group_lags(n_vars, lags)
    oracle = GroupLagOracle(model_fn, background, groups, output_bounds=output_bounds,
                            cache=cache, config_hash=config_hash, logger=logger)
    spec = GameSpec(
        name="group_lag",
        output_bounds=output_bounds,
        r_delta_res=4.0 * (output_bounds[1] - output_bounds[0]) if output_bounds else None,
        M=oracle.M,
        description=f"Group-lag spatiotemporal game: {n_vars} vars x {len(lags)} lags = {M_feat} features",
        extra={"n_vars": n_vars, "lags": list(lags), "M_feat": M_feat},
    )
    return oracle, spec


def _bitmask_plain(mask: np.ndarray) -> int:
    m = np.asarray(mask, dtype=bool)
    bitmask = 0
    for bit in range(len(m)):
        if m[bit]:
            bitmask |= 1 << bit
    return bitmask
