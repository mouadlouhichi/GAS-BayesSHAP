"""Coalition oracle, subset utilities, domain games and the brute-force reference."""

from .brute_force import (
    brute_force_cross_covariance,
    brute_force_prior_covariance,
    brute_force_shapley,
    exact_game_values,
    exact_shapley_from_values,
)
from .domain_games import (
    archetype_game,
    contrastive_game,
    group_lag_game,
    membership_game,
    silhouette_game,
)
from .oracle import CoalitionOracle
from .subsets import (
    all_subsets,
    bitmask_to_mask,
    candidate_pool,
    default_pool_size,
    mask_to_bitmask,
    random_coalition,
    random_subset,
    seed_coalitions,
)

__all__ = [
    "brute_force_cross_covariance",
    "brute_force_prior_covariance",
    "brute_force_shapley",
    "exact_game_values",
    "exact_shapley_from_values",
    "archetype_game",
    "contrastive_game",
    "group_lag_game",
    "membership_game",
    "silhouette_game",
    "CoalitionOracle",
    "all_subsets",
    "bitmask_to_mask",
    "candidate_pool",
    "default_pool_size",
    "mask_to_bitmask",
    "random_coalition",
    "random_subset",
    "seed_coalitions",
]
