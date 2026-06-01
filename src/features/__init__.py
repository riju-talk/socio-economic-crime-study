"""
Delhi Crime Boundaries - Spatial Feature Engineering Module

Implements boundary-aware features for analyzing crime concentration
at socio-economic discontinuities in Delhi wards.
"""

from .boundary_features import (
    BoundaryFeatureEngineer,
    compute_boundary_sharpness,
    compute_max_income_gap,
    compute_neighbor_feature_std,
    compute_income_gradient,
    compute_spatially_lagged_crime,
    compute_all_boundary_features,
)
from .within_ward_heterogeneity import (
    compute_gini_coefficient,
    compute_within_ward_variance,
)
from .temporal_features import (
    compute_moving_average,
    compute_yoy_change,
    add_temporal_features,
)
from .poi_features import (
    compute_poi_density,
    add_poi_features,
)

__all__ = [
    # Boundary features
    "BoundaryFeatureEngineer",
    "compute_boundary_sharpness",
    "compute_max_income_gap",
    "compute_neighbor_feature_std",
    "compute_income_gradient",
    "compute_spatially_lagged_crime",
    "compute_all_boundary_features",
    # Within-ward heterogeneity
    "compute_gini_coefficient",
    "compute_within_ward_variance",
    # Temporal features
    "compute_moving_average",
    "compute_yoy_change",
    "add_temporal_features",
    # POI features
    "compute_poi_density",
    "add_poi_features",
]
