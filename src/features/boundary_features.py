"""
Boundary Feature Engineering Module

Implements formulas for computing socio-economic boundary features
for crime analysis at Delhi ward level.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional


def compute_boundary_sharpness(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    feature_col: str = 'literacy_rate'
) -> pd.Series:
    """
    Compute boundary sharpness for each ward.
    
    Formula:
    BoundarySharpness_i = max(|X_i - X_j| for j in N(i))
    
    Interpretation:
    High → ward sits next to very different neighborhoods
    Low → ward embedded in similar area
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward data with feature column
    adjacency_dict : dict
        Mapping of ward_id -> list of neighbor ward_ids
    feature_col : str
        Name of feature column (e.g., literacy_rate, median_income)
        
    Returns
    -------
    pd.Series
        Boundary sharpness values indexed by ward_id
    """
    boundary_sharpness = {}
    
    for ward_id, neighbors in adjacency_dict.items():
        if ward_id not in gdf.index or len(neighbors) == 0:
            boundary_sharpness[ward_id] = np.nan
            continue
        
        ward_value = gdf.loc[ward_id, feature_col]
        
        # Get neighbor values
        neighbor_values = gdf.loc[
            gdf.index.isin(neighbors), 
            feature_col
        ].dropna()
        
        if len(neighbor_values) == 0:
            boundary_sharpness[ward_id] = np.nan
            continue
        
        # Maximum absolute difference to any neighbor
        diffs = abs(neighbor_values - ward_value)
        boundary_sharpness[ward_id] = diffs.max()
    
    return pd.Series(boundary_sharpness)


def compute_max_income_gap(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    income_col: str = 'median_income'
) -> pd.Series:
    """
    Compute maximum income gap to any neighbor.
    
    Formula:
    MaxIncomeGap_i = max(|Inc_i - Inc_j| for j in N(i))
    
    Interpretation:
    Captures extreme inequality at the border.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward data with income column
    adjacency_dict : dict
        Mapping of ward_id -> list of neighbor ward_ids
    income_col : str
        Name of median income column
        
    Returns
    -------
    pd.Series
        Max income gap values indexed by ward_id
    """
    max_gaps = {}
    
    for ward_id, neighbors in adjacency_dict.items():
        if ward_id not in gdf.index or len(neighbors) == 0:
            max_gaps[ward_id] = np.nan
            continue
        
        ward_income = gdf.loc[ward_id, income_col]
        
        # Get neighbor incomes
        neighbor_incomes = gdf.loc[
            gdf.index.isin(neighbors), 
            income_col
        ].dropna()
        
        if len(neighbor_incomes) == 0:
            max_gaps[ward_id] = np.nan
            continue
        
        # Compute max absolute difference
        gaps = abs(neighbor_incomes - ward_income)
        max_gaps[ward_id] = gaps.max()
    
    return pd.Series(max_gaps)


def compute_neighbor_feature_std(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    feature_col: str = 'median_income'
) -> pd.Series:
    """
    Compute standard deviation of neighbor feature values.
    
    Formula:
    NeighborStd_i = sqrt((1/|N(i)|) * sum((X_j - mean(X_N))^2))
    
    Interpretation:
    High → ward surrounded by diverse neighbors
    Low → surrounded by uniform neighbors
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward data with feature column
    adjacency_dict : dict
        Mapping of ward_id -> list of neighbor ward_ids
    feature_col : str
        Name of feature column
        
    Returns
    -------
    pd.Series
        Neighbor feature std values indexed by ward_id
    """
    neighbor_stds = {}
    
    for ward_id, neighbors in adjacency_dict.items():
        if ward_id not in gdf.index or len(neighbors) == 0:
            neighbor_stds[ward_id] = np.nan
            continue
        
        # Get neighbor values
        neighbor_values = gdf.loc[
            gdf.index.isin(neighbors), 
            feature_col
        ].dropna()
        
        if len(neighbor_values) < 2:
            neighbor_stds[ward_id] = 0.0
            continue
        
        # Compute standard deviation
        neighbor_stds[ward_id] = neighbor_values.std()
    
    return pd.Series(neighbor_stds)


def compute_income_gradient(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    income_col: str = 'median_income'
) -> pd.Series:
    """
    Compute income gradient (directional tension).
    
    Formula:
    IncomeGradient_i = (1/|N(i)|) * sum(Inc_j - Inc_i for j in N(i))
    
    Interpretation:
    Positive → ward poorer than neighbors
    Negative → ward richer than neighbors
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward data with income column
    adjacency_dict : dict
        Mapping of ward_id -> list of neighbor ward_ids
    income_col : str
        Name of median income column
        
    Returns
    -------
    pd.Series
        Income gradient values indexed by ward_id
    """
    gradients = {}
    
    for ward_id, neighbors in adjacency_dict.items():
        if ward_id not in gdf.index or len(neighbors) == 0:
            gradients[ward_id] = np.nan
            continue
        
        ward_income = gdf.loc[ward_id, income_col]
        
        # Get neighbor incomes
        neighbor_incomes = gdf.loc[
            gdf.index.isin(neighbors), 
            income_col
        ].dropna()
        
        if len(neighbor_incomes) == 0:
            gradients[ward_id] = np.nan
            continue
        
        # Mean difference
        gradients[ward_id] = (neighbor_incomes - ward_income).mean()
    
    return pd.Series(gradients)


def compute_spatially_lagged_crime(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    crime_col: str = 'crime_rate'
) -> pd.Series:
    """
    Compute spatially lagged crime rate (control variable).
    
    Formula:
    LagCrime_i = (1/|N(i)|) * sum(C_j for j in N(i))
    
    Interpretation:
    Prevents falsely attributing clustering to inequality.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward data with crime rate column
    adjacency_dict : dict
        Mapping of ward_id -> list of neighbor ward_ids
    crime_col : str
        Name of crime rate column
        
    Returns
    -------
    pd.Series
        Spatially lagged crime rates indexed by ward_id
    """
    lagged_crime = {}
    
    for ward_id, neighbors in adjacency_dict.items():
        if ward_id not in gdf.index or len(neighbors) == 0:
            lagged_crime[ward_id] = np.nan
            continue
        
        # Get neighbor crime rates
        neighbor_crimes = gdf.loc[
            gdf.index.isin(neighbors), 
            crime_col
        ].dropna()
        
        if len(neighbor_crimes) == 0:
            lagged_crime[ward_id] = np.nan
            continue
        
        # Mean neighbor crime rate
        lagged_crime[ward_id] = neighbor_crimes.mean()
    
    return pd.Series(lagged_crime)


class BoundaryFeatureEngineer:
    """
    Comprehensive boundary feature engineering class.
    
    Computes all boundary-aware features in a unified interface.
    """
    
    def __init__(
        self,
        income_col: str = 'median_income',
        literacy_col: str = 'literacy_rate',
        crime_col: str = 'crime_rate',
        slum_col: str = 'slum_percentage'
    ):
        self.income_col = income_col
        self.literacy_col = literacy_col
        self.crime_col = crime_col
        self.slum_col = slum_col
    
    def fit_transform(
        self,
        gdf: gpd.GeoDataFrame,
        adjacency_dict: Dict[str, List[str]]
    ) -> gpd.GeoDataFrame:
        """
        Compute all boundary features.
        
        Parameters
        ----------
        gdf : GeoDataFrame
            Ward data with required columns
        adjacency_dict : dict
            Mapping of ward_id -> list of neighbor ward_ids
            
        Returns
        -------
        GeoDataFrame
            Input GeoDataFrame with added boundary feature columns
        """
        result = gdf.copy()
        
        # Literacy boundary sharpness
        result['boundary_sharpness_literacy'] = compute_boundary_sharpness(
            gdf, adjacency_dict, self.literacy_col
        )
        
        # Income boundary sharpness
        result['boundary_sharpness_income'] = compute_boundary_sharpness(
            gdf, adjacency_dict, self.income_col
        )
        
        # Slum percentage boundary sharpness
        result['boundary_sharpness_slum'] = compute_boundary_sharpness(
            gdf, adjacency_dict, self.slum_col
        )
        
        # Max income gap
        result['max_income_gap'] = compute_max_income_gap(
            gdf, adjacency_dict, self.income_col
        )
        
        # Neighbor income std
        result['neighbor_income_std'] = compute_neighbor_feature_std(
            gdf, adjacency_dict, self.income_col
        )
        
        # Income gradient
        result['income_gradient'] = compute_income_gradient(
            gdf, adjacency_dict, self.income_col
        )
        
        # Spatially lagged crime
        result['lagged_crime'] = compute_spatially_lagged_crime(
            gdf, adjacency_dict, self.crime_col
        )
        
        # Composite boundary index (average of normalized sharpness measures)
        sharpness_cols = [
            'boundary_sharpness_literacy',
            'boundary_sharpness_income',
            'boundary_sharpness_slum'
        ]
        
        # Normalize each sharpness measure
        normalized = []
        for col in sharpness_cols:
            mean_val = result[col].mean()
            std_val = result[col].std()
            if std_val > 0:
                normalized.append((result[col] - mean_val) / std_val)
        
        if len(normalized) > 0:
            result['boundary_index'] = pd.DataFrame(normalized).T.mean(axis=1)
        
        return result


def compute_all_boundary_features(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    income_col: str = 'median_income',
    literacy_col: str = 'literacy_rate',
    crime_col: str = 'crime_rate',
    slum_col: str = 'slum_percentage'
) -> gpd.GeoDataFrame:
    """
    Convenience function to compute all boundary features at once.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward data with required columns
    adjacency_dict : dict
        Mapping of ward_id -> list of neighbor ward_ids
    income_col : str
        Name of median income column
    literacy_col : str
        Name of literacy rate column
    crime_col : str
        Name of crime rate column
    slum_col : str
        Name of slum percentage column
        
    Returns
    -------
    GeoDataFrame
        Input GeoDataFrame with added boundary feature columns
    """
    engineer = BoundaryFeatureEngineer(
        income_col=income_col,
        literacy_col=literacy_col,
        crime_col=crime_col,
        slum_col=slum_col
    )
    return engineer.fit_transform(gdf, adjacency_dict)
