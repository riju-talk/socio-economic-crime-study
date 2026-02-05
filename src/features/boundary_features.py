"""
Boundary Feature Engineering Module

Implements formulas for computing socio-economic boundary features
for crime analysis at census tract level.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List


def compute_boundary_sharpness(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    income_col: str = 'median_income'
) -> pd.Series:
    """
    Compute boundary sharpness for each tract.
    
    Formula:
    BoundarySharpness_i = |Inc_i - (1/|N(i)|) * sum(Inc_j for j in N(i))|
    
    Interpretation:
    High → tract sits next to very different neighborhoods
    Low → tract embedded in similar-income area
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Census tract data with income column
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    income_col : str
        Name of median income column
        
    Returns
    -------
    pd.Series
        Boundary sharpness values indexed by tract_id
    """
    boundary_sharpness = {}
    
    for tract_id, neighbors in adjacency_dict.items():
        if tract_id not in gdf.index or len(neighbors) == 0:
            boundary_sharpness[tract_id] = np.nan
            continue
            
        tract_income = gdf.loc[tract_id, income_col]
        
        # Get neighbor incomes
        neighbor_incomes = gdf.loc[
            gdf.index.isin(neighbors), 
            income_col
        ].dropna()
        
        if len(neighbor_incomes) == 0:
            boundary_sharpness[tract_id] = np.nan
            continue
            
        # Compute mean neighbor income
        mean_neighbor_income = neighbor_incomes.mean()
        
        # Absolute difference
        boundary_sharpness[tract_id] = abs(tract_income - mean_neighbor_income)
    
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
        Census tract data with income column
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    income_col : str
        Name of median income column
        
    Returns
    -------
    pd.Series
        Max income gap values indexed by tract_id
    """
    max_gaps = {}
    
    for tract_id, neighbors in adjacency_dict.items():
        if tract_id not in gdf.index or len(neighbors) == 0:
            max_gaps[tract_id] = np.nan
            continue
            
        tract_income = gdf.loc[tract_id, income_col]
        
        # Get neighbor incomes
        neighbor_incomes = gdf.loc[
            gdf.index.isin(neighbors), 
            income_col
        ].dropna()
        
        if len(neighbor_incomes) == 0:
            max_gaps[tract_id] = np.nan
            continue
            
        # Compute max absolute difference
        gaps = abs(neighbor_incomes - tract_income)
        max_gaps[tract_id] = gaps.max()
    
    return pd.Series(max_gaps)


def compute_neighbor_income_std(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    income_col: str = 'median_income'
) -> pd.Series:
    """
    Compute standard deviation of neighbor incomes.
    
    Formula:
    NeighborIncomeStd_i = sqrt((1/|N(i)|) * sum((Inc_j - mean(Inc_N))^2))
    
    Interpretation:
    High → tract surrounded by diverse-income neighbors
    Low → surrounded by uniform-income neighbors
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Census tract data with income column
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    income_col : str
        Name of median income column
        
    Returns
    -------
    pd.Series
        Neighbor income std values indexed by tract_id
    """
    neighbor_stds = {}
    
    for tract_id, neighbors in adjacency_dict.items():
        if tract_id not in gdf.index or len(neighbors) == 0:
            neighbor_stds[tract_id] = np.nan
            continue
            
        # Get neighbor incomes
        neighbor_incomes = gdf.loc[
            gdf.index.isin(neighbors), 
            income_col
        ].dropna()
        
        if len(neighbor_incomes) < 2:
            neighbor_stds[tract_id] = 0.0
            continue
            
        # Compute standard deviation
        neighbor_stds[tract_id] = neighbor_incomes.std()
    
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
    Positive → tract poorer than neighbors
    Negative → tract richer than neighbors
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Census tract data with income column
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    income_col : str
        Name of median income column
        
    Returns
    -------
    pd.Series
        Income gradient values indexed by tract_id
    """
    gradients = {}
    
    for tract_id, neighbors in adjacency_dict.items():
        if tract_id not in gdf.index or len(neighbors) == 0:
            gradients[tract_id] = np.nan
            continue
            
        tract_income = gdf.loc[tract_id, income_col]
        
        # Get neighbor incomes
        neighbor_incomes = gdf.loc[
            gdf.index.isin(neighbors), 
            income_col
        ].dropna()
        
        if len(neighbor_incomes) == 0:
            gradients[tract_id] = np.nan
            continue
            
        # Mean difference
        gradients[tract_id] = (neighbor_incomes - tract_income).mean()
    
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
        Census tract data with crime rate column
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    crime_col : str
        Name of crime rate column
        
    Returns
    -------
    pd.Series
        Spatially lagged crime rates indexed by tract_id
    """
    lagged_crime = {}
    
    for tract_id, neighbors in adjacency_dict.items():
        if tract_id not in gdf.index or len(neighbors) == 0:
            lagged_crime[tract_id] = np.nan
            continue
            
        # Get neighbor crime rates
        neighbor_crimes = gdf.loc[
            gdf.index.isin(neighbors), 
            crime_col
        ].dropna()
        
        if len(neighbor_crimes) == 0:
            lagged_crime[tract_id] = np.nan
            continue
            
        # Mean neighbor crime rate
        lagged_crime[tract_id] = neighbor_crimes.mean()
    
    return pd.Series(lagged_crime)


def compute_all_boundary_features(
    gdf: gpd.GeoDataFrame,
    adjacency_dict: Dict[str, List[str]],
    income_col: str = 'median_income',
    crime_col: str = 'crime_rate'
) -> gpd.GeoDataFrame:
    """
    Compute all boundary features at once.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Census tract data with income and crime columns
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    income_col : str
        Name of median income column
    crime_col : str
        Name of crime rate column
        
    Returns
    -------
    GeoDataFrame
        Input GeoDataFrame with added boundary feature columns
    """
    result = gdf.copy()
    
    result['boundary_sharpness'] = compute_boundary_sharpness(
        gdf, adjacency_dict, income_col
    )
    result['max_income_gap'] = compute_max_income_gap(
        gdf, adjacency_dict, income_col
    )
    result['neighbor_income_std'] = compute_neighbor_income_std(
        gdf, adjacency_dict, income_col
    )
    result['income_gradient'] = compute_income_gradient(
        gdf, adjacency_dict, income_col
    )
    result['lagged_crime'] = compute_spatially_lagged_crime(
        gdf, adjacency_dict, crime_col
    )
    
    return result
