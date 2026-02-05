"""
Spatial Utilities for Census Tract Analysis

Functions for building adjacency matrices, spatial joins, and
neighborhood operations.
"""

import geopandas as gpd
import pandas as pd
from typing import Dict, List, Tuple
from shapely.geometry import Point


def build_adjacency_dict(
    gdf: gpd.GeoDataFrame,
    id_col: str = 'GEOID',
    contiguity: str = 'queen'
) -> Dict[str, List[str]]:
    """
    Build spatial adjacency dictionary for census tracts.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Census tract geometries
    id_col : str
        Column name containing tract identifiers
    contiguity : str
        Type of contiguity: 'queen' or 'rook'
        - queen: shared edges or vertices
        - rook: shared edges only
        
    Returns
    -------
    dict
        Mapping of tract_id -> list of neighbor tract_ids
    """
    # Ensure geometries are valid
    gdf = gdf.copy()
    gdf['geometry'] = gdf['geometry'].buffer(0)
    
    adjacency = {}
    
    for idx, row in gdf.iterrows():
        tract_id = row[id_col]
        geometry = row['geometry']
        
        if contiguity == 'queen':
            # Queen: intersects (edges or vertices)
            neighbors = gdf[
                (gdf[id_col] != tract_id) & 
                (gdf.geometry.intersects(geometry))
            ][id_col].tolist()
        elif contiguity == 'rook':
            # Rook: shared boundary (edges only)
            neighbors = gdf[
                (gdf[id_col] != tract_id) & 
                (gdf.geometry.touches(geometry))
            ][id_col].tolist()
        else:
            raise ValueError("contiguity must be 'queen' or 'rook'")
        
        adjacency[tract_id] = neighbors
    
    return adjacency


def create_adjacency_matrix(
    adjacency_dict: Dict[str, List[str]],
    tract_ids: List[str]
) -> pd.DataFrame:
    """
    Convert adjacency dictionary to binary matrix.
    
    Parameters
    ----------
    adjacency_dict : dict
        Mapping of tract_id -> list of neighbor tract_ids
    tract_ids : list
        Ordered list of tract identifiers
        
    Returns
    -------
    DataFrame
        Binary adjacency matrix (n_tracts x n_tracts)
    """
    n = len(tract_ids)
    matrix = pd.DataFrame(0, index=tract_ids, columns=tract_ids)
    
    for tract_id, neighbors in adjacency_dict.items():
        if tract_id in matrix.index:
            for neighbor_id in neighbors:
                if neighbor_id in matrix.columns:
                    matrix.loc[tract_id, neighbor_id] = 1
    
    return matrix


def spatial_join_crimes_to_tracts(
    crimes_df: pd.DataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    tract_id_col: str = 'GEOID'
) -> pd.DataFrame:
    """
    Map crime points to census tracts.
    
    Parameters
    ----------
    crimes_df : DataFrame
        Crime records with lat/lon coordinates
    tracts_gdf : GeoDataFrame
        Census tract boundaries
    lat_col : str
        Name of latitude column
    lon_col : str
        Name of longitude column
    tract_id_col : str
        Name of tract ID column
        
    Returns
    -------
    DataFrame
        Crime records with assigned tract_id
    """
    # Filter records with valid coordinates
    valid_coords = crimes_df[[lat_col, lon_col]].notna().all(axis=1)
    crimes_valid = crimes_df[valid_coords].copy()
    
    # Create GeoDataFrame from crime points
    geometry = [
        Point(xy) for xy in zip(
            crimes_valid[lon_col], 
            crimes_valid[lat_col]
        )
    ]
    crimes_gdf = gpd.GeoDataFrame(
        crimes_valid, 
        geometry=geometry,
        crs=tracts_gdf.crs
    )
    
    # Spatial join
    crimes_with_tracts = gpd.sjoin(
        crimes_gdf,
        tracts_gdf[[tract_id_col, 'geometry']],
        how='left',
        predicate='within'
    )
    
    return crimes_with_tracts


def aggregate_crimes_by_tract_year(
    crimes_df: pd.DataFrame,
    tract_id_col: str = 'GEOID',
    date_col: str = 'date',
    crime_type_col: str = 'crime_type'
) -> pd.DataFrame:
    """
    Aggregate crime counts by tract × year.
    
    Parameters
    ----------
    crimes_df : DataFrame
        Crime records with tract_id and date
    tract_id_col : str
        Name of tract ID column
    date_col : str
        Name of date column
    crime_type_col : str
        Name of crime type column (optional)
        
    Returns
    -------
    DataFrame
        Aggregated counts: tract_id × year × crime_count
    """
    # Extract year
    df = crimes_df.copy()
    df['year'] = pd.to_datetime(df[date_col]).dt.year
    
    # Aggregate
    agg_df = df.groupby(
        [tract_id_col, 'year']
    ).size().reset_index(name='crime_count')
    
    return agg_df


def compute_crime_rate(
    crime_counts: pd.DataFrame,
    census_data: pd.DataFrame,
    tract_id_col: str = 'GEOID',
    pop_col: str = 'population'
) -> pd.DataFrame:
    """
    Compute crime rate per 1000 population.
    
    Parameters
    ----------
    crime_counts : DataFrame
        Aggregated crime counts by tract × year
    census_data : DataFrame
        Census tract population data
    tract_id_col : str
        Name of tract ID column
    pop_col : str
        Name of population column
        
    Returns
    -------
    DataFrame
        Crime counts with crime_rate column
    """
    # Merge population data
    result = crime_counts.merge(
        census_data[[tract_id_col, pop_col]],
        on=tract_id_col,
        how='left'
    )
    
    # Compute rate per 1000 population
    result['crime_rate'] = (
        result['crime_count'] / result[pop_col] * 1000
    )
    
    # Handle division by zero
    result['crime_rate'] = result['crime_rate'].replace(
        [float('inf'), -float('inf')], 
        float('nan')
    )
    
    return result
