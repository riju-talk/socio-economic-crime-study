"""
POI (Point of Interest) Feature Engineering Module

Computes built environment features from OpenStreetMap data:
- POI density per ward
- POI diversity
- Distance to nearest facilities
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional
from shapely.geometry import Point


def compute_poi_density(
    poi_gdf: gpd.GeoDataFrame,
    wards_gdf: gpd.GeoDataFrame,
    poi_type_col: str = 'amenity',
    area_col: str = 'area_sqkm'
) -> pd.DataFrame:
    """
    Compute POI density by type for each ward.
    
    Parameters
    ----------
    poi_gdf : GeoDataFrame
        POI points with type column
    wards_gdf : GeoDataFrame
        Ward polygons with area information
    poi_type_col : str
        Column name for POI type
    area_col : str
        Column name for ward area (in sq km)
        
    Returns
    -------
    DataFrame
        POI density by ward and type
    """
    # Spatial join: assign POIs to wards
    poi_with_wards = gpd.sjoin(
        poi_gdf,
        wards_gdf[['ward_id', 'geometry', area_col]],
        how='left',
        predicate='within'
    )
    
    # Count POIs by ward and type
    poi_counts = poi_with_wards.groupby(
        ['ward_id', poi_type_col]
    ).size().unstack(fill_value=0)
    
    # Get ward areas
    ward_areas = wards_gdf[['ward_id', area_col]].set_index('ward_id')
    
    # Merge areas
    poi_counts = poi_counts.merge(
        ward_areas,
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Compute density (per sq km)
    poi_density = poi_counts.copy()
    for col in poi_counts.columns[:-1]:  # Exclude area column
        if area_col in poi_counts.columns:
            poi_density[col] = poi_counts[col] / poi_counts[area_col]
    
    return poi_density.reset_index()


def compute_poi_diversity(
    poi_counts: pd.DataFrame,
    poi_types: List[str]
) -> pd.Series:
    """
    Compute POI diversity index (Shannon entropy) for each ward.
    
    Parameters
    ----------
    poi_counts : DataFrame
        POI counts by ward and type
    poi_types : list
        List of POI type columns
        
    Returns
    -------
    pd.Series
        Diversity index for each ward
    """
    def shannon_entropy(row):
        counts = row[poi_types].values
        total = counts.sum()
        
        if total == 0:
            return 0.0
        
        proportions = counts / total
        # Remove zeros to avoid log(0)
        proportions = proportions[proportions > 0]
        
        entropy = -(proportions * np.log(proportions)).sum()
        max_entropy = np.log(len(poi_types))
        
        if max_entropy == 0:
            return 0.0
        
        return entropy / max_entropy
    
    diversity = poi_counts.apply(shannon_entropy, axis=1)
    return diversity


def compute_distance_to_nearest(
    facility_gdf: gpd.GeoDataFrame,
    wards_gdf: gpd.GeoDataFrame,
    facility_type: str = 'police_station'
) -> pd.Series:
    """
    Compute distance from ward centroid to nearest facility.
    
    Parameters
    ----------
    facility_gdf : GeoDataFrame
        Facility point locations
    wards_gdf : GeoDataFrame
        Ward polygons
    facility_type : str
        Type of facility to filter
        
    Returns
    -------
    pd.Series
        Distance to nearest facility (in km)
    """
    # Filter facilities by type if column exists
    if 'type' in facility_gdf.columns:
        facilities = facility_gdf[facility_gdf['type'] == facility_type].copy()
    else:
        facilities = facility_gdf.copy()
    
    # Compute ward centroids
    wards_centroids = wards_gdf.copy()
    wards_centroids['centroid'] = wards_centroids.geometry.centroid
    
    distances = []
    
    for idx, ward in wards_centroids.iterrows():
        ward_point = ward['centroid']
        
        if len(facilities) == 0:
            distances.append(np.nan)
            continue
        
        # Compute distances to all facilities
        facility_distances = facilities.geometry.distance(ward_point)
        min_distance = facility_distances.min()
        
        # Convert to km (assuming CRS is in meters)
        distances.append(min_distance / 1000)
    
    return pd.Series(distances, index=wards_gdf.index)


def add_poi_features(
    wards_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    poi_categories: Optional[List[str]] = None,
    area_col: str = 'area_sqkm'
) -> gpd.GeoDataFrame:
    """
    Add comprehensive POI features to ward data.
    
    Parameters
    ----------
    wards_gdf : GeoDataFrame
        Ward polygons
    poi_gdf : GeoDataFrame
        POI points with type information
    poi_categories : list, optional
        Specific POI categories to include
    area_col : str
        Column name for ward area
        
    Returns
    -------
    GeoDataFrame
        Input wards with added POI features
    """
    result = wards_gdf.copy()
    
    # Default POI categories
    if poi_categories is None:
        poi_categories = [
            'bank', 'hospital', 'market', 'police_station',
            'school', 'metro_station', 'restaurant', 'pharmacy'
        ]
    
    # Compute POI density
    if 'amenity' in poi_gdf.columns or 'type' in poi_gdf.columns:
        type_col = 'amenity' if 'amenity' in poi_gdf.columns else 'type'
        
        poi_density = compute_poi_density(
            poi_gdf, wards_gdf,
            poi_type_col=type_col,
            area_col=area_col
        )
        
        # Merge density features
        if 'ward_id' in poi_density.columns:
            poi_density = poi_density.set_index('ward_id')
        
        # Add relevant columns
        for cat in poi_categories:
            if cat in poi_density.columns:
                result[f'poi_density_{cat}'] = poi_density[cat]
        
        # Compute total POI density
        poi_cols = [c for c in poi_categories if c in poi_density.columns]
        if len(poi_cols) > 0:
            result['total_poi_density'] = poi_density[poi_cols].sum(axis=1).values
            
            # Compute diversity
            result['poi_diversity'] = compute_poi_diversity(
                poi_density.reset_index(), poi_cols
            ).values
    
    return result


class POIFeatureEngineer:
    """
    Comprehensive POI feature engineering class.
    """
    
    def __init__(
        self,
        poi_categories: Optional[List[str]] = None,
        compute_diversity: bool = True,
        compute_distances: bool = False
    ):
        if poi_categories is None:
            poi_categories = [
                'bank', 'hospital', 'market', 'police_station',
                'school', 'metro_station', 'restaurant', 'pharmacy',
                'gym', 'park', 'temple', 'mosque', 'church'
            ]
        
        self.poi_categories = poi_categories
        self.compute_diversity = compute_diversity
        self.compute_distances = compute_distances
    
    def fit_transform(
        self,
        wards_gdf: gpd.GeoDataFrame,
        poi_gdf: gpd.GeoDataFrame,
        area_col: str = 'area_sqkm'
    ) -> gpd.GeoDataFrame:
        """
        Compute all POI features.
        
        Parameters
        ----------
        wards_gdf : GeoDataFrame
            Ward polygons
        poi_gdf : GeoDataFrame
            POI points
        area_col : str
            Column name for ward area
            
        Returns
        -------
        GeoDataFrame
            Input wards with added POI features
        """
        result = add_poi_features(
            wards_gdf, poi_gdf,
            poi_categories=self.poi_categories,
            area_col=area_col
        )
        
        # Compute distances to key facilities if requested
        if self.compute_distances:
            for facility in ['police_station', 'hospital', 'metro_station']:
                dist_col = f'distance_to_{facility}'
                # This would require a separate facilities GeoDataFrame
                # Implementation depends on data structure
        
        return result
