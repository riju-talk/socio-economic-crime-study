"""
Delhi Ward Shapefile Loader

Loads administrative boundary shapefiles for Delhi wards.
"""

import geopandas as gpd
from pathlib import Path
from typing import Optional


def load_delhi_wards_shapefile(
    shapefile_path: str,
    crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Load Delhi ward shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to shapefile (.shp)
    crs : str
        Target CRS (default: WGS84)
        
    Returns
    -------
    GeoDataFrame
        Ward boundaries with attributes
    """
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject if needed
    if gdf.crs is not None and gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)
    
    return gdf


def process_ward_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and standardize ward geometries.
    
    - Fix invalid geometries
    - Add area calculation
    - Standardize column names
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Raw ward shapefile
        
    Returns
    -------
    GeoDataFrame
        Processed ward boundaries
    """
    result = gdf.copy()
    
    # Fix invalid geometries
    result['geometry'] = result.geometry.buffer(0)
    
    # Compute area (in square kilometers)
    # First project to a suitable CRS for area calculation
    if result.crs.is_geographic:
        # Project to UTM Zone 44N for Delhi
        utm_crs = 'EPSG:24364'
        projected = result.to_crs(utm_crs)
        result['area_sqkm'] = projected.geometry.area / 1e6  # m² to km²
        # Return to original CRS
        result = result.to_crs(gdf.crs)
    else:
        result['area_sqkm'] = result.geometry.area / 1e6
    
    # Standardize column names
    column_mapping = {
        'WARD_CODE': 'ward_id',
        'WARD_NAME': 'ward_name',
        'DISTRICT': 'district',
        'AC_NAME': 'assembly_constituency',
        'PARL_CONST': 'parliamentary_constituency'
    }
    
    result = result.rename(columns={
        k: v for k, v in column_mapping.items() if k in result.columns
    })
    
    # Ensure ward_id is string
    if 'ward_id' in result.columns:
        result['ward_id'] = result['ward_id'].astype(str).str.zfill(3)
    
    return result


def compute_ward_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute ward centroids.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward polygons
        
    Returns
    -------
    GeoDataFrame
        Ward centroids as points
    """
    centroids = gdf.copy()
    centroids['geometry'] = centroids.geometry.centroid
    return centroids


def get_ward_adjacency(
    gdf: gpd.GeoDataFrame,
    contiguity: str = 'queen'
) -> dict:
    """
    Build ward adjacency dictionary from geometries.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Ward polygons
    contiguity : str
        Type of contiguity: 'queen' or 'rook'
        
    Returns
    -------
    dict
        Mapping of ward_id -> list of neighbor ward_ids
    """
    # Ensure valid geometries
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.buffer(0)
    
    adjacency = {}
    
    for idx, row in gdf.iterrows():
        ward_id = row.get('ward_id', str(idx))
        geometry = row['geometry']
        
        if contiguity == 'queen':
            # Queen: shares edge or vertex
            neighbors = gdf[
                (gdf.index != idx) & 
                (gdf.geometry.intersects(geometry))
            ]
        elif contiguity == 'rook':
            # Rook: shares edge only
            neighbors = gdf[
                (gdf.index != idx) & 
                (gdf.geometry.touches(geometry))
            ]
        else:
            raise ValueError("contiguity must be 'queen' or 'rook'")
        
        neighbor_ids = neighbors.get('ward_id', neighbors.index.astype(str)).tolist()
        adjacency[ward_id] = neighbor_ids
    
    return adjacency


class WardShapefileLoader:
    """
    Comprehensive ward shapefile loading class.
    """
    
    def __init__(self, crs: str = 'EPSG:4326'):
        self.crs = crs
        self.gdf = None
        self.adjacency_dict = None
    
    def load(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load and process ward shapefile.
        
        Parameters
        ----------
        shapefile_path : str
            Path to shapefile
            
        Returns
        -------
        GeoDataFrame
            Processed ward boundaries
        """
        self.gdf = load_delhi_wards_shapefile(shapefile_path, crs=self.crs)
        self.gdf = process_ward_geometries(self.gdf)
        return self.gdf
    
    def get_adjacency(self, contiguity: str = 'queen') -> dict:
        """
        Get ward adjacency dictionary.
        
        Parameters
        ----------
        contiguity : str
            Type of contiguity
            
        Returns
        -------
        dict
            Adjacency mapping
        """
        if self.gdf is None:
            raise ValueError("Must call load first")
        
        self.adjacency_dict = get_ward_adjacency(self.gdf, contiguity=contiguity)
        return self.adjacency_dict
    
    def get_centroids(self) -> gpd.GeoDataFrame:
        """
        Get ward centroids.
        
        Returns
        -------
        GeoDataFrame
            Ward centroid points
        """
        if self.gdf is None:
            raise ValueError("Must call load first")
        
        return compute_ward_centroids(self.gdf)
    
    def get_coordinates(self) -> dict:
        """
        Get ward centroid coordinates.
        
        Returns
        -------
        dict
            Mapping of ward_id -> (lon, lat)
        """
        centroids = self.get_centroids()
        coords = {}
        
        for idx, row in centroids.iterrows():
            ward_id = row.get('ward_id', str(idx))
            coords[ward_id] = (row.geometry.x, row.geometry.y)
        
        return coords
