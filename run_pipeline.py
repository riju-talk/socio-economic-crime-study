#!/usr/bin/env python3
"""
Delhi Crime Boundaries - Main Pipeline Runner

This script orchestrates the full data processing and modeling pipeline.
"""

import argparse
import yaml
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_data_pipeline(config: dict):
    """Run data loading and preprocessing pipeline."""
    logger.info("Starting data pipeline...")
    
    from src.data.load_ncrb import NCRBDataLoader
    from src.data.load_census import CensusDataLoader
    from src.data.load_shapefile import WardShapefileLoader
    from src.data.merge_datasets import DatasetMerger
    
    # Load config paths
    paths = config.get('paths', {})
    data_sources = config.get('data_sources', {})
    
    # Initialize loaders
    ncrb_loader = NCRBDataLoader()
    census_loader = CensusDataLoader()
    ward_loader = WardShapefileLoader(crs=config.get('spatial', {}).get('crs', 'EPSG:4326'))
    
    # Load NCRB crime data
    logger.info("Loading NCRB crime data...")
    crime_file = data_sources.get('ncrb_crime', 'data/raw/ncrb_crime_delhi.csv')
    if Path(crime_file).exists():
        crime_df = ncrb_loader.load_and_clean(
            crime_file,
            years=list(range(
                config.get('temporal', {}).get('train_start_year', 2016),
                config.get('temporal', {}).get('test_end_year', 2025)
            ))
        )
        logger.info(f"Loaded {len(crime_df)} crime records")
    else:
        logger.warning(f"Crime data file not found: {crime_file}")
        logger.info("Creating synthetic data for demonstration...")
        crime_df = create_synthetic_crime_data()
    
    # Load Census data
    logger.info("Loading Census 2011 data...")
    census_file = data_sources.get('census_2011', 'data/raw/census_2011_wards.csv')
    if Path(census_file).exists():
        census_df = census_loader.load_and_process(census_file)
        logger.info(f"Loaded {len(census_df)} ward records")
    else:
        logger.warning(f"Census data file not found: {census_file}")
        logger.info("Creating synthetic data for demonstration...")
        census_df = create_synthetic_census_data()
    
    # Load ward shapefile
    logger.info("Loading ward boundaries...")
    shapefile = data_sources.get('ward_shapefile', 'data/raw/delhi_wards.shp')
    if Path(shapefile).exists():
        wards_gdf = ward_loader.load(shapefile)
        adjacency = ward_loader.get_adjacency(
            contiguity=config.get('spatial', {}).get('contiguity', 'queen')
        )
        logger.info(f"Loaded {len(wards_gdf)} wards")
    else:
        logger.warning(f"Shapefile not found: {shapefile}")
        logger.info("Creating synthetic boundaries for demonstration...")
        wards_gdf, adjacency = create_synthetic_wards()
    
    # Merge datasets
    logger.info("Merging datasets...")
    merger = DatasetMerger()
    panel = merger.merge_all(
        crime_df=crime_df,
        census_df=census_df,
        district_df=None,  # Can add NFHS or other district data
        wards_gdf=wards_gdf,
        min_population=config.get('general', {}).get('min_population', 100)
    )
    
    # Validate
    validation = merger.get_validation_report()
    logger.info(f"Panel created: {validation['total_observations']} observations, "
                f"{validation['unique_wards']} wards")
    
    if validation['quality_issues']:
        for issue in validation['quality_issues']:
            logger.warning(f"Quality issue: {issue}")
    
    # Save interim data (drop geometry for parquet compatibility)
    interim_path = Path(paths.get('interim_data', 'data/interim'))
    interim_path.mkdir(parents=True, exist_ok=True)
    
    # Drop geometry column for parquet save (not serializable)
    panel_to_save = panel.drop(columns=['geometry'], errors='ignore')
    panel_to_save.to_parquet(interim_path / 'merged_panel.parquet', index=False)
    logger.info(f"Saved merged panel to {interim_path / 'merged_panel.parquet'}")
    
    return panel, adjacency


def run_feature_engineering(panel, adjacency: dict, config: dict):
    """Run feature engineering pipeline."""
    logger.info("Starting feature engineering...")
    
    from src.features.boundary_features import BoundaryFeatureEngineer
    from src.features.temporal_features import TemporalFeatureEngineer
    from src.features.within_ward_heterogeneity import WithinWardHeterogeneityEngineer
    
    # Boundary features
    logger.info("Computing boundary features...")
    boundary_engineer = BoundaryFeatureEngineer(
        income_col='median_income' if 'median_income' in panel.columns else 'literacy_rate',
        literacy_col='literacy_rate',
        crime_col='crime_rate' if 'crime_rate' in panel.columns else 'crime_count',
        slum_col='slum_percentage' if 'slum_percentage' in panel.columns else None
    )
    
    # For GeoDataFrame
    if hasattr(panel, 'geometry'):
        panel_with_boundaries = boundary_engineer.fit_transform(panel, adjacency)
    else:
        # Convert to simple index-based approach
        panel_indexed = panel.set_index('ward_id')
        panel_with_boundaries = boundary_engineer.fit_transform(panel_indexed, adjacency)
        panel_with_boundaries = panel_with_boundaries.reset_index()
    
    # Temporal features
    logger.info("Computing temporal features...")
    temporal_engineer = TemporalFeatureEngineer(
        ma_windows=[3, 5],
        include_yoy=True,
        include_trend=True,
        include_expanding=True,
        include_volatility=True
    )
    
    value_cols = ['crime_rate'] if 'crime_rate' in panel_with_boundaries.columns else ['crime_count']
    available_cols = [c for c in value_cols if c in panel_with_boundaries.columns]
    
    if available_cols:
        panel_final = temporal_engineer.fit_transform(
            panel_with_boundaries,
            value_cols=available_cols,
            group_cols=['ward_id']
        )
    else:
        panel_final = panel_with_boundaries
    
    # Save processed features (drop geometry for parquet compatibility)
    paths = config.get('paths', {})
    processed_path = Path(paths.get('processed_data', 'data/processed'))
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Drop geometry column for parquet save
    panel_to_save = panel_final.drop(columns=['geometry'], errors='ignore')
    panel_to_save.to_parquet(processed_path / 'features_2024.parquet', index=False)
    logger.info(f"Saved processed features to {processed_path / 'features_2024.parquet'}")
    
    return panel_final


def create_synthetic_crime_data():
    """Create synthetic crime data for demonstration."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    wards = [f"{i:03d}" for i in range(1, 271)]  # ~270 Delhi wards
    years = list(range(2016, 2025))
    
    records = []
    for ward in wards:
        for year in years:
            # Simulate crime counts with some pattern
            base_rate = np.random.poisson(50)
            trend = (year - 2016) * np.random.uniform(-2, 5)
            crime_count = max(0, base_rate + trend + np.random.normal(0, 10))
            
            records.append({
                'ward_id': ward,
                'year': year,
                'crime_count': int(crime_count),
                'crime_head': np.random.choice(['THEFT', 'BURGLARY', 'ASSAULT', 'ROBBERY'])
            })
    
    return pd.DataFrame(records)


def create_synthetic_census_data():
    """Create synthetic Census data for demonstration."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    wards = [f"{i:03d}" for i in range(1, 271)]
    
    records = []
    for ward in wards:
        records.append({
            'ward_id': ward,
            'ward_name': f'Ward {ward}',
            'district': np.random.choice(['Central Delhi', 'North Delhi', 'South Delhi', 
                                          'East Delhi', 'West Delhi', 'New Delhi']),
            'population': np.random.randint(10000, 200000),
            'literacy_rate': np.random.uniform(60, 95),
            'male_literacy': np.random.uniform(65, 98),
            'female_literacy': np.random.uniform(55, 95),
            'sc_population': np.random.randint(0, 50000),
            'st_population': np.random.randint(0, 10000),
            'slum_population': np.random.randint(0, 100000),
            'total_workers': np.random.randint(5000, 100000),
            'households': np.random.randint(2000, 40000)
        })
    
    df = pd.DataFrame(records)
    # Add area for density calculation
    df['area_sqkm'] = np.random.uniform(1, 50, len(df))
    
    return df


def create_synthetic_wards():
    """Create synthetic ward boundaries for demonstration."""
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    import numpy as np
    
    np.random.seed(42)
    
    # Create approximate grid of wards for Delhi
    wards = []
    adjacency = {}
    
    # Delhi approximate bounds
    lat_min, lat_max = 28.4, 28.8
    lon_min, lon_max = 76.8, 77.3
    
    rows = 15
    cols = 18
    
    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols
    
    idx = 0
    grid = np.zeros((rows, cols), dtype=object)
    
    for r in range(rows):
        for c in range(cols):
            ward_id = f"{idx+1:03d}"
            
            # Create simple polygon
            lat1 = lat_min + r * lat_step
            lat2 = lat_min + (r + 1) * lat_step
            lon1 = lon_min + c * lon_step
            lon2 = lon_min + (c + 1) * lon_step
            
            poly = Polygon([
                (lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2)
            ])
            
            wards.append({
                'ward_id': ward_id,
                'ward_name': f'Ward {ward_id}',
                'geometry': poly
            })
            
            grid[r, c] = ward_id
            idx += 1
    
    gdf = gpd.GeoDataFrame(wards, crs='EPSG:4326')
    
    # Build adjacency
    for r in range(rows):
        for c in range(cols):
            ward_id = grid[r, c]
            neighbors = []
            
            # Check all 8 directions (queen contiguity)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(grid[nr, nc])
            
            adjacency[ward_id] = neighbors
    
    return gdf, adjacency


def main():
    parser = argparse.ArgumentParser(description='Delhi Crime Boundaries Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'data', 'features', 'models'],
        default='all',
        help='Pipeline step to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.step in ['all', 'data']:
        panel, adjacency = run_data_pipeline(config)
    
    if args.step in ['all', 'features']:
        if 'panel' not in locals():
            # Reload from interim
            import pandas as pd
            panel = pd.read_parquet('data/interim/merged_panel.parquet')
            # Need to rebuild adjacency - simplified for now
            adjacency = {}
        
        panel_final = run_feature_engineering(panel, adjacency, config)
    
    if args.step == 'models':
        logger.info("Model training coming soon...")
    
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()
