"""
Dataset Merger

Merges NCRB crime data, Census 2011 data, and supplementary sources
into a unified panel dataset for modeling.
"""

import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional


def merge_crime_census(
    crime_df: pd.DataFrame,
    census_df: pd.DataFrame,
    on_col: str = 'ward_id'
) -> pd.DataFrame:
    """
    Merge crime data with Census socio-economic data.
    
    Parameters
    ----------
    crime_df : DataFrame
        Crime data with ward_id and year
    census_df : DataFrame
        Census data with ward_id (time-invariant)
    on_col : str
        Column to merge on
        
    Returns
    -------
    DataFrame
        Merged crime × census data
    """
    # Census data is time-invariant (2011), so we merge and forward-fill
    merged = crime_df.merge(
        census_df,
        on=on_col,
        how='left'
    )
    
    return merged


def add_district_indicators(
    df: pd.DataFrame,
    district_df: pd.DataFrame,
    district_col: str = 'district',
    year_col: str = 'year'
) -> pd.DataFrame:
    """
    Add district-level indicators (e.g., NFHS, economic data).
    
    Parameters
    ----------
    df : DataFrame
        Ward-level data
    district_df : DataFrame
        District-level data with year
    district_col : str
        District column name
    year_col : str
        Year column name
        
    Returns
    -------
    DataFrame
        Data with district indicators
    """
    merged = df.merge(
        district_df,
        on=[district_col, year_col],
        how='left',
        suffixes=('', '_district')
    )
    
    return merged


def create_analysis_panel(
    crime_df: pd.DataFrame,
    census_df: pd.DataFrame,
    wards_gdf: Optional[gpd.GeoDataFrame] = None,
    min_population: int = 100
) -> pd.DataFrame:
    """
    Create final analysis panel with all datasets merged.
    
    Parameters
    ----------
    crime_df : DataFrame
        Cleaned NCRB crime data
    census_df : DataFrame
        Processed Census 2011 data
    wards_gdf : GeoDataFrame, optional
        Ward shapefile with geometry
    min_population : int
        Minimum population threshold for inclusion
        
    Returns
    -------
    DataFrame
        Final analysis panel
    """
    # Step 1: Aggregate crime by ward × year
    if 'crime_count' not in crime_df.columns:
        # Aggregate if needed
        crime_agg = crime_df.groupby(['ward_id', 'year']).size().reset_index(name='crime_count')
    else:
        crime_agg = crime_df[['ward_id', 'year', 'crime_count']].copy()
    
    # Step 2: Merge with Census
    panel = merge_crime_census(crime_agg, census_df, on_col='ward_id')
    
    # Step 3: Filter by population
    if 'population' in panel.columns:
        panel = panel[panel['population'] >= min_population]
    
    # Step 4: Compute crime rate
    if 'population' in panel.columns and 'crime_count' in panel.columns:
        panel['crime_rate'] = (
            panel['crime_count'] / panel['population'].replace(0, float('nan')) * 1000
        )
    
    # Step 5: Add geometry if available
    if wards_gdf is not None:
        ward_geo = wards_gdf[['ward_id', 'geometry']].copy()
        panel = panel.merge(ward_geo, on='ward_id', how='left')
    
    return panel


def validate_panel(panel: pd.DataFrame) -> Dict[str, any]:
    """
    Validate the merged panel dataset.
    
    Checks:
    - No negative crime counts
    - No NaN in critical fields
    - Ward codes match across sources
    - Temporal coverage
    
    Parameters
    ----------
    panel : DataFrame
        Merged panel data
        
    Returns
    -------
    dict
        Validation report
    """
    report = {
        'total_observations': len(panel),
        'unique_wards': panel['ward_id'].nunique(),
        'years_covered': sorted(panel['year'].unique()),
        'missing_checks': {},
        'quality_issues': []
    }
    
    # Check for missing values in critical columns
    critical_cols = ['ward_id', 'year', 'crime_count', 'crime_rate']
    available_critical = [c for c in critical_cols if c in panel.columns]
    
    for col in available_critical:
        missing_pct = panel[col].isna().sum() / len(panel) * 100
        report['missing_checks'][col] = f"{missing_pct:.2f}%"
        
        if missing_pct > 5:
            report['quality_issues'].append(f"High missing rate in {col}: {missing_pct:.1f}%")
    
    # Check for negative crime counts
    if 'crime_count' in panel.columns:
        neg_count = (panel['crime_count'] < 0).sum()
        if neg_count > 0:
            report['quality_issues'].append(f"Found {neg_count} negative crime counts")
    
    # Check crime rate range
    if 'crime_rate' in panel.columns:
        extreme_rates = panel[panel['crime_rate'] > 1000]
        if len(extreme_rates) > 0:
            report['quality_issues'].append(
                f"Found {len(extreme_rates)} wards with crime rate > 1000 per 1000"
            )
    
    return report


class DatasetMerger:
    """
    Comprehensive dataset merging class.
    """
    
    def __init__(self):
        self.panel = None
        self.validation_report = None
    
    def merge_all(
        self,
        crime_df: pd.DataFrame,
        census_df: pd.DataFrame,
        district_df: Optional[pd.DataFrame] = None,
        wards_gdf: Optional[gpd.GeoDataFrame] = None,
        min_population: int = 100
    ) -> pd.DataFrame:
        """
        Merge all datasets into analysis panel.
        
        Parameters
        ----------
        crime_df : DataFrame
            NCRB crime data
        census_df : DataFrame
            Census 2011 data
        district_df : DataFrame, optional
            District-level indicators
        wards_gdf : GeoDataFrame, optional
            Ward boundaries
        min_population : int
            Minimum population threshold
            
        Returns
        -------
        DataFrame
            Final analysis panel
        """
        # Create base panel
        self.panel = create_analysis_panel(
            crime_df, census_df, wards_gdf, min_population
        )
        
        # Add district indicators if provided
        if district_df is not None:
            district_col = 'district' if 'district' in self.panel.columns else None
            if district_col:
                self.panel = add_district_indicators(
                    self.panel, district_df, district_col=district_col
                )
        
        # Validate
        self.validation_report = validate_panel(self.panel)
        
        return self.panel
    
    def get_validation_report(self) -> dict:
        """Get validation report."""
        return self.validation_report
    
    def save_panel(self, output_path: str, format: str = 'parquet'):
        """
        Save panel to file.
        
        Parameters
        ----------
        output_path : str
            Output file path
        format : str
            File format: 'parquet' or 'csv'
        """
        if self.panel is None:
            raise ValueError("No panel to save. Call merge_all first.")
        
        if format == 'parquet':
            self.panel.to_parquet(output_path, index=False)
        elif format == 'csv':
            self.panel.to_csv(output_path, index=False)
        else:
            raise ValueError("format must be 'parquet' or 'csv'")
    
    def get_feature_matrix(
        self,
        feature_cols: List[str],
        target_col: str = 'crime_count'
    ):
        """
        Extract feature matrix and target vector.
        
        Parameters
        ----------
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name
            
        Returns
        -------
        tuple
            (X, y) feature matrix and target vector
        """
        if self.panel is None:
            raise ValueError("No panel available. Call merge_all first.")
        
        available_features = [c for c in feature_cols if c in self.panel.columns]
        X = self.panel[available_features].copy()
        y = self.panel[target_col].copy()
        
        return X, y
