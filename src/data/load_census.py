"""
Census 2011 Data Loader

Loads and processes Census 2011 data for Delhi wards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_census_2011(file_path: str) -> pd.DataFrame:
    """
    Load Census 2011 ward-level data.
    
    Parameters
    ----------
    file_path : str
        Path to Census 2011 CSV file
        
    Returns
    -------
    DataFrame
        Raw Census 2011 data
    """
    df = pd.read_csv(file_path)
    return df


def process_census_wards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean Census 2011 ward data.
    
    Standardizes column names, handles missing values,
    and computes derived variables.
    
    Parameters
    ----------
    df : DataFrame
        Raw Census 2011 data
        
    Returns
    -------
    DataFrame
        Processed ward-level data
    """
    result = df.copy()
    
    # Standardize column names
    column_mapping = {
        'WARD_CODE': 'ward_id',
        'WARD_NAME': 'ward_name',
        'DISTRICT': 'district',
        'POPULATION': 'population',
        'MALE_POPULATION': 'male_population',
        'FEMALE_POPULATION': 'female_population',
        'LITERACY_RATE': 'literacy_rate',
        'MALE_LITERACY': 'male_literacy',
        'FEMALE_LITERACY': 'female_literacy',
        'SC_POPULATION': 'sc_population',
        'ST_POPULATION': 'st_population',
        'WORKERS': 'total_workers',
        'MAIN_WORKERS': 'main_workers',
        'MARGINAL_WORKERS': 'marginal_workers',
        'NON_WORKERS': 'non_workers',
        'HOUSEHOLDS': 'households',
        'SLUM_POPULATION': 'slum_population'
    }
    
    result = result.rename(columns={
        k: v for k, v in column_mapping.items() if k in result.columns
    })
    
    # Compute derived variables
    # Sex ratio (females per 1000 males)
    if 'male_population' in result.columns and 'female_population' in result.columns:
        result['sex_ratio'] = (
            result['female_population'] / result['male_population'].replace(0, np.nan) 
            * 1000
        )
    
    # Population density (if area available)
    if 'population' in result.columns and 'area_sqkm' in result.columns:
        result['population_density'] = (
            result['population'] / result['area_sqkm'].replace(0, np.nan)
        )
    
    # Worker participation rate
    if 'population' in result.columns and 'total_workers' in result.columns:
        result['worker_participation_rate'] = (
            result['total_workers'] / result['population'].replace(0, np.nan) * 100
        )
    
    # SC/ST percentage
    if 'population' in result.columns:
        if 'sc_population' in result.columns:
            result['sc_pct'] = (
                result['sc_population'] / result['population'].replace(0, np.nan) * 100
            )
        if 'st_population' in result.columns:
            result['st_pct'] = (
                result['st_population'] / result['population'].replace(0, np.nan) * 100
            )
    
    # Slum percentage
    if 'population' in result.columns and 'slum_population' in result.columns:
        result['slum_percentage'] = (
            result['slum_population'] / result['population'].replace(0, np.nan) * 100
        )
    
    # Standardize ward IDs
    if 'ward_id' in result.columns:
        result['ward_id'] = result['ward_id'].astype(str).str.zfill(3)
    
    # Convert literacy rate to percentage (if stored as 0-1)
    if 'literacy_rate' in result.columns:
        # Check if values are in 0-1 range
        if result['literacy_rate'].max() <= 1:
            result['literacy_rate'] = result['literacy_rate'] * 100
    
    return result


def merge_with_district_data(
    ward_df: pd.DataFrame,
    district_df: pd.DataFrame,
    district_col: str = 'district'
) -> pd.DataFrame:
    """
    Merge ward-level data with district-level indicators (e.g., NFHS).
    
    Parameters
    ----------
    ward_df : DataFrame
        Ward-level Census data
    district_df : DataFrame
        District-level data (NFHS, etc.)
    district_col : str
        Column name for district identifier
        
    Returns
    -------
    DataFrame
        Merged data
    """
    merged = ward_df.merge(
        district_df,
        on=district_col,
        how='left',
        suffixes=('_ward', '_district')
    )
    
    return merged


class CensusDataLoader:
    """
    Comprehensive Census 2011 data loading class.
    """
    
    def __init__(self):
        self.processed_data = None
    
    def load_and_process(self, file_path: str) -> pd.DataFrame:
        """
        Load and process Census 2011 data.
        
        Parameters
        ----------
        file_path : str
            Path to Census 2011 CSV
            
        Returns
        -------
        DataFrame
            Processed ward-level data
        """
        raw = load_census_2011(file_path)
        self.processed_data = process_census_wards(raw)
        return self.processed_data
    
    def get_socioeconomic_features(self) -> pd.DataFrame:
        """
        Extract key socio-economic features for modeling.
        
        Returns
        -------
        DataFrame
            Selected socio-economic columns
        """
        if self.processed_data is None:
            raise ValueError("Must call load_and_process first")
        
        feature_cols = [
            'ward_id',
            'population',
            'population_density',
            'literacy_rate',
            'male_literacy',
            'female_literacy',
            'sex_ratio',
            'sc_pct',
            'st_pct',
            'slum_percentage',
            'worker_participation_rate',
            'households'
        ]
        
        available_cols = [c for c in feature_cols if c in self.processed_data.columns]
        return self.processed_data[available_cols].copy()
