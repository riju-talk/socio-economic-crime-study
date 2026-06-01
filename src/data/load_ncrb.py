"""
NCRB Crime Data Loader

Loads and cleans crime data from National Crime Records Bureau (NCRB)
for Delhi wards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import yaml


def load_ncrb_crime_data(
    file_path: str,
    years: Optional[List[int]] = None,
    crime_types: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load NCRB crime data from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to NCRB crime data CSV
    years : list, optional
        Filter to specific years
    crime_types : list, optional
        Filter to specific crime types
        
    Returns
    -------
    DataFrame
        Raw NCRB crime data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Filter by years if specified
    if years is not None and 'year' in df.columns:
        df = df[df['year'].isin(years)]
    
    # Filter by crime types if specified
    if crime_types is not None and 'crime_head' in df.columns:
        df = df[df['crime_head'].isin(crime_types)]
    
    return df


def clean_ncrb_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize NCRB crime data.
    
    Handles:
    - Missing values
    - Inconsistent ward names/codes
    - Crime head standardization
    - Negative values (shouldn't exist but sometimes do)
    
    Parameters
    ----------
    df : DataFrame
        Raw NCRB crime data
        
    Returns
    -------
    DataFrame
        Cleaned crime data
    """
    result = df.copy()
    
    # Standardize column names
    column_mapping = {
        'WARD_NAME': 'ward_name',
        'WARD_CODE': 'ward_id',
        'YEAR': 'year',
        'CRIME_HEAD': 'crime_head',
        'COUNT': 'crime_count',
        'DISTRICT': 'district'
    }
    
    result = result.rename(columns={
        k: v for k, v in column_mapping.items() if k in result.columns
    })
    
    # Remove negative counts
    if 'crime_count' in result.columns:
        result = result[result['crime_count'] >= 0]
    
    # Handle missing values
    # For crime counts, fill with 0 (likely means no crimes reported)
    if 'crime_count' in result.columns:
        result['crime_count'] = result['crime_count'].fillna(0)
    
    # Drop rows with missing critical fields
    critical_cols = ['ward_id', 'year', 'crime_head']
    available_critical = [c for c in critical_cols if c in result.columns]
    result = result.dropna(subset=available_critical)
    
    # Standardize crime head names
    if 'crime_head' in result.columns:
        result['crime_head'] = result['crime_head'].str.upper().str.strip()
        
        # Map common variations
        crime_head_mapping = {
            'MURDER': 'MURDER',
            'MURDER & NON-NEGL. MANSLAUGHTER': 'MURDER',
            'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER': 'MURDER',
            'RAPE': 'RAPE',
            'KIDNAPPING & ABDUCTION': 'KIDNAPPING & ABDUCTION',
            'DACOITY': 'DACOITY',
            'ROBBERY': 'ROBBERY',
            'BURGLARY': 'BURGLARY',
            'THEFT': 'THEFT',
            'AUTO THEFT': 'MOTOR VEHICLE THEFT',
            'MOTOR VEHICLE THEFT': 'MOTOR VEHICLE THEFT',
            'ARSON': 'ARSON',
            'HURT': 'HURT',
            'GRIEVIOUS HURT': 'HURT',
            'ASSAULT ON WOMEN': 'ASSAULT ON WOMEN',
            'CRUELTY BY HUSBAND': 'CRUELTY BY HUSBAND',
            'DOWRY DEATHS': 'DOWRY DEATHS',
            'MOLESTATION': 'MOLESTATION',
            'SEXUAL HARASSMENT': 'SEXUAL HARASSMENT'
        }
        
        result['crime_head'] = result['crime_head'].map(
            lambda x: crime_head_mapping.get(x, x)
        )
    
    # Standardize ward IDs (ensure string format)
    if 'ward_id' in result.columns:
        result['ward_id'] = result['ward_id'].astype(str).str.zfill(3)
    
    # Ensure year is integer
    if 'year' in result.columns:
        result['year'] = result['year'].astype(int)
    
    return result


def aggregate_crime_by_ward_year(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregate crime counts by ward and year.
    
    Parameters
    ----------
    df : DataFrame
        Cleaned crime data
    group_cols : list, optional
        Additional columns to group by
        
    Returns
    -------
    DataFrame
        Aggregated crime counts by ward × year
    """
    base_cols = ['ward_id', 'year']
    
    if group_cols is not None:
        group_cols = base_cols + group_cols
    else:
        group_cols = base_cols
    
    # Filter to available columns
    available_cols = [c for c in group_cols if c in df.columns]
    available_cols.append('crime_count')
    
    agg_df = df[available_cols].groupby(available_cols[:-1])['crime_count'].sum()
    agg_df = agg_df.reset_index()
    
    return agg_df


def filter_violent_crimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to violent crime categories only.
    
    Parameters
    ----------
    df : DataFrame
        Crime data with crime_head column
        
    Returns
    -------
    DataFrame
        Filtered to violent crimes
    """
    violent_crimes = [
        'MURDER',
        'RAPE',
        'KIDNAPPING & ABDUCTION',
        'DACOITY',
        'ROBBERY',
        'ARSON',
        'HURT',
        'ASSAULT ON WOMEN',
        'DOWRY DEATHS',
        'MOLESTATION',
        'SEXUAL HARASSMENT'
    ]
    
    return df[df['crime_head'].isin(violent_crimes)]


def filter_property_crimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to property crime categories only.
    
    Parameters
    ----------
    df : DataFrame
        Crime data with crime_head column
        
    Returns
    -------
    DataFrame
        Filtered to property crimes
    """
    property_crimes = [
        'BURGLARY',
        'THEFT',
        'MOTOR VEHICLE THEFT',
        'CRIMINAL BREACH OF TRUST',
        'CHEATING',
        'FORGERY'
    ]
    
    return df[df['crime_head'].isin(property_crimes)]


class NCRBDataLoader:
    """
    Comprehensive NCRB data loading and cleaning class.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None
    ):
        if config_path is None:
            config_path = 'configs/config.yaml'
        
        self.config = self._load_config(config_path)
        self.violent_crimes = self.config.get('general', {}).get(
            'violent_crimes',
            ['Murder', 'Rape', 'Kidnapping & Abduction', 'Dacoity', 
             'Robbery', 'Arson', 'Hurt']
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def load_and_clean(
        self,
        file_path: str,
        years: Optional[List[int]] = None,
        filter_violent: bool = False
    ) -> pd.DataFrame:
        """
        Load, clean, and optionally filter NCRB data.
        
        Parameters
        ----------
        file_path : str
            Path to NCRB data
        years : list, optional
            Years to include
        filter_violent : bool
            Whether to filter to violent crimes only
            
        Returns
        -------
        DataFrame
            Cleaned crime data
        """
        # Load raw data
        df = load_ncrb_crime_data(
            file_path,
            years=years or list(range(2016, 2025))
        )
        
        # Clean
        df = clean_ncrb_data(df)
        
        # Filter to violent crimes if requested
        if filter_violent:
            df = filter_violent_crimes(df)
        
        return df
    
    def create_panel(
        self,
        df: pd.DataFrame,
        include_all_years: bool = True
    ) -> pd.DataFrame:
        """
        Create balanced panel data (all wards × all years).
        
        Parameters
        ----------
        df : DataFrame
            Cleaned crime data
        include_all_years : bool
            Whether to include all years in range
            
        Returns
        -------
        DataFrame
            Balanced panel with zeros for missing observations
        """
        # Get unique wards and years
        wards = df['ward_id'].unique()
        
        if include_all_years:
            min_year = df['year'].min()
            max_year = df['year'].max()
            years = list(range(min_year, max_year + 1))
        else:
            years = df['year'].unique()
        
        # Create complete grid
        panel = pd.MultiIndex.from_product(
            [wards, years],
            names=['ward_id', 'year']
        ).to_frame(index=False)
        
        # Merge with actual data
        panel = panel.merge(
            df[['ward_id', 'year', 'crime_count']],
            on=['ward_id', 'year'],
            how='left'
        )
        
        # Fill missing with 0
        panel['crime_count'] = panel['crime_count'].fillna(0)
        
        return panel
