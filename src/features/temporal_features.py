"""
Temporal Feature Engineering Module

Computes time-based features for crime forecasting:
- Moving averages
- Year-over-year changes
- Trend features
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_moving_average(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    window: int = 3,
    date_col: str = 'year'
) -> pd.Series:
    """
    Compute moving average of a value within groups.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with ward × year observations
    value_col : str
        Column to compute moving average on
    group_cols : list
        Columns to group by (e.g., ['ward_id'])
    window : int
        Window size for moving average
    date_col : str
        Time column for ordering
        
    Returns
    -------
    pd.Series
        Moving average values
    """
    # Sort by group and time
    df_sorted = df.sort_values(group_cols + [date_col])
    
    # Compute rolling mean
    ma = df_sorted.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    return ma


def compute_yoy_change(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    date_col: str = 'year'
) -> pd.Series:
    """
    Compute year-over-year change (percentage).
    
    Parameters
    ----------
    df : DataFrame
        Panel data with ward × year observations
    value_col : str
        Column to compute YoY change on
    group_cols : list
        Columns to group by (e.g., ['ward_id'])
    date_col : str
        Time column for ordering
        
    Returns
    -------
    pd.Series
        YoY percentage change
    """
    # Sort by group and time
    df_sorted = df.sort_values(group_cols + [date_col])
    
    # Compute pct_change
    yoy = df_sorted.groupby(group_cols)[value_col].transform(
        lambda x: x.pct_change()
    )
    
    return yoy


def compute_trend_feature(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = 'year'
) -> pd.Series:
    """
    Compute linear trend (time index) within each group.
    
    Useful for capturing secular trends.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    group_cols : list
        Columns to group by
    date_col : str
        Time column
        
    Returns
    -------
    pd.Series
        Time trend (0, 1, 2, ...)
    """
    df_sorted = df.sort_values(group_cols + [date_col])
    
    trend = df_sorted.groupby(group_cols).cumcount()
    
    return trend


def compute_expanding_mean(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    date_col: str = 'year'
) -> pd.Series:
    """
    Compute expanding mean (all historical data up to current point).
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    value_col : str
        Column to compute expanding mean on
    group_cols : list
        Columns to group by
    date_col : str
        Time column
        
    Returns
    -------
    pd.Series
        Expanding mean values
    """
    df_sorted = df.sort_values(group_cols + [date_col])
    
    expanding = df_sorted.groupby(group_cols)[value_col].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    
    return expanding


def compute_volatility(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    window: int = 3,
    date_col: str = 'year'
) -> pd.Series:
    """
    Compute rolling volatility (standard deviation).
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    value_col : str
        Column to compute volatility on
    group_cols : list
        Columns to group by
    window : int
        Window size
    date_col : str
        Time column
        
    Returns
    -------
    pd.Series
        Rolling standard deviation
    """
    df_sorted = df.sort_values(group_cols + [date_col])
    
    vol = df_sorted.groupby(group_cols)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=2).std()
    )
    
    return vol


def add_temporal_features(
    df: pd.DataFrame,
    value_col: str = 'crime_rate',
    group_cols: Optional[List[str]] = None,
    ma_window: int = 3,
    include_expanding: bool = True,
    include_volatility: bool = True
) -> pd.DataFrame:
    """
    Add all temporal features to a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with ward × year observations
    value_col : str
        Primary value column (e.g., crime_rate)
    group_cols : list, optional
        Columns to group by. Defaults to ['ward_id']
    ma_window : int
        Window size for moving average
    include_expanding : bool
        Whether to include expanding mean
    include_volatility : bool
        Whether to include rolling volatility
        
    Returns
    -------
    DataFrame
        Input DataFrame with added temporal features
    """
    result = df.copy()
    
    if group_cols is None:
        group_cols = ['ward_id']
    
    # Moving average
    result[f'{value_col}_ma{ma_window}'] = compute_moving_average(
        result, value_col, group_cols, window=ma_window
    )
    
    # Year-over-year change
    result[f'{value_col}_yoy'] = compute_yoy_change(
        result, value_col, group_cols
    )
    
    # Trend
    result['time_trend'] = compute_trend_feature(result, group_cols)
    
    # Expanding mean
    if include_expanding:
        result[f'{value_col}_expanding_mean'] = compute_expanding_mean(
            result, value_col, group_cols
        )
    
    # Volatility
    if include_volatility:
        result[f'{value_col}_volatility'] = compute_volatility(
            result, value_col, group_cols, window=ma_window
        )
    
    return result


class TemporalFeatureEngineer:
    """
    Comprehensive temporal feature engineering class.
    """
    
    def __init__(
        self,
        ma_windows: List[int] = [3, 5],
        include_yoy: bool = True,
        include_trend: bool = True,
        include_expanding: bool = True,
        include_volatility: bool = True
    ):
        self.ma_windows = ma_windows
        self.include_yoy = include_yoy
        self.include_trend = include_trend
        self.include_expanding = include_expanding
        self.include_volatility = include_volatility
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all temporal features for multiple value columns.
        
        Parameters
        ----------
        df : DataFrame
            Panel data
        value_cols : list
            List of columns to compute temporal features on
        group_cols : list, optional
            Columns to group by
            
        Returns
        -------
        DataFrame
            Input DataFrame with added temporal features
        """
        result = df.copy()
        
        if group_cols is None:
            group_cols = ['ward_id']
        
        for value_col in value_cols:
            # Multiple moving averages
            for window in self.ma_windows:
                result[f'{value_col}_ma{window}'] = compute_moving_average(
                    result, value_col, group_cols, window=window
                )
            
            # YoY change
            if self.include_yoy:
                result[f'{value_col}_yoy'] = compute_yoy_change(
                    result, value_col, group_cols
                )
            
            # Expanding mean
            if self.include_expanding:
                result[f'{value_col}_expanding_mean'] = compute_expanding_mean(
                    result, value_col, group_cols
                )
            
            # Volatility
            if self.include_volatility:
                result[f'{value_col}_volatility'] = compute_volatility(
                    result, value_col, group_cols, window=self.ma_windows[0]
                )
        
        # Time trend (only once)
        if self.include_trend:
            result['time_trend'] = compute_trend_feature(result, group_cols)
        
        return result
