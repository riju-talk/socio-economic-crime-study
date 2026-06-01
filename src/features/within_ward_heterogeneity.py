"""
Within-Ward Heterogeneity Features

Computes internal heterogeneity measures for each ward,
complementing boundary features with intra-ward inequality metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_gini_coefficient(
    values: pd.Series,
    weights: Optional[pd.Series] = None
) -> float:
    """
    Compute Gini coefficient for income/literacy distribution.
    
    Formula:
    Gini = (sum_i sum_j |x_i - x_j|) / (2 * n * sum(x))
    
    Parameters
    ----------
    values : pd.Series
        Values (e.g., income brackets, literacy rates by sub-area)
    weights : pd.Series, optional
        Population weights for each value
        
    Returns
    -------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    if len(values) == 0 or values.sum() == 0:
        return np.nan
    
    # Remove NaN
    mask = values.notna()
    values = values[mask]
    
    if weights is not None:
        weights = weights[mask]
    else:
        weights = pd.Series(np.ones(len(values)), index=values.index)
    
    if len(values) == 0:
        return np.nan
    
    # Sort by values
    sorted_idx = values.argsort()
    values_sorted = values.iloc[sorted_idx]
    weights_sorted = weights.iloc[sorted_idx]
    
    # Cumulative share of population
    cum_pop = weights_sorted.cumsum()
    total_pop = cum_pop.iloc[-1]
    
    # Cumulative share of income
    weighted_values = values_sorted * weights_sorted
    cum_income = weighted_values.cumsum()
    total_income = cum_income.iloc[-1]
    
    if total_income == 0:
        return np.nan
    
    # Gini formula
    gini = 1 - (cum_income / total_income).sum() * (weights_sorted / total_pop).sum()
    gini = gini + (weights_sorted * values_sorted / total_income).sum() * (cum_pop / total_pop).shift(1).fillna(0).sum()
    
    # Simplified approximation
    n = len(values_sorted)
    if n < 2:
        return 0.0
    
    rank = np.arange(1, n + 1)
    gini_approx = (2 * (rank * values_sorted.values).sum()) / (n * values_sorted.sum()) - (n + 1) / n
    
    return max(0.0, min(1.0, gini_approx))


def compute_within_ward_variance(
    gdf: pd.DataFrame,
    ward_col: str = 'ward_id',
    value_col: str = 'literacy_rate'
) -> pd.Series:
    """
    Compute within-ward variance of a feature.
    
    Useful when you have sub-ward level data (e.g., census blocks).
    
    Parameters
    ----------
    gdf : DataFrame
        Data with sub-ward observations
    ward_col : str
        Column name for ward identifier
    value_col : str
        Column name for the feature value
        
    Returns
    -------
    pd.Series
        Within-ward variance indexed by ward_id
    """
    variance_by_ward = gdf.groupby(ward_col)[value_col].var()
    return variance_by_ward


def compute_entropy_index(
    proportions: pd.Series
) -> float:
    """
    Compute entropy-based diversity index.
    
    Formula:
    Entropy = -sum(p_i * log(p_i)) / log(k)
    
    Normalized to [0, 1] where 1 = maximum diversity.
    
    Parameters
    ----------
    proportions : pd.Series
        Proportions of different categories (must sum to 1)
        
    Returns
    -------
    float
        Entropy index (0 = no diversity, 1 = maximum diversity)
    """
    # Remove zeros and NaN
    p = proportions[(proportions > 0) & (proportions.notna())]
    
    if len(p) == 0:
        return np.nan
    
    k = len(p)
    if k == 1:
        return 0.0
    
    # Shannon entropy
    entropy = -(p * np.log(p)).sum()
    
    # Normalize by maximum possible entropy
    max_entropy = np.log(k)
    
    return entropy / max_entropy


def compute_herfindahl_index(
    proportions: pd.Series
) -> float:
    """
    Compute Herfindahl-Hirschman Index (HHI) for concentration.
    
    Formula:
    HHI = sum(p_i^2)
    
    Parameters
    ----------
    proportions : pd.Series
        Proportions of different categories (must sum to 1)
        
    Returns
    -------
    float
        HHI (0 = perfect competition, 1 = monopoly)
    """
    p = proportions[proportions.notna()]
    return (p ** 2).sum()


class WithinWardHeterogeneityEngineer:
    """
    Compute within-ward heterogeneity features.
    """
    
    def __init__(
        self,
        income_cols: Optional[List[str]] = None,
        literacy_cols: Optional[List[str]] = None,
        occupation_cols: Optional[List[str]] = None
    ):
        self.income_cols = income_cols or []
        self.literacy_cols = literacy_cols or []
        self.occupation_cols = occupation_cols or []
    
    def fit_transform(self, gdf: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all within-ward heterogeneity features.
        
        Parameters
        ----------
        gdf : DataFrame
            Ward-level data with sub-category columns
            
        Returns
        -------
        DataFrame
            Input DataFrame with added heterogeneity features
        """
        result = gdf.copy()
        
        # Income Gini (if income distribution columns exist)
        if self.income_cols and all(col in gdf.columns for col in self.income_cols):
            result['income_gini'] = gdf.apply(
                lambda row: compute_gini_coefficient(row[self.income_cols]),
                axis=1
            )
        
        # Literacy entropy (if literacy by category exists)
        if self.literacy_cols and all(col in gdf.columns for col in self.literacy_cols):
            result['literacy_entropy'] = gdf.apply(
                lambda row: compute_entropy_index(row[self.literacy_cols]),
                axis=1
            )
        
        # Occupational diversity (if occupation columns exist)
        if self.occupation_cols and all(col in gdf.columns for col in self.occupation_cols):
            result['occupation_hhi'] = gdf.apply(
                lambda row: compute_herfindahl_index(row[self.occupation_cols]),
                axis=1
            )
        
        return result


def compute_all_heterogeneity_features(
    gdf: pd.DataFrame,
    income_cols: Optional[List[str]] = None,
    literacy_cols: Optional[List[str]] = None,
    occupation_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to compute all within-ward heterogeneity features.
    
    Parameters
    ----------
    gdf : DataFrame
        Ward-level data
    income_cols : list, optional
        Columns for income distribution
    literacy_cols : list, optional
        Columns for literacy distribution
    occupation_cols : list, optional
        Columns for occupation distribution
        
    Returns
    -------
    DataFrame
        Input DataFrame with added heterogeneity features
    """
    engineer = WithinWardHeterogeneityEngineer(
        income_cols=income_cols,
        literacy_cols=literacy_cols,
        occupation_cols=occupation_cols
    )
    return engineer.fit_transform(gdf)
