"""
Delhi Crime Boundaries - Data Loading Module

Handles loading and cleaning of NCRB crime data, Census 2011 data,
and supplementary sources for Delhi ward-level analysis.
"""

from .load_ncrb import load_ncrb_crime_data, clean_ncrb_data
from .load_census import load_census_2011, process_census_wards
from .load_shapefile import load_delhi_wards_shapefile
from .merge_datasets import DatasetMerger, create_analysis_panel

__all__ = [
    "load_ncrb_crime_data",
    "clean_ncrb_data",
    "load_census_2011",
    "process_census_wards",
    "load_delhi_wards_shapefile",
    "DatasetMerger",
    "create_analysis_panel",
]
