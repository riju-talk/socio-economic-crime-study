"""
Configuration for Violent Crime Categories

Defines crime type mappings based on NYPD complaint data.
"""

# Primary violent crime categories for analysis
VIOLENT_CRIME_CATEGORIES = [
    'MURDER & NON-NEGL. MANSLAUGHTER',
    'RAPE',
    'ROBBERY',
    'FELONY ASSAULT',
    'ASSAULT 3 & RELATED OFFENSES'
]

# Property crime categories (secondary analysis)
PROPERTY_CRIME_CATEGORIES = [
    'BURGLARY',
    'GRAND LARCENY',
    'PETIT LARCENY',
    'GRAND LARCENY OF MOTOR VEHICLE'
]

# Categories to exclude (enforcement bias)
EXCLUDED_CATEGORIES = [
    'DANGEROUS DRUGS',
    'PROSTITUTION & RELATED OFFENSES',
    'LOITERING',
    'GAMBLING'
]

# Years for analysis
ANALYSIS_YEARS = list(range(2018, 2023))  # 2018-2022

# Minimum population threshold for inclusion
MIN_POPULATION = 100

# CRS for NYC (NAD83 / New York Long Island)
NYC_CRS = 'EPSG:2263'
