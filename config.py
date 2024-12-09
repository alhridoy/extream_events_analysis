"""Configuration settings for ECMWF forecast analysis"""

# AWS S3 Configuration
S3_CONFIG = {
    'BUCKET': 'ecmwf-forecasts',
    'PREFIX': '00z/ifs/0p25/oper/',
    'START_DATE': '2024-02-29',  # Starting date
    'START_HOUR': 0,             # Starting hour (UTC)
    'FORECAST_DAYS': 7           # Analyzing one week of forecasts
}

# Region of Interest (Continental USA)
REGION_CONFIG = {
    'LAT_MIN': 25,    # Southern tip of Florida
    'LAT_MAX': 49,    # Northern border with Canada
    'LON_MIN': -125,  # West Coast
    'LON_MAX': -67,   # East Coast
    'PROJECTION': 'LambertConformal',  # Map projection for visualization
    'CENTRAL_LON': -96,  # Central longitude for projection
    'CENTRAL_LAT': 37,   # Central latitude for projection
}

# Analysis Parameters
ANALYSIS_CONFIG = {
    'TEMP_ANOMALY_THRESHOLD': 5.0,     # Temperature anomaly threshold (Kelvin)
    'WIND_SPEED_THRESHOLD': 15.0,      # Wind speed threshold (m/s)
    'MIN_EVENT_DURATION': 6,           # Minimum duration for persistent events (hours)
    'PARALLEL_CHUNKS': {'time': 1, 'latitude': -1, 'longitude': -1},  # Chunking for parallel processing
    'BASELINE_PERIOD': '1991-2020',    # Climatological baseline period
    'CONFIDENCE_LEVEL': 0.95,          # Confidence level for statistical analysis
    'COMPOUND_EVENT_WINDOW': 24        # Time window for compound event analysis (hours)
}

# Cache Configuration
CACHE_CONFIG = {
    'CACHE_DIR': 'cache',
    'OUTPUT_DIR': 'output',
    'CLIMATOLOGY_PATH': 'data/era5_climatology.zarr',  # Path to climatology data
    'CHUNK_SIZE': '100MB',             # Chunk size for dask operations
    'COMPRESSION': {                   # Compression settings for output files
        'complevel': 5,
        'shuffle': True
    }
}

# Visualization Settings
VIZ_CONFIG = {
    'DPI': 300,                      # Resolution for output figures
    'COLORMAP_TEMP': 'RdBu_r',       # Colormap for temperature anomalies
    'COLORMAP_WIND': 'viridis',      # Colormap for wind speed
    'FIGURE_SIZE': (15, 10),         # Default figure size
    'MAP_FEATURES': {                # Map features to include
        'coastlines': True,
        'borders': True,
        'states': True,
        'gridlines': True
    }
}
