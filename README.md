### Overview
This project implements a Python application for processing and analyzing ECMWF forecast data, focusing on temperature anomalies and extreme weather conditions.

### Features

#### Data Processing
- Efficient loading of ECMWF IFS  7 days forecast data (0.25-degree resolution)
- Temperature anomaly calculation using WeatherBench2 climatology
- Identification of extreme weather conditions:
- Wind speeds exceeding 15 m/s
- Temperature anomalies beyond ±5 K


#### Analysis Capabilities
- Calculates temperature anomalies globally
- Identifies extreme weather events for example wind_speed
- Detects compound events (multiple extreme conditions occurring simultaneously)
- Processes wind patterns and their relationship to temperature changes

#### Visualization & Output
- Generates publication-quality maps and plots
- Creates detailed CSV reports of extreme event locations
- Visualizes temperature anomalies with wind overlay
- Produces compound event analysis maps


### Technical Stack

#### Core Technologies
- Python 3.8+
- xarray/dask for parallel data processing
- Cartopy for geospatial visualization
- NumPy/Pandas for numerical computations

#### Cloud & Storage
- AWS S3 (boto3) for data storage
- Google Cloud Storage (gcsfs) for WeatherBench2 data
- Zarr format for efficient data handling
- fsspec for cloud storage abstraction

#### Weather Data Processing
- ECMWF IFS forecast data (0.25° resolution)
- WeatherBench2 climatology integration
- GRIB format handling via eccodes
- Automated data validation and QC

#### Visualization
- Matplotlib for publication-quality plots
- Lambert Conformal projection
- Interactive map generation
- CSV exports for detailed analysis

#### Performance Features
- Dask distributed computing
- Chunked streaming
- Lazy loading
- Dynamic memory management
- Intelligent caching system

#### Monitoring
- Comprehensive logging
- Error handling with retries
- Resource monitoring
- Health checks for memory, disk, and network
- Dask cluster status tracking
   

### Features Implemented
- Efficient loading of ECMWF IFS forecast data (0.25-degree resolution)
- Temperature anomaly calculation using WeatherBench2 climatology
- Identification of extreme weather conditions:
- Wind speeds exceeding 15 m/s
- Temperature anomalies beyond ±5 K
- Visualization with Lambert Conformal projection for US

### Technical Implementation

#### Data Access Strategy
1. **S3 Access Implementation**
   - Configured boto3 with retry logic (max_attempts=3)
   - Anonymous access fallback for public data
   - Structured data access with prefix patterns

2. **GCP Access**
   - Used gcsfs for WeatherBench2 climatology data
   - Implemented zarr format access
   - Lazy loading with xarray

#### Performance Optimizations
1. **Local Caching**
   - Implemented cache directory for downloaded files
   - Files only downloaded if not in cache
   - Cache persists across sessions

2. **Memory Management**
   - Lazy loading with xarray/dask
   - Selective variable loading using GRIB filter_by_keys
   - 6-hourly timesteps instead of hourly data

### Installation Guide

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
# Edit .env with your credentials 
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### Running the Analysis

2. **Run Complete Analysis**
```bash
python main.py
```

### Analysis Results


![Temperature Anomalies](output/analysis_plots.png)
#### Temperature Anomalies (Continental US)
- Analysis Period: February 29, 2024 - March 6, 2024
- Total extreme locations: 7,031 grid points


#### High Wind Events

- Total high wind locations: 1,321 grid points


#### Compound Events
![Compound Events](output/compound_events.png)
- Locations with both extreme temperature and high winds: 18

