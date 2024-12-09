on### Project Overview
This project implements a Python application for processing and analyzing ECMWF  forecast data, focusing on temperature anomalies and extreme weather conditions. inditilization of the forecast was chosses as 20240229 and region was selected as USA.

### Implementation Challenges
The most significant challenge was handling ECMWF operational forecast data from S3:

1. **Data Format Complexity**
   - GRIB2 files required careful handling due to multiple fields and levels
   - Each forecast step needed separate processing and field extraction
   - Variable naming conventions differed between GRIB and climatology data

2. **Processing Time Optimization**
   - Initial loads took 3-5 minutes per forecast step
   - Implemented parallel processing reducing time to ~45 seconds
   - Added caching layer for subsequent runs

3. **Data Availability**
   - Operational forecasts(0.25-degree grid) only available from February 2024
   - Used February 29, 2024 initialization for analysis
   

### Features Implemented
- Efficient loading of ECMWF IFS forecast data (0.25-degree resolution)
- Temperature anomaly calculation using WeatherBench2 climatology
- Identification of extreme weather conditions:
  - Wind speeds exceeding 15 m/s
  - Temperature anomalies beyond ±5 K
- Visualization with Lambert Conformal projection for Continental US

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
# Copy example environment file
cp .env.example .env

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

#### Temperature Anomalies (Continental US)
![Temperature Anomalies](output/temperature_anomalies.png)
- Analysis Period: February 29, 2024 - March 6, 2024
- Maximum positive anomaly: +8.40K
- Maximum negative anomaly: -8.40K
- Total extreme locations: 7,031 grid points


#### High Wind Events
![Wind Events](output/wind_events.png)
- Maximum wind speed: 19.06 m/s
- Total high wind locations: 1,321 grid points


#### Compound Events
![Compound Events](output/compound_events.png)
- Locations with both extreme temperature and high winds: 18

### Implementation Answers to Interview Questions

1. **Large Dataset Handling**
    - Collecting weatherbench2 data was straightforwarded as it was stored in public bucket in zarr format
    - Collecing data from s3 was the most challening due to file format and processing time. Also operations forecast was avibale after 2024 Feb
   - Implemented lazy loading with xarray
   - Used dask for parallel processing
   - Local caching system for frequently accessed data
   - Selective variable loading with GRIB filters

2. **Data Transfer Optimization**
   - Local cache implementation
   - 6-hourly data selection
   - Lazy evaluation of operations
   - Region-specific data loading (Continental US only)

3. **Scalability**
   - Modular variable mapping system
   - Configurable processing parameters
   - Memory-efficient processing pipeline
   - Extensible data validation system

4. **System Reliability**
   - Comprehensive error handling
   - Data validation checks
   - Detailed logging system
   - Retry mechanism for cloud storage access
   - Cache management for failed operations