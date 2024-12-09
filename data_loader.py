import boto3
import os
import logging
from botocore.config import Config
from dotenv import load_dotenv
import xarray as xr
import numpy as np
import gcsfs
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)
load_dotenv()

class ECMWFDataLoader:
    def __init__(self):
        """Initialize the ECMWF data loader."""
        config = Config(
            retries=dict(
                max_attempts=3
            )
        )
        
       
        if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                config=config
            )
            logger.info("Using AWS credentials from environment variables")
        else:
            logger.warning("AWS credentials not found in environment variables.")
            logger.info("Using anonymous access (this might not work for all data)")
            self.s3_client = boto3.client('s3', config=config)
    
    def get_forecast_data(self, date_str):
        """Get 7-day forecast data from a single initialization date."""
        try:
            bucket = 'ecmwf-forecasts'
            prefix = f'{date_str}/00z/ifs/0p25/oper/'  # Using 00Z initialization
            
            # Map our variable names to GRIB shortNames
            variable_map = {
                't2m': '2t',   
                'u10': '10u',  
                'v10': '10v' 
            }
            
            logger.info(f"Loading 7-day forecast initialized on {date_str} 00Z")
            
            # Define forecast steps (0 to 168 hours in 6-hour steps)
            forecast_steps = list(range(0, 169, 6))  # 0, 6, 12, ..., 168
            logger.info(f"Will load {len(forecast_steps)} forecast steps: {forecast_steps}")
            
            # First, list all available files for this initialization
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.error(f"No data found for initialization date {date_str}")
                return None
            
            # Get available forecast steps
            available_files = {}
            for obj in response['Contents']:
                if obj['Key'].endswith('-oper-fc.grib2'):
                    # Extract the forecast step from the filename
                    step = int(obj['Key'].split('-')[-3].replace('h', ''))
                    if step in forecast_steps:  # Only include 6-hourly steps
                        available_files[step] = obj['Key']
            
            if not available_files:
                logger.error(f"No forecast files found for initialization date {date_str}")
                return None
            
            logger.info(f"Found {len(available_files)} forecast steps at 6-hour intervals")
            
            # Initialize list to store datasets for each forecast step
            step_datasets = []
            
            # Process available forecast steps
            for step in sorted(available_files.keys()):
                file_key = available_files[step]
                logger.info(f"Processing forecast step +{step}h: {file_key}")
                
                # Download and load the GRIB file
                local_path = os.path.join('cache', os.path.basename(file_key))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if not os.path.exists(local_path):
                    logger.info(f"Downloading {file_key} to {local_path}")
                    self.s3_client.download_file(bucket, file_key, local_path)
                
                # Load each variable separately for this timestep
                step_vars = {}
                for var_name, grib_name in variable_map.items():
                    try:
                        import cfgrib
                        ds_var = cfgrib.open_dataset(
                            local_path,
                            filter_by_keys={'shortName': grib_name},
                            backend_kwargs={'indexpath': ''}
                        )
                        
                        if ds_var is None or len(ds_var.data_vars) == 0:
                            logger.error(f"No data found for variable {grib_name} at step +{step}h")
                            continue
                        
                        # Drop unnecessary coordinates and rename variable
                        if 'heightAboveGround' in ds_var.coords:
                            ds_var = ds_var.drop('heightAboveGround')
                        
                        ds_var = ds_var.rename({list(ds_var.data_vars)[0]: var_name})
                        step_vars[var_name] = ds_var[var_name]
                        
                    except Exception as e:
                        logger.error(f"Error loading {grib_name} at step +{step}h: {str(e)}")
                        continue
                
                if len(step_vars) == len(variable_map):
                    # Create a dataset with all variables for this timestep
                    coords = {
                        'latitude': step_vars[list(step_vars.keys())[0]].latitude,
                        'longitude': step_vars[list(step_vars.keys())[0]].longitude,
                        'time': step_vars[list(step_vars.keys())[0]].time
                    }
                    step_ds = xr.Dataset(step_vars, coords=coords)
                    step_datasets.append(step_ds)
            
            if not step_datasets:
                logger.error("No valid forecast data found")
                return None
            
            # Combine all timesteps into a single dataset
            forecast = xr.concat(step_datasets, dim='time')
            logger.info(f"Successfully loaded forecast data from {forecast.time.min().values} to {forecast.time.max().values}")
            logger.info(f"Forecast shape: {forecast.t2m.shape}")
            
            # Verify the forecast covers 7 days
            time_range = (forecast.time.max() - forecast.time.min()).values
            expected_range = np.timedelta64(168, 'h')  # 7 days
            if time_range < expected_range:
                logger.warning(f"Forecast only covers {time_range} instead of expected {expected_range}")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error getting forecast data: {str(e)}")
            raise
    
    def _verify_dataset(self, ds):
        """Perform comprehensive quality checks on the dataset."""
        logger.info("Performing dataset verification...")
        
        # Check coordinates
        required_coords = ['latitude', 'longitude', 'time']
        for coord in required_coords:
            if coord not in ds.coords:
                raise ValueError(f"Missing required coordinate: {coord}")
        
        # Check variables
        required_vars = ['t2m', 'u10', 'v10']
        for var in required_vars:
            if var not in ds:
                raise ValueError(f"Missing required variable: {var}")
        
        # Check for NaN values
        for var in ds.data_vars:
            nan_count = ds[var].isnull().sum().values
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in {var}")
        
        # Check coordinate ranges
        lat_range = ds.latitude.values
        lon_range = ds.longitude.values
        logger.info(f"Latitude range: [{lat_range.min():.2f}, {lat_range.max():.2f}]")
        logger.info(f"Longitude range: [{lon_range.min():.2f}, {lon_range.max():.2f}]")
        
        # Check data ranges
        if 't2m' in ds:
            temp_range = ds.t2m.values
            logger.info(f"Temperature range: [{temp_range.min():.2f}, {temp_range.max():.2f}] K")
            if temp_range.min() < 180 or temp_range.max() > 330:  # Reasonable temperature range in Kelvin
                logger.warning("Temperature values outside expected range")
        
        if 'u10' in ds and 'v10' in ds:
            wind_speed = np.sqrt(ds.u10**2 + ds.v10**2)
            logger.info(f"Wind speed range: [{wind_speed.min().values:.2f}, {wind_speed.max().values:.2f}] m/s")
            if wind_speed.max() > 100:  # Reasonable wind speed range
                logger.warning("Wind speed values unusually high")
    
    def download_file(self, s3_path, local_path):
        """Download a file from S3."""
        try:
            logger.info(f"Downloading {s3_path} to {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file('ecmwf-forecasts', s3_path, local_path)
            logger.info("Download successful")
            return True
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def load_forecast(self, file_path):
        """Load forecast data from GRIB file."""
        try:
            # Load temperature data (2m height)
            t2m_ds = xr.open_dataset(
                file_path,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 2,
                    'stepType': 'instant',
                    'dataType': 'cf'
                }}
            )
            
            # Load wind data (10m height)
            wind_ds = xr.open_dataset(
                file_path,
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 10,
                    'stepType': 'instant',
                    'dataType': 'cf'
                }}
            )
            
            # Drop unnecessary coordinates
            t2m_ds = t2m_ds.drop_vars('heightAboveGround')
            wind_ds = wind_ds.drop_vars('heightAboveGround')
            
            # Merge datasets
            ds = xr.merge([t2m_ds, wind_ds])
            
            return ds
            
        except Exception as e:
            logger.error(f"Error loading forecast data: {str(e)}")
            raise
    
    def find_forecast_files(self, date):
        """Find forecast files for a given date."""
        # Check if date is after February 2024
        min_date = datetime(2024, 2, 1)
        if date < min_date:
            raise ValueError("Only data after February 2024 is supported")

        date_str = date.strftime('%Y%m%d')
        prefix = f'{date_str}/00z/ifs/0p25/oper/'
        
        # List objects in the bucket with the prefix
        response = self.s3_client.list_objects_v2(
            Bucket='ecmwf-forecasts',
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        # Get file paths
        files = [obj['Key'] for obj in response['Contents']]
        
        # Download files to cache
        local_files = []
        for file in files:
            local_path = os.path.join('cache', os.path.basename(file))
            os.makedirs('cache', exist_ok=True)
            
            if not os.path.exists(local_path):
                self.s3_client.download_file('ecmwf-forecasts', file, local_path)
            
            local_files.append(local_path)
        
        return local_files
    
    def load_weatherbench_climatology(self):
        """Load climatology data from WeatherBench2."""
        logger.info("Loading WeatherBench2 climatology data...")
        
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem(token='anon')
            
            # List available variables
            mapper = fs.get_mapper('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr/')
            ds = xr.open_zarr(mapper)
            
            logger.info(f"Available variables in WeatherBench2: {list(ds.data_vars)}")
            logger.info(f"Available coordinates in WeatherBench2: {list(ds.coords)}")
            logger.info(f"Dataset dimensions: {ds.dims}")
            
            # Rename variables to match forecast data
            ds = ds.rename({
                '2m_temperature': 't2m',
                '10m_u_component_of_wind': 'u10',
                '10m_v_component_of_wind': 'v10'
            })
            
            # Keep temperature in Kelvin (no conversion needed)
            logger.info(f"Climatology temperature range: {ds['t2m'].min().values:.2f}K to {ds['t2m'].max().values:.2f}K")
            
            # Calculate wind speed
            wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
            logger.info(f"Climatology wind speed range: {wind_speed.mean().values:.2f} m/s")
            
            return ds
            
        except Exception as e:
            logger.error(f"Error loading WeatherBench2 data: {str(e)}")
            raise

    def get_climatology_data(self, date_str=None):
        """
        Load climatology data for temperature comparison.
        
        Args:
            date_str (str, optional): Date string in YYYYMMDD format. If not provided,
                                    uses today's date.
        """
        logger.info("Loading climatology data")
        
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
            
        # Load the climatology dataset
        ds = self.load_weatherbench_climatology()
        
        # Convert date_str to datetime
        date = datetime.strptime(date_str, '%Y%m%d')
        
        # Find climatology for this date
        climatology = self.find_climatology(ds, date)
        
        logger.info("Successfully loaded climatology data")
        return climatology

    def find_climatology(self, ds, date):
        """Find climatology for a given date."""
        date = pd.to_datetime(date)
        
        # Get the day of year
        day_of_year = date.dayofyear
        
        # Get the hour (rounded to nearest 6-hour interval)
        hour = (date.hour // 6) * 6
        
        # Select data for this day and hour
        climatology = ds.sel(
            dayofyear=day_of_year,
            hour=hour,
            method='nearest'
        )
        
        return climatology

    def get_weekly_climatology(self, date_str):
        """Get climatology for the entire 7-day forecast period."""
        try:
            # Convert initialization date to datetime
            start_date = datetime.strptime(date_str, '%Y%m%d')
            
            # Load the full climatology dataset
            ds = self.load_weatherbench_climatology()
            
            # Initialize list to store climatology for each forecast step
            clim_steps = []
            
            # Get climatology for each 6-hour step up to 7 days
            for hour in range(0, 169, 6):  # 0 to 168 hours in 6-hour steps
                # Calculate the valid time for this forecast step
                valid_time = start_date + pd.Timedelta(hours=hour)
                
                # Get climatology for this specific time
                step_clim = self.find_climatology(ds, valid_time)
                clim_steps.append(step_clim)
            
            # Combine all timesteps into a single dataset
            climatology = xr.concat(clim_steps, dim='time')
            
            logger.info(f"Successfully loaded weekly climatology from {climatology.time.min().values} to {climatology.time.max().values}")
            logger.info(f"Climatology shape: {climatology.t2m.shape}")
            
            return climatology
            
        except Exception as e:
            logger.error(f"Error getting weekly climatology: {str(e)}")
            raise
