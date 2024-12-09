"""
ECMWF Forecast Analysis
----------------------
Analyzes ECMWF forecast data for temperature anomalies and high wind events.
"""

import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from data_loader import ECMWFDataLoader
import gcsfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_climatology():
    """Load ERA5 climatology data from Google Cloud Storage."""
    try:
        logger.info("Loading climatology data...")
        fs = gcsfs.GCSFileSystem(token='anon')
        mapper = fs.get_mapper('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr/')
        ds = xr.open_zarr(mapper)
        
        
        logger.info(f"Original coordinates:")
        logger.info(f"Latitude range: [{ds.latitude.min().values}, {ds.latitude.max().values}]")
        logger.info(f"Longitude range: [{ds.longitude.min().values}, {ds.longitude.max().values}]")
        
        # Rename variables to match forecast names
        ds = ds.rename({
            '2m_temperature': 't2m',
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10'
        })
        
        
        if ds.longitude.max() > 180:
            logger.info("Converting longitude coordinates from [0, 360) to [-180, 180)")
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby('longitude')
            logger.info(f"New longitude range: [{ds.longitude.min().values}, {ds.longitude.max().values}]")
        
        # Select region of interest (Continental USA)
        ds = ds.sel(
            latitude=slice(49, 25),    # North to South
            longitude=slice(-125, -67)  # West to East
        )
        
        logger.info(f"Selected coordinates:")
        logger.info(f"Latitude range: [{ds.latitude.min().values}, {ds.latitude.max().values}]")
        logger.info(f"Longitude range: [{ds.longitude.min().values}, {ds.longitude.max().values}]")
        logger.info(f"Loaded climatology with shape: {ds.t2m.shape}")
        
        return ds
    except Exception as e:
        logger.error(f"Error loading climatology: {e}")
        raise

def compute_weekly_average(data: xr.DataArray, time_dim: str = 'time') -> xr.DataArray:
    """
    Compute weekly averages from a time series.
    
    Args:
        data: Input DataArray
        time_dim: Name of time dimension
    
    Returns:
        Weekly averaged DataArray
    """
    try:
        # Convert time to datetime if needed
        if not np.issubdtype(data[time_dim].dtype, np.datetime64):
            data[time_dim] = pd.to_datetime(data[time_dim])
        
        # Group by week and compute mean
        weekly = data.resample({time_dim: '7D'}).mean()
        return weekly
        
    except Exception as e:
        logger.error(f"Error computing weekly average: {e}")
        raise

def compute_anomalies(forecast, climatology):
    """
    Compute temperature anomalies using weekly averages.
    
    Args:
        forecast: Forecast dataset
        climatology: Climatology dataset
    
    Returns:
        Temperature anomalies
    """
    try:
        logger.info("Computing temperature anomalies with weekly averaging...")
        
        # Ensure coordinates are aligned
        forecast = forecast.sel(
            latitude=slice(49, 25),
            longitude=slice(-125, -67)
        )
        
        # Get the days and hours from forecast
        forecast_days = forecast.time.dt.dayofyear
        forecast_hours = forecast.time.dt.hour
        
        logger.info(f"Selecting climatology data for days {forecast_days.min().values} to {forecast_days.max().values}")
        
        # Select matching climatology data
        clim_subset = climatology.sel(
            dayofyear=forecast_days,
            hour=forecast_hours,
            method='nearest'
        )
        
        # Interpolate climatology to forecast grid if needed
        if not np.array_equal(forecast.latitude, clim_subset.latitude) or \
           not np.array_equal(forecast.longitude, clim_subset.longitude):
            logger.info("Interpolating climatology to forecast grid")
            clim_subset = clim_subset.interp(
                latitude=forecast.latitude,
                longitude=forecast.longitude
            )
        
        # Create time coordinate for climatology matching forecast times
        clim_subset = clim_subset.assign_coords(time=forecast.time)
        
        # Compute weekly averages
        logger.info("Computing weekly averages...")
        forecast_weekly = compute_weekly_average(forecast.t2m)
        clim_weekly = compute_weekly_average(clim_subset.t2m)
        
        # Compute anomalies
        anomalies = forecast_weekly - clim_weekly
        
        logger.info(f"Computed anomalies with shape: {anomalies.shape}")
        return anomalies
        
    except Exception as e:
        logger.error(f"Error computing anomalies: {e}")
        raise

def compute_wind_speed(forecast):
    """Compute wind speed from U and V components."""
    try:
        logger.info("Computing wind speeds...")
        
        
        logger.info(f"Wind forecast coordinates:")
        logger.info(f"Latitude range: [{forecast.latitude.min().values}, {forecast.latitude.max().values}]")
        logger.info(f"Longitude range: [{forecast.longitude.min().values}, {forecast.longitude.max().values}]")
        
        # Select region of interest
        forecast = forecast.sel(
            latitude=slice(49, 25),
            longitude=slice(-125, -67)
        )
        
        logger.info(f"Selected wind forecast coordinates:")
        logger.info(f"Latitude range: [{forecast.latitude.min().values}, {forecast.latitude.max().values}]")
        logger.info(f"Longitude range: [{forecast.longitude.min().values}, {forecast.longitude.max().values}]")
        
       
        wind_speed = np.sqrt(forecast.u10**2 + forecast.v10**2)
        
        logger.info(f"Computed wind speeds with shape: {wind_speed.shape}")
        return wind_speed
    except Exception as e:
        logger.error(f"Error computing wind speed: {e}")
        raise

def save_extreme_locations(wind_speed, anomalies, output_dir):
    """Save locations of extreme events."""
    try:
        
        wind_speed = wind_speed.compute()
        anomalies = anomalies.compute()
        
        # Find extreme wind locations
        high_winds = wind_speed > 15  # m/s
        wind_locs = []
        
        # Get high wind points
        high_wind_points = wind_speed.where(high_winds, drop=True)
        if high_wind_points.size:
            for lat in high_wind_points.latitude.values:
                for lon in high_wind_points.longitude.values:
                    # Get all values for this lat/lon and take the maximum
                    speed_values = wind_speed.sel(latitude=lat, longitude=lon)
                    if isinstance(speed_values, xr.DataArray):
                        speed = float(speed_values.max().values)
                    else:
                        speed = float(speed_values)
                    if speed > 15:  
                        wind_locs.append({
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'wind_speed': speed
                        })
        
        # Find extreme temperature locations
        temp_locs = []
        extreme_temps = (anomalies > 5) | (anomalies < -5)
        extreme_temp_points = anomalies.where(extreme_temps, drop=True)
        if extreme_temp_points.size:
            for lat in extreme_temp_points.latitude.values:
                for lon in extreme_temp_points.longitude.values:
                    # Get all values for this lat/lon and take the maximum absolute anomaly
                    anom_values = anomalies.sel(latitude=lat, longitude=lon)
                    if isinstance(anom_values, xr.DataArray):
                        anom = float(anom_values.max().values)
                    else:
                        anom = float(anom_values)
                    if abs(anom) > 5:  
                        temp_locs.append({
                            'latitude': float(lat),
                            'longitude': float(lon),
                            'temperature_anomaly': anom
                        })
        
        # Save to files
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(wind_locs).to_csv(f'{output_dir}/extreme_winds.csv', index=False)
        pd.DataFrame(temp_locs).to_csv(f'{output_dir}/extreme_temperatures.csv', index=False)
        
        logger.info(f"Found {len(wind_locs)} locations with high winds")
        logger.info(f"Found {len(temp_locs)} locations with extreme temperatures")
        
    except Exception as e:
        logger.error(f"Error saving extreme locations: {e}")
        raise

def create_visualization(wind_speed, anomalies, output_dir):
    """Create visualization of wind speeds and temperature anomalies with extreme events highlighted."""
    try:
        logger.info("Creating visualizations...")
        
       
        wind_speed = wind_speed.compute()
        anomalies = anomalies.compute()
        
        # Take the maximum value across time dimension for visualization
        wind_speed_max = wind_speed.max(dim='time')
        anomalies_max = anomalies.max(dim='time')
        
        
        proj = ccrs.LambertConformal(
            central_longitude=-96,
            central_latitude=37
        )
        
        
        plt.figure(figsize=(20, 25), dpi=300)
        
        # Plot 1: Clean temperature anomalies without markers
        ax1 = plt.subplot(311, projection=proj)
        im1 = plt.pcolormesh(
            anomalies_max.longitude,
            anomalies_max.latitude,
            anomalies_max.values,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=-10,
            vmax=10,
            shading='auto'
        )
        plt.colorbar(im1, ax=ax1, label='Maximum Temperature Anomaly (K)', extend='both')
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES, linewidth=0.5)
        ax1.set_title('Temperature Anomalies')
        
        # Plot 2: Temperature anomalies with extreme markers
        ax2 = plt.subplot(312, projection=proj)
        im2 = plt.pcolormesh(
            anomalies_max.longitude,
            anomalies_max.latitude,
            anomalies_max.values,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=-10,
            vmax=10,
            shading='auto'
        )
        plt.colorbar(im2, ax=ax2, label='Maximum Temperature Anomaly (K)', extend='both')
        
       
        extreme_temp_mask = (anomalies_max > 5) | (anomalies_max < -5)
        extreme_temp_points = np.where(extreme_temp_mask)
        if len(extreme_temp_points[0]) > 0:
            lats = anomalies_max.latitude.values[extreme_temp_points[0]]
            lons = anomalies_max.longitude.values[extreme_temp_points[1]]
            ax2.scatter(
                lons,
                lats,
                transform=ccrs.PlateCarree(),
                color='black',
                marker='x',
                s=50,
                label='Extreme Temperature'
            )
        ax2.coastlines()
        ax2.add_feature(cfeature.STATES, linewidth=0.5)
        ax2.set_title('Temperature Anomalies with Extreme Events Marked')
        
        # Plot 3: Wind speeds
        ax3 = plt.subplot(313, projection=proj)
        im3 = plt.pcolormesh(
            wind_speed_max.longitude,
            wind_speed_max.latitude,
            wind_speed_max.values,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            vmin=0,
            vmax=30,
            shading='auto'
        )
        plt.colorbar(im3, ax=ax3, label='Maximum Wind Speed (m/s)', extend='max')
        
       
        high_wind_mask = wind_speed_max > 15
        high_wind_points = np.where(high_wind_mask)
        if len(high_wind_points[0]) > 0:
            lats = wind_speed_max.latitude.values[high_wind_points[0]]
            lons = wind_speed_max.longitude.values[high_wind_points[1]]
            ax3.scatter(
                lons,
                lats,
                transform=ccrs.PlateCarree(),
                color='red',
                marker='x',
                s=50,
                label='High Winds'
            )
        ax3.coastlines()
        ax3.add_feature(cfeature.STATES, linewidth=0.5)
        ax3.set_title('Maximum Wind Speeds with High Wind Events Marked')
        
       
        for ax in [ax1, ax2, ax3]:
            ax.gridlines(draw_labels=True)
            ax.legend()
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/analysis_plots.png', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_dir}/analysis_plots.png")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise

def create_compound_events_visualization(wind_speed, anomalies, output_dir):
    """Create visualization of compound events (extreme temperatures and high winds)."""
    try:
        logger.info("Creating compound events visualization...")
        
       
        wind_speed = wind_speed.compute()
        anomalies = anomalies.compute()
    
        wind_speed_max = wind_speed.max(dim='time')
        anomalies_max = anomalies.max(dim='time')
        
        
        proj = ccrs.LambertConformal(
            central_longitude=-96,
            central_latitude=37
        )
        
        
        plt.figure(figsize=(20, 10), dpi=300)
        
        
        
        ax1 = plt.subplot(121, projection=proj)
        
        
        extreme_temp_mask = (anomalies_max > 5) | (anomalies_max < -5)
        high_wind_mask = wind_speed_max > 15
        
        compound_events = np.zeros_like(anomalies_max)
        compound_events[extreme_temp_mask] = 1  # Temperature only
        compound_events[high_wind_mask] = 2     # Wind only
        compound_events[extreme_temp_mask & high_wind_mask] = 3  # Both
        
       
        colors = ['lightgray', 'blue', 'red', 'purple']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        
        im1 = plt.pcolormesh(
            anomalies_max.longitude,
            anomalies_max.latitude,
            compound_events,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=-0.5,
            vmax=3.5
        )
        
        
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3])
        cbar1.set_ticklabels(['No Events', 'Extreme Temperature', 'High Winds', 'Both'])
        
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES, linewidth=0.5)
        ax1.gridlines(draw_labels=True)
        ax1.set_title('Distribution of Extreme Events')
        
        # 2. Intensity plot
        ax2 = plt.subplot(122, projection=proj)
        
        # Create intensity score (normalized combination of temperature and wind anomalies)
        temp_intensity = np.abs(anomalies_max) / 5  # Normalize by threshold
        wind_intensity = wind_speed_max / 15        # Normalize by threshold
        combined_intensity = (temp_intensity + wind_intensity) / 2
        
        
        im2 = plt.pcolormesh(
            anomalies_max.longitude,
            anomalies_max.latitude,
            combined_intensity,
            transform=ccrs.PlateCarree(),
            cmap='YlOrRd',
            vmin=0,
            vmax=2
        )
        
        plt.colorbar(im2, ax=ax2, label='Combined Event Intensity\n(>1 indicates extreme conditions)')
        
        ax2.coastlines()
        ax2.add_feature(cfeature.STATES, linewidth=0.5)
        ax2.gridlines(draw_labels=True)
        ax2.set_title('Combined Event Intensity')
        
        
        plt.suptitle(f'Compound Events Analysis\nTime Range: {wind_speed.time.min().values} to {wind_speed.time.max().values}',
                    y=1.02)
        
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/compound_events.png', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Compound events visualization saved to {output_dir}/compound_events.png")
        
    except Exception as e:
        logger.error(f"Error creating compound events visualization: {e}")
        raise

def main():
    """Main analysis function."""
    try:
       
        loader = ECMWFDataLoader()
        
        # Load forecast data 
        start_date = '20240229'  
        logger.info(f"Loading 7-day forecast initialized on {start_date} 00Z")
        
        forecast = loader.get_forecast_data(start_date)
        if forecast is None:
            raise ValueError("Failed to load forecast data")
        
        logger.info(f"Loaded forecast from {forecast.time.min().values} to {forecast.time.max().values}")
        logger.info(f"Forecast contains {len(forecast.time)} timesteps")
        
        
        climatology = load_climatology()
        
        anomalies = compute_anomalies(forecast, climatology)
       
        wind_speed = compute_wind_speed(forecast)
        
        save_extreme_locations(wind_speed, anomalies, 'output')

        create_visualization(wind_speed, anomalies, 'output')
        create_compound_events_visualization(wind_speed, anomalies, 'output')
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 