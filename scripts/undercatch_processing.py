

import xarray as xr
import pandas as pd
from scipy.spatial import KDTree
from geopy.distance import geodesic
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from pathlib import Path

def ec_earth_to_csv(netcdf_file, csv_file):
    # Open the NetCDF file
    ds = xr.open_dataset(netcdf_file)
    
    # Extract the variables
    station_numbers = ds['station_number'].values
    latitudes = ds['latitude'].values
    longitudes = ds['longitude'].values
    
    # Combine the station ID parts into a single string for each station
    station_ids = [''.join(ds['station_ID'].isel(station_number=i).astype(str).values.flatten()) for i in range(len(station_numbers))]
    
    # Update to remove "GHCN_" prefix
    station_ids = [station_id.replace('GHCN_', '') for station_id in station_ids]

    # Create a DataFrame
    data = {
        'station_number': station_numbers,
        'station_id': station_ids,
        'latitude': latitudes,
        'longitude': longitudes
    }
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

def match_stations(file1_path, file2_path, output_path, lat1_col, lon1_col, id1_col, lat2_col, lon2_col, id2_col, max_distance_km=5):
    # Load the datasets
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Drop rows with missing latitude or longitude
    df1 = df1.dropna(subset=[lat1_col, lon1_col])
    df2 = df2.dropna(subset=[lat2_col, lon2_col])

    # Prepare the results list
    matches = []

    # Create KDTree for the second set of stations
    coords2 = df2[[lat2_col, lon2_col]].values
    tree2 = KDTree(coords2)

    # Iterate through the first set of stations
    for i, row1 in df1.iterrows():
        lat1 = row1[lat1_col]
        lon1 = row1[lon1_col]
        id1 = row1[id1_col]

        # Find the nearest station within the max distance
        dist, idx = tree2.query((lat1, lon1), distance_upper_bound=max_distance_km / 111)  # convert km to degrees

        # Check if a valid match is found
        if idx != len(coords2):
            row2 = df2.iloc[idx]
            id2 = row2[id2_col]

            # Calculate the precise geodesic distance
            exact_dist = geodesic((lat1, lon1), (row2[lat2_col], row2[lon2_col])).km
            if exact_dist <= max_distance_km:
                matches.append({
                    id1_col: id1,
                    id2_col: id2,
                    'Distance_km': exact_dist
                })

    # Convert the matches to a DataFrame and save to CSV
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(output_path, index=False)


def calculate_CE_backup(U,Tair,height='gh',shield='SA', T_thres=5):
    """
    Calculate the expression CE = e^(-a(U)(1 - tan^(-1)(b(Tair))) + c)
    
    Parameters:
    a (float): coefficient a
    U (float): variable U
    b (float): coefficient b
    Tair (float): air temperature Tair
    c (float): coefficient c
    
    Returns:
    float: the calculated CE value
    """

    if height == 'gh':
        U_thres=7.2
        if shield == 'UN':
            a = 0.0785
            b = 0.729
            c = 0.407
        elif shield == 'SA':
            a = 0.0348
            b = 1.366
            c = 0.779

    elif height == '10m':
        U_thres=9
        if shield == 'UN':
            a = 0.0623
            b = 0.776
            c = 0.431
        elif shield == 'SA':
            a = 0.0281
            b = 1.628
            c = 0.837
    else:
        raise ValueError("Invalid value for height or shield")

    if U > U_thres:
        U = U_thres

    CE = math.exp(-a * U * ((1 - math.atan(b * Tair))+ c))

    if Tair > T_thres:
        CE = 1

    return CE

def apply_undercatch_backup(prcp, CE):
    """
    Apply the undercatch correction to precipitation data
    
    Parameters:
    prcp (float): the precipitation value
    CE (float): the catch efficiency value
    
    Returns:
    float: the corrected precipitation value
    float: the catch efficiency value
    """
    if prcp > 0:
        corrected_prcp = prcp / CE
    else:
        corrected_prcp = 0

    return corrected_prcp
def calculate_CE(U, Tair, height='gh', shield='SA', T_thres=5):
    """
    Vectorized version of the catch efficiency calculation.

    Parameters:
    U (xarray.DataArray): Wind speed data.
    Tair (xarray.DataArray): Air temperature data.
    height (str): Measurement height ('gh' or '10m').
    shield (str): Shield type ('SA' or 'UN').
    T_thres (float): Temperature threshold above which CE is set to 1.
    
    Returns:
    xarray.DataArray: Calculated CE values.
    """

    if height == 'gh':
        U_thres = 7.2
        if shield == 'UN':
            a = 0.0785
            b = 0.729
            c = 0.407
        elif shield == 'SA':
            a = 0.0348
            b = 1.366
            c = 0.779

    elif height == '10m':
        U_thres = 9
        if shield == 'UN':
            a = 0.0623
            b = 0.776
            c = 0.431
        elif shield == 'SA':
            a = 0.0281
            b = 1.628
            c = 0.837
    else:
        raise ValueError("Invalid value for height or shield")

    # Limit wind speed to threshold
    if U <= U_thres:
        U > U_thres
    #U = U.where(U <= U_thres, U_thres)

    # Vectorized computation of CE
    CE = np.exp(-a * U * (1 - np.arctan(b * Tair) + c))

    # Set CE to 1 where Tair > T_thres
    if Tair > T_thres:
        CE = 1
    #CE = CE.where(Tair <= T_thres, 1)

    return CE

def apply_undercatch(prcp, CE):
    """
    Vectorized undercatch correction for precipitation data.

    Parameters:
    prcp (xarray.DataArray): Precipitation data.
    CE (xarray.DataArray): Catch efficiency data.
    
    Returns:
    xarray.DataArray: Corrected precipitation data.
    """
    corrected_prcp = prcp / CE
    if prcp == 0:
        corrected_prcp = 0
    #corrected_prcp = corrected_prcp.where(prcp > 0, 0)  # Apply correction only where prcp > 0
    return corrected_prcp

def correct_precipitation(input_file, output_file):
    """
    Efficiently corrects the precipitation data in the netCDF file using vectorized operations.

    Parameters:
    input_file (str): Path to the input netCDF file.
    output_file (str): Path to save the output netCDF file with corrected precipitation.
    """
    # Open the dataset
    ds = xr.open_dataset(input_file)

    # Rename the original prcp to original_prcp
    ds = ds.rename({'prcp': 'original_prcp'})

    # Get the variables: original_prcp (precipitation), wind (U), and tmean (Tair)
    prcp = ds['original_prcp']
    wind = ds['wind']
    Tair = ds['tmean']

    # Calculate the catch efficiency (CE) in a vectorized way
    CE = calculate_CE(wind, Tair)

    # Apply the undercatch correction in a vectorized way
    corrected_prcp = apply_undercatch(prcp, CE)

    # Add the corrected prcp to the dataset
    ds['prcp'] = corrected_prcp

    # Save the corrected dataset to a new file
    ds.to_netcdf(output_file)

    print(f"Corrected precipitation data saved to {output_file}")

def merge_station_datasets(input_files, output_file):
    """
    Merges multiple NetCDF files containing station data on 'station_number' and checks that 
    all 'station_ID' are unique post-merge. The merged dataset is saved to the specified output file.

    Parameters:
    files (list of str): List of file paths to the NetCDF files to be merged.
    output_file (str): File path to save the merged NetCDF file.
    
    Returns:
    bool: True if all station_IDs are unique after merging, False otherwise.
    """
    # Load all datasets
    datasets = [xr.open_dataset(file) for file in input_files]

    # Check uniqueness of station_ID in each dataset before merging
    for i, ds in enumerate(datasets):
        station_ids = ds['station_ID'].values
        if len(np.unique(station_ids)) != len(station_ids):
            raise ValueError(f"Dataset {input_files[i]} contains non-unique station_IDs.")
    
    # Merge datasets on 'station_number'
    merged_ds = xr.merge(datasets)

    # Check uniqueness of station_ID in the merged dataset
    merged_station_ids = merged_ds['station_ID'].values
    is_unique = len(np.unique(merged_station_ids)) == len(merged_station_ids)
    
    if not is_unique:
        raise ValueError("Station_IDs are not unique after merging.")

    # Save the merged dataset to the specified output file
    merged_ds.to_netcdf(output_file)
    print(f"Merged dataset saved to {output_file}")

def create_test_figure():

    # Create some data
    U = np.linspace(0, 10, 100)
    Tair = [-20,-2,-0.5,0,0.5,2,5]

    # Calculate the catch efficiency
    CE = np.zeros((len(Tair), len(U)))
    for i, t in enumerate(Tair):
        for j, u in enumerate(U):
            CE[i, j] = calculate_CE(U=u, Tair=t)
    

    # Create a figure and axis
    fig, ax = plt.subplots()
    for i in range(len(Tair)):
        ax.plot(U, CE[i], label=f"T-Air = {Tair[i]}")
        
    ax.set_xlabel('U')
    ax.set_ylabel('CE')
    ax.set_title('Catch Efficiency at Gauge Height (m s^-1)')
    # add a grid
    ax.grid(True)
    # set x-axis and y-axis limits
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()

def calculate_daily_averages_from_eccc_paper(hourly_data_file, daily_data_file):

    # Define columns
    columns = ["YYYYMMDDThhmm", "Unadj_P(mm)", "Tair(C)", "Wind(m/s)", "Wind_Flag", 
               "CE", "UTF_Adj_P(mm)", "CODECON(mm)", "UTF_Adj+CODECON_P(mm)", "Adj_Flag"]
    
    # Read the data file
    df = pd.read_csv(hourly_data_file, delim_whitespace=True, skiprows=2, names=columns)
    
    # Replace -99999 with NaN
    df.replace(-99999, np.nan, inplace=True)
    
    # Extract date part from datetime
    df['Date'] = df['YYYYMMDDThhmm'].str[:8]
    
    # Calculate daily averages, ignoring flags
    daily_avg = df.groupby('Date').agg({
        "Unadj_P(mm)": "sum",
        "Tair(C)": "mean",
        "Wind(m/s)": "mean",
        "CE": "mean",
        "UTF_Adj_P(mm)": "sum",
        "CODECON(mm)": "mean",
        "UTF_Adj+CODECON_P(mm)": "sum"
    }).reset_index()
    
    # Write daily averages to a CSV file
    daily_avg.to_csv(daily_data_file, index=False)
    print(f"Daily averages have been written to {daily_data_file}")

def extract_and_separate_stations(input_file, output_file_with_wind, output_file_without_wind, output_file_no_prcp_tmean_trange, start_date=None, end_date=None):
    """
    Extracts stations from the input netCDF file and separates them based on:
    1. Stations with `prcp`, `tmean`, `trange`, and `wind` data.
    2. Stations with `prcp`, `tmean`, `trange` but no `wind` data.
    3. Stations missing one or more of `prcp`, `tmean`, or `trange`.
    
    Optionally filters data within a user-defined date range.

    Parameters:
    input_file (str): Path to the input netCDF file.
    output_file_with_wind (str): Path to the output netCDF file for stations with wind data.
    output_file_without_wind (str): Path to the output netCDF file for stations without wind data.
    output_file_no_prcp_tmean_trange (str): Path to the output netCDF file for stations missing `prcp`, `tmean`, or `trange` data.
    start_date (str): Start of the date range (in 'YYYY-MM-DD' format). Optional.
    end_date (str): End of the date range (in 'YYYY-MM-DD' format). Optional.
    """
    # Open the input netCDF file
    ds = xr.open_dataset(input_file)

    # Apply date range filtering if start_date and end_date are provided
    if start_date or end_date:
        ds = ds.sel(time=slice(start_date, end_date))
    
    # Check that the variables prcp, tmean, trange exist in the dataset
    required_vars = ['prcp', 'tmean', 'trange']
    missing_vars = [var for var in required_vars if var not in ds.variables]
    
    if missing_vars:
        raise ValueError(f"The following required variables are missing from the dataset: {missing_vars}")
    
    # Ensure station_ID is handled correctly (convert to fixed-length string if necessary)
    if ds['station_ID'].dtype == object:  # assuming station_ID contains strings
        ds['station_ID'] = ds['station_ID'].astype('str')  # Convert to string type
    
    # Check if wind data exists
    has_wind = 'wind' in ds.variables
    
    # Extract station numbers where prcp, tmean, trange are present across all times
    stations_with_data = ds['station_number'].where(
        (ds['prcp'].notnull().all(dim='time')) & 
        (ds['tmean'].notnull().all(dim='time')) & 
        (ds['trange'].notnull().all(dim='time')),
        drop=True
    )
    
    # Extract station numbers where prcp, tmean, or trange are missing across any time
    stations_without_data = ds['station_number'].where(
        (ds['prcp'].isnull().any(dim='time')) | 
        (ds['tmean'].isnull().any(dim='time')) | 
        (ds['trange'].isnull().any(dim='time')),
        drop=True
    )
    
    # Split stations based on the presence of wind data
    if has_wind:
        stations_with_wind = ds['station_number'].where(
            (ds['prcp'].notnull().all(dim='time')) & 
            (ds['tmean'].notnull().all(dim='time')) & 
            (ds['trange'].notnull().all(dim='time')) & 
            (ds['wind'].notnull().all(dim='time')),
            drop=True
        )
        stations_without_wind = ds['station_number'].where(
            (ds['prcp'].notnull().all(dim='time')) & 
            (ds['tmean'].notnull().all(dim='time')) & 
            (ds['trange'].notnull().all(dim='time')) & 
            (ds['wind'].isnull().all(dim='time')),
            drop=True
        )
    else:
        stations_with_wind = None
        stations_without_wind = stations_with_data
    
    # Subset datasets
    if stations_with_wind is not None:
        subset_with_wind = ds.sel(station_number=stations_with_wind)
        subset_without_wind = ds.sel(station_number=stations_without_wind)
        
        # Save the subsets to new netCDF files
        subset_with_wind.to_netcdf(output_file_with_wind)
        subset_without_wind.to_netcdf(output_file_without_wind)
    else:
        ds.sel(station_number=stations_without_wind).to_netcdf(output_file_without_wind)
    
    # Save the stations without prcp, tmean, or trange data
    if len(stations_without_data) > 0:
        subset_no_prcp_tmean_trange = ds.sel(station_number=stations_without_data)
        subset_no_prcp_tmean_trange.to_netcdf(output_file_no_prcp_tmean_trange)

def find_nearest_grid_point(grid_lats, grid_lons, station_lat, station_lon):
    """
    Find the nearest grid point in the dataset for a given station's latitude and longitude.

    Parameters:
    grid_lats (np.array): Latitude values from the gridded dataset.
    grid_lons (np.array): Longitude values from the gridded dataset.
    station_lat (float): The latitude of the station.
    station_lon (float): The longitude of the station.

    Returns:
    tuple: Indices of the nearest latitude and longitude in the dataset.
    """
    lat_diff = np.abs(grid_lats - station_lat)
    lon_diff = np.abs(grid_lons - station_lon)
    
    nearest_lat_idx = lat_diff.argmin()
    nearest_lon_idx = lon_diff.argmin()

    return nearest_lat_idx, nearest_lon_idx

def add_variable_to_stations(stations_ds, grid_ds, grid_var_name,station_var_name, lat_name='latitude', lon_name='longitude'):
    """
    Add data from a gridded dataset for a specific variable to the stations dataset by matching each station with the nearest grid point.

    Parameters:
    stations_ds (xarray.Dataset): Dataset containing station information (latitude, longitude).
    grid_ds (xarray.Dataset): Gridded dataset containing the variable data.
    variable_name (str): The variable name to be added from the gridded dataset (e.g., 'windspd', 'temperature').
    lat_name (str): The name of the latitude dimension in the gridded dataset.
    lon_name (str): The name of the longitude dimension in the gridded dataset.
    time_name (str): The name of the time dimension in the gridded dataset.

    Returns:
    xarray.Dataset: The stations dataset with added variable data.
    """
    grid_lats = grid_ds[lat_name].values
    grid_lons = grid_ds[lon_name].values

    # Create an empty array for the variable data to store for each station
    var_data = np.full(stations_ds['prcp'].shape, np.nan)  # Same shape as time x stations

    for i, (station_lat, station_lon) in enumerate(zip(stations_ds['latitude'].values, stations_ds['longitude'].values)):
        # Find nearest grid point for the current station
        nearest_lat_idx, nearest_lon_idx = find_nearest_grid_point(grid_lats, grid_lons, station_lat, station_lon)
        
        # Extract the variable data from the nearest grid point
        var_at_grid_point = grid_ds[grid_var_name].isel({lat_name: nearest_lat_idx, lon_name: nearest_lon_idx}).values
        
        # Match time dimensions and add the data for this station
        var_data[i, :] = var_at_grid_point[:len(stations_ds['time'].values)]
    
    # Add the variable data to the stations dataset
    stations_ds[station_var_name] = (('station_number', 'time'), var_data)

    return stations_ds

def add_gridded_data_to_stations(input_file, grid_file_path, output_file, grid_var_name, station_var_name,lat_name='latitude', lon_name='longitude'):
    """
    Main function to load station and gridded dataset data, add a specified variable to the stations dataset, and save to a new NetCDF file.

    Parameters:
    input_file (str): Path to the input NetCDF file containing station data.
    grid_file_path (str): Path to the gridded dataset (assumed to be in NetCDF format).
    variable_name (str): The variable name to extract from the gridded dataset (e.g., 'windspd', 'temperature').
    output_file (str): Path to the output NetCDF file to save the modified data with the added variable.
    lat_name (str): The name of the latitude dimension in the gridded dataset.
    lon_name (str): The name of the longitude dimension in the gridded dataset.
    time_name (str): The name of the time dimension in the gridded dataset.
    """
    # Load station data
    stations_ds = xr.open_dataset(input_file)

    # Load ERA5 dataset (assuming multiple files in the folder)
    grid_files = [os.path.join(grid_file_path, f) for f in os.listdir(grid_file_path) if f.endswith('.nc')]
    grid_ds = xr.open_mfdataset(grid_files, combine='by_coords')

    # Add the specified variable from the gridded dataset to the station dataset
    updated_stations_ds = add_variable_to_stations(stations_ds, grid_ds, grid_var_name,station_var_name, lat_name, lon_name)

    # Save updated dataset to the output file
    updated_stations_ds.to_netcdf(output_file)

if __name__ == '__main__':
    plot = create_test_figure()
    hourly_data_file = '/Users/dcasson/Data/pems/undercatch/3050519_UTF_hly_prec.txt'
    daily_data_file_csv = '/Users/dcasson/Data/pems/undercatch/3050519_daily.csv'
    #calculate_daily_averages_from_eccc_paper(hourly_data_file, daily_data_file_csv)

    """
    canada_station_inventory = pd.read_csv('/Users/dcasson/Data/pems/station_data/CanadaStationInventory.csv')
    ghcnd_stations = pd.read_csv('/Users/dcasson/Data/pems/station_data/ghcnd-stations.csv')

    matches_bow= match_stations(
        national_df=canada_station_inventory,
        ghcnd_df=ghcnd_stations,
        national_lat_col='Latitude (Decimal Degrees)',
        national_lon_col='Longitude (Decimal Degrees)',
        ghcnd_lat_col='Latitude',
        ghcnd_lon_col='Longitude',
        station_prefix='CA',
        threshold_km=1
    )
    matches_bow.to_csv('/Users/dcasson/Data/pems/station_data/matches_bow.csv', index=False)
    """