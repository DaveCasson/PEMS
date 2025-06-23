import os
import re
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
from rasterio.mask import mask
from rasterio.features import geometry_window
from rasterio.warp import transform_geom
from datetime import datetime

def read_tif_files(directory, pattern="SWE"):
    tif_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
    # Update to pattern check if "SWE" in filename
    tif_files = [f for f in tif_files if pattern in f]
    return tif_files

def extract_date_from_filename(filename):
    date_str = re.search(r'\d{8}', filename).group()
    return datetime.strptime(date_str, '%Y%m%d')

def reproject_geometries(hrus, src_crs):
    hrus_reprojected = hrus.to_crs(src_crs)
    return hrus_reprojected

def calculate_mean_snow_depth(tif_file, hrus):
    with rasterio.open(tif_file) as src:
        src_crs = src.crs.to_string()
        hrus_reprojected = reproject_geometries(hrus, src_crs)

        mean_snow_depths = []
        for hru in hrus_reprojected.itertuples():
            hru_mask = [hru.geometry]
            snow_depth_hru, _ = mask(src, hru_mask, crop=True, filled=True, nodata=-9999)
            snow_depth_hru = np.ma.masked_equal(snow_depth_hru, -9999)
            mean_snow_depth = snow_depth_hru.mean()
            mean_snow_depths.append((hru.HRU_ID, mean_snow_depth))

    return mean_snow_depths

def aggregate_snow_depths(tif_files, hrus):
    snow_depths = {}
    for tif_file in tif_files:
        date = extract_date_from_filename(tif_file)
        mean_snow_depths = calculate_mean_snow_depth(tif_file, hrus)
        for hru_id, mean_snow_depth in mean_snow_depths:
            if hru_id not in snow_depths:
                snow_depths[hru_id] = []
            snow_depths[hru_id].append((date, mean_snow_depth))
    return snow_depths

def write_to_netcdf(snow_data,var_name, output_file):

    hru_ids = list(snow_data.keys())
    dates = sorted({date for values in snow_data.values() for date, _ in values})

    snow_data_output = np.full((len(hru_ids), len(dates)), np.nan)

    for i, hru_id in enumerate(hru_ids):
        for date, mean_snow_depth in snow_data[hru_id]:
            j = dates.index(date)
            snow_data_output[i, j] = mean_snow_depth

    dataset = xr.Dataset(
        {
            var_name: (["HRU_ID", "time"], snow_data_output),
        },
        coords={
            "HRU_ID": hru_ids,
            "time": dates
        }
    )
    dataset.to_netcdf(output_file)

def main():
    directory = '/Users/dcasson/Data/snow_data/tuolumne_lidar/'
    hrus_file = '//Users/dcasson/Data/summa_snakemake/hydrofabric_tuolumne/watershed_tools/tuolumne_hydrofabric_gru.gpkg'
    snow_depth_file = '/Users/dcasson/Data/snow_data/tuolumne_lidar/tuolumne_snow_depth_data.nc'
    swe_file = '/Users/dcasson/Data/snow_data/tuolumne_lidar/tuolumne_swe_data.nc'

    hrus = gpd.read_file(hrus_file)

    swe_files = read_tif_files(directory,'SWE')
    swe = aggregate_snow_depths(swe_files, hrus)

    snow_depth_files = read_tif_files(directory,'SD')
    sd = aggregate_snow_depths(snow_depth_files, hrus)

    write_to_netcdf(sd, "snow_depth", snow_depth_file)
    write_to_netcdf(swe, "swe", swe_file)

if __name__ == "__main__":
    main()