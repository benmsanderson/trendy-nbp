#!/usr/bin/env python3
"""
Process TRENDY model NBP files to calculate area-weighted global annual means.

Handles the diverse time encodings across TRENDY models by inferring temporal
resolution from the number of timesteps and deriving years from actual time values.

Output: CSV files in global_means/ with columns [model, experiment, year, nbp_global_mean]
Units are preserved as-is from each model (typically kg m-2 s-1).
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_lat_lon_names(ds):
    """Find the latitude and longitude dimension/coordinate names."""
    lat_name = None
    lon_name = None
    for name in ['latitude', 'lat']:
        if name in ds.dims:
            lat_name = name
            break
    for name in ['longitude', 'lon']:
        if name in ds.dims:
            lon_name = name
            break
    return lat_name, lon_name


def area_weighted_global_mean(ds, var_name='nbp'):
    """
    Calculate area-weighted global mean using cos(lat) weighting.
    Returns a 1D array of length n_time.
    """
    data = ds[var_name]
    lat_name, lon_name = get_lat_lon_names(ds)

    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find lat/lon dims in {list(ds.dims)}")

    lat_vals = ds[lat_name].values
    weights = np.cos(np.deg2rad(lat_vals))
    weights = np.maximum(weights, 0)  # clip negative weights near poles

    # Build xarray-compatible weight array
    w = xr.DataArray(weights, dims=[lat_name])
    # Weighted mean over spatial dims
    weighted_mean = data.weighted(w).mean(dim=[lat_name, lon_name])
    return weighted_mean.values


def derive_years(ds, n_times):
    """
    Derive an integer year for every timestep, handling the many different
    time encodings in TRENDY models.

    Strategy:
      1. If time values look like fractional years (range 1600-2100), use floor.
      2. If units contain a reference date, convert offsets to years.
      3. Fall back to counting from 1700.

    Returns: np.array of integer years for each timestep
    """
    time_var = ds['time']
    t = time_var.values.astype(float)
    units = time_var.attrs.get('units', '')
    calendar = time_var.attrs.get('calendar', 'standard').lower()

    # Determine days-per-year from calendar
    if '365' in calendar or 'noleap' in calendar:
        dpyr = 365.0
    elif '360' in calendar:
        dpyr = 360.0
    else:
        dpyr = 365.25

    # ---- Case 1: Time values look like actual years (fractional) ----
    # e.g. ELM-FATES: [1701.0, 1701.083, ...], VISIT-UT: [1700.042, ...]
    if np.nanmin(t) > 1500 and np.nanmax(t) < 2200:
        return np.floor(t).astype(int)

    # ---- Case 2: Parse reference date from units string ----
    year_match = re.search(r'since\s+(?:AD\s+)?(\d{1,4})', units)
    if year_match:
        ref_year = int(year_match.group(1))

        if 'month' in units.lower():
            # t = months since ref_year
            return np.floor(ref_year + t / 12.0).astype(int)

        elif 'hour' in units.lower():
            days = t / 24.0
            return np.floor(ref_year + days / dpyr).astype(int)

        elif 'day' in units.lower():
            return np.floor(ref_year + t / dpyr).astype(int)

        elif 'year' in units.lower():
            return np.floor(ref_year + t).astype(int)

    # ---- Case 3: No parseable units - infer from timestep count ----
    if n_times > 400:
        # Monthly, assume ends ~2024
        n_years = n_times // 12
        start_year = max(1700, 2024 - n_years + 1)
        return np.repeat(np.arange(start_year, start_year + n_years), 12)[:n_times]
    else:
        # Annual
        start_year = 1700
        return np.arange(start_year, start_year + n_times)


def process_file(filepath, output_dir):
    """Process a single NetCDF file and save annual area-weighted global means."""
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")

    # Skip mean_annual files
    if 'mean_annual' in filename or 'mean-annual' in filename:
        print(f"  Skipping - pre-computed annual file")
        return None

    # Parse model name and experiment
    match = re.match(r'^(.+)_(S\d+)_nbp\.nc', filename)
    if not match:
        print(f"  Skipping - unexpected filename format")
        return None

    model_name = match.group(1)
    experiment = match.group(2)

    try:
        ds = xr.open_dataset(filepath, decode_times=False)

        if 'time' not in ds.dims:
            print(f"  Skipping - no time dimension")
            ds.close()
            return None

        n_times = ds.sizes['time']

        # Handle special case: CARDAMOM has a 'Time' variable with better metadata
        if 'Time' in ds.data_vars:
            time_attrs = ds['Time'].attrs
            long_name = time_attrs.get('long_name', '')
            if 'since' in long_name:
                date_match = re.search(r'since\s+\d{2}/\d{2}/(\d{4})', long_name)
                if date_match:
                    ref_year = int(date_match.group(1))
                    days = ds['Time'].values
                    per_step_years = np.floor(ref_year + days / 365.25).astype(int)

                    global_mean = area_weighted_global_mean(ds)
                    df_monthly = pd.DataFrame({'year': per_step_years, 'nbp': global_mean})
                    df_annual = df_monthly.groupby('year')['nbp'].mean().reset_index()
                    df_annual.columns = ['year', 'nbp_global_mean']
                    df_annual['model'] = model_name
                    df_annual['experiment'] = experiment
                    df_annual = df_annual[['model', 'experiment', 'year', 'nbp_global_mean']]

                    output_file = output_dir / f"{model_name}_{experiment}_annual_global_mean.csv"
                    df_annual.to_csv(output_file, index=False)
                    print(f"  Saved {len(df_annual)} years ({df_annual.year.min()}-{df_annual.year.max()}) to {output_file.name}")
                    ds.close()
                    return df_annual

        # Derive year per timestep
        per_step_years = derive_years(ds, n_times)

        # Compute area-weighted global mean
        global_mean = area_weighted_global_mean(ds)

        # Build DataFrame and aggregate to annual
        df_all = pd.DataFrame({'year': per_step_years, 'nbp': global_mean})
        df_annual = df_all.groupby('year')['nbp'].mean().reset_index()
        df_annual.columns = ['year', 'nbp_global_mean']
        df_annual['model'] = model_name
        df_annual['experiment'] = experiment
        df_annual = df_annual[['model', 'experiment', 'year', 'nbp_global_mean']]

        # Sanity check
        yr_min, yr_max = df_annual.year.min(), df_annual.year.max()
        if yr_max > 2030 or yr_min < 1600:
            print(f"  WARNING: suspicious year range {yr_min}-{yr_max}")

        output_file = output_dir / f"{model_name}_{experiment}_annual_global_mean.csv"
        df_annual.to_csv(output_file, index=False)
        print(f"  Saved {len(df_annual)} years ({yr_min}-{yr_max}) to {output_file.name}")

        ds.close()
        return df_annual

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    output_dir = Path('global_means')
    output_dir.mkdir(exist_ok=True)

    nc_files = sorted(glob.glob('*_nbp.nc'))
    print(f"Found {len(nc_files)} NetCDF files\n")

    results = []
    for filepath in nc_files:
        result = process_file(filepath, output_dir)
        if result is not None:
            results.append(result)

    print(f"\nSuccessfully processed {len(results)} files -> {output_dir}/")


if __name__ == '__main__':
    main()
