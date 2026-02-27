#!/usr/bin/env python3
"""
Process TRENDY model NBP files to decadal-mean fields on a common 2°×2° grid.

For each model × experiment file:
  1. Open the raw NetCDF, derive years for each timestep
  2. Standardise coordinate names and longitude convention (-180..180)
  3. Compute decadal means (temporal averages within each decade)
  4. Regrid to a common 2°×2° regular lat-lon grid (bilinear interp)
  5. Save as a compact NetCDF in  decadal_grids/

Target grid:  90 lat (-89,-87,...,89)  ×  180 lon (-179,-177,...,179)
Units are preserved (kg C m-2 s-1).
"""

import xarray as xr
import numpy as np
import glob
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Target grid ─────────────────────────────────────────────────────
TARGET_LAT = np.arange(-89, 90, 2, dtype=np.float64)   # 90 points
TARGET_LON = np.arange(-179, 180, 2, dtype=np.float64)  # 180 points

# ── Helpers ─────────────────────────────────────────────────────────

def get_lat_lon_names(ds):
    """Return (lat_dim_name, lon_dim_name) used in *ds*."""
    lat_name = lon_name = None
    for n in ds.dims:
        nl = n.lower()
        if nl.startswith('lat') and lat_name is None:
            lat_name = n
        elif nl.startswith('lon') and lon_name is None:
            lon_name = n
    return lat_name, lon_name


def standardise_coords(ds, lat_name, lon_name):
    """
    Rename lat/lon dims to 'lat'/'lon', convert lon to -180..180,
    sort both axes ascending.

    Also handles files where the dimension uses index values (0,1,2,...)
    but real coordinates live in a separate 'latitude'/'longitude' variable.
    """
    # ── Replace index-valued dims with real coordinate values ────────
    lat_vals = ds[lat_name].values.astype(float)
    lon_vals = ds[lon_name].values.astype(float)

    # Detect index-like lat: values are 0,1,2,... or very small range
    lat_looks_indexed = (lat_vals.min() >= 0 and lat_vals.max() < 1000
                         and np.allclose(lat_vals, np.arange(len(lat_vals))))
    lon_looks_indexed = (lon_vals.min() >= 0 and lon_vals.max() < 1000
                         and np.allclose(lon_vals, np.arange(len(lon_vals))))

    if lat_looks_indexed:
        # Look for a real latitude coordinate (1-D, same length)
        for cname in ['latitude', 'lat_values']:
            if cname in ds and cname != lat_name and ds[cname].ndim == 1 and len(ds[cname]) == len(lat_vals):
                ds = ds.assign_coords({lat_name: ds[cname].values})
                break

    if lon_looks_indexed:
        for cname in ['longitude', 'lon_values']:
            if cname in ds and cname != lon_name and ds[cname].ndim == 1 and len(ds[cname]) == len(lon_vals):
                ds = ds.assign_coords({lon_name: ds[cname].values})
                break

    # ── Rename to 'lat' / 'lon' ─────────────────────────────────────
    rename_map = {}
    if lat_name != 'lat':
        rename_map[lat_name] = 'lat'
    if lon_name != 'lon':
        rename_map[lon_name] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)

    # Convert 0..360 → -180..180
    lon_vals = ds['lon'].values
    if lon_vals.max() > 180:
        ds = ds.assign_coords(lon=((ds['lon'] + 180) % 360 - 180))
        ds = ds.sortby('lon')

    ds = ds.sortby('lat')
    return ds


def derive_years(ds, n_times):
    """
    Derive an integer year for every timestep.
    Mirrors the logic in process_global_means.py — handles
    months-since, days-since, years-since, fractional-year, and
    CARDAMOM's special 'Time' variable.
    """
    # ── CARDAMOM special case ──
    if 'Time' in ds.data_vars:
        long_name = ds['Time'].attrs.get('long_name', '')
        date_match = re.search(r'since\s+\d{2}/\d{2}/(\d{4})', long_name)
        if date_match:
            ref_year = int(date_match.group(1))
            days = ds['Time'].values.astype(float)
            return np.floor(ref_year + days / 365.25).astype(int)

    time_var = ds['time']
    t = time_var.values.astype(float)
    units = time_var.attrs.get('units', '')
    calendar = time_var.attrs.get('calendar', 'standard').lower()

    if '365' in calendar or 'noleap' in calendar:
        dpyr = 365.0
    elif '360' in calendar:
        dpyr = 360.0
    else:
        dpyr = 365.25

    # Fractional years (e.g. VISIT-UT, ELM-FATES)
    if np.nanmin(t) > 1500 and np.nanmax(t) < 2200:
        return np.floor(t).astype(int)

    # Parse "since <year>" from units
    year_match = re.search(r'since\s+(?:AD\s+)?(\d{1,4})', units)
    if year_match:
        ref_year = int(year_match.group(1))
        if 'month' in units.lower():
            return np.floor(ref_year + t / 12.0).astype(int)
        elif 'hour' in units.lower():
            return np.floor(ref_year + t / 24.0 / dpyr).astype(int)
        elif 'day' in units.lower():
            return np.floor(ref_year + t / dpyr).astype(int)
        elif 'year' in units.lower():
            return np.floor(ref_year + t).astype(int)

    # Fallback
    if n_times > 400:
        n_years = n_times // 12
        start_year = max(1700, 2024 - n_years + 1)
        return np.repeat(np.arange(start_year, start_year + n_years), 12)[:n_times]
    else:
        return np.arange(1700, 1700 + n_times)


# ── Main processing ─────────────────────────────────────────────────

def process_file(filepath, output_dir):
    """
    Open one ``{MODEL}_{EXP}_nbp.nc`` file, compute decadal means on
    the common 2°×2° grid, and save the result.
    """
    filename = os.path.basename(filepath)

    # Skip non-standard files
    if 'mean_annual' in filename or 'mean-annual' in filename:
        print(f"  Skipping {filename} (pre-computed file)")
        return False

    match = re.match(r'^(.+)_(S\d+)_nbp\.nc', filename)
    if not match:
        print(f"  Skipping {filename} (unexpected name)")
        return False

    model_name = match.group(1)
    experiment  = match.group(2)

    # Skip if output already exists
    out_name = f"{model_name}_{experiment}_decadal_nbp.nc"
    out_path = output_dir / out_name
    if out_path.exists():
        print(f"  {out_name} already exists – skipping")
        return True

    print(f"Processing {model_name} {experiment} ...", end=' ', flush=True)

    try:
        ds = xr.open_dataset(filepath, decode_times=False)

        if 'time' not in ds.dims:
            print("no time dim – skipped")
            ds.close()
            return False

        n_times = ds.sizes['time']
        years = derive_years(ds, n_times)

        lat_name, lon_name = get_lat_lon_names(ds)
        if lat_name is None or lon_name is None:
            print(f"could not identify lat/lon dims {list(ds.dims)} – skipped")
            ds.close()
            return False

        # Standardise to 'lat', 'lon', -180..180
        ds = standardise_coords(ds, lat_name, lon_name)

        nbp = ds['nbp']  # (time, lat, lon)  — order may vary

        # Add 'year' coordinate and a 'decade' grouping variable
        ds = ds.assign_coords(year=('time', years))
        decade = (years // 10) * 10
        ds = ds.assign_coords(decade=('time', decade))

        # ── Decadal mean ────────────────────────────────────────────
        # Load into memory to avoid repeated disk I/O during groupby
        nbp_data = ds['nbp'].load()
        dec_mean = nbp_data.groupby('decade').mean(dim='time')
        # dec_mean has dims (decade, lat, lon)

        # ── Regrid to 2°×2° ─────────────────────────────────────────
        dec_regrid = dec_mean.interp(
            lat=TARGET_LAT, lon=TARGET_LON,
            method='linear',
            kwargs={'fill_value': None},   # extrapolate NaN at edges
        )

        # ── Assemble output dataset ─────────────────────────────────
        decades = np.sort(np.unique(decade))
        out = xr.Dataset(
            {
                'nbp': (['decade', 'lat', 'lon'], dec_regrid.values.astype(np.float32)),
            },
            coords={
                'decade': decades,
                'lat': TARGET_LAT,
                'lon': TARGET_LON,
            },
        )
        out['nbp'].attrs = {
            'long_name': 'Net Biome Productivity (decadal mean)',
            'units': 'kg m-2 s-1',
            'positive': 'land carbon sink',
        }
        out.attrs = {
            'model': model_name,
            'experiment': experiment,
            'description': f'Decadal-mean NBP from {model_name} {experiment}, '
                           f'regridded to 2°×2° regular lat-lon grid.',
            'source': filename,
        }

        # ── Save ────────────────────────────────────────────────────
        out_name = f"{model_name}_{experiment}_decadal_nbp.nc"
        out_path = output_dir / out_name

        encoding = {
            'nbp': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
            'decade': {'dtype': 'int32'},
            'lat': {'dtype': 'float64'},
            'lon': {'dtype': 'float64'},
        }
        out.to_netcdf(out_path, encoding=encoding)

        size_kb = out_path.stat().st_size / 1024
        print(f"{len(decades)} decades, {size_kb:.0f} KB → {out_name}")
        ds.close()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    output_dir = Path('decadal_grids')
    output_dir.mkdir(exist_ok=True)

    nc_files = sorted(glob.glob('*_nbp.nc'))
    print(f"Found {len(nc_files)} NetCDF files\n")

    ok = 0
    for f in nc_files:
        if process_file(f, output_dir):
            ok += 1

    print(f"\nDone – processed {ok} files → {output_dir}/")


if __name__ == '__main__':
    main()
