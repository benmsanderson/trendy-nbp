#!/usr/bin/env python3
"""
Build pivot-table CSVs from per-model global mean files.

Reads global_means/*_annual_global_mean.csv and produces one CSV per experiment
in pivot_tables/, with models as rows and years as columns.

Values are converted from kg C m-2 s-1 to PgC/yr.
"""

import pandas as pd
import glob
from pathlib import Path

# Load all per-model CSVs
csv_files = sorted(glob.glob('global_means/*_annual_global_mean.csv'))
print(f"Found {len(csv_files)} CSV files")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Convert from area-weighted global mean (kg C m-2 s-1) to global total (PgC/yr)
SECONDS_PER_YEAR = 365.25 * 24 * 3600
EARTH_LAND_AREA = 1.496e14  # m²
KG_TO_PG = 1e-12
df['nbp_global_mean'] = df['nbp_global_mean'] * EARTH_LAND_AREA * SECONDS_PER_YEAR * KG_TO_PG

# Output directory
out_dir = Path('pivot_tables')
out_dir.mkdir(exist_ok=True)

# One CSV per experiment: models as rows, years as columns
for exp in ['S0', 'S1', 'S2', 'S3']:
    sub = df[df['experiment'] == exp]
    pivot = sub.pivot_table(index='model', columns='year', values='nbp_global_mean')
    pivot.columns = pivot.columns.astype(int)
    pivot = pivot.sort_index()
    outfile = out_dir / f'nbp_{exp}.csv'
    pivot.to_csv(outfile)
    print(f'{outfile}: {pivot.shape[0]} models x {pivot.shape[1]} years '
          f'({pivot.columns.min()}-{pivot.columns.max()})')
