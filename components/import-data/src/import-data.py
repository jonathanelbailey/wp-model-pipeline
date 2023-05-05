#!/usr/bin/env python3

import argparse
from pathlib import Path
import nfl_data_py as nfl
import pandas as pd

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Download file from public HTTP/S URL')
# Paths must be passed in, not hardcoded
parser.add_argument('--starting-season',
                    type=str,
                    required=True,
                    help='Beginning of the range of seasons to import')
parser.add_argument('--ending-season',
                    type=str,
                    required=True,
                    help='End of the range of seasons to import')
parser.add_argument('--table-output-path',
                    type=str,
                    required=True,
                    help='Path of the local file where the Output data should be written.')
args = parser.parse_args()

print(f'Program arguments: {args}')

# Creating the directory where the output file is created (the directory
# may or may not exist).
output_dir = Path(args.table_output_path).parent
try:
    output_dir.mkdir(parents=True, exist_ok=True)
except Exception as ex:
    raise RuntimeError(f'Error creating output directory {output_dir}: {ex}')

data = pd.concat([nfl.import_pbp_data(range(args.starting_season, args.ending_season))])
nfl.clean_nfl_data(data)
data.head()
try:
    data.to_csv(args.table_output_path, index=False)
except OSError as ose:
    raise RuntimeError(f'Error creating output file '
                       f'{args.table_output_path}: {ose}')
