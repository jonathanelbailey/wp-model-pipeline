#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import jinja2
import json
from epa import (
    get_epa,
    lag_epa_one_period,
    calculate_ewma,
    calculate_dynamic_window_ewma,
    merge_data,
    plot_epa,
    get_schedule
)

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Count rows in input file')
# Paths must be passed in, not hardcoded
parser.add_argument('--input-file-path',
                    type=str,
                    required=True,
                    help='Path of the file to be processed.')
parser.add_argument('--preprocessed-data-path',
                    type=str,
                    required=True,
                    help='Path of the preprocessed data output.')
parser.add_argument('--epa-graph-output',
                    type=str,
                    required=True,
                    help='Path of the rendered EPA Average Graph.')
args = parser.parse_args()

print('Input arguments:')
print(args)

# Creating the directory where the output file is created (the directory
# may or may not exist).
preprocessed_data_output_dir = Path(args.preprocessed_data_path).parent
epa_graph_output_dir = Path(args.epa_graph_output).parent

try:
    preprocessed_data_output_dir.mkdir(parents=True, exist_ok=True)
except Exception as ex:
    raise RuntimeError(f'Error creating output directory {preprocessed_data_output_dir}: {ex}')

try:
    epa_graph_output_dir.mkdir(parents=True, exist_ok=True)
except Exception as ex:
    raise RuntimeError(f'Error creating output directory {epa_graph_output_dir}: {ex}')

data = pd.read_csv(args.input_file_path)
print("CSV HEAD:\n")
data.head()

# seperate EPA in to rushing offense, rushing defense, passing offense, passing defense for each team
rushing_offense_epa = get_epa(data, 'posteam', 'rush_attempt')
rushing_defense_epa = get_epa(data, 'defteam', 'rush_attempt')
passing_offense_epa = get_epa(data, 'posteam', 'pass_attempt')
passing_defense_epa = get_epa(data, 'defteam', 'pass_attempt')

# lag EPA one period back
rushing_offense_epa['epa_shifted'] = lag_epa_one_period(rushing_offense_epa, 'posteam')
rushing_defense_epa['epa_shifted'] = lag_epa_one_period(rushing_defense_epa, 'defteam')
passing_offense_epa['epa_shifted'] = lag_epa_one_period(passing_offense_epa, 'posteam')
passing_defense_epa['epa_shifted'] = lag_epa_one_period(passing_defense_epa, 'defteam')

# In each case, calculate EWMA with a static window and dynamic window and assign it as a column
rushing_offense_epa['ewma'] = calculate_ewma(rushing_offense_epa, 'posteam')
rushing_offense_epa['ewma_dynamic_window'] = calculate_dynamic_window_ewma(rushing_offense_epa, 'posteam')

rushing_defense_epa['ewma'] = calculate_ewma(rushing_defense_epa, 'defteam')
rushing_defense_epa['ewma_dynamic_window'] = calculate_dynamic_window_ewma(rushing_defense_epa, 'defteam')

passing_offense_epa['ewma'] = calculate_ewma(passing_offense_epa, 'posteam')
passing_offense_epa['ewma_dynamic_window'] = calculate_dynamic_window_ewma(passing_offense_epa, 'posteam')

passing_defense_epa['ewma'] = calculate_ewma(passing_defense_epa, 'defteam')
passing_defense_epa['ewma_dynamic_window'] = calculate_dynamic_window_ewma(passing_defense_epa, 'defteam')

# Merge all the data together
offense_epa = merge_data(rushing_offense_epa, passing_offense_epa, 'posteam')
defense_epa = merge_data(rushing_defense_epa, passing_defense_epa, 'defteam')

epa = offense_epa.merge(defense_epa, on=['team', 'season', 'week'], suffixes=('_offense', '_defense'))

# remove the first season of data
epa = epa.loc[epa['season'] != epa['season'].unique()[0], :]

epa = epa.reset_index(drop=True)

df = get_schedule(data, epa)

print("Preprocessed data HEAD:\n")
df.head()

fig = plot_epa(epa, 'LA')
fig.savefig(args.epa_graph_output, format='svg', dpi=1200)
df.to_csv(args.preprocessed_data_path, index=False)