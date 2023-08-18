import pandas as pd
import numpy as np


def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted EPA with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i + 1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]

    return pd.Series(values, index=x.index)


def lag_epa_one_period(df, mode):
    return df.groupby(mode)['epa'].shift()


def calculate_ewma(df, mode):
    return df.groupby(mode)['epa_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())


def calculate_dynamic_window_ewma(df, mode):
    return df.groupby(mode).apply(dynamic_window_ewma).values


def merge_epa_data(rush_df, pass_df, mode):
    return rush_df.merge(pass_df, on=[mode, 'season', 'week'], suffixes=('_rushing', '_passing')).rename(
        columns={mode: 'team'})