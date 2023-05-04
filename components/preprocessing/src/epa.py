import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker


def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted EPA with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i+1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]

    return pd.Series(values, index=x.index)


def get_epa(df, mode, play_type):
    return df.loc[df[play_type] == 1, :].groupby([mode, 'season', 'week'], as_index=False)['epa'].mean()


def lag_epa_one_period(df, mode):
    return df.groupby(mode)['epa'].shift()


def calculate_ewma(df, mode):
    return df.groupby(mode)['epa_shifted'].transform(lambda x: x.ewm(min_periods=1, span=10).mean())


def calculate_dynamic_window_ewma(df, mode):
    return df.groupby(mode).apply(dynamic_window_ewma).values


def merge_data(rush_df, pass_df, mode):
    return rush_df.merge(pass_df, on=[mode, 'season', 'week'], suffixes=('_rushing', '_passing')).rename(columns={mode: 'team'})


def plot_epa(df, team):
    tm = df.loc[df['team'] == team, :].assign(
        season_week = lambda x: 'w' + x.week.astype(str) + ' (' + x.season.astype(str) + ')'
    ).set_index('season_week')
    fig, ax = plt.subplots()
    loc = plticker.MultipleLocator(base=16)
    ax.xaxis.set_major_locator(loc)
    ax.tick_params(axis='x', rotation=75)
    ax.plot(tm['epa_shifted_passing_offense'], lw=1, alpha=0.5)
    ax.plot(tm['ewma_dynamic_window_passing_offense'], lw=2)
    ax.plot(tm['ewma_passing_offense'], lw=2);
    plt.axhline(y=0, color='red', lw=1.5, alpha=0.5)
    ax.legend(['Passing EPA', 'EWMA on EPA with dynamic window', 'Static 10 EWMA on EPA'])
    ax.set_title(f'{team} Passing EPA per play')
    return plt


def get_schedule(dataset_df, epa_df):
    schedule = dataset_df[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates().reset_index(drop=True).assign(
        home_team_win = lambda x: (x.home_score > x.away_score).astype(int))
    return schedule.merge(epa_df.rename(columns={'team': 'home_team'}), on=['home_team', 'season', 'week']).merge(
        epa_df.rename(columns={'team': 'away_team'}), on=['away_team', 'season', 'week'], suffixes=('_home', '_away'))