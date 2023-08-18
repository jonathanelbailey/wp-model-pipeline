from matplotlib import pyplot as plt
import matplotlib.ticker as plticker


def get_epa(df, mode, play_type):
    return df.loc[df[play_type] == 1, :].groupby([mode, 'season', 'week'], as_index=False)['epa'].mean()


def plot_passing_epa(df, team):
    tm = df.loc[df['team'] == team, :].assign(
        season_week=lambda x: 'w' + x.week.astype(str) + ' (' + x.season.astype(str) + ')'
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