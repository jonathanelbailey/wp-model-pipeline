import itertools
import numpy as np


def predict_season(df, season, clf, features):
    season_df = df.loc[(df['season'] == season)].assign(
        predicted_winner = lambda x: clf.predict(x[features]),
        home_team_win_probability = lambda x: clf.predict_proba(x[features])[:,1])[
        ['home_team', 'away_team', 'week', 'predicted_winner', 'home_team_win_probability', 'home_team_win']]
    season_df['actual_winner'] = season_df.apply(lambda x: x.home_team if x.home_team_win else x.away_team, axis=1)
    season_df['predicted_winner'] = season_df.apply(lambda x: x.home_team if x.predicted_winner == 1 else x.away_team, axis=1)
    season_df['win_probability'] = season_df.apply(
        lambda x: x.home_team_win_probability if x.predicted_winner == x.home_team else 1 - x.home_team_win_probability, axis=1)
    season_df['correct_prediction'] = (season_df['predicted_winner'] == season_df['actual_winner']).astype(int)
    season_df = season_df.drop(columns=['home_team_win_probability', 'home_team_win'])
    return season_df.sort_values(by='win_probability', ascending=False).reset_index(drop=True)

def get_correct_predictions(df):
    correct = df.loc[df['correct_prediction'] == 1].groupby('week')['correct_prediction'].sum()
    num_games = df.groupby('week')['correct_prediction'].size()
    return correct / num_games


def get_season_prediction_outcomes(df, correct_predictions):
    return df.loc[df['week'] == correct_predictions.idxmax()].sort_values(by='win_probability', ascending=False)


def get_full_season_prediction_outcomes(df):
    return df.loc[df['week'] > 17]


def ewma(data, window):
    """
    Calculate the most recent value for EWMA given an array of data and a window size
    """
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    scale = 1 / alpha_rev
    n = data.shape[0]
    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0] * alpha_rev**(r+1)
    pw0 = alpha * alpha_rev**(n-1)
    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out[-1]


def get_pregame_predictions(df, season, home, away, clf):
    sb_df = df.loc[(df['season'] == season)]
    offense = sb_df.loc[(sb_df['posteam'] == home) | (sb_df['posteam'] == away)]
    defense = sb_df.loc[(sb_df['defteam'] == home) | (sb_df['defteam'] == away)]
    rushing_offense = offense.loc[offense['rush_attempt'] == 1].groupby(['posteam', 'week'], as_index=False)['epa'].mean().rename(columns={'posteam': 'team'})
    passing_offense = offense.loc[offense['pass_attempt'] == 1].groupby(['posteam', 'week'], as_index=False)['epa'].mean().rename(columns={'posteam': 'team'})
    rushing_defense = defense.loc[defense['rush_attempt'] == 1].groupby(['defteam', 'week'], as_index=False)['epa'].mean().rename(columns={'defteam': 'team'})
    passing_defense = defense.loc[defense['pass_attempt'] == 1].groupby(['defteam', 'week'], as_index=False)['epa'].mean().rename(columns={'defteam': 'team'})
    super_bowl = np.zeros(8)

    for i, (tm, stat_df) in enumerate(itertools.product([home, away], [rushing_offense, passing_offense, rushing_defense, passing_defense])):
        ewma_value = ewma(stat_df.loc[stat_df['team'] == tm]['epa'].values, 20)
        super_bowl[i] = ewma_value

    predicted_winner = clf.predict(super_bowl.reshape(1, 8))[0]
    predicted_proba = clf.predict_proba(super_bowl.reshape(1, 8))[0]
    winner = home if predicted_winner else away
    loser = home if not predicted_winner else away
    win_prob = predicted_proba[-1] if predicted_winner else predicted_proba[0]
    lose_prob = predicted_proba[-1] if not predicted_winner else predicted_proba[0]
    print(f'Model predicts {winner} will win and has a {round(win_prob*100, 2)}% win probability.  {loser} has {round(lose_prob*100, 2)}%')