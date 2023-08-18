import numpy as np


def rename_column(df, column_name, new_name):
    df = df.rename(columns={column_name: new_name})
    return df


def update_column_type(df, column_name, type):
    df = df.astype({column_name: type})
    return df


def set_index(df, index):
    df = df.set_index(index)
    return df


def reset_index(df):
    df = df.reset_index(drop=True)
    return df


def join_data_frames_by_index(left_df, right_df, index, column_name):
    set_index(left_df, index)
    set_index(right_df, index)
    for idx, row in left_df.iterrows():
        if idx in right_df.index.tolist():
            right_df.at[idx, column_name] = row[column_name]

    result = reset_index(right_df)

    return result


def prepare_wp_data(pbp):
    pbp = pbp.groupby('game_id').apply(lambda x: x.assign(
        receive_2h_ko = np.where((x['qtr'] <= 2) & (x['posteam'] == x['defteam'].dropna().iloc[0]), 1, 0)
    ))
    pbp = pbp.assign(
        posteam_spread = np.where(pbp['home'] == 1, pbp['spread_line'], -1 * pbp['spread_line']),
        elapsed_share = (3600 - pbp['game_seconds_remaining']) / 3600,
        spread_time = pbp['posteam_spread'] * np.exp(-4 * pbp['elapsed_share']),
        Diff_Time_Ratio = pbp['score_differential'] / (np.exp(-4 * pbp['elapsed_share']))
    )

    return pbp