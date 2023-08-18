import nfl_data_py as nfl
from preprocessing import (
    rename_column,
    update_column_type,
    set_index,
    reset_index,
    join_data_frames_by_index
)


def rename_qbr_game_id(df, column_name='espn'):
    df = rename_column(df, column_name)

    return df