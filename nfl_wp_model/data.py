from typing import Callable, Optional

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from pandas import DataFrame


class Data:
    def __init__(self, import_function: Optional[Callable] = None, start_season: Optional[int] = None,
                 end_season: Optional[int] = None, clean: bool = False, import_csv: Optional[str] = None) -> DataFrame:
        """Imports Data from External Source

        Args:
            import_function (Callable): Import Function used to gather data
            start_season (int): Starting season year
            end_season (int): Ending season year
            clean (bool): clean data after import, default True
        Returns:
            self
        """
        self.csv_path: Optional[str] = import_csv
        self.clean: bool = clean
        if self.csv_path is not None:
            self.read_csv()
        else:
            self.import_function: Callable = import_function
            self.start_season: int = start_season
            self.end_season: int = end_season
            self.seasons_range: list[int] = list(range(self.start_season, self.end_season, 1))
            self.data = None
            self.get_data()
        if not isinstance(self.clean, bool):
            raise TypeError(f"Object {self.clean} passed to 'clean' argument is not bool.")
        elif self.clean is True:
            self.clean_data()

    def get_data(self):
        self.data = self.import_function(self.seasons_range)

    def clean_data(self):
        self.data = nfl.clean_nfl_data(self.data)

    def read_csv(self):
        self.data = pd.read_csv(self.csv_path, low_memory=False)


class Preprocess:
    def __init__(self, df: DataFrame, test_data=False) -> DataFrame:
        self._raw_data: DataFrame = df
        self.data: DataFrame = self._raw_data
        self.selected_columns: list[str] = [
            'label', 'game_id', 'home_team', 'away_team', 'season', 'receive_2h_ko', 'spread_time', 'home',
            'half_seconds_remaining', 'game_seconds_remaining', 'diff_time_ratio', 'score_differential',
            'down', 'ydstogo', 'yardline_100', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining'
        ]
        self.preprocess_data(test_data=test_data)

    def preprocess_data(self, test_data=False):
        self._set_column_types(['home_team', 'away_team'], 'category')
        self._add_label_column()
        self._add_home_column()
        self._add_receive_2h_ko_column()
        self._add_posteam_spread_column()
        self._add_elapsed_share_column()
        self._add_spread_time_column()
        self._add_diff_time_ratio_column()
        self._filter_rows_with_missing_values(test_data=test_data)
        self.select_columns()

    def select_columns(self):
        df = self.data.copy()
        df = df[self.selected_columns]
        self.data = df.copy()

    def _set_column_types(self, columns, type):
        df = self.data.copy()
        df[columns] = df[columns].astype(type)
        self.data = df.copy()

    def _add_label_column(self):
        df = self.data.copy()
        df['label'] = df.apply(lambda row: 1 if (row['result'] > 0 and row['posteam'] == row['home_team']) or (
                row['result'] < 0 and row['posteam'] == row['away_team']) else 0, axis=1)
        self.data = df.copy()

    def _add_home_column(self):
        df = self.data.copy()
        df['home'] = df.apply(lambda row: 1 if row['posteam'] == row['home_team'] else 0, axis=1)
        self.data = df.copy()

    def _add_receive_2h_ko_column(self):
        df = self.data.copy()
        df = df.groupby('game_id', group_keys=False).apply(lambda x: x.assign(
            receive_2h_ko=np.where((x['qtr'] <= 2) & (x['posteam'] == x['defteam'].dropna().iloc[0]), 1, 0)
        ))
        self.data = df.copy()

    def _add_posteam_spread_column(self):
        df = self.data.copy()
        df = df.assign(
            posteam_spread=np.where(df['home'] == 1, df['spread_line'], -1 * df['spread_line'])
        )
        self.data = df.copy()

    def _add_elapsed_share_column(self):
        df = self.data.copy()
        df = df.assign(
            elapsed_share=(3600 - df['game_seconds_remaining']) / 3600
        )
        self.data = df.copy()

    def _add_spread_time_column(self):
        df = self.data.copy()
        df = df.assign(
            spread_time=df['posteam_spread'] * np.exp(-4 * df['elapsed_share'])
        )
        self.data = df.copy()

    def _add_diff_time_ratio_column(self):
        df = self.data.copy()
        df = df.assign(
            diff_time_ratio=df['score_differential'] / (np.exp(-4 * df['elapsed_share']))
        )
        self.data = df.copy()

    def _filter_rows_with_missing_values(self, test_data=False):
        df = self.data.copy()
        if test_data is True:
            df = df.dropna(
                subset=['down', 'game_seconds_remaining', 'yardline_100', 'score_differential'])
        else:
            df = df.dropna(
                subset=['down', 'game_seconds_remaining', 'yardline_100', 'score_differential', 'result', 'posteam'])
            df = df[df['result'] != 0]
        df = df[df['qtr'] <= 4]
        self.data = df.copy()
