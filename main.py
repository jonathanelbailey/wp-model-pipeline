import nfl_data_py as nfl

from nfl_wp_model.data import Data
from nfl_wp_model.data import Preprocess

# pbp_data = Data(import_csv='tests/test_data/test_pbp_df.csv', clean=True)
pbp_data = Data(nfl.import_pbp_data, 1999, 2023, clean=True)
pbp_df = Preprocess(pbp_data.data)

print(pbp_df.data.dtypes)
print(pbp_df.data.head())

from nfl_wp_model.model import Model

cross_validation = Model(pbp_df.data, train=True, maximize=True, test=True, train_season_subset=2022, num_splits=5,
                         init_points_factor=1, n_iter_factor=10, num_rounds=15000)
