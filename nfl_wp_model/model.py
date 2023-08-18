import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GroupKFold

from nfl_wp_model.data import Preprocess


class Model:
    def __init__(self, df, train_season_subset, num_splits=None, init_points_factor=None, n_iter_factor=None,
                 num_rounds=None, train=False, maximize=False, test=False):
        self.data = df
        self.test_data = None
        self.test_data_subset = None
        self.train_data = None
        self.train_labels = None
        self.folds = None
        self.unnecessary_columns = ['season', 'game_id', 'label']
        self.bounds = {
            'colsample_bytree': (0.1, 1),
            'eta': (0.01, 0.3),
            'gamma': (0, 20),
            'max_depth': (1, 16),
            'min_child_weight': (0, 48),
            'subsample': (0.1, 1)
        }
        self.bayesian_opt = None
        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'sampling_method': 'gradient_based',
            'eval_metric': 'auc',
            'eta': 0.2453,
            'gamma': 9.956,
            'subsample': 0.9023,
            'max_depth': 10,
            'min_child_weight': 30.66
        }
        self.model = None
        self.feature_importance = None
        self.test_data_predictions = None
        self.logloss_score = None
        self.accuracy = None
        self.num_rounds = num_rounds
        if train:
            self.generate_training_data(train_season_subset, num_splits)
            if maximize:
                self.begin_bayesian_optimization(init_points_factor, n_iter_factor)
                self.get_optimized_parameters()
            if test:
                self.generate_test_data(seasons_subset=train_season_subset)
            self.train(num_rounds)

    def generate_training_data(self, seasons_subset, num_splits=5):
        df = self.data[self.data['season'] < seasons_subset]
        self.train_labels = df['label']
        group_kfold = GroupKFold(n_splits=num_splits)
        self.folds = list(group_kfold.split(X=df, y=df['game_id'], groups=df['game_id']))
        df = df.drop(self.unnecessary_columns, axis=1)
        self.train_data = xgb.DMatrix(data=df, label=self.train_labels, enable_categorical=True, nthread=-1)

    def generate_test_data(self, df=None, seasons_subset=None):
        if seasons_subset is not None:
            df = self.data.copy()
            df = df[df['season'] >= seasons_subset]
            df.reset_index(drop=True, inplace=True)
        self.test_data = df.copy()
        df_subset = df.copy()
        df_subset = df_subset.drop(self.unnecessary_columns, axis=1)
        self.test_data_subset = xgb.DMatrix(data=df_subset, enable_categorical=True, nthread=-1)

    @staticmethod
    def generate_test_dataframe(df_dict):
        df = pd.DataFrame(df_dict)
        processed_df = Preprocess(df, test_data=True)
        return processed_df.data

    def _scoring_function(self, colsample_bytree, eta, max_depth, min_child_weight, subsample, gamma):

        monotone_constraints = {
            'home_team': 0,
            'away_team': 0,
            'receive_2h_ko': 0,
            'spread_time': 1,
            'home': 0,
            'half_seconds_remaining': 0,
            'game_seconds_remaining': 0,
            'diff_time_ratio': 1,
            'score_differential': 1,
            'down': -1,
            'ydstogo': -1,
            'yardline_100': -1,
            'posteam_timeouts_remaining': 1,
            'defteam_timeouts_remaining': -1
        }

        params = {
            'booster': 'gbtree',
            'colsample_bytree': colsample_bytree,
            'eta': eta,
            'max_depth': int(max_depth),
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'gamma': gamma,
            'sampling_method': 'gradient_based',
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'eval_metric': 'logloss',
            # 'monotone_constraints': monotone_constraints
        }

        xgbcv = xgb.cv(
            params=params,
            dtrain=self.train_data,
            folds=self.folds,
            num_boost_round=self.num_rounds,
            early_stopping_rounds=50
        )
        return max((xgbcv['test-logloss-mean'] * -1))

    def begin_bayesian_optimization(self, init_points_factor=1, n_iter_factor=10):
        bayesian_opt = BayesianOptimization(f=self._scoring_function, pbounds=self.bounds)
        bayesian_opt.maximize(init_points=len(self.bounds) * init_points_factor,
                              n_iter=len(self.bounds) * n_iter_factor)
        self.bayesian_opt = bayesian_opt

    def get_optimized_parameters(self):
        optimized_params = self.bayesian_opt.max['params']
        optimized_params['max_depth'] = int(optimized_params['max_depth'])
        self.params.update(optimized_params)

    def train(self, num_rounds=15000):
        model = xgb.train(params=self.params, dtrain=self.train_data, num_boost_round=num_rounds, verbose_eval=2)
        self.model = model

    def get_importance(self):
        self.feature_importance = self.model.get_score(importance_type='weight')
        xgb.plot_importance(self.feature_importance)

    def get_scores(self):
        preds = pd.DataFrame(self.model.predict(self.test_data_subset),
                             columns=['wp'])
        self.test_data_predictions = pd.concat([preds, self.test_data], axis=1)
        self.logloss_score = log_loss(self.test_data_predictions['label'], self.test_data_predictions['wp'])
        self.accuracy = accuracy_score(self.test_data_predictions['label'], self.test_data_predictions['wp'] > 0.5)

    def plot_calibration_score(self):
        self.test_data_predictions['bin_pred_prob'] = np.round(self.test_data_predictions['wp'] / 0.05) * 0.05
        calibration_data = self.test_data_predictions.groupby('bin_pred_prob').agg(
            n_plays=('label', 'count'),
            n_wins=('label', lambda x: np.sum(x == 1))
        ).reset_index()
        calibration_data['bin_actual_prob'] = calibration_data['n_wins'] / calibration_data['n_plays']
        ann_text = pd.DataFrame({
            'x': [0.25, 0.75],
            'y': [0.75, 0.25],
            'lab': ["More times\nthan expected", "Fewer times\nthan expected"]
        })
        plt.figure(figsize=(10, 6))

        sns.scatterplot(data=calibration_data, x='bin_pred_prob', y='bin_actual_prob', size='n_plays')
        sns.regplot(data=calibration_data, x='bin_pred_prob', y='bin_actual_prob', scatter=False, lowess=True)
        plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Estimated win probability")
        plt.ylabel("Observed win probability")
        plt.title("Win Prob Calibration Plot")
        plt.text(0.25, 0.75, "More times\nthan expected", fontsize=10)
        plt.text(0.75, 0.25, "Fewer times\nthan expected", fontsize=10)
        plt.legend(['Perfect Calibration', 'Plays'])
        plt.tight_layout()
        plt.show()

    def predict(self, df_dict):
        df = self.generate_test_dataframe(df_dict)
        self.generate_test_data(df)
        win_probability = self.model.predict(self.test_data_subset)
        return win_probability
