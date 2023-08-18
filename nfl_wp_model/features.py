from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt


WP_MODEL_FEATURES = [
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "yardline_100",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "game_half",
    "down",
    "ydstogo",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "score_differential_post"
]

def get_data_with_features(df, season, features, target):
    df = df.dropna()
    x = df.loc[df['season'] != season, features].values
    y = df.loc[df['season'] != season, target].values
    clf = LogisticRegression()
    return x, y, clf


def get_most_important_features(df):
    features = [column for column in df.columns if 'ewma' in column and 'dynamic' in column]
    return features


def get_accuracy_scores(x, y, clf, cv, scoring):
    accuracy_scores = cross_val_score(clf, x, y, cv=cv)
    log_losses = cross_val_score(clf, x, y, cv=cv, scoring=scoring)
    model_accuracy = np.mean(accuracy_scores)
    neg_log_loss = np.mean(log_losses)

    return accuracy_scores, model_accuracy, log_losses, neg_log_loss


def plot_most_important_features(features, clf):
    fig, ax = plt.subplots()
    feature_names = ['_'.join(feature_name.split('_')[3:]) for feature_name in features]
    coef_ = clf.coef_[0]
    features_coef_sorted = sorted(zip(feature_names, coef_), key=lambda x: x[-1], reverse=True)
    features_sorted = [feature for feature, _ in features_coef_sorted]
    coef_sorted = [coef for _, coef in features_coef_sorted]
    ax.set_title('Feature Importance')
    ax.barh(features_sorted, coef_sorted)
    plt.show()
    return plt