import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def get_most_important_features(df):
    features = [column for column in df.columns if 'ewma' in column and 'dynamic' in column]
    return features


def get_data_with_features(df, season, features, target):
    df = df.dropna()
    X = df.loc[df['season'] != season, features].values
    y = df.loc[df['season'] != season, target].values
    clf = LogisticRegression()
    clf.fit(X, y)
    return X, y, clf


def get_accuracy_scores(X, y, clf, cv, scoring):
    accuracy_scores = cross_val_score(clf, X, y, cv=cv)
    log_losses = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    model_accuracy = np.mean(accuracy_scores)
    neg_log_loss = np.mean(log_losses)

    return accuracy_scores, model_accuracy, log_losses, neg_log_loss


def plot_most_important_features(features, clf):
    fig, ax = plt.subplots()
    feature_names = ['_'.join(feature_name.split('_')[3:]) for feature_name in features]
    coef_ = clf.coef_[0]
    features_coef_sorted = sorted(zip(feature_names, coef_), key=lambda x:x[-1], reverse=True)
    features_sorted = [feature for feature, _ in features_coef_sorted]
    coef_sorted = [coef for _, coef in features_coef_sorted]
    ax.set_title('Feature Importance')
    ax.barh(features_sorted, coef_sorted)
    plt.show()
    return plt