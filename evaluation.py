# Noam Basha 208847228

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
import math


def precision_10(test_set, cf, is_user_based=True):
    unique_users_len = len(test_set.userId.unique())
    test_set = test_set.loc[test_set['rating'] >= 4]
    unique_users = test_set.userId.unique()

    hits = 0
    for user in unique_users:
        user_movies = test_set.loc[test_set['userId'] == user]['movieId'].tolist()
        user_10_predictions = cf.predict_k_movie_id(user, 10, is_user_based)
        hits += len(set(user_movies).intersection(user_10_predictions))

    val = hits / (10 * unique_users_len)
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based=True):
    unique_users_len = len(test_set.userId.unique())
    test_set = test_set.loc[test_set['rating'] >= 4]
    unique_users = test_set.userId.unique()

    score = 0
    for user in unique_users:
        user_movies = test_set.loc[test_set['userId'] == user]['movieId'].tolist()
        user_10_predictions = cf.predict_k_movie_id(user, 10, is_user_based)

        for idx, predicted_movie in enumerate(user_10_predictions):
            if predicted_movie in user_movies:
                score += 1 / (idx + 1)

    val = score / unique_users_len
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based=True):
    if is_user_based:
        actual = cf.user_based_matrix
    else:
        actual = cf.item_based_matrix

    predicted = pd.pivot_table(test_set, index="userId", columns="movieId").to_numpy()

    all_predicted = []
    all_actual = []

    for user_id in test_set.userId.unique():
        user_id_sorted_index = cf.user_ids_sorted.index(int(user_id))
        predicted_i = predicted[user_id_sorted_index]
        not_nan_indices = np.argwhere(~np.isnan(predicted_i))

        predicted_i = predicted_i[not_nan_indices]
        actual_i = actual[user_id_sorted_index][not_nan_indices]

        all_predicted.extend(predicted_i)
        all_actual.extend(actual_i)

    val = math.sqrt(mean_squared_error(all_predicted, all_actual))
    print("RMSE: " + str(val))
