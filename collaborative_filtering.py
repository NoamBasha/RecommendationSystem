# Noam Basha 208847228

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp


class collaborative_filtering:
    def __init__(self):
        self.ratings_matrix = None

        self.user_based_matrix = None
        self.item_based_matrix = None

        self.mean_user_rating = None
        self.ratings_diff = None

        self.ratings = None
        self.movies = None

        self.user_ids_sorted = None
        self.movie_ids_sorted = None
        self.movie_ids_sorted_np = None

    """
    def create_fake_user(self, ratings):
        # Adding ratings same as user 283225   
        ratings.loc[ratings.shape[0]] = [283238, 50, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 288, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 762, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 1260, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 1377, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 1617, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 1792, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 1909, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 1953, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 2126, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 6874, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 6957, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 6953, 1.0]
        ratings.loc[ratings.shape[0]] = [283238, 14, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 17, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 25, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 62, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 105, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 147, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 175, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 193, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 222, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 249, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 277, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 300, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 307, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 337, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 381, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 475, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 524, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 802, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 902, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 926, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 1041, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 1293, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 1358, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 1680, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 1952, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 1962, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 2020, 5.0]
        ratings.loc[ratings.shape[0]] = [283238, 3186, 5.0]

        # ratings.loc[ratings.shape[0]] = [283238,3424,5.0]
        # ratings.loc[ratings.shape[0]] = [283238,3556,5.0]
        # ratings.loc[ratings.shape[0]] = [283238,3967,5.0]
        # ratings.loc[ratings.shape[0]] = [283238,4308,5.0]
        # ratings.loc[ratings.shape[0]] = [283238,4995,5.0]

        return ratings
    """

    def create_matrix_base(self, data):
        self.ratings = data[0]
        self.movies = data[1]
        # self.ratings = self.create_fake_user(self.ratings)
        self.user_ids_sorted = sorted(list(set(self.ratings['userId'])))
        self.movie_ids_sorted = sorted(list(set(self.ratings['movieId'])))
        self.movie_ids_sorted_np = np.array(self.movie_ids_sorted)
        self.ratings_matrix = pd.pivot_table(self.ratings, index="userId", columns="movieId")
        self.mean_user_rating = self.ratings_matrix.mean(axis=1).to_numpy().reshape(-1, 1)
        self.ratings_diff = (self.ratings_matrix - self.mean_user_rating)
        self.ratings_diff[np.isnan(self.ratings_diff)] = 0

    def create_user_based_matrix(self, data):
        self.create_matrix_base(data)

        user_similarity = 1 - pairwise_distances(self.ratings_diff, metric='cosine')

        predictions = self.mean_user_rating + user_similarity.dot(self.ratings_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T
        predictions = np.nan_to_num(predictions)

        self.user_based_matrix = predictions

    def create_item_based_matrix(self, data):
        self.create_matrix_base(data)

        item_similarity = 1 - pairwise_distances(self.ratings_diff.T, metric='cosine')
        predictions = self.mean_user_rating + self.ratings_diff.dot(item_similarity) / np.array(
            [np.abs(item_similarity).sum(axis=1)])
        predictions = predictions.to_numpy()
        predictions = np.nan_to_num(predictions)

        self.item_based_matrix = predictions

    def predict_k_movie_id(self, user_id, k, is_user_based=True):
        user_id_sorted_index = self.user_ids_sorted.index(int(user_id))

        user_ratings_matrix_row = self.ratings_matrix.iloc[user_id_sorted_index].to_numpy()
        rated_indicies = np.where(~np.isnan(user_ratings_matrix_row))[0]
        user_rated_movies = self.movie_ids_sorted_np[rated_indicies]

        if is_user_based:
            predictions = self.user_based_matrix
        else:
            predictions = self.item_based_matrix

        user_id_prediction = predictions[int(user_id_sorted_index)]

        user_predictions_and_movies = list(zip(self.movie_ids_sorted, user_id_prediction))
        user_predictions_and_movies.sort(key=lambda tup: tup[1], reverse=True)
        user_all_movies = [movie for (movie, rating) in user_predictions_and_movies]

        movies_predictions = []
        counter = 0
        for movie in user_all_movies:
            if movie not in user_rated_movies:
                movies_predictions.append(movie)
                counter += 1
                if counter == k:
                    break

        return movies_predictions

    def get_titles(self, predictions):
        titles = []
        for movie in predictions:
            title = self.movies[self.movies['movieId'] == movie]['title'].iloc[0]
            titles.append(title)
        return titles

    def predict_movies(self, user_id, k, is_user_based=True):
        movies_predictions = self.predict_k_movie_id(user_id, k, is_user_based)

        return self.get_titles(movies_predictions)
