# Noam Basha 208847228

import sys
import matplotlib.pyplot as plt
import seaborn as sns


def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def get_max_key_and_value(dict):
    values = list(dict.values())
    keys = list(dict.keys())
    max_value = max(values)
    max_value_index = values.index(max_value)
    max_key = keys[max_value_index]
    return max_key, max_value


def get_min_key_and_value(dict):
    values = list(dict.values())
    keys = list(dict.keys())
    min_value = min(values)
    min_value_index = values.index(min_value)
    min_key = keys[min_value_index]
    return min_key, min_value


def print_data(data):
    # data = (rating, movies)
    ratings = data[0]

    # Getting the userId column
    user_id_list = ratings['userId']
    # removing duplicates
    user_id_list_no_dup = list(dict.fromkeys(user_id_list))
    # How many unique users rated the movies?
    num_unique_users_ratings = len(user_id_list_no_dup)

    # Getting the movieId column
    movie_id_list = ratings['movieId']
    # removing duplicates.
    movie_id_list_no_dup = list(dict.fromkeys(movie_id_list))
    # How many unique movies were rated?
    num_unique_movies_rated = len(movie_id_list_no_dup)

    # How many ratings are in the file?
    num_ratings = len(user_id_list)

    # dictionary where [key: movie_id, value: num_of_ratings]
    movies_ratings_dict = {}
    for movie_id in movie_id_list:
        if movie_id in movies_ratings_dict.keys():
            movies_ratings_dict[movie_id] += 1
        else:
            movies_ratings_dict[movie_id] = 1
    # min and max ratings for movie_id
    movies_max_key, movies_max_value = get_max_key_and_value(movies_ratings_dict)
    movies_min_key, movies_min_value = get_min_key_and_value(movies_ratings_dict)

    # dictionary where [key: user_id, value: num_of_ratings]
    users_ratings_dict = {}
    for user_id in user_id_list:
        if user_id in users_ratings_dict.keys():
            users_ratings_dict[user_id] += 1
        else:
            users_ratings_dict[user_id] = 1
    # min and max ratings for movie_id
    users_max_key, users_max_value = get_max_key_and_value(users_ratings_dict)
    users_min_key, users_min_value = get_min_key_and_value(users_ratings_dict)

    # Printing the result:
    # print(f'Users unique ratings: {num_unique_users_ratings}')
    # print(f'Movies unique ratings: {num_unique_movies_rated}')
    # print(f'All ratings: {num_ratings}')
    # print(f'Max movie value ratings: {movies_max_value}')
    # print(f'Min movie value ratings: {movies_min_value}')
    # print(f'Max user value ratings: {users_max_value}')
    # print(f'Min user value ratings: {users_min_value}')


def plot_data(data, plot=True):
    # data = (rating, movies)
    ratings = data[0]

    # Getting the movieId column
    ratings_list = ratings['rating']
    # removing duplicates.
    ratings_list_no_dup = list(dict.fromkeys(ratings_list))
    ratings_list_no_dup_sorted = sorted(ratings_list_no_dup)

    # dictionary where [key: user_id, value: num_of_ratings]
    ratings_dict = {}
    for rating in ratings_list:
        if rating in ratings_dict.keys():
            ratings_dict[rating] += 1
        else:
            ratings_dict[rating] = 1

    sorted_ratings_dict = {}
    for key in sorted(ratings_dict):
        sorted_ratings_dict[key] = ratings_dict[key]

    x_axis = ratings_list_no_dup_sorted
    y_axis = list(sorted_ratings_dict.values())

    plt.plot(x_axis, y_axis)
    plt.xlabel("Rating")
    plt.ylabel("Value")
    plt.title("Ratings Distribution")

    if plot:
        plt.show()
