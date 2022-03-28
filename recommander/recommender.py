"""
contains various implementations for recommending movies
"""

import pandas as pd
import pandas as pd
from utils import make_user_vector, movies, ratings, model, model_nn, get_popular_ratings


# recommender_systems_intro_filled.ipynb
def recommend_random(query, movies, k=10):
    """
    Recommends a list of k random movie ids
    """

    # 1. candiate generation

    # filter out movies that the user has allready seen
    movies_seen = movies.set_index('movieId').drop(query.keys())

    # 2. "scoring"
    # calculate a random sample of movies

    return movies_seen.sample(n=k, random_state=1).index


# recommender_systems_intro_filled.ipynb
def recommend_popular(query, ratings, k=10):
    """
    Recommend a list of k movie ids that are the most popular
    """
    # 1. candiate generation
    # filter out movies that the user has allready seen
    ratings_seen = ratings.set_index('movieId').drop(query.keys())
    # filter out movies that have been watched by less than 20/50/100... users
    ratings_per_movie = ratings_seen.groupby('movieId')['userId'].count()
    considered_movies = ratings_per_movie[ratings_per_movie > 50]
    ratings_considered = ratings_seen.loc[considered_movies.index]
    ratings_considered.reset_index(inplace=True)

    # 2. scoring
    # calculate the average rating for each movie
    ave_r_per_movie = ratings_considered.groupby('movieId')['rating'].mean()

    # 3. ranking
    # return the top-k highst rated movie ids or titles
    scores = ave_r_per_movie.sort_values(ascending=False)

    return scores.head(k).index


# matrix_factorization_filled.ipynb
def recommend_nmf(query, k=10):
    """
    Recommend a list of k movie ids based on a trained NMF model
    """
    # generate user_vec
    shape = model.components_.shape[1]
    user_vec = make_user_vector(query, shape)
    #calculate the score
    scores = model.inverse_transform(model.transform(user_vec))
    scores = pd.Series(scores[0])
    # give recommendations
    scores[query.keys()] = 0
    scores = scores.sort_values(ascending=False)
    recommendations = scores.head(k).index
    return recommendations


# neighborhood_based_filtering.ipynb
def recommend_neighbors(query, ratings, k=10):
    """
    Recommend a list of k movie ids based on the most similar users
    """
    # construct a user vector
    shape_nn = model_nn.n_features_in_
    user_vec_nn = make_user_vector(query, shape_nn)
    # 2. scoring
    # find n neighbors
    # calculate their average rating
    distances, userIds = model_nn.kneighbors(user_vec_nn, n_neighbors=k, return_distance=True)
    userIds = userIds[0]
    distances = distances[0]
    ratings_pop = get_popular_ratings(ratings, threshold=30)
    neighborhood = ratings_pop.set_index('userId').loc[userIds]
    scores = neighborhood.groupby('movieId')['rating'].mean()
    # 3. ranking
    # filter out movies allready seen by the user
    scores.loc[scores.index.isin(query.keys())] = 0
    # return the top-k highst rated movie ids or titles
    scores.sort_values(ascending=False, inplace=True)
    return scores.head(k).index


if __name__=='__main__':
    # list of liked movies
    #query = [1, 34, 56, 21]
    #print(recommend_random(query, movies))
    '''
    query = {
        # movieId, rating
        4470:5,
        48:5,
        594:5,
        27619:5,
        152081:5,
        595:5,
        616:5,
        1029:5
    }
    '''
    query = {
        # movieId, rating
        4470:5,
        48:5,
        594:5,
    }
    query_test = {
        # movieId, rating
        4470:5,
        48:5,
        594:5,
        27619:5,
        152081:5,
        595:5,
        616:5,
        1029:5
    }
    test_query = {4407: 5, 1613: 5, 156605: 5}
    #recs1 = recommend_nmf(test_query, k=10)
    #recs2 = recommend_popular(query, ratings, k=10)
    recs3 = recommend_neighbors(query_test, ratings, k=10)

