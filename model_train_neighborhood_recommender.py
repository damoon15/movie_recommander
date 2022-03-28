import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import sklearn
import pickle
from utils import make_user_vector

import os
#%%
# movie given by the user
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

#%%
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
#%%
movies.set_index('movieId').loc[query.keys()]
#%%
ratings_per_movie = ratings.groupby('movieId')['rating'].count()
popular_movies = ratings_per_movie[ratings_per_movie>30]
ratings = ratings.loc[ratings['movieId'].isin(popular_movies.index)]
R = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
#%%Training
sorted(sklearn.neighbors.VALID_METRICS_SPARSE['brute'])
#%%
model_nn = NearestNeighbors(metric='cosine')
model_nn.fit(R)

#%%Save the trained model
with open('./nn_recommender.pkl', 'wb') as file:
    pickle.dump(model_nn, file)

#%%read the model from hard drive
with open('./nn_recommender.pkl', 'rb') as file:
    model_nn = pickle.load(file)

#%%
shape = model_nn.n_features_in_
user_vec = make_user_vector(query, shape)

#%%calculate the score
distances, userIds = model_nn.kneighbors(user_vec, n_neighbors=10, return_distance=True)
distances = distances[0]
userIds = userIds[0]

#%% extract the ratings of the similar users from the original data
neighborhood = ratings.set_index('userId').loc[userIds]

#%%score calculation
scores = neighborhood.groupby('movieId')['rating'].mean()

#%% give recommendations
scores.loc[scores.index.isin(query.keys())] = 0
scores.sort_values(ascending=False, inplace=True)
#%%
scores_10 = scores.head(10)
recommendations = movies.set_index('movieId').loc[scores_10.index]