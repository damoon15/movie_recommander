import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix
import sklearn
import pickle
from sklearn.decomposition import NMF
#%%
#implement a baseline recommander




#%%
#Derive a user-item matrix
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

#%%
#user-item matrix wide format
user_item_wide = pd.pivot_table(ratings,
                                values='rating',
                                index='userId',
                                columns='movieId'
)
user_item_wide

#%%
#user-item matrix
user_item_sparse = csr_matrix((ratings['rating'],
                (ratings['userId'], ratings['movieId'])))
user_item_sparse
print(ratings.shape)

#%%
# collabrative filtering with Matrix factorization

#%%
#Non Negative Matrix Factorization for Recommender Systems

#%%
#1- Model Development
ratings_per_movie = ratings.groupby('movieId')['userId'].count()
ratings_per_movie
#%%
popular_movies = ratings_per_movie.loc[ratings_per_movie > 20]
popular_movies
#%%
ratings = ratings.set_index('movieId').loc[popular_movies.index]
ratings = ratings.reset_index()
ratings
#%%
# sparse matrix
R = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
R
#%%
#Training
#%%
model = NMF(n_components=55, init='nndsvd', max_iter=10000, tol=0.01, verbose=2)
model.fit(R)

#%%
#Model Inspection
P = model.transform(R)
Q = model.components_
P.shape, Q.shape
#%%
R_hat = model.inverse_transform(model.transform((R)))
R_hat
#%%
#rmse_manual
err_manual = np.sqrt(np.sum(np.square(R-R_hat)))
#rmse_model
#%%
err_model = model.reconstruction_err_

#%%
#Model deployment
with open('./nmf_recommender.pkl', 'wb') as file:
    pickle.dump(model, file)
!ls

#%%
with open ('nmf_recommender.pkl', 'rb') as file:
    model = pickle.load(file)

#%%
#%%
query = {
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
#goal : to create a user_vector with the same shape of our sparse(R) matrix
data = list(query.values())
# we consider one user, so we have on row. length of row: number of cols:
row_ind = [0]*len(data)
# columns = movieIds (5 movies)
col_ind = list(query.keys())
# for one user, thus shape = (1, number of cols of R)
user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, R.shape[1]))

#%%
scores = model.inverse_transform(model.transform(user_vec))
scores = pd.Series(scores[0])

#%%
# Give Recommendations
scores[query.keys()] = 0
scores = scores.sort_values(ascending=False)
recommendations = scores.head(10).index
recommendations
#%%
movies.set_index('movieId').loc[recommendations]
#%%
