'''
import data here and have utility functions that could help
'''
from thefuzz import process, fuzz
import pandas as pd
from scipy.sparse import csr_matrix
import pickle

movies = pd.read_csv('./data/movies.csv')
ratings = pd.read_csv('./data/ratings.csv')

#model = ...
with open ('./nmf_recommender.pkl', 'rb') as file:
    model = pickle.load(file)
with open ('./nn_recommender.pkl', 'rb') as file:
    model_nn = pickle.load(file)

#print(movies.head(5))

def movie_title_search(fuzzy_title, movies):
    '''
    does a fuzzy search and returns best matched movie
    '''
    matches = process.extractBests(fuzzy_title, movies, limit=1, scorer=fuzz.token_set_ratio)
    return matches

def movie_to_id(title, movies):
    '''
    converts movie title to id for use in algorithms
    '''
    return movieId

def id_to_movie(movieId, movies):
    '''
    converts movie Id to title
    '''
    titles = movies.set_index('movieId').loc[movieId]['title'].tolist()
    return titles

def make_user_vector(query, shape):
    '''
    creates user vector from query
    '''
    data = list(query.values())             # the ratings of the new user
    row_ind = [0]*len(data)              # we use just a single row 0 for this user
    col_ind = list(query.keys())                           # the columns (=movieId) of the ratings
    user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, shape))
    return user_vec

def get_popular_movies(ratings, threshold):
    '''
    filtr out movies with low number of ratings
    '''
    ratings_per_movie = ratings.groupby('movieId')['rating'].count()
    popular_movies = ratings_per_movie[ratings_per_movie>threshold]
    movies_pop = movies.loc[movies['movieId'].isin(popular_movies.index)]
    return movies_pop
def get_popular_ratings(ratings, threshold):
    '''
    filtr out ratings with low number of ratings
    '''
    ratings_per_movie = ratings.groupby('movieId')['rating'].count()
    popular_movies = ratings_per_movie[ratings_per_movie>threshold]
    ratings_pop = ratings.loc[ratings['movieId'].isin(popular_movies.index)]
    return ratings_pop


if __name__ == '__main__':
    #fuzzy_matches = movie_title_search('star cars', movies.set_index('movieId')['title'])
    #print(fuzzy_matches)
    #check = id_to_movie([364, 588], movies)
    query = {
        # movieId, rating
        4470:5,
        48:5,
        594:5,
    }
    test_query = {4407: 5, 1613: 5, 156605: 5}
    #check = make_user_vector(query, model_nn.n_features_in_)
    #check = make_user_vector(test_query, model.components_.shape[1])
    a = get_popular_movies(ratings, threshold=30)

#%%
