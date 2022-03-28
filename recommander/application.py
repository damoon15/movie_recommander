from flask import Flask, request, render_template
from recommender import recommend_random, recommend_neighbors, recommend_nmf, recommend_popular
from utils import movies, movie_title_search, ratings, id_to_movie, get_popular_movies
#%%
# we instanciate the flask oject, (__name__) used to declare this file as the home script 
app = Flask(__name__)

# each view (or webpage) is defined by a function and app.route tells flask at which URL to render it
@app.route('/')
def landing_page():
    return render_template('landing_page.html')
#%%
@app.route('/recommender/')
def recommender():
    # request gives us access to the URL arguments
    # print(request.args)
   
    # access each with different key variables
    # recs = request.args['movies']
    # dd1 = request.args['t']
    # dd2 = request.args['ia']
    # this method used for more than one movie with all the smae key varibale
    user_movies = request.args.getlist('movies')

    #uses our movie title search function from utils to match the exact title
    all_titles = movies.set_index('movieId')['title']
    #exclusively for nn_recommender to filter out movies with low number of ratings
    popular_movies = get_popular_movies(ratings, threshold=30)
    popular_titles = popular_movies.set_index('movieId')['title']
    #having two different matched titles, one from all movies, one from popular movies
    matched_titles = [movie_title_search(user_movie, all_titles) for user_movie in user_movies]
    pop_matched_titles = [movie_title_search(user_movie, popular_titles) for user_movie in user_movies]
    #get the movieIds to make user_dict and eventually user_vec
    matched_ids = [matched_title[0][2] for matched_title in matched_titles]
    pop_matched_ids = [matched_title[0][2] for matched_title in pop_matched_titles]
    
    #create user_dict to be used to make the uservector for our recommendations
    user_query = dict(zip(matched_ids, len(user_movies)*[5]))
    print(user_query)
    pop_user_query = dict(zip(pop_matched_ids, len(user_movies)*[5]))
    print(pop_user_query)

    
    # to do: create uservec with the user_dict data
    # make recommendation
    #######################choose one of the recoomenders#########################
    #recs = recommend_random(user_query, movies, k=10)
    #recs = recommend_popular(test_query, ratings, k=10)
    #recs = recommend_nmf(user_query, k=10)
    recs = recommend_neighbors(pop_user_query, ratings, k=10)
    titles = id_to_movie(recs, movies)
    print(titles)


    #pass recs to html and render
    return render_template('recommender.html', matched_titles=titles)

#%%
# ensures the code below is only executed when this file is directly run
if __name__=='__main__':
    # runs app and sets debug level on so that the errors are shows in the terminal
    # and the server restarts when any changes are made to this file!
    app.run(debug=True)

