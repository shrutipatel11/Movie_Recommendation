import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Read the ratings from the csv file
ratings = pd.read_csv("ratings.csv")
# Read the movie id and its associated movie title
movie_df = pd.read_csv('movies.csv')[['movieId', 'title']].set_index('movieId')

# Filter the number of users
ratings = ratings.loc[ratings['userId'] < 16]

# Join the ratings and movie data frames to get the movie titles
ratings = ratings.join(movie_df, on=["movieId"], how="inner")

# Create a matrix with rows as users and columns as movies. The values represent ratings.
crosstab = pd.crosstab(ratings.userId, ratings.title, values=ratings['rating'], aggfunc=np.sum).fillna(0)
# Find similarities between users using cosine similarity
user_similarities = pd.DataFrame(cosine_similarity(crosstab), index=crosstab.index, columns=crosstab.index)

# Transpose the crosstab matrix for getting similarities between movies
crosstab_transpose = crosstab.transpose()
# Find similarity between movies using cosine similarity
movie_similarities = pd.DataFrame(cosine_similarity(crosstab_transpose), index=crosstab_transpose.index, columns=crosstab_transpose.index)


# Read my ratings from the file and recommend movies
my_ratings = pd.read_csv("my_ratings.csv")
my_ratings = my_ratings.join(movie_df, on=["movieId"], how="inner")

frames = [ratings, my_ratings]
new_frame = pd.concat(frames)
cross = pd.crosstab(new_frame.userId, new_frame.title, values=new_frame['rating'], aggfunc=np.sum).fillna(0)
us = pd.DataFrame(cosine_similarity(cross), index=cross.index, columns=cross.index)

# Get the top 2 user ids that matched the most with you
matched_users = us[611].sort_values(ascending=False)[1:3]
matched_users_id = [ids for ids,sim in matched_users.items()]

# Get the movie ids of the movies that I rated/viewed
watched_movie_ids = set(my_ratings.loc[my_ratings['userId'] == 611]['movieId'])

# Get the movie ids of the users that are matched
matched_user_movies = ratings.loc[ratings['userId'].isin(matched_users_id)]
matched_user_movie_ids = set(matched_user_movies['movieId'])

movies_left_to_watch = matched_user_movie_ids - watched_movie_ids
user_watched_movies = matched_user_movies.loc[matched_user_movies['movieId'].isin(movies_left_to_watch)]
movie_recommendations = user_watched_movies.sort_values(by=['rating'], ascending=False)['title'].unique()[:5]

# Print the top 5 movie recommendations
print("Your recommendations are : ", movie_recommendations)

