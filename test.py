'''''
Created on Nov 17, 2012 
 
@Author: Dennis Wu
@E-mail: hansel.zh@gmail.com
@Homepage: http://blog.csdn.net/wuzh670 
 
Data set download from : http://www.grouplens.org/system/files/ml-100k.zip 
 
MovieLens data sets were collected by the GroupLens Research Project
at the University of Minnesota.The data was collected through the MovieLens web site
(movielens.umn.edu) during the seven-month period from September 19th,
1997 through April 22nd, 1998. 
 
This data set consists of:
    * 100,000 ratings (1-5) from 943 users on 1682 movies.
    * Each user has rated at least 20 movies.
    * Simple demographic info for the users  
 
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of
              user id | item id | rating | timestamp.
              The time stamps are unix seconds since 1/1/1970 UTC
u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.
'''  
 
from operator import itemgetter, attrgetter
from math import sqrt  
 
def load_data():  
 
    filename_user_movie = 'data/u.data'
    filename_movieInfo = 'data/u.item'  
 
    user_movie = {}
    for line in open(filename_user_movie):
        (userId, itemId, rating, timestamp) = line.strip().split('\t')
        user_movie.setdefault(userId,{})
        user_movie[userId][itemId] = float(rating)  
 
    movies = {}
    for line in open(filename_movieInfo):
        (movieId, movieTitle) = line.split('|')[0:2]
        movies[movieId] = movieTitle  
 
    return user_movie, movies  
 
def average_rating(user):
    average = 0
    for u in user_movie[user].keys():
        average += user_movie[user][u]
    average = average * 1.0 / len(user_movie[user].keys())
    return average  
 
def calUserSim(user_movie):  
 
    # build inverse table for movie_user
    movie_user = {}
    for ukey in user_movie.keys():
        for mkey in user_movie[ukey].keys():
            if mkey not in movie_user:
                movie_user[mkey] = []
            movie_user[mkey].append(ukey)  
 
    # calculated co-rated movies between users
    C = {}
    for movie, users in movie_user.items():
        for u in users:
            C.setdefault(u,{})
            for n in users:
                if u == n:
                    continue
                C[u].setdefault(n,[])
                C[u][n].append(movie)  
 
    # calculate user similarity (perason correlation)
    userSim = {}
    for u in C.keys():  
 
        for n in C[u].keys():  
 
            userSim.setdefault(u,{})
            userSim[u].setdefault(n,0)  
 
            average_u_rate = average_rating(u)
            average_n_rate = average_rating(n)  
 
            part1 = 0
            part2 = 0
            part3 = 0
            for m in C[u][n]:  

                part1 += (user_movie[u][m]-average_u_rate)*(user_movie[n][m]-average_n_rate)*1.0
                part2 += pow(user_movie[u][m]-average_u_rate, 2)*1.0
                part3 += pow(user_movie[n][m]-average_n_rate, 2)*1.0  
 
            part2 = sqrt(part2)
            part3 = sqrt(part3)
            if part2 == 0:
                part2 = 0.001
            if part3 == 0:
                part3 = 0.001
            userSim[u][n] = part1 / (part2 * part3)
    return userSim  
 
def getRecommendations(user, user_movie, movies, userSim, N):
    pred = {}
    interacted_items = user_movie[user].keys()
    print("user", user)
    print("user_movie", user_movie)
    average_u_rate = average_rating(user)
    sumUserSim = 0
    for n, nuw in sorted(userSim[user].items(),key=itemgetter(1),reverse=True)[0:N]:
        average_n_rate = average_rating(n)
        for i, nrating in user_movie[n].items():
            # filter movies user interacted before
            if i in interacted_items:
                continue
            pred.setdefault(i,0)
            pred[i] += nuw * (nrating - average_n_rate)
        sumUserSim += nuw
 
    for i, rating in pred.items():
        pred[i] = average_u_rate + (pred[i]*1.0) / sumUserSim  
 
    # top-10 pred
    pred = sorted(pred.items(), key=itemgetter(1), reverse=True)[0:10]
    return pred
 
if __name__ == "__main__":  
 
    # load data
    user_movie, movies = load_data()  
 
    # Calculate user similarity
    userSim = calUserSim(user_movie)
 
    # Recommend
    pred = getRecommendations('', user_movie, movies, userSim, 20)  
 
    # display recommend result (top-10 results)
    for i, rating in pred:
        print("film: %s,  rating: %s".format(movies[i], rating))