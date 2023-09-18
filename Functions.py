import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy import spatial

#Total amount of movies and users in the database
total_movies = 1682
total_users = 943

def get_ratings(userID:int, df:pd.DataFrame) -> list[int]:
    """
    Given a userID, this funcion returns a list of integers containing the ratings of that user.
    In the returned list, the i-th element contains the rating of the movie whose ID is equal to i+1.
    If the user has not rated the movie, the rating is considered to be 0.

    Args:
        userID: the ID of the user for whom the ratings need to be retrieved.
        df: the DataFrame containing the rating information.

    Returns:
        A list of integers representing the ratings of the specified user.
    """

    #Creating ratings list
    ratings_list = []

    #Getting a DataFrame with the user ratings
    user_condition = df["UserID"] == userID
    user_df = df[user_condition]

    for i in range(1, total_movies + 1):
        #Extracting the ratings (0 for non-rated movies).
        if i in user_df["ItemID"].values:
            rating = user_df[ user_df["ItemID"]==i ]["Rating"].values[0]
            ratings_list.append(rating)
        else:
            ratings_list.append(0)

    return ratings_list


def euc_sim_users(userID:int, df: pd.DataFrame, min_common:int=4, max_distance:float=15.0, top:int=25) -> List[Tuple]:
    """
    Given a userID, this function returns a list of tuples, containing the ID of the 'similar users' and
    their 'distance' to the user with the given userID.
    
    As a first filter, this function selects all those users who have rated a minimum amount of movies amoung those
    rated by the given user. The minimum amount is represented by the variable min_common and it is given as an argument (4 by default).

    After the first filtration, this function computes the Euclidean distance between two points, one of them having as 
    coordinates the ratings of the given user, and the second one having as coordinates the ratings of a selected user. We are
    only considering the rating for those movies that have been rated by both users at the same time.

    Finally, the function returns a list with tuples, having as a first coordinate the ID of those users whose distance is
    small enough to be considered a 'similar user' (specified through the variable max_distance, 15 by default), and the 
    value of that distance as a second coordinate. The list is sorted in descending order, in such a way that the 'more similar'
    users are placed first. The 'top' variable indicates how many users are wished to be returned, being 25 by default.

    If no 'similar users' have been found, a message is printed and an empty list is returned.

    Args:
        userID: ID of the user for whom the 'similar users' shall be found.
        df: the DataFrame containing the users and items ID as well as the rating information.
        min_common: minimum amount of common movies rated.
        max_distance: maximum distance at which a user should be from the principal user to be considered 'simmilar'.
        top: integer number to indicate the maximum number of similar users to be returned.
    
    Returns:
        A list with tuples, containing the ID of 'similar users' and their distance to the given user. 
    """
    
    #Getting ratings for the principal user
    user_ratings = get_ratings(userID, df)

    #Selecting movies rated by the principal user
    selected_movies = []
    selected_movies = df.loc[df['UserID'] == userID, 'ItemID'].values

    #Creating a list to store similar users
    similar_users = []

    for i in range(1, total_users + 1):
        if i != userID:
            #Getting ratings of a candidate to similar user
            second_user_ratings = get_ratings(i, df)

            common_movies_ratings = []

            #Checking common movies
            for movie in selected_movies:
                if second_user_ratings[movie-1] != 0:
                    common_movies_ratings.append(user_ratings[movie-1]-second_user_ratings[movie-1])

            #We ask a minim amount of "min_common" movies in common.        
            if len(common_movies_ratings) >= min_common:
                distance = np.linalg.norm(np.array(common_movies_ratings))

                #In order for a user to be considered similar to the principal one, we ask for a maximum distance.
                if distance <= max_distance:
                    similar_users.append((i, distance))
    
    #Checking if similar users were found
    if len(similar_users) == 0:
        print(f"It was not possible to find similar users to UserID={userID}")
        return similar_users
    
    #Sorting similar users by distance
    else:
        sorted_similar_users = sorted(similar_users, key = lambda x: x[1])
        return sorted_similar_users[:top]
    
def cos_sim_users(userID:int, df: pd.DataFrame, min_common:int=4, max_distance:float=0.5, top:int=25)-> List[Tuple]:
    """
    Given a userID, this function returns a list of tuples, containing the ID of the 'similar users' and
    their 'distance' to the user with the given userID.
    
    As a first filter, this function selects all those users who have rated a minimum amount of movies amoung those
    rated by the given user. The minimum amount is represented by the variable min_common and it is given as an argument (4 by default).

    After the first filtration, this function computes the cosine distance between two points, one of them having as 
    coordinates the ratings of the given user, and the second one having as coordinates the ratings of a selected user. We are
    only considering the rating for those movies that have been rated by both users at the same time.

    Finally, the function returns a list with tuples, having as a first coordinate the ID of those users whose distance is
    small enough to be considered a 'similar user' (specified through the variable max_distance, 0.5 by default), and the 
    value of that distance as a second coordinate. The list is sorted in descending order, in such a way that the 'more similar'
    users are placed first. The 'top' variable indicated how many users are wished to be returned, being 25 by default.

    If no 'similar users' have been found, a message is printed and an empty list is returned.

    Args:
        userID: ID of the user for whom the 'similar users' shall be found.
        df: the DataFrame containing the users and items ID as well as the rating information.
        min_common: minimum amount of common movies rated.
        max_distance: maximum distance at which a user should be from the principal user to be considered 'simmilar'.
        top: integer number to indicate the maximum number of similar users to be returned.
    
    Returns:
        A list with tuples, containing the ID of 'similar users' and their distance to the given user. 
    """
    #Getting ratings for the principal user
    user_ratings = get_ratings(userID, df)

    #Selecting movies rated by the principal user.
    selected_movies = []
    selected_movies = df.loc[df['UserID'] == userID, 'ItemID'].values

    #Creating a list to store similar users.
    similar_users = []

    for i in range(1, total_users + 1):
        if i != userID:
            #Getting ratings of a candidate to similar user.
            second_user_ratings = get_ratings(i, df)

            #The cosine of the angle between those vectors will be computed later.
            vector1 = []
            vector2 = []

            #Checking common movies
            for movie in selected_movies:
                if second_user_ratings[movie-1] != 0:
                    vector1.append(user_ratings[movie-1])
                    vector2.append(second_user_ratings[movie-1])

            #We ask a minim amount of "min_common" movies in common.  
            if len(vector1) >= min_common:
                distance = spatial.distance.cosine(vector1, vector2)

                #In order for a user to be considered similar to the principal one, we ask for a maximum distance.
                if distance <= max_distance:
                    similar_users.append((i, distance))
    
    #Checking if similar users were found
    if len(similar_users) == 0:
        print(f"It was not possible to find similar users to UserID={userID}")
        return similar_users
    
    else:

        #Sorting similar users by distance
        sorted_similar_users = sorted(similar_users, key = lambda x: x[1])
        return sorted_similar_users[:top]
    

def rating_prediction(userID:int, movieIDs:list[int], df: pd.DataFrame, min_common:int=4, max_distance:float=15.0, top:int=25) -> List[float]:
    """
    As an argument of the function, we get a userID and a list of movies (as a list of integers representing their ID).
    
    This function then computes the prediction of the ratings that the given user would give to the movies in the list.
    
    The prediction is computed as the weighted average of the ratings that similar users (computed using the euclidean distance
    between their raings coordinates) have given to each movie in the list (if so).

    The weights are given by the inverse of the euclidean distance between the given user and the similar users rating coordinates.

    The function returns then a list with the predictions, keeping the order of the list "movieIDs" given as an argument.


    Args:
        userID (int): ID of the user for whom the ratings must be predicted.
        movieIDs (list[int]): IDs of the movies for which we want to predict the user's ratings.
        df (pd.DataFrame): the DataFrame containing the users and items ID as well as the rating information.
        min_common: minimum amount of common movies rated.
        max_distance: maximum distance at which a user should be from the principal user to be considered 'simmilar'.
        top: integer number to indicate the maximum number of similar users to be returned.

    Returns:
        List[float]: list of predictions, keeping the order of the list "movieIDs" given as an argument.
    """

    #Finding similar users.
    similar_users = euc_sim_users(userID, df, min_common=min_common, max_distance=max_distance, top=top)
    similar_users_ID = [user[0] for user in similar_users]

    #Computing the weigths. If the distance is 0, the weight is taken as 2.
    weights = []
    for user in similar_users:
        if user[1]==0:
            weights.append(2)
        else:
            weights.append(1/user[1])

    #Dictionary storing similar users IDs and their weights.
    similar_users_dic = dict(zip(similar_users_ID, weights))

    predictions = []

    #Computing rate prediction for each movie in the list "movieIDs".
    for movieID in movieIDs:
        condition = (df["UserID"].isin(similar_users_ID)) & (df["ItemID"] == movieID)
        similar_users_df = df[condition]

        ratings_sum = 0
        weight_control = 0

        #We require a minimum of 3 similar users in order to compute the rating prediction. If not, we take 0 as the
        #rating prediction.
        if len(similar_users_df) < 3:
            predictions.append(0)
        else:
            for index, row in similar_users_df.iterrows():
                weight = similar_users_dic[row["UserID"]]
                ratings_sum += row["Rating"]*weight
                weight_control += weight

            #The prediction is the weighted average of the ratings of similar users.
            avg = ratings_sum/weight_control
            predictions.append(avg)

    return predictions

def rating_prediction_cos(userID:int, movieIDs:list[int], df: pd.DataFrame, min_common:int=5, max_distance:float=0.5, top:int=10) -> List[float]:
    """
    As an argument of the function, we get a userID and a list of movies (as a list of integers representing their ID).
    
    This function then computes the prediction of the ratings that the given user would give to the movies in the list.
    
    The prediction is computed as the weighted average of the ratings that similar users (computed using the cosine distance
    between their raings coordinates) have given to each movie in the list (if so).

    The weights are given by the cosine similarity between the given user and the similar users rating coordinates.

    The function returns then a list with the predictions, keeping the order of the list "movieIDs" given as an argument.


    Args:
        userID (int): ID of the user for whom the ratings must be predicted.
        movieIDs (list[int]): IDs of the movies for which we want to predict the user's ratings.
        df (pd.DataFrame): the DataFrame containing the users and items ID as well as the rating information.
        min_common: minimum amount of common movies rated.
        max_distance: maximum distance at which a user should be from the principal user to be considered 'simmilar'.
        top: integer number to indicate the maximum number of similar users to be returned.

    Returns:
        List[float]: list of predictions, keeping the order of the list "movieIDs" given as an argument.
    """

    #Finding similar users.
    similar_users = cos_sim_users(userID, df, min_common=min_common, max_distance=max_distance, top=top)
    similar_users_ID = [user[0] for user in similar_users]

    #Storing the similar users IDs and their cosine similarity.
    cosine_similarity = [1-user[1] for user in similar_users]
    similar_users_dic = dict(zip(similar_users_ID, cosine_similarity))

    predictions = []

    #Computing rate prediction for each movie in the list "movieIDs".
    for movieID in movieIDs:
        condition = (df["UserID"].isin(similar_users_ID)) & (df["ItemID"] == movieID)
        similar_users_df = df[condition]
        # print(similar_users_df)

        # print("MovieID: ", movieID)

        ratings_sum = 0
        similarity_control = 0

        #We require a minimum of 3 similar users in order to compute the rating prediction. If not, we take 0 as the
        #rating prediction.
        if len(similar_users_df) < 3:
            predictions.append(0)
        else:
            for index, row in similar_users_df.iterrows():
                similarity = similar_users_dic[row["UserID"]]
                ratings_sum += row["Rating"] * similarity
                similarity_control += similarity

            #The prediction is the weighted average of the ratings of similar users.
            avg = ratings_sum/similarity_control
            predictions.append(avg)
    
    return(predictions)