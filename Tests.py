import pandas as pd
import Functions as Fct

#Extracting from the database the base and test files.
file_path_data = "ml-100k/ml-100k/u5.base"
file_path_test = "ml-100k/ml-100k/u5.test"
column_names = [
    "UserID",
    "ItemID",
    "Rating",
    "Time"
]
df_data = pd.read_csv(file_path_data, delimiter='\t', header=None, names=column_names)
df_test = pd.read_csv(file_path_test, delimiter='\t', header=None, names=column_names)

############################################################
#####Testing predictions given by the Euclidean method.#####
############################################################
correct_predictions = 0
total_predictions = 0
rounded_rating_error = 0
error = 0
total_errors = 0

for i in range(len(df_test["UserID"].unique())):

    #Filtering the dataframe by the user of interest.
    userID = df_test["UserID"].unique()[i]
    user_df = df_test[ df_test["UserID"]==userID ]

    #Creating lists of movieIDs and the real ratings of the users.
    movieIDs = list(user_df["ItemID"].values)
    real_ratings = list(user_df["Rating"].values)

    #Computing the predicted ratings
    ratings = Fct.rating_prediction(userID, movieIDs, df_data, min_common=4, max_distance=15.0, top=25)
    ratings_length = len(ratings)

    #Comparing real/predicted ratings
    total_predictions += ratings_length
    for j in range(ratings_length):

        #If the rating was predicted as 0, meaning that there were not enough similar users,
        #we may choose another way to predict the rating.
        if ratings[j]==0:

            #If no info is available on the base file, then we choose as a prediction the
            #most frequent rating of the database: 4.
            if len(df_data[ df_data["ItemID"]==movieIDs[j] ])==0:
                ratings[j]=4

            else:
                #If information is avilable in the base file, we choose as a prediction the
                #average rating of the film.
                ratings[j] = df_data[ df_data["ItemID"]==movieIDs[j] ]["Rating"].mean()
        
        #Computing the correct predictions and the errors.
        if round(ratings[j])==real_ratings[j]:
            correct_predictions += 1
        else:
            rounded_rating_error += abs(round(ratings[j])-real_ratings[j])
            error += abs(ratings[j]-real_ratings[j])
            total_errors += 1

#Printing the results.
print("Euclidean Method:\n")
print(f"Accuracy: {round((correct_predictions/total_predictions)*100,2)}%")
print(f"Average Error: {rounded_rating_error/total_errors}")
print(f"MAE: {error/total_errors}")



#########################################################
#####Testing predictions given by the Cosine method.#####
#########################################################

correct_predictions = 0
total_predictions = 0
rounded_rating_error = 0
error = 0
total_errors = 0

for i in range(len(df_test["UserID"].unique())):

    #Filtering the dataframe by the user of interest.
    userID = df_test["UserID"].unique()[i]
    user_df = df_test[ df_test["UserID"]==userID ]

    #Creating lists of movieIDs and the real ratings of the users.
    movieIDs = list(user_df["ItemID"].values)
    real_ratings = list(user_df["Rating"].values)

    #Computing the predicted ratings
    ratings = Fct.rating_prediction_cos(userID, movieIDs, df_data, min_common=4, max_distance=0.5, top=25)
    ratings_length = len(ratings)

    #Comparing real/predicted ratings
    total_predictions += ratings_length
    for j in range(ratings_length):

        #If the rating was predicted as 0, meaning that there were not enough similar users,
        #we may choose another way to predict the rating.
        if ratings[j]==0:

            #If no info is available on the base file, then we choose as a prediction the
            #most frequent rating of the database: 4.
            if len(df_data[ df_data["ItemID"]==movieIDs[j] ])==0:
                ratings[j]=4

            else:
            #If information is avilable in the base file, we choose as a prediction the
            #average rating of the film.
                ratings[j] = df_data[ df_data["ItemID"]==movieIDs[j] ]["Rating"].mean()

        #Computing the correct predictions and the errors.
        if round(ratings[j])==real_ratings[j]:
            correct_predictions += 1
        else:
            rounded_rating_error += abs(round(ratings[j])-real_ratings[j])
            error += abs(ratings[j]-real_ratings[j])
            total_errors += 1

#Printing the results.
print("Cosine Method:\n")
print(f"Accuracy: {round((correct_predictions/total_predictions)*100,2)}%")
print(f"Average Error: {rounded_rating_error/total_errors}")
print(f"MAE: {error/total_errors}")