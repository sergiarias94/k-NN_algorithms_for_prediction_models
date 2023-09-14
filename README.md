This repository exposes the final project developed during the last week of the Ironhack data analysis bootcamp.

The goal of the project was to create a simple k-NN based algorithms to predict the movie ratings of a user based on ratings made by users that rate similarly. Two methods were implemented to achieve that purpose.

1.	We consider two users to be similar if the Euclidean distance between their rating coordinates is sufficiently small. Given a list of similar users, the way of predicting a movie rating is by performing a weighted average of the ratings made by the similar users. The weights are chosen to be the inverse of the distance between the users.

2.	As a second approach, we consider two users to be similar if their cosine distance is sufficiently small. The cosine distance is computed as the difference between 1 and the cosine of the angle that form the vectors joining the users rating coordinates and the center of coordinates. Once a list of similar users is found, we compute the predicted rating as a weighted average, identically to method 1. However, in this case, the weights are given by the cosine similarity, that is, the difference between 1 and the cosine distance.

To test the methods, the MovieLens 100K Database provided by GroupLens was used, which can be found here: https://grouplens.org/datasets/movielens/

In the repository, there is available the file “Functions.py” where you can find the functions defining the different stages of the algorithms: finding similar users and predicting the movie ratings. In addition, the file “Tests.py” contains the code to test the two methods on the MovieLens 100K Database.
