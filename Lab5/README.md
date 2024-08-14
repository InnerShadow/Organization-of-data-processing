# Lab5
## Create a recommendation system to recommend movies.

### [**Code**](/Lab5/lab5.ipynb)

### [**Dataset**](https://www.kaggle.com/datasets/rajmehra03/movielens100k)

### Procedure 

1. A dataset containing movies and their ratings was obtained, and an EDA analysis of the data was performed.
   
2. A collaborative recommendation system was built based on the data.

3. Quality metrics for each model, such as MSE, RMSE, MAE, MSlogE, and R², were measured.

4. A class was created for hyperparameter tuning using the BayesSearchCV method. The results of the hyperparameter tuning were visualized with graphs.

5. A model with the best hyperparameters was created and trained. The model was tested on a new user, created manually.

6. An attempt was made to switch from a regression task to a multi-class classification task, but the results were worse.

7. An attempt was made to add an additional input that contained the inverse frequency of movie appearances.

8. An attempt was made to solve the problem using classical ML methods, such as SVD.

9. One additional input was added for both the user and the movie. For the user, this included statistical characteristics of their ratings — mean, median, standard deviation, 1st and 3rd quartiles, and skewness. For movies, TF-IDF based on genres was added.