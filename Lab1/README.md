# Lab 1

## As of today, cardiovascular diseases stand as one of the primary causes of mortality globally. Your task involves processing a collected dataset comprising risk factors for cardiovascular diseases, followed by the development of a binary classification algorithm to determine whether an individual has the condition.

### [**Code**](/Lab1/lab1.ipynb)

### [**Data set**](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

### Data Cleaning
Data was extracted from the "heart_2022_with_nans" dataset. To grasp the overall picture, a heatmap of missing values was generated. Subsequently, records with less than 30 non-null features out of 40 were dropped. Following that, all entries without data on the target variable were removed. Duplicates were then discarded. The data was further divided into three groups: the first group comprised categorical features with missing values, where the missing values were dropped; the second group consisted of categorical features with missing values, where the gaps were filled with the mode; and the third group encompassed numerical features, with missing values replaced by the median. Boxplots were generated for numerical features, and values falling into the outlier category were dropped. Histograms of the distribution of each feature were constructed. A class imbalance was observed in the target variable, and the down-sampling method was applied. Correlation matrices were then created using both Pearson and Spearman methods.

### Model Fitting
The data was split into training and testing sets and normalized using standard normalization. The first model, DecisionTreeClassifier, with a maximum tree depth of 3, was employed. Metrics were collected, a confusion matrix was created, and an ROC curve was plotted. Subsequently, the data underwent dimensionality reduction using PCA, and a second decision tree model with unlimited depth was executed. It was observed that the model became overfit. Next, the Support Vector Machine (SVM) method with a linear kernel was applied, showing superior metrics compared to the decision tree. An attempt was made to reduce the feature space to the top 10 features identified by PCA as significant.

### Results

| Model # | Classifire | Accuracy | Precision | Recall | F1 | AUC-ROC |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | Decision tree | 0.77 | 0.75 | 0.81 | 0.78 | 0.85 |
| 2 | Decision tree | 0.70 | 0.71 | 0.71 | 0.71 | 0.70 |
| 3 | SVM | 0.79 | 0.84 | 0.71 | 0.77 | 0.89 |
| 4 | SVM | 0.78 | 0.78 | 0.78 | 0.78 | 0.86 |