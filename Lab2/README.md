# Lab2

## Email spam not only wastes users' time but also poses additional threats, such as phishing, extortion through false information, distribution of viral and trojan programs, etc. Your task is to develop a classifier that can distinguish spam emails based on message data analysis. The dataset includes emails labeled as either containing or not containing spam.

### [**Code**](/Lab2/Lab2.ipynb)

### [**Data set**](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset)

### Data Preprocessing & NLP for Logistic Regression

1. A dataset was obtained from a CSV file. After analyzing the dataset, two non-informative columns, "Unnamed: 0.1" and "Unnamed: 0," were dropped, leaving us with the columns Body and Label, where the latter represents the target variable.
2. In the dataset, 313 exact duplicates were identified and subsequently removed. The text was then converted to lowercase. Analyzing a specific row revealed that all messages start with the construction "subject:," so using regular expressions, the subject was extracted into a separate column and removed from the Body.
3. Three features were created: the number of characters, words, and sentences in the Body.
4. Based on this data, boxplots and a correlation matrix were generated. Outliers based on the number of sentences (approximately 3k rows) were removed, and a correlation matrix for the dataset without outliers was constructed. The correlations improved, leading to the decision to completely remove outliers. It's worth noting that in this dataset, correlations between different parameters of message length and the target variable are very small.
5. Punctuation and non-ASCII alphabet symbols were removed from the textual data.
6. Stop words were removed from the textual data, and stemming was applied to generalize words present in the text.
7. Word Clouds were generated for spam and non-spam messages.
8. Since the task involves classification, it was decided to extract topics from the text, specifically 16 topics using the LDA method.
9. Pearson and Kendall correlation matrices were constructed for topics and labels, revealing significant correlation.
10.  For the Subject column, TF-IDF vectorization was applied, retaining the top 1000 most significant words.
11. No class imbalance was detected, so the decision was made to split the data into a training and testing set.
12. Feature scaling using standard scaling was applied to the features from step 3. Subsequently, PCA was applied to reduce the dimensionality of the feature space by a factor of 4.

### Logistic Regression Training

1. Logistic regression with L2 regularization was trained, yielding an F1-score of 0.95, look like overfitting. Metrics such as precision, recall, and accuracy were also computed.
2. A confusion matrix was constructed.
3. ROC curves were plotted, and the ROC-AUC metric was calculated.
4. A second logistic regression model with L1 regularization was trained, showing similar results.
5. Try random forest on this task, less overfitting, but still not ok.
   
### Results
| Model # | F1-score | precision | recall | accuracy | ROC-AUC |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.95 | 0.95 | 0.95 | 0.95 | 0.9875 |
| 2 | 0.95 | 0.95 | 0.95 | 0.95 | 0.9880 |
| 3 | 0.94 | 0.94 | 0.94 | 0.94 | ------ |


### Data Preprocessing & NLP for Neural Network Model with LSTM Layer

1. Before dropping textual information during logistic regression data preparation, it was saved for this moment. Now, we will tokenize it using the Tokenizer from the Keras library, both for Body and Subject.
2. Padding was applied to achieve a consistent format, using 500 words for message bodies and 125 words for subjects. Padding type 'pre' was used to aid model training.
3. Data was split into training and testing sets. 'y' was transformed into a one-hot vector.
4. A neural network model with two inputs was created. The first input for the subject was connected to a fully connected neural network with 2 neurons, followed by a GRU block with 2 neurons. BatchNormalization and Dropout were applied to prevent overfitting. This was connected to the subject output GRU block. The second input for the body was connected to a fully connected neural network with 4 neurons, followed by an LSTM block with 4 neurons. BatchNormalization and Dropout were applied to prevent overfitting, and this was connected to the body output LSTM block. Outputs from LSTM and GRU blocks were merged and fed to the output layer with a fully connected layer having 2 neurons (2 class labels) and a softmax activation function. Leaky ReLU was used as the activation function on LSTM and GRU to prevent dead neurons.
5. The model was compiled using the Adam optimizer and categorical_crossentropy loss function.
6. EarlyStopping was implemented on the validation set to avoid overfitting.
7. The model was trained for 34 epochs.
8. Loss and accuracy plots were visualized.
9. Metrics including precision, recall, F1-score, and ROC-AUC were computed. A confusion matrix and ROC curves were also generated.

### Results
| Model # | F1-score | precision | recall | accuracy | ROC-AUC |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 0.90 | 0.97 | 0.84 | 0.90 | 0.9826 |