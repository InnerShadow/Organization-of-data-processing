# Lab4
## Create and train a convolutional Siamese neural network to verify the authenticity of an arbitrary handwritten signature.

### [**Code**](/Lab4/lab4.ipynb)

### [**Dataset**](https://www.kaggle.com/robinreni/signature-verification-dataset)

### Procedure 

1. The dataset was obtained and an EDA analysis of the images was conducted. The images had varying sizes; the median values for height and width were found. Subsequently, all images were reshaped to these dimensions.
   
2. A generator was created to feed batches of images into the neural network.

3. A Siamese convolutional neural network was built for authenticity verification of signatures. TensorBoard was used for logging the network's performance.

4. Quality metrics such as precision, recall, f1-score, accuracy, ROC-AUC, Kappa, MCC were measured. Additionally, error matrices, ROC curves, and precision-recall curves were plotted.

5. A search for the best model parameters was conducted.

6. Augmentation techniques were applied to expand the dataset.

7. Wavelet transformation and the application of a thresholding filter were used for edge detection and noise reduction, with the selection of an appropriate wavelet.