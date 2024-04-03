# Lab3
## The dataset contains data on sales of 5 different products and air temperature on those days. The frequency of observations for different products does not coincide.

### [**Code**](/Lab3/Lab3.ipynb)

### [**Data set**](https://www.kaggle.com/datasets/soumyadiptadas/products-sales-timeseries-data)

### Procedure 

1. A dataset was obtained, which was divided into a training and testing set. That is, the first 80% of samples constitute the training set, while the last 20% form the testing set.
2. A time series plot was constructed.
3. The forecasting horizon for the testing set was selected.
4. Performance metrics were calculated for each model: MAE, medAE, MSE, MSLE, MAPE.
5. A naive forecast was made using the strategy - last. The results were not very good.
6. A naive forecast model was built using the strategy - mean for the last 20 observations.
7. A naive forecast was made with a seasonality of 4 lags.
8. A forecast was made using exponential smoothing.
9. The presence of a trend was explicitly specified in the exponential smoothing method.
10. Coefficient auto-tuning method was applied to the exponential smoothing model.
11. Autocorrelation and partial autocorrelation plots of the original time series were constructed - the series is non-stationary.
12. An ADF test was conducted to confirm that the series is non-stationary.
13. The series was differenced to check if it became stationary. The series is now stationary. An ADF test is conducted to confirm this.
14. Based on the autocorrelation and partial autocorrelation plots, as well as using auto ARIMA, coefficients for the ARIMA model were determined - 2, 1, 0.
15. A simple LSTM model was created to predict the next value.
16. RandomizedSearch was applied for hyperparameter tuning of the LSTM model. The found parameters are: lag count - 4, number of neurons per layer - 14, number of layers 
