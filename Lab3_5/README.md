# Lab3_5
## Get dataset of Power Consumption. Choose one zone and create forecasting model to predict future power Consumption. Use both statistics models and deep learning methods.

### [**Code**](/Lab3_5/lab3_5.ipynb)

### [**Data set**](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption/data)

### Procedure 

1. The dataset was split into a training and testing set, with the first 80% of samples comprising the training set and the last 20% forming the testing set.
2. A time series plot was created.
3. The forecasting horizon was chosen based on the testing set.
4. Metrics including MAE, medAE, MSE, MSLE, and MAPE were collected for each model.
5. A naive forecast was made using the strategy - last. The results were not very good.
6. A model for naive forecasting was built using the strategy - mean for the last 30 observations.
7. A forecast was made using exponential smoothing.
8. The presence of a trend was explicitly specified in the exponential smoothing method.
9. The coefficient auto-tuning method was applied to the exponential smoothing model.
10. Autocorrelation and partial autocorrelation plots of the original time series were constructed - the series is non-stationary.
11. An ADF test was conducted to confirm that the series is non-stationary.
12. The series was differenced to check if it became stationary. The series is now stationary. An ADF test was conducted to confirm this.
13. Based on the autocorrelation and partial autocorrelation plots, as well as using auto ARIMA, coefficients for the ARIMA model were determined - 0, 1, 1.
14. A residual plot, a histogram of residual distribution, an autocorrelation plot of residuals, and a partial autocorrelation plot of residuals were created.
15. A SARIMA model with coefficients 0, 1, 1 and seasonal coefficients 0, 1, 1, 31 (indicating one month) was built.
16. A simple LSTM model was created to predict the next value.
17. A GRU model with the same architecture was created.
18. A model with two inputs - direct and inverted - for the GRU model was created.
19. A model with three inputs - direct, inverted for the GRU layers and one to attentions with LSTM layers was created.
20. The performance of each model was displayed, and the training graph was created.