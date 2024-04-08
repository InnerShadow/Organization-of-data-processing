# Predict how much BY rubels will 1 USD cost

## Task: Predicting the future exchange rate of the Belarusian ruble against the dollar.

## [**Datasets**](https://www.nbrb.by/statistics/rates/ratesdaily) 

## Usage

To obtain training for the prediction model, you should use the *train.py* module. 

```{bash}
python3 train.py
```

To predict future values of the Belarusian ruble exchange rate, you need to run the *predict.py* module. This will launch a web interface on Flask, where there will be images depicting how the model was trained and how it predicts future values. You can open the images in a higher resolution for detailed viewing. Then, there will be a window to enter the number of days for which you want to make a forecast, as well as a "Predict" button, which will load the model and make predictions for the next n days. After prediction, a graph and actual values rounded to 3 decimal places for the Belarusian ruble will be displayed. Need to upload an XLS document downloaded from the official website of the National Bank of the Republic of Belarus, you need to drag and drop the file into the designated window. [Link](https://www.nbrb.by/statistics/rates/ratesdaily)

```{bash}
python3 predict.py
```

![](/USDpredict/Show_interface.png)

## [**IpyNb**](/USDpredict/usdPredict.ipynb)

1. A dataset was obtained from the official website of the National Bank of the Republic of Belarus. Training data consisted of records from 2004 to 2022. The spike in the Belarusian ruble due to the pandemic was excluded. Then the data were merged into a single table.
2. The data were divided into training and testing sets. 80% of the original dataset was used for training, and 20% for testing with the latest values.
3. A naive forecast was made using the last method.
4. A naive forecast was made using the mean method, with a window size of 20 days.
5. A naive forecast was made with weekly seasonality.
6. A naive forecast was made with monthly seasonality.
7. A forecast was made using exponential smoothing.
8. A forecast was made using exponential smoothing with explicit trend initialization.
9. Auto ETS was applied to tune the parameters of the exponential smoothing model.
10. Autocorrelation function and partial autocorrelation function plots were generated.
11. The Dickey-Fuller test for stationarity was conducted, indicating non-stationarity of the series.
12. Integration order was increased by 1. The series became stationary.
13. Parameter tuning algorithm for ARIMA and SARIMA models was launched. Found coefficients: 1, 1, 3.
14. Residual plot and distribution were plotted.
15. Data were scaled using min-max scaler. A neural network model with 5 LSTM layers was built.
16. ReduceLROnPlateau method was applied during neural network training to decrease the learning rate at minima.
17. A neural network model with 5 LSTM layers was built.
18. A model with two inputs - direct and inverse - and 4 GRU layers for each input was built.
19. An additional Attention input was added to the previous model, resulting in worse metrics.
20. RandomizedSearchCV method was applied for hyperparameter tuning with 25 iterations and 2 repetitions per iteration. Optimal hyperparameters were found.
21. A scatter plot was created to visualize the model's performance in predicting the dollar exchange rate.

## Results

| Model # | Feature of the model | MAPE |
|:-:|:-:|:-:|
| 1 | Naive(last) | 0.03734 |
| 2 | Naive(mean) | 0.02076 |
| 3 | Naive(sp=7) | 0.03549 |
| 4 | Naive(sp=30) | 0.01793 |
| 5 | ETS | 0.03734 |
| 6 | ETS(trean) | 0.01820 |
| 7 | AutoETS | 0.03829 |
| 8 | ARIMA(1, 1, 3) | - |
| 9 | LSTM | 0.00399 |
| 10 | LSTM(ReduceLROnPlateau) | 0.002090 |
| 11 | GRU(ReduceLROnPlateau) | 0.001982 |
| 12 | GRU(direct+inverse inputs) | 0.001938 |
| 13 | GRU(direct+inverse+attention inputs) | 0.001992 |
