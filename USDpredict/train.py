import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras

from keras.models import Model, save_model
from keras.layers import Dense, GRU, concatenate, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error

def grabData() -> pd.DataFrame:
    df = pd.read_excel('2022_day_ru.xls')
    df = df.drop([0, 1, 2, 3])
    df = df['Unnamed: 58']
    df = df.astype('float32')
    df = df[150 : ].reset_index(drop = True)

    df_2023 = pd.read_excel('2023_day_ru.xls')
    df_2023 = df_2023.drop([0, 1, 2, 3])
    df_2023 = df_2023['Unnamed: 58']
    df_2023 = df_2023.astype('float32')
    df = pd.concat([df, df_2023]).reset_index(drop = True)

    df_2024 = pd.read_excel('2024_day_ru.xls')
    df_2024 = df_2024.drop([0, 1, 2, 3])
    df_2024 = df_2024['Unnamed: 58']
    df_2024 = df_2024.astype('float32')
    df = pd.concat([df, df_2024]).reset_index(drop = True)
    return df


if __name__ == '__main__':
    df = grabData()
    df.plot()
    plt.imsave("OriginalDataSet.png")

    scaler = MinMaxScaler(feature_range = (0, 1))
    data = scaler.fit_transform(df.values.reshape(-1, 1))

    time_steps = 32
    n_neurons = 32

    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : (i + time_steps), 0])
        y.append(data[i + time_steps, 0])

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[ : split], X[split : ]
    y_train, y_test = y[ : split], y[split : ]
    X_train_rev = X_train[:, ::-1]
    X_test_rev = X_test[:, ::-1]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    X_train_rev = X_train_rev.reshape(X_train_rev.shape[0], X_train_rev.shape[1], 1)
    X_test_rev = X_test_rev.reshape(X_test_rev.shape[0], X_test_rev.shape[1], 1)

    inp1 = Input(shape = ((time_steps, 1)))
    direct = GRU(units = n_neurons, return_sequences = True, activation = 'elu')(inp1)
    direct = GRU(units = n_neurons, return_sequences = True, activation = 'elu')(direct)
    direct = GRU(units = n_neurons, return_sequences = True, activation = 'elu')(direct)
    direct = GRU(units = n_neurons, activation = 'elu')(direct)

    inp2 = Input(shape = ((time_steps, 1)))
    reverse = GRU(units = n_neurons, return_sequences = True, activation = 'elu')(inp2)
    reverse = GRU(units = n_neurons, return_sequences = True, activation = 'elu')(reverse)
    reverse = GRU(units = n_neurons, return_sequences = True, activation = 'elu')(reverse)
    reverse = GRU(units = n_neurons, activation = 'elu')(reverse)

    concat = concatenate([direct, reverse])
    concat = Dense(n_neurons * 16, activation = 'elu')(concat)
    output = Dense(units = 1)(concat)

    model = Model(inputs = [inp1, inp2], outputs = [output])
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, min_lr = 0.0001)
    tensorboard = TensorBoard(log_dir = "./TensorboardDir/", write_images = True)

    mlflow.set_tracking_uri("./MlflowRuns/")
    mlflow.set_experiment("USD predictions")

    history = model.fit([X_train, X_train_rev], y_train, epochs = 75, batch_size = 1,
                        validation_data = ([X_test, X_test_rev], y_test), 
                        callbacks = [early_stopping, reduce_lr, tensorboard])

    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.imsave("lossModelPlot.png")

    predicted_values = model.predict([X_test, X_test_rev])
    predicted_values = scaler.inverse_transform(predicted_values)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mlflow.log_metric("MAE", mean_absolute_error(y_test, predicted_values))
    mlflow.log_metric("MSE", mean_squared_error(y_test, predicted_values))
    mlflow.log_metric("MSLE", mean_squared_log_error(y_test, predicted_values))
    mlflow.log_metric("medAE", median_absolute_error(y_test, predicted_values))
    mlflow.log_metric("MAPE", mean_absolute_percentage_error(y_test, predicted_values))
    mlflow.keras.log_model(model, "USDpredictions")
    mlflow.end_run()
    model.save("./Model.h5")

    plt.plot(df.index[ : -len(y_test)], df.iloc[ : -len(y_test)], label = 'Train')
    plt.plot(df.index[-len(y_test) : ], df.iloc[-len(y_test) : ], label = 'Test')
    plt.plot(df.index[-len(predicted_values) : ], predicted_values, label = 'Predicted')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.imsave("PlotOfPredictedValues.png")

    plt.plot(np.arange(3.10, 3.3, 0.01), np.arange(3.10, 3.3, 0.01), c = 'red', alpha = 0.5)
    plt.scatter(y_test, predicted_values, 
                c = np.abs(y_test - predicted_values), cmap = 'viridis')
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.imsave("ScatterResidualsPlot.png")

