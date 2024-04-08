import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def index():
    image_dir = 'static/images'
    images = os.listdir(image_dir)
    return render_template('index.html', images = images)


@app.route('/upload', methods = ['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = file.filename
        file.save(os.path.join('static/uploads', filename))
        return 'File uploaded successfully', 200


@app.route('/run_function', methods=['POST'])
def run_function():
    input_value = request.form['input_field_name']
    result = predict(int(input_value))
    result = [float(pred) for pred in result]
    return jsonify(result)


def predict(input_value : int):
    time_steps = 32

    xls_file = glob.glob(os.path.join("static/uploads/", '*.xls'))[0]
    df = pd.read_excel(xls_file)
    df = df.drop([0, 1, 2, 3])
    df = df['Unnamed: 58']
    df = df.astype('float32').reset_index(drop = True)

    model = load_model("Model.h5")
    scaler = joblib.load("./Scaler.pkl")
    predictions = []

    for i in range(input_value):
        X_inp = np.array(df[-time_steps : ].values).reshape(1, -1).T
        X_rev = X_inp[::-1]
        X_inp = scaler.transform(X_inp)[np.newaxis, :]
        X_rev = scaler.transform(X_rev)[np.newaxis, :]

        prediction = model.predict([X_inp, X_rev])
        predictions.append(scaler.inverse_transform(prediction)[0, 0])
        df = pd.concat([df, pd.Series(scaler.inverse_transform(prediction)[0, 0])], ignore_index = True)

    visualize_predictions(df[:-input_value], predictions)
    return predictions


def visualize_predictions(original_values, predicted_values):
    plt.figure(figsize = (10, 6))
    plt.plot(original_values, label = 'Original', color = 'blue')
    plt.plot(range(len(original_values) - 1, len(original_values) + len(predicted_values) - 1), 
             predicted_values, label = 'Predicted', color = 'red')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.title('Original vs Predicted Values')
    plt.legend()
    plt.savefig("static/generated/Predictions.png")
    plt.close()


if __name__ == '__main__':
    app.run(debug = True)

