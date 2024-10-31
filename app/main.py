from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import math

app = Flask(__name__)
CORS(app)

# Load the scaler and model
scaler = joblib.load('./models/scaler.pkl')
scaler_mp3 = joblib.load('./models/scaler_v+ma+sdi_mp3.pkl')
model = tf.keras.models.load_model('./models/lstm_model_v+ma_ws3.h5')
model_mp3 = tf.keras.models.load_model('./models/lstm_model_v+ma+sdi_ws3_mp3.h5')

def generateMovingAverage(data, seq_length):
    movingAverages = []
    for i in range(len(data)):
        if (i < seq_length-1):
            movingAverages.append(0)
            continue
        seq = data['Close'][i-(seq_length-1):i+1]
        movingAverages.append(round(seq.sum()/3, 6))
    data = data.assign(movingAverage=movingAverages)
    return data

def generateStandardDeviationIndicator(data, seq_length):
    standardDeviationIndicators = []
    for i in range(len(data)):
        if (i < seq_length-1):
            standardDeviationIndicators.append(0)
            continue
        averagePrice = round(sum(data['Close'][i-(seq_length-1):i+1])/seq_length, 6)
        sumDeviation = 0
        for j in range(i-(seq_length-1), i+1):
            sumDeviation += math.pow(data['Close'][j] - averagePrice, 2)
        variance = round(sumDeviation/(seq_length-1), 6)
        standardDeviationIndicators.append(math.sqrt(variance))
    data = data.assign(standardDeviationIndicator=standardDeviationIndicators)
    return data

def expand_predictions(predictions, original_shape):
    expanded = np.zeros((predictions.shape[0], original_shape[1]))
    expanded[:, [0, 1, 2, 3]] = predictions  # Place predictions in the correct columns
    return expanded

def expand_predictions_multiple(predictions, original_shape):
    batch_size, pred_length, num_features = predictions.shape
    expanded = np.zeros((batch_size * pred_length, original_shape[1]))
    
    for i in range(pred_length):
        expanded[i::pred_length, :num_features] = predictions[:, i, :]
        
    return expanded


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    windowSize = request.json['windowSize']
    predictionDay = request.json['predictionDay']
    feature = {
        'Open': data['open'],
        'High': data['high'],
        'Low': data['low'],
        'Close': data['close'],
        'Volume': data['volume']
        }
    df = pd.DataFrame(feature)
    if (predictionDay == 3):
        df = generateMovingAverage(df, windowSize)
        df = generateStandardDeviationIndicator(df, 3)
        data_scaled = scaler_mp3.transform(df[windowSize-1:])
        data_scaled = data_scaled.reshape(1, 3, 7)
        prediction = model_mp3.predict(data_scaled)
        expanded_predictions = expand_predictions_multiple(prediction, df.shape)
        predictions_inverse = scaler_mp3.inverse_transform(expanded_predictions)[:, :4]
        print(predictions_inverse)
        print(expanded_predictions.shape)
        return jsonify({'prediction': predictions_inverse.tolist()})

    else:
        df = generateMovingAverage(df, windowSize)
        data_scaled = scaler.transform(df[windowSize-1:])
        data_scaled = data_scaled.reshape(1, 3, 6)
        prediction = model.predict(data_scaled)
        expanded_predictions = expand_predictions(prediction, df.shape)
        predictions_inverse = scaler.inverse_transform(expanded_predictions)[:, [0, 1, 2, 3]]
        return jsonify({'prediction': predictions_inverse.tolist()[0]})

@app.route('/getall', methods=['GET'])
def getAll():
    print('tess')
    df = pd.read_csv('./data/BTCPredictionDataForVisualization.csv')
    tes = jsonify({'y': df['Year'].tolist(), 'm': df['Month'].tolist(), 'd': df['Day'].tolist(), 'o': df['Open'].tolist(), 'h': df['High'].tolist(), 'l': df['Low'].tolist(), 'c': df['Close'].tolist()})
    return tes

if __name__ == '__main__':
    app.run(debug=True)