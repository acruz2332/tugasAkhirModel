from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import math

app = Flask(__name__)
CORS(app, origins=["https://tugasakhir-production.up.railway.app"])

model = None
scaler = None

def load_resources(day):
    global model, scaler
    if day == 3:
        scaler = joblib.load('./models/scaler_ma+ewma+rsi_mp3.pkl')
        model = tf.keras.models.load_model('./models/gru_model_ma+ewma+rsi_ws3_mp3.h5')
    else:
        scaler = joblib.load('./models/scaler.pkl')
        model = tf.keras.models.load_model('./models/lstm_model_v+ma_ws3.h5')

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

def generateExponentialMovingAverage(data, seq_length):
    exponentialMovingAverages = []
    mul = 2/(seq_length+1)
    for i in range(len(data)):
        if (i < seq_length-1):
            exponentialMovingAverages.append(0)
            continue
        if (i == seq_length-1):
            seq = data['Close'][i-(seq_length-1):i+1]
            movAverage = round(seq.sum()/3, 6)
            exponentialMovingAverage = ((data['Close'][i] - movAverage))*mul + movAverage
            exponentialMovingAverages.append(round(exponentialMovingAverage, 6))
            continue
        exponentialMovingAverage = ((data['Close'][i] - exponentialMovingAverages[-1])*mul) + exponentialMovingAverages[-1]
        exponentialMovingAverages.append(round(exponentialMovingAverage, 6))
    data = data.assign(exponentialMovingAverage=exponentialMovingAverages)
    return data

def generateRelativeStrengthIndex(data, seq_length):
    relativeStrengthIndexs = []
    for i in range(len(data)):
        if (i < seq_length-1):
            relativeStrengthIndexs.append(0)
            continue
        slicedData = data['Close'][i-(seq_length-1):i+1].reset_index(drop=True)
        priceChanges = []
        for j in range(seq_length-1):
            priceChanges.append(round(slicedData[j+1]-slicedData[j], 6))
        positiveGained = [i if i > 0 else 0 for i in priceChanges]
        negativeGained = [abs(i) if i < 0 else 0 for i in priceChanges]
        avgPositiveGained = round(sum(positiveGained)/(seq_length-1), 6)
        avgNegativeGained = round(sum(negativeGained)/(seq_length-1), 6)
        if avgNegativeGained == 0:
            relativeStrengthIndex = 100
        else:
            relativeStrength = round(avgPositiveGained/avgNegativeGained, 6)
            relativeStrengthIndex = round(100 - (100/(1 + relativeStrength)), 6)
        relativeStrengthIndexs.append(relativeStrengthIndex)
    data = data.assign(relativeStrengthIndex=relativeStrengthIndexs)
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
    expanded[:, [0, 1, 2, 3]] = predictions
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
    load_resources(predictionDay)
    if (predictionDay == 3):
        df = df.drop(['Volume'], axis=1)
        df = generateMovingAverage(df, windowSize)
        df = generateExponentialMovingAverage(df, windowSize)
        df = generateRelativeStrengthIndex(df, windowSize)
        data_scaled = scaler.transform(df[windowSize-1:])
        data_scaled = data_scaled.reshape(1, 3, 7)
        prediction = model.predict(data_scaled)
        expanded_predictions = expand_predictions_multiple(prediction, df.shape)
        predictions_inverse = scaler.inverse_transform(expanded_predictions)[:, :4]
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


@app.after_request
def apply_csp(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "connect-src 'self' https://tugasakhir-production.up.railway.app; "
        "script-src 'self'; "
        "img-src 'self'; "
        "style-src 'self'"
    )
    return response


if __name__ == '__main__':
    app.run(debug=True)