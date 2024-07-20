from flask import Flask, request, jsonify
import numpy as np
from tensorflow.python.keras.models import load_model
import joblib

app = Flask(__name__)

# Load models and scalers
models = {}
scalers = {}

tickers = ['AAPL', 'MSFT', 'GOOGL']
for ticker in tickers:
    models[ticker] = load_model(f'{ticker}_lstm_model.h5')
    scalers[ticker] = joblib.load(f'{ticker}_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data['ticker']
    input_data = np.array(data['data']).reshape(-1, 1)

    # Scale the input data
    scaled_data = scalers[ticker].transform(input_data)

    # Make prediction
    prediction = models[ticker].predict(scaled_data)

    # Inverse scale the prediction
    scaled_prediction = scalers[ticker].inverse_transform(prediction)

    return jsonify(scaled_prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)