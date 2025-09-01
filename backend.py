import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from flask import Flask, request, jsonify
from flask_cors import CORS 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# ------------------ Data Generation ------------------

def create_sample_data(days=30):
    routes = [
        {'route_id': 'R1', 'route_name': 'Blue Line', 'route_type': 'Rail', 'capacity': 200},
        {'route_id': 'B15', 'route_name': 'Bus 15', 'route_type': 'Bus', 'capacity': 50},
        {'route_id': 'B42', 'route_name': 'Bus 42', 'route_type': 'Bus', 'capacity': 50},
    ]

    stops = [
        {'stop_id': 'S1', 'stop_name': 'Downtown Station', 'major': True},
        {'stop_id': 'S2', 'stop_name': 'University Plaza', 'major': False},
        {'stop_id': 'S3', 'stop_name': 'Stadium', 'major': True},
        {'stop_id': 'S4', 'stop_name': 'Shopping Mall', 'major': False},
        {'stop_id': 'S5', 'stop_name': 'Business District', 'major': True},
        {'stop_id': 'S6', 'stop_name': 'Airport Terminal', 'major': True},
    ]

    records = []
    np.random.seed(42)
    random.seed(42)

    for day_idx in range(days):
        date = datetime.now() - timedelta(days=days - day_idx)
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        temperature = 65 + 15 * np.sin(day_idx / 365 * 2 * np.pi) + np.random.normal(0, 5)
        weather_condition = random.choices(['Clear', 'Rain', 'Cloudy'], weights=[0.7, 0.15, 0.15])[0]
        weather_numeric = {'Clear':1, 'Cloudy':2, 'Rain':3}[weather_condition]

        has_event = random.random() < 0.1
        event_impact = 2.5 if has_event else 1.0

        for hour in range(6, 23):
            is_morning_rush = 1 if hour in [7,8,9] else 0
            is_evening_rush = 1 if hour in [17,18,19] else 0
            is_rush_hour = 1 if is_morning_rush or is_evening_rush else 0

            for route in routes:
                route_type_encoded = 1 if route['route_type']=='Rail' else 0
                vehicle_capacity = route['capacity']

                for stop in stops:
                    major_stop = 1 if stop['major'] else 0

                    # Base passengers
                    base = 20 if route_type_encoded == 0 else 35
                    base *= (2.5 if is_rush_hour else 1)
                    base *= (1.8 if major_stop else 1)
                    base *= event_impact
                    base *= (1.3 if weather_condition == 'Rain' else 1.0)
                    base *= (0.7 if is_weekend else 1.0)

                    passengers = max(0, int(np.random.normal(base, base * 0.2)))
                    crowd_ratio = min(passengers / vehicle_capacity, 1.5)

                    records.append({
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'is_weekend': is_weekend,
                        'route_type_encoded': route_type_encoded,
                        'vehicle_capacity': vehicle_capacity,
                        'temperature': temperature,
                        'weather_numeric': weather_numeric,
                        'event_impact': event_impact,
                        'has_event': 1 if has_event else 0,
                        'is_morning_rush': is_morning_rush,
                        'is_evening_rush': is_evening_rush,
                        'major_stop': major_stop,
                        'passenger_count': passengers,
                        'crowding_ratio': crowd_ratio
                    })

    return pd.DataFrame(records)

# ------------------ Training ------------------

def train_and_save_models():
    df = create_sample_data(days=60)
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'route_type_encoded', 'vehicle_capacity',
                    'temperature', 'weather_numeric', 'event_impact', 'has_event',
                    'is_morning_rush', 'is_evening_rush', 'major_stop']

    X = df[feature_cols]
    y_passengers = df['passenger_count']
    y_crowding = df['crowding_ratio']

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_passengers, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crowding, test_size=0.2, random_state=42)

    passenger_model = RandomForestRegressor(n_estimators=100, random_state=42)
    passenger_model.fit(X_train_p, y_train_p)

    crowding_model = RandomForestRegressor(n_estimators=100, random_state=42)
    crowding_model.fit(X_train_c, y_train_c)

    # Save models
    if not os.path.exists('models'):
        os.mkdir('models')

    with open('models/crowd_prediction_models.pkl', 'wb') as f:
        pickle.dump({
            'passenger_count_model': passenger_model,
            'crowding_ratio_model': crowding_model,
            'feature_columns': feature_cols
        }, f)
    print("Models trained and saved to 'models/crowd_prediction_models.pkl'")

train_and_save_models()

# ------------------ Flask API ------------------

app = Flask(__name__)
CORS(app)

with open('models/crowd_prediction_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

passenger_model = models_data['passenger_count_model']
crowding_model = models_data['crowding_ratio_model']
feature_columns = models_data['feature_columns']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = []
    for col in feature_columns:
        if col not in data:
            return jsonify({"error":f"Missing feature: {col}"}), 400
        val = data[col]
        if isinstance(val, str):
            if val.lower() == 'true':
                val = 1
            elif val.lower() == 'false':
                val = 0
            else:
                try:
                    val = float(val)
                except:
                    pass
        features.append(val)
    features = np.array(features).reshape(1, -1)

    passenger_pred = passenger_model.predict(features)[0]
    crowd_ratio_pred = crowding_model.predict(features)[0]

    crowd_ratio_pred = min(max(crowd_ratio_pred, 0), 1.5)

    if crowd_ratio_pred < 0.3:
        level = 'Low'
    elif crowd_ratio_pred < 0.7:
        level = 'Medium'
    else:
        level = 'High'

    return jsonify({
        'predicted_passengers': int(round(passenger_pred)),
        'predicted_crowding_ratio': round(crowd_ratio_pred, 2),
        'crowd_level': level
    })

if __name__ == '__main__':
    print("Server starting on http://localhost:5000")
    app.run(debug=True)