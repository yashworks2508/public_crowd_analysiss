import pickle
import pandas as pd
import numpy as np

# --------- LOAD TRAINED MODELS ----------
with open('models/crowd_prediction_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

passenger_model = models_data['passenger_count_model']
crowding_model = models_data['crowding_ratio_model']
feature_columns = models_data['feature_columns']

# --------- GENERATE TEST DATA ----------
# This matches your data structure in backend.py. You can adjust hours/days as needed.
def create_test_data(days=7):
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
    np.random.seed(100)
    for day_idx in range(days):
        for hour in range(6, 23):
            day_of_week = day_idx % 7
            is_weekend = 1 if day_of_week >= 5 else 0
            temperature = 72
            weather_numeric = 1
            event_impact = 1.0
            has_event = 0
            for route in routes:
                route_type_encoded = 1 if route['route_type']=='Rail' else 0
                vehicle_capacity = route['capacity']
                for stop in stops:
                    major_stop = 1 if stop['major'] else 0
                    is_morning_rush = 1 if hour in [7,8,9] else 0
                    is_evening_rush = 1 if hour in [17,18,19] else 0
                    is_rush_hour = 1 if is_morning_rush or is_evening_rush else 0
                    base = 20 if route_type_encoded == 0 else 35
                    base *= (2.5 if is_rush_hour else 1)
                    base *= (1.8 if major_stop else 1)
                    base *= event_impact
                    passengers = max(0, int(np.random.normal(base, base * 0.2)))
                    crowd_ratio = min(passengers / vehicle_capacity, 1.5)
                    records.append({
                        'route_name': route['route_name'],
                        'stop_name': stop['stop_name'],
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'is_weekend': is_weekend,
                        'route_type_encoded': route_type_encoded,
                        'vehicle_capacity': vehicle_capacity,
                        'temperature': temperature,
                        'weather_numeric': weather_numeric,
                        'event_impact': event_impact,
                        'has_event': has_event,
                        'is_morning_rush': is_morning_rush,
                        'is_evening_rush': is_evening_rush,
                        'major_stop': major_stop,
                        'actual_passenger_count': passengers,
                        'actual_crowding_ratio': crowd_ratio
                    })
    return pd.DataFrame(records)

df = create_test_data(days=7)  # One week of test data

# ------------- PREDICTION AND EVALUATION --------------
X = df[feature_columns]
y_true = df['actual_passenger_count']

# Predict
pred_passengers = passenger_model.predict(X)
pred_crowding = crowding_model.predict(X)

# ----------- BUILD RESULT TABLE -------------
df_result = df.copy()
df_result['predicted_passenger_count'] = pred_passengers.round().astype(int)
df_result['predicted_crowding_ratio'] = pred_crowding.round(2)
df_result['abs_error_passengers'] = np.abs(df_result['actual_passenger_count'] - df_result['predicted_passenger_count'])
df_result['abs_error_crowding'] = np.abs(df_result['actual_crowding_ratio'] - df_result['predicted_crowding_ratio'])

# -------------- SAVE TO CSV -----------------
output_path = 'model_results.csv'  # Output will be in your project directory
df_result.to_csv(output_path, index=False)

# ------------ PRINT METRICS -----------------
from sklearn.metrics import mean_absolute_error, r2_score

mae_passengers = mean_absolute_error(df_result['actual_passenger_count'], df_result['predicted_passenger_count'])
r2_passengers = r2_score(df_result['actual_passenger_count'], df_result['predicted_passenger_count'])

print(f"Saved: {output_path}")
print(f"Passenger Prediction MAE: {mae_passengers:.2f}")
print(f"Passenger Prediction R²: {r2_passengers:.2f}")