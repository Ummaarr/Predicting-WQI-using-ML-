import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

def calculate_qi(value, standard):
    return (value / standard) * 100

def calculate_wqi(parameter_values, weights, standards):
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]
    quality_indices = [calculate_qi(parameter_values[i], standards[i]) for i in range(len(parameter_values))]
    s_values = [normalized_weights[i] * quality_indices[i] for i in range(len(parameter_values))]
    return sum(s_values)

# Standardize column names mapping
column_mapping = {
    'pH': 'PH', 'Cond. (μS)': 'Cond', 'TDS(mg/ l)': 'TDS', 'DO(mg/l)': 'DOX',
    'Ca(mg/ l)': 'Ca', 'Mg(mg/l)': 'Mg', 'Na(m g/l)': 'Na', 'K(mg/l)': 'K',
    'Cl(mg/l)': 'Cl', 'HCO3(mg/l)': 'HCO3', 'SO4(mg/l)': 'SO4', 'PO4(mg/l)': 'PO4'
}

def preprocess_data(data):
    # Rename columns
    data.rename(columns=column_mapping, inplace=True)

    # Define parameters, weights, and standards for WQI
    parameters = ['PH', 'Cond', 'TDS', 'DOX', 'Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4', 'HCO3', 'PO4']
    weights = [4, 3, 3, 4, 3, 2, 2, 1, 3, 2, 4, 4]
    standards = [8.5, 1000, 500, 6, 75, 30, 200, 12, 250, 200, 250, 0.1]

    # Check for missing columns
    missing_cols = [param for param in parameters if param not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Handle missing and infinite values
    for param in parameters:
        # Replace infinite values with NaN
        data[param] = data[param].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median
        data[param] = data[param].fillna(data[param].median())

    # Calculate WQI
    data['WQI'] = data.apply(lambda row: calculate_wqi([row[param] for param in parameters], weights, standards), axis=1)

    # Convert mg/L to meq/L
    epsilon = 1e-10
    data['Ca_meq'] = (data['Ca'] / 40) * 2
    data['Mg_meq'] = (data['Mg'] / 24) * 2
    data['Na_meq'] = data['Na'] / 23
    data['K_meq'] = data['K'] / 39.1
    data['HCO3_meq'] = data['HCO3'] / 61

    # Calculate additional parameters
    # Calculate additional parameters
    data['SAR'] = ((data['Na_meq']) ** 2) / np.sqrt((data['Ca_meq'] )+ (data['Mg_meq'] + epsilon))
    data['PI'] = ((data['Na_meq'] + (np.sqrt(data['HCO3_meq'] + epsilon))) /
                  (np.sqrt(data['Ca_meq'] + data['Mg_meq'] + data['Na_meq'] + epsilon)))
    data['Na_percent'] = ((data['Na_meq'] + data['K_meq']) /
                          (data['Ca_meq'] + data['Mg_meq'] + data['Na_meq'] + data['K_meq']))
    data['RSC'] = data['HCO3_meq'] - (data['Ca_meq'] + data['Mg_meq'])

    return data, parameters

def create_ml_models(X_train, y_train, X_test, y_test, target_name):
    """
    Create and train both MLP and LSTM models with cross-validation
    """
    # Prepare data (handle any remaining NaN or inf)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=1.0, neginf=0.0)

    # MLP Model
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    # LSTM Model
    tf.keras.backend.clear_session()
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    lstm = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm.fit(
        X_train_lstm, y_train,
        epochs=100,
        batch_size=min(32, X_train.shape[0] // 2),
        validation_data=(X_test_lstm, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    return mlp, lstm

def main():
    # Load and preprocess data
    data = pd.read_csv('data.csv')
    data, features = preprocess_data(data)

    # Target variables
    target_variables = ['WQI', 'SAR', 'PI', 'Na_percent', 'RSC']

    # Scale features and targets
    feature_scaler = MinMaxScaler()
    X = feature_scaler.fit_transform(data[features])

    # Initialize dictionary to store results
    prediction_results = {}

    # Train models and generate predictions for each target
    for target in target_variables:
        print(f"\nProcessing {target}...")

        # Prepare target data
        y = data[target].values
        y_scaled = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

        # Train models
        mlp_model, lstm_model = create_ml_models(X_train, y_train, X_test, y_test, target)

        # Prepare full dataset for prediction
        X_full_scaled = feature_scaler.transform(data[features])
        X_full_lstm = X_full_scaled.reshape(X_full_scaled.shape[0], 1, X_full_scaled.shape[1])

        # Generate predictions
        mlp_preds_scaled = mlp_model.predict(X_full_scaled)
        lstm_preds_scaled = lstm_model.predict(X_full_lstm).flatten()

        # Inverse transform predictions
        mlp_scaler = MinMaxScaler().fit(data[target].values.reshape(-1, 1))
        lstm_scaler = MinMaxScaler().fit(data[target].values.reshape(-1, 1))

        mlp_preds = mlp_scaler.inverse_transform(mlp_preds_scaled.reshape(-1, 1)).flatten()
        lstm_preds = lstm_scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()

        # Store results
        prediction_results[target] = {
            'MLP_Predictions': mlp_preds,
            'LSTM_Predictions': lstm_preds
        }

        # Calculate and print error metrics
        mlp_mse = np.mean((data[target] - mlp_preds) ** 2)
        lstm_mse = np.mean((data[target] - lstm_preds) ** 2)
        print(f"{target} Mean Squared Error:")
        print(f"  MLP MSE: {mlp_mse:.4f}")
        print(f"  LSTM MSE: {lstm_mse:.4f}")

    # Add predictions to original dataframe
    for target in target_variables:
        data[f'{target}_MLP_Pred'] = prediction_results[target]['MLP_Predictions']
        data[f'{target}_LSTM_Pred'] = prediction_results[target]['LSTM_Predictions']

    # Save results
    data.to_csv('water_quality_predictions.csv', index=False)
    print("\nPredictions saved to water_quality_predictions.csv")

    # === Add model comparison and evaluation metrics ===
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from dtaidistance import dtw
    import seaborn as sns

    # Load predictions from DataFrame (already in memory as 'data')
    comparison_data = []

    for target in target_variables:
        actual = data[target].values
        mlp_pred = data[f'{target}_MLP_Pred']
        lstm_pred = data[f'{target}_LSTM_Pred']

        comparison_data.append({
            'Parameter': target,
            'MLP_RMSE': np.sqrt(mean_squared_error(actual, mlp_pred)),
            'LSTM_RMSE': np.sqrt(mean_squared_error(actual, lstm_pred)),
            'MLP_R2': r2_score(actual, mlp_pred),
            'LSTM_R2': r2_score(actual, lstm_pred),
            'MLP_DTW': dtw.distance(actual, mlp_pred),
            'LSTM_DTW': dtw.distance(actual, lstm_pred)
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('model_comparison_metrics.csv', index=False)

    # Plot RMSE
    comparison_df.plot(x='Parameter', y=['MLP_RMSE', 'LSTM_RMSE'], kind='bar', title='RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rmse_comparison.png')
    plt.close()

    # Plot R2
    comparison_df.plot(x='Parameter', y=['MLP_R2', 'LSTM_R2'], kind='bar', title='R² Score Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('r2_comparison.png')
    plt.close()

    # Heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(comparison_df.set_index('Parameter'), annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Heatmap of Model Evaluation Metrics')
    plt.tight_layout()
    plt.savefig('model_metrics_heatmap.png')
    plt.close()

    # Visualization
    plt.figure(figsize=(15, 12))
    for i, target in enumerate(target_variables):
        plt.subplot(len(target_variables), 1, i + 1)
        plt.plot(data.index, data[target], 'b-', label='Actual')
        plt.plot(data.index, data[f'{target}_MLP_Pred'], 'r--', label='MLP Prediction')
        plt.plot(data.index, data[f'{target}_LSTM_Pred'], 'g--', label='LSTM Prediction')
        plt.title(f'{target} - Actual vs Predicted')
        plt.legend()

    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()