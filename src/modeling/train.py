"""Script to train predictive maintenance models for different machine types."""

import pandas as pd
import numpy as np
from datetime import datetime
import random
import os
import sys
import argparse
import joblib # For saving sklearn models/pipelines
from pathlib import Path
import json

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest # Added import

# XGBoost
import xgboost as xgb

# TensorFlow / Keras for LSTM
# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
# Conditional import for Keras components based on TF version
try:
    # TensorFlow 2.16+ uses keras.src
    from keras.src.models import Sequential
    from keras.src.layers import LSTM, Dense, Dropout
    from keras.src.callbacks import EarlyStopping
except ImportError:
    # Older TensorFlow versions use keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping


# Add project root to sys.path to allow importing custom modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import custom modules
from src.modeling.feature_engineering import apply_feature_engineering, extract_base_features # Added extract_base_features
from src.modeling.predict_maintenance import suggest_maintenance_from_predictions, DEFAULT_HEALTH_THRESHOLD

# --- Configuration & Constants ---
RANDOM_SEED = 42
RESULTS_DIR = project_root / "results"
MODELS_DIR = project_root / "src" / "modeling" / "saved_models" # Added models directory path
DATA_DIR = project_root / "src" / "data_generation_health_index"
N_SPLITS_CV = 5       # Number of splits for TimeSeriesSplit Cross-Validation
N_ITER_SEARCH = 20    # Number of iterations for RandomizedSearchCV (increase for better results)
TEST_SIZE_PERCENT = 0.2 # Percentage of data held out for final testing

# Feature Engineering Params
LAGS = [1, 3, 6, 12, 24] # e.g., 1 hour ago, 3 hours ago, etc.
WINDOW_SIZES = [6, 12, 24] # Rolling windows (e.g., 6 hours, 12 hours)

# LSTM Params
LSTM_WINDOW_SIZE = 24 # How many past hours to look at for predicting the next hour
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 64
LSTM_PATIENCE = 5 # Early stopping patience

# --- Helper Functions ---

def load_data(machine_type):
    """Loads the generated CSV data for a specific machine type."""
    file_path = DATA_DIR / f"{machine_type}_health_data.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def get_sensor_columns(df):
    """Identifies sensor columns (excluding metadata and target)."""
    exclude_cols = ['timestamp', 'machine_id', 'health_index', 'simulation_type']
    sensor_cols = [col for col in df.columns if col not in exclude_cols and '_lag_' not in col and '_roll_' not in col and col not in ['hour', 'dayofweek', 'dayofyear', 'month']]
    print(f"Identified sensor columns: {sensor_cols}")
    return sensor_cols

def create_lstm_sequences(data, sequence_length):
    """Reshapes data into sequences for LSTM input (features + target)."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :-1]) # Features sequence
        y.append(data[i + sequence_length, -1])    # Target at the end of sequence
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Builds the LSTM model architecture."""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Output layer (predicting health_index)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("LSTM Model Summary:")
    model.summary()
    return model

def evaluate_model(y_true, y_pred, model_name, machine_type, results_path):
    """Calculates regression metrics and saves results."""
    # Ensure metrics are standard Python floats for JSON serialization
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

    print(f"\n--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save metrics
    metrics_file = results_path / f"{model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

    # --- Plotting --- 
    plt.figure(figsize=(12, 6))
    
    # Actual vs Predicted Scatter
    plt.subplot(1, 2, 1) 
    plt.scatter(y_true, y_pred, alpha=0.3, label='Predictions', s=10)
    plt.plot([min(y_true.min(), 0), max(y_true.max(), 1)], [min(y_true.min(), 0), max(y_true.max(), 1)], 'r--', label='Ideal Fit')
    plt.title(f'{model_name} - Actual vs. Predicted')
    plt.xlabel("Actual Health Index")
    plt.ylabel("Predicted Health Index")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.gca().set_aspect('equal', adjustable='box') # Make axes equal
    
    # Residual Plot
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Health Index')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.tight_layout()

    plot_file = results_path / f"{model_name}_evaluation_plots.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Evaluation plots saved to: {plot_file}")

    return metrics

def save_pipeline(pipeline, model_name, machine_type, results_path):
    """Saves the scikit-learn pipeline (scaler + model)."""
    pipeline_file = results_path / f"{model_name}_pipeline.joblib"
    joblib.dump(pipeline, pipeline_file)
    print(f"Pipeline (scaler + {model_name}) saved to: {pipeline_file}")

def save_lstm_components(scaler, model, model_name, machine_type, results_path):
    """Saves the LSTM scaler and model separately."""
    scaler_file = results_path / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to: {scaler_file}")

    model_file = results_path / f"{model_name}_model.keras" # Use .keras format
    model.save(model_file)
    print(f"LSTM model saved to: {model_file}")

# --- Anomaly Detection Training ---

def train_anomaly_detector(df, machine_type, sensor_cols, results_path, model_save_dir):
    """Trains an Isolation Forest model for anomaly detection on normal data."""
    print("\n--- Starting Anomaly Detector Training ---")
    model_save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    # 1. Filter for 'normal' data (assuming 'simulation_type' column exists)
    if 'simulation_type' in df.columns:
        normal_df = df[df['simulation_type'] == 'normal'].copy()
        if normal_df.empty:
            print("Warning: No data with simulation_type='normal' found. Using all data for anomaly training.")
            normal_df = df.copy() # Fallback to using all data if no 'normal' data
        else:
            print(f"Filtered for 'normal' data. Shape: {normal_df.shape}")
    else:
        print("Warning: 'simulation_type' column not found. Using all data for anomaly training.")
        normal_df = df.copy()

    if normal_df.empty:
        print("Error: No data available for anomaly detection training. Skipping.")
        return

    # 2. Select a primary sensor and set timestamp as index
    # For simplicity, using the first sensor column identified
    if not sensor_cols:
        print("Error: No sensor columns identified. Cannot train anomaly detector.")
        return
    primary_sensor = sensor_cols[0]
    print(f"Using primary sensor for anomaly detection: {primary_sensor}")

    if 'timestamp' not in normal_df.columns:
         print("Error: 'timestamp' column required for feature extraction. Skipping anomaly training.")
         return
         
    # Ensure timestamp is datetime and set as index for feature extraction
    normal_df['timestamp'] = pd.to_datetime(normal_df['timestamp'])
    normal_df = normal_df.set_index('timestamp').sort_index()
    sensor_series = normal_df[primary_sensor]

    # 3. Extract Base Features
    print("Extracting base features (rolling mean/std)...")
    # Use the window size defined in feature_engineering or define one here
    try:
        # Assuming ROLLING_WINDOW_SIZE is accessible or redefine it
        from src.modeling.feature_engineering import ROLLING_WINDOW_SIZE
        anomaly_features = extract_base_features(sensor_series, window=ROLLING_WINDOW_SIZE)
    except ImportError:
         print("Warning: ROLLING_WINDOW_SIZE not found in feature_engineering. Using default window=10.")
         anomaly_features = extract_base_features(sensor_series, window=10) # Use a default

    if anomaly_features.empty or anomaly_features.isnull().all().all():
        print("Error: Feature extraction resulted in empty or all-NaN DataFrame. Skipping anomaly training.")
        return
        
    # Drop any remaining NaNs after feature extraction's internal handling
    initial_rows = len(anomaly_features)
    anomaly_features = anomaly_features.dropna()
    if anomaly_features.empty:
        print(f"Error: All rows removed after dropna post-feature-extraction ({initial_rows} initial). Skipping anomaly training.")
        return
    print(f"Features extracted. Shape: {anomaly_features.shape}. Rows removed by dropna: {initial_rows - len(anomaly_features)}")


    # 4. Instantiate and Train Isolation Forest
    # Contamination: expected proportion of outliers. 'auto' lets sklearn decide.
    # Or set a specific value like 0.01 (1%) if you have an estimate.
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination='auto', # Or e.g., 0.01
                                 random_state=RANDOM_SEED,
                                 n_jobs=-1) # Use all available CPU cores
    iso_forest.fit(anomaly_features)
    print("Isolation Forest training complete.")

    # 5. Save the Model
    model_filename = model_save_dir / f"anomaly_model_{machine_type}.joblib"
    joblib.dump(iso_forest, model_filename)
    print(f"Anomaly detection model saved to: {model_filename}")

# --- Main Training Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Predictive Maintenance Models")
    parser.add_argument("machine_type", type=str, help="Type of machine to train the model for (e.g., siemens_motor)")
    parser.add_argument("--isolation-only", action="store_true", help="Only retrain anomaly detection Isolation Forest models")
    args = parser.parse_args()
    machine_type = args.machine_type

    print(f"===== Starting Training for: {machine_type} =====")
    start_time_script = datetime.now()

    # --- 1. Setup Paths ---
    machine_results_dir = RESULTS_DIR / machine_type
    machine_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {machine_results_dir}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure models directory exists

    # --- 2. Load and Prepare Data ---
    df = load_data(machine_type)
    sensor_cols = get_sensor_columns(df)

    # If the isolation-only flag is provided, retrain only the anomaly detection model and exit.
    if args.isolation_only:
        print("Retraining anomaly detection Isolation Forest model only...")
        train_anomaly_detector(df.copy(), machine_type, sensor_cols, machine_results_dir, MODELS_DIR)  # Pass a copy of df
        print("Isolation Forest retraining complete.")
        exit(0)

    print("\nApplying feature engineering for Health Index Prediction...")
    # This part prepares data for the health index regression models (Phase 2 style)
    df_featured = apply_feature_engineering(df, sensor_cols, LAGS, WINDOW_SIZES)

    # Define features (X) and target (y)
    target_col = 'health_index'
    exclude_cols_final = ['timestamp', 'machine_id', 'simulation_type', target_col] + sensor_cols
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols_final]

    X = df_featured[feature_cols].astype(np.float32) # Ensure float32 for consistency
    y = df_featured[target_col].astype(np.float32)

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target column: {target_col}")
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # --- 3. Time-Series Split (Train/Test) ---
    # Hold out the last portion for final testing
    test_size = int(len(X) * TEST_SIZE_PERCENT)
    train_size = len(X) - test_size
    
    # Ensure indices are aligned before splitting
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    timestamps_test = df_featured['timestamp'].iloc[train_size:].reset_index(drop=True)
    print(f"\nTrain set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- 4. Scaling (for non-LSTM models) ---
    # Scaler for RF and XGBoost - will be part of their pipeline
    # LSTM scaler is separate as it needs inverse transform later if needed
    scaler_skl = StandardScaler() # Scaler used within RF/XGB pipelines
    scaler_lstm = StandardScaler() # Scaler used explicitly for LSTM

    X_train_scaled_lstm = scaler_lstm.fit_transform(X_train)
    X_test_scaled_lstm = scaler_lstm.transform(X_test)
    print("\nData scaled for LSTM using StandardScaler.")

    # Setup for Cross-Validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    scoring = 'neg_root_mean_squared_error' # Lower is better, hence negative
    
    # Store evaluation results
    all_metrics = {}
    best_estimators = {}

    # =========================================
    # --- 5. Model Training: RandomForest ---
    # =========================================
    print("\n--- Training Random Forest Regressor ---")
    start_time_rf = datetime.now()

    rf_pipeline = Pipeline([
        ('scaler', scaler_skl), # Use the shared scaler
        ('model', RandomForestRegressor(n_estimators=100, # Use a reasonable default
                                     max_depth=20,     # Use a reasonable default
                                     min_samples_split=5, # Use a reasonable default
                                     min_samples_leaf=3,  # Use a reasonable default
                                     random_state=RANDOM_SEED,
                                     n_jobs=-1)) # Use all cores
    ])

    # Fit the pipeline directly
    print("Fitting RandomForest pipeline...")
    rf_pipeline.fit(X_train, y_train)
    best_rf_pipeline = rf_pipeline # Use the fitted pipeline

    # Evaluate RF on Test Set
    print("Evaluating RandomForest on test set...")
    y_pred_rf = best_rf_pipeline.predict(X_test)
    rf_metrics = evaluate_model(y_test, y_pred_rf, "RandomForest", machine_type, machine_results_dir)
    all_metrics['RandomForest'] = rf_metrics # Store metrics
    best_estimators['RandomForest'] = best_rf_pipeline # Store the pipeline

    # Save RF Model (using the helper function)
    save_pipeline(best_rf_pipeline, "RandomForest", machine_type, machine_results_dir) # Save to results dir

    end_time_rf = datetime.now()
    print(f"Random Forest training and evaluation time: {end_time_rf - start_time_rf}")

    # =======================================
    # --- 6. Model Training: XGBoost ---
    # =======================================
    print("\n===== Training XGBoost (No Hyperparameter Tuning) ====") # Modified print
    start_time_xgb = datetime.now()

    # Define base model and pipeline with default/reasonable parameters
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                 n_estimators=100, # Sensible default
                                 learning_rate=0.1, # Common default
                                 max_depth=5,       # Common default
                                 random_state=RANDOM_SEED, 
                                 n_jobs=-1,
                                 early_stopping_rounds=10) # Add early stopping
    xgb_pipeline = Pipeline([
        ('scaler', scaler_skl),
        ('xgb', xgb_model)
    ])

    # --- Skip Randomized Search CV ---
    # xgb_param_dist = { ... }
    # xgb_random_search = RandomizedSearchCV(...)
    # xgb_random_search.fit(X_train, y_train)
    # xgb_best_pipeline = xgb_random_search.best_estimator_
    # --- End Skip ---

    # Train the pipeline using early stopping
    print("Training XGBoost pipeline with early stopping...")

    # 1. Split training data for early stopping validation
    # Use last 10% of training data as validation set for early stopping
    temp_train_size = int(len(X_train) * 0.9)
    X_train_temp, X_val_temp = X_train.iloc[:temp_train_size], X_train.iloc[temp_train_size:]
    y_train_temp, y_val_temp = y_train.iloc[:temp_train_size], y_train.iloc[temp_train_size:]

    # 2. Fit the scaler on the *entire* training set first
    print("Fitting scaler on full training data...")
    xgb_pipeline.named_steps['scaler'].fit(X_train)

    # 3. Scale the temporary training and validation sets using the fitted scaler
    print("Scaling temporary train/validation sets for XGBoost early stopping...")
    X_train_temp_scaled = xgb_pipeline.named_steps['scaler'].transform(X_train_temp)
    X_val_temp_scaled = xgb_pipeline.named_steps['scaler'].transform(X_val_temp)
    eval_set = [(X_val_temp_scaled, y_val_temp)]

    # 4. Fit *only* the XGBoost model step with early stopping, using the scaled data
    print("Fitting XGBoost model with early stopping...")
    xgb_pipeline.named_steps['xgb'].fit(X_train_temp_scaled, y_train_temp,
                                        eval_set=eval_set,
                                        verbose=False) # Suppress verbose output during fit

    # 5. The pipeline now contains the fitted scaler and the early-stopped XGB model.
    #    No further fitting of the pipeline is needed.
    xgb_best_pipeline = xgb_pipeline # Use this pipeline

    # Evaluate on the test set (Pipeline handles scaling automatically)
    print("\nEvaluating XGBoost on test set...") # Modified print
    xgb_predictions = xgb_best_pipeline.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_predictions, 'XGBoost', machine_type, machine_results_dir)
    all_metrics['XGBoost'] = xgb_metrics
    best_estimators['XGBoost'] = xgb_best_pipeline

    # Save the pipeline
    save_pipeline(xgb_best_pipeline, 'XGBoost', machine_type, machine_results_dir)

    print(f"XGBoost training and evaluation time: {datetime.now() - start_time_xgb}")

    # =====================================
    # --- 7. Model Training: LSTM ---
    # =====================================
    print("\n===== Training LSTM ====")
    start_time_lstm = datetime.now()

    # --- Prepare data for LSTM ---
    # Combine features and target for sequence creation
    train_data_lstm = np.hstack((X_train_scaled_lstm, y_train.values.reshape(-1, 1)))
    test_data_lstm = np.hstack((X_test_scaled_lstm, y_test.values.reshape(-1, 1)))

    print(f"Creating LSTM sequences with window size: {LSTM_WINDOW_SIZE}...")
    X_train_lstm, y_train_lstm = create_lstm_sequences(train_data_lstm, LSTM_WINDOW_SIZE)
    X_test_lstm, y_test_lstm = create_lstm_sequences(test_data_lstm, LSTM_WINDOW_SIZE)
    
    # Adjust test set target to match LSTM output length
    y_test_lstm_eval = y_test.iloc[LSTM_WINDOW_SIZE:]

    print(f"LSTM Train shapes: X={X_train_lstm.shape}, y={y_train_lstm.shape}")
    print(f"LSTM Test shapes: X={X_test_lstm.shape}, y={y_test_lstm.shape}")
    print(f"LSTM Test target for eval shape: {y_test_lstm_eval.shape}")
    
    if X_train_lstm.shape[0] == 0 or X_test_lstm.shape[0] == 0:
        print("Error: Not enough data to create LSTM sequences after feature engineering and splitting.")
        print("Skipping LSTM training.")
    else:
        # --- Build and Train LSTM ---
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        lstm_model = build_lstm_model(input_shape)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=LSTM_PATIENCE, restore_best_weights=True)
        
        print("Training LSTM model...")
        history = lstm_model.fit(
            X_train_lstm,
            y_train_lstm,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.1, # Use last 10% of training sequences for validation
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'LSTM Model Loss ({machine_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        history_plot_file = machine_results_dir / "LSTM_training_history.png"
        plt.savefig(history_plot_file)
        plt.close()
        print(f"LSTM training history plot saved to: {history_plot_file}")

        # Evaluate LSTM
        print("\nEvaluating LSTM on test set...")
        lstm_predictions = lstm_model.predict(X_test_lstm).flatten()
        
        # Ensure prediction length matches evaluation target length
        if len(lstm_predictions) != len(y_test_lstm_eval):
             print(f"Warning: LSTM prediction length ({len(lstm_predictions)}) does not match evaluation target length ({len(y_test_lstm_eval)}). Check sequence generation.")
             # Attempt to align if off by a small amount (e.g., off-by-one)
             min_len = min(len(lstm_predictions), len(y_test_lstm_eval))
             lstm_predictions = lstm_predictions[:min_len]
             y_test_lstm_eval = y_test_lstm_eval[:min_len]
             
        lstm_metrics = evaluate_model(y_test_lstm_eval, lstm_predictions, 'LSTM', machine_type, machine_results_dir)
        all_metrics['LSTM'] = lstm_metrics
        best_estimators['LSTM'] = lstm_model # Store the model itself

        # Save the scaler and model
        save_lstm_components(scaler_lstm, lstm_model, 'LSTM', machine_type, machine_results_dir)

        print(f"LSTM training and evaluation time: {datetime.now() - start_time_lstm}")

    # =========================================
    # --- 8. Final Summary & Maintenance Prediction ---
    # =========================================
    print("\n===== Training Summary ====")
    for model_name, metrics in all_metrics.items():
        print(f"--- {model_name} ---")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  R2 Score: {metrics['R2 Score']:.4f}")

    # Example: Maintenance suggestion based on the best model (lowest RMSE)
    best_model_name = min(all_metrics, key=lambda k: all_metrics[k]['RMSE'])
    print(f"\nBest performing model based on RMSE: {best_model_name}")

    # Get predictions from the best model for the test set
    if best_model_name == 'LSTM':
         if 'LSTM' in best_estimators:
              best_preds = best_estimators['LSTM'].predict(X_test_lstm).flatten()
              # Align length if needed
              min_len = min(len(best_preds), len(y_test_lstm_eval))
              best_preds = best_preds[:min_len]
         else: 
             print("LSTM model not available for maintenance prediction.")
             best_preds = None
    elif best_model_name in best_estimators:
        best_pipeline = best_estimators[best_model_name]
        best_preds = best_pipeline.predict(X_test)
    else:
        print(f"Could not find best model '{best_model_name}' predictions.")
        best_preds = None

    if best_preds is not None:
         maintenance_suggestion = suggest_maintenance_from_predictions(best_preds)
         print(f"\nMaintenance Suggestion (based on {best_model_name} predictions on test set):")
         print(maintenance_suggestion)
         
         # Plot predictions of the best model over time
         plt.figure(figsize=(15, 6))
         if best_model_name == 'LSTM':
              plot_timestamps = timestamps_test[LSTM_WINDOW_SIZE:LSTM_WINDOW_SIZE+len(best_preds)] # Align timestamps
              plot_actual = y_test_lstm_eval[:len(best_preds)] # Align actuals
         else:
              plot_timestamps = timestamps_test[:len(best_preds)] # Align timestamps
              plot_actual = y_test[:len(best_preds)] # Align actuals
              
         plt.plot(plot_timestamps, plot_actual, label='Actual Health Index', alpha=0.7)
         plt.plot(plot_timestamps, best_preds, label=f'{best_model_name} Predicted Health Index', alpha=0.7)
         plt.axhline(DEFAULT_HEALTH_THRESHOLD, color='r', linestyle='--', label=f'Maintenance Threshold ({DEFAULT_HEALTH_THRESHOLD})')
         plt.title(f'{best_model_name} Predictions vs Actual ({machine_type} - Test Set)')
         plt.xlabel('Timestamp')
         plt.ylabel('Health Index')
         plt.legend()
         plt.grid(True)
         plt.ylim(0, 1.1)
         best_model_plot_file = machine_results_dir / f"{best_model_name}_test_set_predictions.png"
         plt.savefig(best_model_plot_file)
         plt.close()
         print(f"Best model predictions plot saved to: {best_model_plot_file}")
         

    print(f"\nTotal script execution time: {datetime.now() - start_time_script}")
    print(f"===== Training Complete for: {machine_type} =====")

"""Script to train predictive maintenance models for different machine types.""" 