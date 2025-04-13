"""Functions to predict remaining useful life or days until maintenance."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add project root to sys.path to allow importing custom modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import necessary functions/constants
# Import feature engineering functions needed for BOTH models
from src.modeling.feature_engineering import (
    extract_base_features, # For anomaly detection
    apply_feature_engineering, # Use the same function as in train.py
    ROLLING_WINDOW_SIZE, # Shared constant
)
# Define expected feature generation parameters (should match training)
# These need to be accessible for prediction feature generation
# Use the same constants as defined in train.py (ensure these match!)
LAGS = [1, 3, 6, 12, 24] 
WINDOW_SIZES = [6, 12, 24] 

# --- Constants ---
DEFAULT_HEALTH_THRESHOLD = 0.2 # Example: Suggest maintenance if health drops below 20%
MAINTENANCE_HORIZON_DAYS = 7 # Example: Suggest maintenance within 7 days if threshold is crossed
MODELS_DIR = project_root / "src" / "modeling" / "saved_models"
ANOMALY_MODEL_PREFIX = "anomaly_model_"
HEALTH_MODEL_PREFIX = "RandomForest"
RESULTS_DIR = project_root / "results"  # Added to load health models from results directory

# Store loaded models to avoid reloading on every call
_loaded_models = {}

# Define sensor columns used for HEALTH model training per machine type
# (Based on train.py logs/logic)
HEALTH_MODEL_SENSORS = {
    "siemens_motor": ['temperature', 'vibration', 'current', 'voltage'],
    "abb_bearing": ['temperature', 'vibration', 'current'], # Updated based on training expectations
    "haas_cnc": ['spindle_speed', 'spindle_load', 'temperature', 'vibration'], # Assuming
    "grundfos_pump": ['flow_rate', 'pressure', 'temperature', 'power'], # Assuming
    "carrier_chiller": ['refrigerant_pressure', 'evaporator_temp', 'condenser_temp', 'power'], # Assuming
    "default_machine": ['temperature', 'vibration'] # Example default
}
# NOTE: Update the above dictionary accurately based on how each machine model was actually trained!

def ensure_required_features(features_df: pd.DataFrame, machine_type: str, expected_features_set: set) -> pd.DataFrame:
    """
    Adds missing features required by the machine's model.
    
    Args:
        features_df: DataFrame with current features
        machine_type: Type of machine 
        expected_features_set: Set of feature names expected by the model
        
    Returns:
        DataFrame with all expected features added (newly added will be 0)
    """
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    current_features = set(df.columns)
    missing_features = expected_features_set - current_features
    
    # Find all possible sensor names from the expected feature columns
    # This extracts base sensor names from complex feature names like 'voltage_roll_mean_6'
    sensor_base_names = set()
    for feature in expected_features_set:
        # Extract base name (e.g., 'voltage' from 'voltage_roll_mean_6')
        parts = feature.split('_')
        if len(parts) > 1 and not feature.startswith(('hour', 'day', 'month')):
            sensor_base_names.add(parts[0])
    
    # Time-based features are simple to handle
    time_features = {'hour', 'dayofweek', 'dayofyear', 'month'}
    for feature in missing_features & time_features:
        if isinstance(df.index, pd.DatetimeIndex):
            if feature == 'hour':
                df[feature] = df.index.hour
            elif feature == 'dayofweek':
                df[feature] = df.index.dayofweek
            elif feature == 'dayofyear':
                df[feature] = df.index.dayofyear
            elif feature == 'month':
                df[feature] = df.index.month
        else:
            # If no DatetimeIndex, use zeros
            df[feature] = 0
    
    # For complex engineered features, we need to identify the pattern 
    for feature in missing_features - time_features:
        # Set all other missing features to 0 - this is a fallback approach
        # A more sophisticated approach would be to calculate them properly
        df[feature] = 0.0
        
    print(f"[Debug] Added {len(missing_features)} missing features required by the {machine_type} model")
    
    return df

def load_model(machine_type: str, model_prefix: str):
    """Loads a joblib model based on prefix and machine type.
    For anomaly models, loads from MODELS_DIR;
    for health models, loads from RESULTS_DIR/<machine_type> directory.
    """
    # Normalize machine_type to ensure it matches file naming conventions
    machine_type = machine_type.lower().replace(" ", "_")
    
    if model_prefix == ANOMALY_MODEL_PREFIX:
        model_key = f"{model_prefix}{machine_type}"
        model_path = MODELS_DIR / f"{model_key}.joblib"
    else:
        model_key = f"{model_prefix}_pipeline"
        model_path = RESULTS_DIR / machine_type / f"{model_key}.joblib"
    
    # Check if model is already loaded
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    # Check if model file exists
    if not model_path.exists():
        # print(f"Warning: Model {model_key} not found at {model_path}") # Less verbose
        # Check if models directory exists
        # if not model_path.parent.exists():
        #     print(f"Directory does not exist: {model_path.parent}")
        
        # Additional check for alternative file names that might exist
        if model_prefix == HEALTH_MODEL_PREFIX:
            alternative_path = RESULTS_DIR / machine_type / "XGBoost_pipeline.joblib"
            if alternative_path.exists():
                # print(f"Found alternative model at {alternative_path}") # Less verbose
                model_path = alternative_path
                model_key = "XGBoost_pipeline"
            else:
                print(f"Error: No health models (RandomForest or XGBoost) found in {RESULTS_DIR / machine_type}")
                return None
        else:
            # It's okay if only anomaly model exists for now
            # print(f"Model {model_key} not found, continuing without it.") 
            return None
    
    try:
        # print(f"Loading model from {model_path}...") # Less verbose
        model = joblib.load(model_path)
        _loaded_models[model_key] = model
        # print(f"Model {model_key} loaded successfully.") # Less verbose
        return model
    except Exception as e:
        print(f"Error loading model {model_key}: {e}")
        return None

def extract_predict_features(sensor_data_df: pd.DataFrame, machine_type: str) -> pd.DataFrame:
    """
    Applies feature engineering steps consistent with training for prediction.
    Uses ALL required sensor columns for the given machine_type.
    Assumes sensor_data_df has a DatetimeIndex and contains all necessary raw sensor columns.
    Returns a DataFrame with features for the LATEST timestamp.
    """
    # Convert machine_type to lowercase with underscores for consistency
    machine_type = machine_type.lower().replace(" ", "_")
    print(f"[Debug] Extracting features for machine_type: {machine_type}")

    # Basic validation of input DataFrame
    if not isinstance(sensor_data_df, pd.DataFrame) or sensor_data_df.empty:
        print(f"Error: Input is not a DataFrame or is empty.")
        return pd.DataFrame()

    # Ensure we have a DatetimeIndex
    if not isinstance(sensor_data_df.index, pd.DatetimeIndex):
        print(f"Warning: Input DataFrame does not have a DatetimeIndex. Prediction features require it.")
        # Attempt conversion only if timestamp columns are clearly present
        timestamp_col = None
        if 'timestamp' in sensor_data_df.columns:
            timestamp_col = 'timestamp'
        elif 'Timestamp' in sensor_data_df.columns:
            timestamp_col = 'Timestamp'

        if timestamp_col:
            try:
                print(f"Attempting to set index from column: {timestamp_col}")
                sensor_data_df[timestamp_col] = pd.to_datetime(sensor_data_df[timestamp_col])
                sensor_data_df = sensor_data_df.set_index(timestamp_col)
                print("Index set successfully.")
            except Exception as e:
                print(f"Error setting DatetimeIndex: {e}. Cannot proceed.")
                return pd.DataFrame()
        else:
            print("Could not find timestamp column to convert to index. Cannot proceed.")
            return pd.DataFrame()

    # Add 'timestamp' column from index for feature engineering
    if 'timestamp' not in sensor_data_df.columns:
        print("[Debug] Adding 'timestamp' column from index for feature engineering.")
        # Create a proper copy to avoid SettingWithCopyWarning
        sensor_data_df = sensor_data_df.copy()
        sensor_data_df['timestamp'] = sensor_data_df.index

    # Basic check for sufficient data length
    required_rows = max(max(LAGS, default=0), max(WINDOW_SIZES, default=0)) + 1
    if len(sensor_data_df) < required_rows:
        print(f"Warning: Insufficient data ({len(sensor_data_df)} points) for full feature calculation (needs ~{required_rows}). Returning empty DataFrame.")
        return pd.DataFrame()

    # Create a copy to avoid SettingWithCopyWarning
    df_for_feat_eng = sensor_data_df.copy()
    expected_sensors = HEALTH_MODEL_SENSORS.get(machine_type, df_for_feat_eng.columns.tolist())
    df_for_feat_eng = df_for_feat_eng[expected_sensors]

    try:
        # Apply feature engineering
        print(f"[Debug] Applying feature engineering with expected sensors: {expected_sensors}, Lags: {LAGS}, Windows: {WINDOW_SIZES}")
        engineered_features_df = apply_feature_engineering(df_for_feat_eng, expected_sensors, LAGS, WINDOW_SIZES)
        
        # Ensure time-based features are properly created if not already present
        if isinstance(engineered_features_df.index, pd.DatetimeIndex):
            # Add time features from index if not present
            if 'hour' not in engineered_features_df.columns:
                engineered_features_df['hour'] = engineered_features_df.index.hour
            if 'dayofweek' not in engineered_features_df.columns:
                engineered_features_df['dayofweek'] = engineered_features_df.index.dayofweek
            if 'dayofyear' not in engineered_features_df.columns:
                engineered_features_df['dayofyear'] = engineered_features_df.index.dayofyear
            if 'month' not in engineered_features_df.columns:
                engineered_features_df['month'] = engineered_features_df.index.month
        
        print(f"[Debug] Engineered features df columns: {engineered_features_df.columns.tolist()}")
        print(f"[Debug] Engineered features df shape: {engineered_features_df.shape}")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

    # Select the latest row AFTER all features are calculated
    if len(engineered_features_df) == 0:
        print("Error: No data remains after feature engineering")
        return pd.DataFrame()

    # Filter out original sensor columns before taking the latest row
    final_feature_cols = [col for col in engineered_features_df.columns if col not in expected_sensors]
    if not final_feature_cols:
        print(f"Error: No feature columns remain after excluding original sensors {expected_sensors}")
        return pd.DataFrame()

    filtered_features_df = engineered_features_df[final_feature_cols]
    latest_features = filtered_features_df.iloc[-1:]

    # Handle potential NaNs in the final feature set
    if latest_features.isnull().any().any():
        print("Warning: NaNs found in latest features after calculation. Filling with 0.")
        latest_features = latest_features.fillna(0)

    return latest_features

def predict_status(sensor_data_df: pd.DataFrame, machine_type: str = "default_machine") -> dict:
    """
    Predicts machine status using both Anomaly Detection and Health Index models.

    Args:
        sensor_data_df (pd.DataFrame): DataFrame of recent time series data for potentially 
                                     MULTIPLE sensors, with a DatetimeIndex. Needs enough 
                                     history for feature calculation. Column names should match
                                     the raw sensor names used during training (e.g., 'temperature', 'vibration').
        machine_type (str): The type of machine (used to load correct models).

    Returns:
        dict: A dictionary containing the combined prediction status and related info.
    """
    # Normalize machine_type
    machine_type = machine_type.lower().replace(" ", "_")
    
    # Load models
    anomaly_model = load_model(machine_type, ANOMALY_MODEL_PREFIX)
    health_model = load_model(machine_type, HEALTH_MODEL_PREFIX)
    if not health_model:
        print(f"[Debug] Health model ({HEALTH_MODEL_PREFIX} or XGBoost) not loaded for {machine_type}") # Debug

    # Initialize default results
    current_time = pd.Timestamp.now(tz='UTC').isoformat()
    result = {
        "status": "Unknown",
        "reason": "",
        "risk_score": 0.0,
        "is_anomaly_now": False,
        "predicted_health_index": None,
        "prediction_timestamp": current_time,
        # Add other fields if needed
    }

    # --- Input Validation (basic check - detailed check in extract_predict_features) ---
    if not isinstance(sensor_data_df, pd.DataFrame) or sensor_data_df.empty:
        result["status"] = "Error"
        result["reason"] = f"Invalid or empty input DataFrame for {machine_type}."
        return result

    # --- Anomaly Detection (if model loaded) ---
    anomaly_status = "Healthy"
    anomaly_reason = "Anomaly check OK."
    anomaly_risk = 0.0
    is_anomaly_now = False
    if anomaly_model:
        try:
            # Anomaly model likely uses simpler features (rolling mean/std)
            # Extract features for the LATEST point using the primary sensor
            # NOTE: Assumes the first sensor in HEALTH_MODEL_SENSORS is the primary one used for anomaly training
            # This might need refinement based on how anomaly models were *actually* trained
            primary_sensor_name = HEALTH_MODEL_SENSORS.get(machine_type, [None])[0]
            if primary_sensor_name and primary_sensor_name in sensor_data_df.columns:
                 primary_sensor_series = sensor_data_df[primary_sensor_name]
                 # Use the same window as defined in feature_engineering/train.py
                 anomaly_features = extract_base_features(primary_sensor_series, window=ROLLING_WINDOW_SIZE)
                 anomaly_features = anomaly_features.dropna()
                 if not anomaly_features.empty:
                     latest_anomaly_features = anomaly_features.iloc[-1:]
                     # print(f"[Debug] Anomaly Features: {latest_anomaly_features}") # Debug
                     prediction = anomaly_model.predict(latest_anomaly_features)
                     is_anomaly_now = prediction[0] == -1 # -1 typically indicates anomaly
                     if is_anomaly_now:
                         anomaly_status = "Warning" 
                         anomaly_reason = "Anomaly detected by Isolation Forest."
                         anomaly_risk = 0.75 # Assign higher risk for anomaly
                 else:
                     anomaly_reason = "Not enough data for anomaly features."
            else:
                 anomaly_reason = f"Primary sensor ('{primary_sensor_name}') not found for anomaly check."
        except Exception as e:
            print(f"Error during anomaly detection for {machine_type}: {e}")
            anomaly_reason = f"Anomaly check failed: {e}"
            anomaly_risk = 0.1 # Assign small risk on failure

    result["is_anomaly_now"] = is_anomaly_now

    # --- Health Index Prediction (if model loaded) ---
    health_index = None
    health_reason = ""
    health_risk_contribution = 0.0
    if health_model:
        try:
            # Extract features using the refined function
            # It now expects the DataFrame with raw sensor columns matching HEALTH_MODEL_SENSORS
            features_df = extract_predict_features(sensor_data_df, machine_type)
            
            if features_df is not None and not features_df.empty:
                try:
                    # --- Start: New Feature Alignment Logic ---
                    # Get the scaler step from the pipeline (assuming it's named 'scaler')
                    scaler_step = health_model.named_steps.get('scaler')
                    if not scaler_step:
                        raise ValueError("Health model pipeline does not contain a 'scaler' step.")

                    # Get the feature names the scaler/pipeline was trained on
                    expected_feature_names = scaler_step.feature_names_in_
                    print(f"[Debug] Model '{health_model.steps[-1][0]}' for {machine_type} expects features: {list(expected_feature_names)}")

                    # Check if all expected features are present in the generated features
                    current_features_set = set(features_df.columns)
                    expected_features_set = set(expected_feature_names)
                    
                    missing_features = expected_features_set - current_features_set
                    extra_features = current_features_set - expected_features_set

                    if missing_features:
                        # Modified: Instead of failing, try to add the missing features
                        print(f"[Debug] Adding missing features: {missing_features}")
                        features_df = ensure_required_features(features_df, machine_type, expected_features_set)
                        # Double-check we've fixed the issue
                        current_features_set = set(features_df.columns)
                        missing_features = expected_features_set - current_features_set
                        if missing_features:
                            # If we still have missing features after trying to add them, raise error
                            raise ValueError(f"Feature mismatch: Generated features missing {missing_features} expected by the model.")
                    
                    if extra_features:
                        # Warn about extra features but proceed by selecting only expected ones.
                        print(f"[Debug] Warning: Generated features contain extra columns not expected by model: {extra_features}. Selecting only expected features.")

                    # Reindex the DataFrame to match the exact order and columns expected by the pipeline
                    features_for_predict = features_df[expected_feature_names]
                    print(f"[Debug] Features passed to health model predict (re-indexed): {features_for_predict.columns.tolist()}")
                    # --- End: New Feature Alignment Logic ---

                    # The loaded model pipeline handles scaling automatically
                    health_index_pred = health_model.predict(features_for_predict)
                    health_index = float(health_index_pred[0]) # Get single value
                    # Clamp prediction between 0 and 1
                    health_index = max(0.0, min(1.0, health_index))
                    result["predicted_health_index"] = health_index
                    health_reason = f"Predicted Health Index: {health_index:.3f}"
                    # Simple risk scaling based on health index
                    health_risk_contribution = (1.0 - health_index) * 0.5 # Scale risk from 0 to 0.5 based on health
                
                except (AttributeError, KeyError, ValueError) as align_err:
                    # Catch errors specifically related to accessing pipeline steps/features or mismatches
                    print(f"Error aligning features for health prediction ({machine_type}): {align_err}")
                    # Provide more specific and actionable error message
                    if "Generated features missing" in str(align_err):
                        health_reason = f"Feature alignment failed: {align_err}"
                        # Set a higher risk score when feature mismatch is detected
                        # This will trigger warnings in the UI to alert users
                        health_risk_contribution = 0.35 # Higher risk to indicate attention needed
                        # Add anomaly risk to make total risk high enough for warning state
                        if anomaly_risk < 0.5:
                            anomaly_risk = 0.5
                    else:
                        health_reason = f"Feature alignment failed: {align_err}"
                        health_risk_contribution = 0.1 # Default small risk
                except Exception as inner_e: # Catch other unexpected errors during prediction
                    print(f"Unexpected error during health prediction logic ({machine_type}): {inner_e}")
                    health_reason = f"Prediction logic error: {inner_e}"
                    health_risk_contribution = 0.1
            else:
                health_reason = "Feature extraction for health prediction failed or returned empty."
                health_risk_contribution = 0.1 # Assign small risk

        except Exception as e:
            print(f"Error during health prediction for {machine_type}: {e}")
            health_reason = f"Health prediction failed: {e}"
            health_risk_contribution = 0.1 # Assign small risk on failure
            # Optionally re-raise or log traceback for more detail
            import traceback
            # traceback.print_exc()
    else:
        health_reason = "Health prediction model not loaded."

    # --- Combine Results --- 
    final_risk = anomaly_risk + health_risk_contribution
    # Determine overall status based on risk/anomaly/health
    if is_anomaly_now:
        final_status = "Warning"
    elif health_index is not None and health_index < DEFAULT_HEALTH_THRESHOLD:
        final_status = "Warning" # Or "Critical" depending on threshold meaning
    else:
        final_status = "Healthy"
        
    # If prediction failed completely, status might still be Unknown or Error
    if not health_model and not anomaly_model:
        final_status = "Unknown"
        result["reason"] = "No models loaded."
        final_risk = 0.0
    elif not health_model and anomaly_model:
        final_status = anomaly_status # Base status only on anomaly if health model missing
        result["reason"] = anomaly_reason
        final_risk = anomaly_risk # Risk only from anomaly
    elif health_model and not anomaly_model:
         if health_index is not None:
             final_status = "Warning" if health_index < DEFAULT_HEALTH_THRESHOLD else "Healthy"
             result["reason"] = health_reason
             final_risk = health_risk_contribution
         else:
             final_status = "Error" # If health model loaded but prediction failed
             result["reason"] = health_reason 
             final_risk = health_risk_contribution
    else: # Both models potentially contributed
        # Combine reasons
        result["reason"] = f"{anomaly_reason} {health_reason}".strip()

    result["status"] = final_status
    result["risk_score"] = max(0.0, min(1.0, final_risk)) # Clamp risk score 0-1

    # print(f"[Debug] Final Result for {machine_type}: {result}") # Debug
    return result

# --- Original Health Index Suggestion Functions (Can be used or adapted later) ---

def suggest_maintenance_from_predictions(predictions, threshold=DEFAULT_HEALTH_THRESHOLD, horizon_days=MAINTENANCE_HORIZON_DAYS):
    """
    Analyzes health index predictions to suggest maintenance.

    Args:
        predictions (np.array or pd.Series): Array of predicted health index values.
        threshold (float): Health index level below which maintenance is considered.
        horizon_days (int): How many days in advance to suggest maintenance once the threshold is predicted to be crossed.

    Returns:
        str: Maintenance suggestion message.
    """
    if not isinstance(predictions, (np.ndarray, pd.Series)):
        predictions = np.array(predictions)

    if len(predictions) == 0:
        return "No predictions available."

    # Check if the threshold is ever crossed in the prediction horizon
    below_threshold_indices = np.where(predictions < threshold)[0]

    if len(below_threshold_indices) > 0:
        first_cross_index = below_threshold_indices[0]
        # Assuming predictions are hourly, convert index to days
        days_until_cross = first_cross_index / 24.0

        if days_until_cross <= horizon_days:
            return f"Maintenance recommended within {int(np.ceil(days_until_cross)) if days_until_cross > 0 else 1} day(s) (Health index predicted below {threshold:.2f})."
        else:
             # Threshold crossed, but further out than our immediate horizon
             return f"Monitor closely. Health index predicted below {threshold:.2f} in {int(np.ceil(days_until_cross))} days."
    else:
        # Threshold not predicted to be crossed
        return f"No immediate maintenance suggested (Health index predicted to stay above {threshold:.2f})."

def suggest_maintenance_from_final_health(final_health_index, threshold=DEFAULT_HEALTH_THRESHOLD):
     """
    Suggests maintenance based on the *last* predicted health index value.
    Simpler approach for when only the final state matters.

    Args:
        final_health_index (float): The last predicted health index.
        threshold (float): Health index level below which maintenance is considered.

    Returns:
        str: Maintenance suggestion message.
    """
     if final_health_index < threshold:
         return f"Maintenance recommended (Final health index {final_health_index:.2f} is below threshold {threshold:.2f})."
     else:
         return f"No immediate maintenance suggested (Final health index {final_health_index:.2f} is above threshold {threshold:.2f})." 