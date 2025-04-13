"""
Improved Test script for Predictive Maintenance Models

This script loads the trained models and runs inference on realistic dummy data
that better matches the characteristics of the training data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

def generate_dummy_data(machine_type, condition="normal", num_samples=5):
    """
    Generate dummy data for a specific machine type with realistic values.
    
    Args:
        machine_type: Type of machine to generate data for
        condition: "normal" or "abnormal" to simulate normal or potential failure conditions
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with dummy data
    """
    # Create a base timestamp
    base_timestamp = datetime.now()
    timestamps = [base_timestamp + timedelta(hours=i) for i in range(num_samples)]
    
    # Create machine IDs
    machine_ids = [f"{machine_type}_{i+1}" for i in range(num_samples)]
    
    # Base dataframe with timestamps and machine IDs
    data = {
        'timestamp': timestamps,
        'machine_id': machine_ids,
        # Add maintenance and anomaly columns with 0 values
        'maintenance': [0] * num_samples,
        'anomaly': [0] * num_samples,
        'failure': [0] * num_samples  # We don't know the actual failure status
    }
    
    # Add machine-specific sensor data with realistic distributions based on analysis
    if machine_type == "siemens_motor":
        mean_temp = 70 if condition == "normal" else 95
        mean_vibration = 3 if condition == "normal" else 8
        
        data.update({
            'temperature': np.random.normal(mean_temp, 5, num_samples),
            'vibration': np.random.normal(mean_vibration, 1, num_samples),
            'current': np.random.normal(60, 5, num_samples),
            'voltage': np.random.normal(380, 2, num_samples)
        })
        
    elif machine_type == "abb_bearing":
        mean_vibration = 1 if condition == "normal" else 3
        mean_acoustic = 60 if condition == "normal" else 85
        
        data.update({
            'vibration': np.random.normal(mean_vibration, 0.2, num_samples),
            'temperature': np.random.normal(53, 3, num_samples),
            'acoustic': np.random.normal(mean_acoustic, 5, num_samples)
        })
        
    elif machine_type == "haas_cnc":
        mean_spindle = 40 if condition == "normal" else 70
        mean_vibration = 1.5 if condition == "normal" else 4
        
        data.update({
            'spindle_load': np.random.normal(mean_spindle, 5, num_samples),
            'vibration': np.random.normal(mean_vibration, 0.5, num_samples),
            'temperature': np.random.normal(55, 2, num_samples),
            'acoustic': np.random.normal(65, 3, num_samples)
        })
        
    elif machine_type == "grundfos_pump":
        # For abnormal, decrease flow rate and increase power
        mean_flow = 80 if condition == "normal" else 40
        mean_power = 10 if condition == "normal" else 18
        
        data.update({
            'pressure': np.random.normal(15, 1, num_samples),
            'flow_rate': np.random.normal(mean_flow, 5, num_samples),
            'temperature': np.random.normal(55, 1, num_samples),
            'power': np.random.normal(mean_power, 2, num_samples)
        })
        
    elif machine_type == "carrier_chiller":
        # For abnormal, increase condenser temp and refrigerant pressure
        mean_cond_temp = 40 if condition == "normal" else 52
        mean_ref_press = 15 if condition == "normal" else 25
        
        data.update({
            'refrigerant_pressure': np.random.normal(mean_ref_press, 1, num_samples),
            'condenser_temp': np.random.normal(mean_cond_temp, 2, num_samples),
            'evaporator_temp': np.random.normal(8, 0.5, num_samples),
            'power': np.random.normal(0.6, 0.05, num_samples)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure all values are within realistic ranges
    # Apply min/max constraints based on machine type
    apply_min_max_constraints(df, machine_type)
    
    return df

def apply_min_max_constraints(df, machine_type):
    """Apply minimum and maximum constraints to sensor values."""
    constraints = {
        "siemens_motor": {
            "temperature": (60, 120),
            "vibration": (1, 15),
            "current": (10, 100),
            "voltage": (370, 400)
        },
        "abb_bearing": {
            "vibration": (0.2, 5),
            "temperature": (40, 80),
            "acoustic": (40, 90)
        },
        "haas_cnc": {
            "spindle_load": (20, 95),
            "vibration": (0.5, 7),
            "temperature": (45, 85),
            "acoustic": (50, 90)
        },
        "grundfos_pump": {
            "pressure": (5, 25),
            "flow_rate": (20, 130),
            "temperature": (45, 70),
            "power": (3, 22)
        },
        "carrier_chiller": {
            "refrigerant_pressure": (8, 30),
            "condenser_temp": (35, 60),
            "evaporator_temp": (5, 15),
            "power": (0.5, 1.5)
        }
    }
    
    if machine_type in constraints:
        for col, (min_val, max_val) in constraints[machine_type].items():
            if col in df.columns:
                df[col] = df[col].clip(min_val, max_val)

def engineer_siemens_motor_features(df, normal_ranges):
    """Create features for Siemens motors."""
    features = df.copy()
    required_cols = ['temperature', 'vibration', 'current', 'voltage']
    
    # Add normalized features based on normal operating ranges
    for col in required_cols:
        if col in features.columns:
            min_val, max_val = normal_ranges["siemens_motor"][col]
            features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
            features[f'{col}_deviation'] = features[col].apply(
                lambda x: max(0, min_val - x) / min_val if x < min_val 
                else max(0, x - max_val) / max_val if x > max_val else 0
            )
    
    # Calculate apparent power
    if 'current' in features.columns and 'voltage' in features.columns:
        features['apparent_power'] = features['voltage'] * features['current']
    
    # Create time-based features
    if 'timestamp' in features.columns:
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
    
    # Create rolling window features
    for col in required_cols:
        if col in features.columns:
            features[f'{col}_rolling_mean_12h'] = features[col]  # No history for dummy data
            features[f'{col}_rolling_std_12h'] = 0  # No history for dummy data
    
    # Temperature-to-vibration ratio
    if 'temperature' in features.columns and 'vibration' in features.columns:
        features['temp_vibration_ratio'] = features['temperature'] / features['vibration'].replace(0, 0.001)
    
    # Remove non-feature columns
    drop_columns = ['timestamp', 'machine_id', 'maintenance', 'anomaly']
    X = features.drop(columns=drop_columns + ['failure'], errors='ignore')
    
    return X

def engineer_abb_bearing_features(df, normal_ranges):
    """Create features for ABB bearings."""
    features = df.copy()
    required_cols = ['vibration', 'temperature', 'acoustic']
    
    # Add normalized features based on normal operating ranges
    for col in required_cols:
        if col in features.columns:
            min_val, max_val = normal_ranges["abb_bearing"][col]
            features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
            features[f'{col}_deviation'] = features[col].apply(
                lambda x: max(0, min_val - x) / min_val if x < min_val 
                else max(0, x - max_val) / max_val if x > max_val else 0
            )
    
    # Create time-based features
    if 'timestamp' in features.columns:
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
    
    # Create rolling window features
    for col in required_cols:
        if col in features.columns:
            features[f'{col}_rolling_mean_12h'] = features[col]  # No history for dummy data
            features[f'{col}_rolling_std_12h'] = 0  # No history for dummy data
    
    # Create acoustic-to-vibration ratio
    if 'acoustic' in features.columns and 'vibration' in features.columns:
        features['acoustic_vibration_ratio'] = features['acoustic'] / features['vibration'].replace(0, 0.001)
    
    # Remove non-feature columns
    drop_columns = ['timestamp', 'machine_id', 'maintenance', 'anomaly']
    X = features.drop(columns=drop_columns + ['failure'], errors='ignore')
    
    return X

def engineer_haas_cnc_features(df, normal_ranges):
    """Create features for HAAS CNC machines."""
    features = df.copy()
    required_cols = ['spindle_load', 'vibration', 'temperature', 'acoustic']
    
    # Add normalized features based on normal operating ranges
    for col in required_cols:
        if col in features.columns:
            min_val, max_val = normal_ranges["haas_cnc"][col]
            features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
            features[f'{col}_deviation'] = features[col].apply(
                lambda x: max(0, min_val - x) / min_val if x < min_val 
                else max(0, x - max_val) / max_val if x > max_val else 0
            )
    
    # Create time-based features
    if 'timestamp' in features.columns:
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
    
    # Create rolling window features
    for col in required_cols:
        if col in features.columns:
            features[f'{col}_rolling_mean_12h'] = features[col]  # No history for dummy data
            features[f'{col}_rolling_std_12h'] = 0  # No history for dummy data
    
    # Create spindle efficiency indicator
    if all(col in features.columns for col in ['spindle_load', 'temperature', 'vibration']):
        features['spindle_efficiency'] = features['spindle_load'] / (features['temperature'] * features['vibration'])
        
    # Acoustic-to-vibration ratio
    if 'acoustic' in features.columns and 'vibration' in features.columns:
        features['acoustic_vibration_ratio'] = features['acoustic'] / features['vibration'].replace(0, 0.001)
    
    # Remove non-feature columns
    drop_columns = ['timestamp', 'machine_id', 'maintenance', 'anomaly']
    X = features.drop(columns=drop_columns + ['failure'], errors='ignore')
    
    return X

def engineer_grundfos_pump_features(df, normal_ranges):
    """Create features for Grundfos pumps."""
    features = df.copy()
    required_cols = ['pressure', 'flow_rate', 'temperature', 'power']
    
    # Add normalized features based on normal operating ranges
    for col in required_cols:
        if col in features.columns:
            min_val, max_val = normal_ranges["grundfos_pump"][col]
            features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
            features[f'{col}_deviation'] = features[col].apply(
                lambda x: max(0, min_val - x) / min_val if x < min_val 
                else max(0, x - max_val) / max_val if x > max_val else 0
            )
    
    # Create time-based features
    if 'timestamp' in features.columns:
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
    
    # Create rolling window features
    for col in required_cols:
        if col in features.columns:
            features[f'{col}_rolling_mean_12h'] = features[col]  # No history for dummy data
            features[f'{col}_rolling_std_12h'] = 0  # No history for dummy data
    
    # Calculate pump efficiency related metrics
    if 'pressure' in features.columns and 'flow_rate' in features.columns and 'power' in features.columns:
        features['hydraulic_power'] = features['pressure'] * features['flow_rate']
        features['pump_efficiency'] = features['hydraulic_power'] / features['power'].replace(0, 0.001)
        features['efficiency_change'] = 0  # No history for dummy data
    
    # Remove non-feature columns
    drop_columns = ['timestamp', 'machine_id', 'maintenance', 'anomaly']
    X = features.drop(columns=drop_columns + ['failure'], errors='ignore')
    
    return X

def engineer_carrier_chiller_features(df, normal_ranges):
    """Create features for Carrier chillers."""
    features = df.copy()
    required_cols = ['refrigerant_pressure', 'condenser_temp', 'evaporator_temp', 'power']
    
    # Add normalized features based on normal operating ranges
    for col in required_cols:
        if col in features.columns:
            min_val, max_val = normal_ranges["carrier_chiller"][col]
            features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
            features[f'{col}_deviation'] = features[col].apply(
                lambda x: max(0, min_val - x) / min_val if x < min_val 
                else max(0, x - max_val) / max_val if x > max_val else 0
            )
    
    # Create time-based features
    if 'timestamp' in features.columns:
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
    
    # Create rolling window features
    for col in required_cols:
        if col in features.columns:
            features[f'{col}_rolling_mean_12h'] = features[col]  # No history for dummy data
            features[f'{col}_rolling_std_12h'] = 0  # No history for dummy data
    
    # Calculate temperature differential
    if 'condenser_temp' in features.columns and 'evaporator_temp' in features.columns:
        features['temp_differential'] = features['condenser_temp'] - features['evaporator_temp']
        
    # Calculate COP
    if 'temp_differential' in features.columns and 'power' in features.columns:
        features['approx_cop'] = features['evaporator_temp'] / features['temp_differential']
        
    # Pressure-temperature relationship
    if 'refrigerant_pressure' in features.columns and 'condenser_temp' in features.columns:
        features['pressure_temp_ratio'] = features['refrigerant_pressure'] / features['condenser_temp']
    
    # Remove non-feature columns
    drop_columns = ['timestamp', 'machine_id', 'maintenance', 'anomaly']
    X = features.drop(columns=drop_columns + ['failure'], errors='ignore')
    
    return X

def prepare_features(machine_data, machine_type, normal_ranges):
    """Apply machine-specific feature engineering."""
    df = machine_data.copy()
    
    # Apply machine-specific feature engineering based on machine type
    if "siemens_motor" in machine_type:
        X = engineer_siemens_motor_features(df, normal_ranges)
    elif "abb_bearing" in machine_type:
        X = engineer_abb_bearing_features(df, normal_ranges)
    elif "haas_cnc" in machine_type:
        X = engineer_haas_cnc_features(df, normal_ranges)
    elif "grundfos_pump" in machine_type:
        X = engineer_grundfos_pump_features(df, normal_ranges)
    elif "carrier_chiller" in machine_type:
        X = engineer_carrier_chiller_features(df, normal_ranges)
    else:
        print(f"Warning: Unknown machine type '{machine_type}'.")
        X = df.copy()
    
    # Handle NaNs that might have been introduced during feature engineering
    X = X.fillna(0)
    
    return X

def load_model(machine_type):
    """Load the saved model for the machine type."""
    model_path = f'models/{machine_type}_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
    
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and 'model' in model_data:
            print(f"Loaded model for {machine_type} with {len(model_data['feature_names'])} features")
            return model_data
        else:
            print(f"Loaded model for {machine_type} (legacy format)")
            return {'model': model_data, 'feature_names': None}
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def test_model(machine_type, model_data, dummy_data):
    """Test the model with dummy data."""
    if model_data is None:
        print(f"No model available for {machine_type}")
        return
    
    # Define normal ranges for feature engineering
    normal_ranges = {
        "siemens_motor": {
            "temperature": (60, 80),
            "vibration": (1, 5),
            "current": (10, 100),
            "voltage": (370, 400)
        },
        "abb_bearing": {
            "vibration": (0.2, 2),
            "temperature": (40, 60),
            "acoustic": (40, 70)
        },
        "haas_cnc": {
            "spindle_load": (20, 60),
            "vibration": (0.5, 3),
            "temperature": (45, 65),
            "acoustic": (50, 80)
        },
        "grundfos_pump": {
            "pressure": (2, 25),
            "flow_rate": (20, 130),
            "temperature": (40, 70),
            "power": (0.37, 22)
        },
        "carrier_chiller": {
            "refrigerant_pressure": (8, 25),
            "condenser_temp": (35, 45),
            "evaporator_temp": (5, 10),
            "power": (0.5, 0.7)
        }
    }
    
    # Engineer features for the test data
    X = prepare_features(dummy_data, machine_type, normal_ranges)
    
    # Ensure the test data has all the features the model was trained on
    if model_data['feature_names'] is not None:
        required_features = set(model_data['feature_names'])
        available_features = set(X.columns)
        
        # Find missing features
        missing_features = required_features - available_features
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features. Adding zeros for missing features.")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0
                
        # Only keep the features the model was trained on
        X = X[model_data['feature_names']]
    
    # Make predictions
    try:
        model = model_data['model']
        prediction_proba = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'machine_id': dummy_data['machine_id'],
            'failure_probability': prediction_proba,
            'failure_predicted': predictions,
            # Include some key sensor values
            **{col: dummy_data[col] for col in dummy_data.columns 
               if col not in ['machine_id', 'timestamp', 'failure', 'maintenance', 'anomaly']}
        })
        
        return results
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def main():
    # List of machine types to test
    machine_types = [
        "siemens_motor",
        "abb_bearing",
        "haas_cnc",
        "grundfos_pump",
        "carrier_chiller"
    ]
    
    print("Testing Predictive Maintenance Models with Realistic Dummy Data\n")
    
    for machine_type in machine_types:
        print(f"\n{'='*50}")
        print(f"Testing {machine_type} model")
        print(f"{'='*50}")
        
        # Load the model
        model_data = load_model(machine_type)
        if model_data is None:
            continue
        
        # Generate dummy data - normal operating conditions
        print("\nGenerating normal operating data...")
        normal_data = generate_dummy_data(machine_type, condition="normal", num_samples=3)
        normal_results = test_model(machine_type, model_data, normal_data)
        
        if normal_results is not None:
            print("\nPredictions for normal operating conditions:")
            print(normal_results)
        
        # Generate dummy data - abnormal conditions (potential failures)
        print("\nGenerating abnormal operating data...")
        abnormal_data = generate_dummy_data(machine_type, condition="abnormal", num_samples=3)
        abnormal_results = test_model(machine_type, model_data, abnormal_data)
        
        if abnormal_results is not None:
            print("\nPredictions for abnormal operating conditions:")
            print(abnormal_results)
            
        # Print comparison
        if normal_results is not None and abnormal_results is not None:
            print("\nSummary:")
            print(f"Average failure probability (normal): {normal_results['failure_probability'].mean():.4f}")
            print(f"Average failure probability (abnormal): {abnormal_results['failure_probability'].mean():.4f}")
            print(f"Difference: {abnormal_results['failure_probability'].mean() - normal_results['failure_probability'].mean():.4f}")

if __name__ == "__main__":
    main() 