"""Script to test trained models against various edge cases and scenarios."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
import argparse
import joblib
from pathlib import Path
import tensorflow as tf
import warnings

# Suppress specific warnings if needed (e.g., from joblib or tensorflow)
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import necessary functions
try:
    from src.modeling.feature_engineering import apply_feature_engineering
    from redundant.data_generation.machine_configs import MACHINE_CONFIGS
    # Import the constants from the training script
    from src.modeling.train import LAGS, WINDOW_SIZES
    # Import base generation functions (assuming they are callable independently)
    # We might need to adapt these slightly if they rely heavily on class state
    from src.data_generation_health_index.generate_siemens_motor import (
        generate_normal_data,
        apply_cyclical_pattern,
        apply_degradation,
        # apply_maintenance_effect # Might not be needed directly for simple test cases
    )
    from src.modeling.predict_maintenance import suggest_maintenance_from_predictions, suggest_maintenance_from_final_health, DEFAULT_HEALTH_THRESHOLD
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure the script is run from the project root or that paths are correct.")
    sys.exit(1)

# --- Configuration ---
RESULTS_DIR = project_root / "results"
TEST_DURATION_HOURS = 7 * 24 # Generate 1 week (168 hours) of data for each test case
RANDOM_SEED = 123 # Use a different seed for testing
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# LSTM Params needed for sequence creation during testing
# TODO: Ideally, load these from a shared config or the training results
LSTM_WINDOW_SIZE = 24


print(f"Test script initialized. Project root: {project_root}")
print("NOTE: Test case generation uses functions adapted from data generation scripts.")

# --- Test Case Generation (Simplified & Adapted) ---

def generate_test_data(machine_type, scenario, duration_hours, failure_type=None):
    """Generates specific test data scenarios."""
    print(f"\nGenerating test case: {scenario} for {machine_type} ({duration_hours} hours)")
    if machine_type not in MACHINE_CONFIGS:
        print(f"Error: Machine type '{machine_type}' not found in MACHINE_CONFIGS.")
        return pd.DataFrame() # Return empty dataframe
        
    machine_config = MACHINE_CONFIGS[machine_type]
    sensor_config = machine_config["sensors"]
    failure_patterns = machine_config["failure_patterns"]
    sampling_interval_hours = 1 # Assuming hourly
    
    start_time = datetime.now() - timedelta(hours=duration_hours) 
    timestamps = [start_time + timedelta(hours=i * sampling_interval_hours)
                  for i in range(duration_hours)]
    
    # Generate baseline normal data using the imported function
    try:
        data = generate_normal_data(timestamps, sensor_config)
    except Exception as e:
        print(f"Error during generate_normal_data: {e}")
        return pd.DataFrame()
        
    health_index = np.ones(duration_hours)
    
    # Apply base cyclical patterns (machine-specific)
    # Use a dictionary to map machine type to its cyclical pattern config
    cyclical_patterns_config = {
        "siemens_motor": [("current", 0.15, 24)],
        "abb_bearing": [("temperature", 0.1, 24)],
        "haas_cnc": [("spindle_load", 0.25, 8)],
        "grundfos_pump": [("flow_rate", 0.2, 12), ("power", 0.15, 12)],
        "carrier_chiller": [("condenser_temp", 0.05, 24)] # Default from training script
    }
    
    if machine_type in cyclical_patterns_config:
        for sensor, amp, period in cyclical_patterns_config[machine_type]:
            try:
                data = apply_cyclical_pattern(timestamps, data, sensor, amplitude=amp, period_hours=period, start_date=start_time)
            except Exception as e:
                 print(f"Error applying cyclical pattern to {sensor}: {e}")
                 # Continue without this pattern if it fails
    else:
        print(f"Warning: No specific cyclical pattern defined for {machine_type} in test script.")

    
    # Apply scenario-specific modifications
    try:
        if scenario == "normal":
            pass # Already generated normal data
        elif scenario == "early_failure" and failure_type:
            if failure_type not in failure_patterns:
                print(f"Error: Failure type '{failure_type}' not found for {machine_type}.")
                return pd.DataFrame()
            pattern = failure_patterns[failure_type]
            # Apply degradation over the *entire* test duration
            data, health_index = apply_degradation(timestamps, data, pattern, duration_hours, machine_type)
        elif scenario == "late_failure" and failure_type:
            if failure_type not in failure_patterns:
                print(f"Error: Failure type '{failure_type}' not found for {machine_type}.")
                return pd.DataFrame()
            pattern = failure_patterns[failure_type]
            # Apply degradation only in the last 30% of the duration
            degradation_start_index = int(duration_hours * 0.7)
            if degradation_start_index >= duration_hours: degradation_start_index = duration_hours -1 # Ensure valid index
            
            # Create copies to modify only the tail end
            temp_data = {k: v[degradation_start_index:].copy() for k, v in data.items() if k in sensor_config}
            temp_timestamps = timestamps[degradation_start_index:]
            
            if not temp_timestamps: # Check if list is empty
                 print("Warning: 'late_failure' scenario results in empty degradation segment.")
            else:    
                 degraded_part, health_part = apply_degradation(temp_timestamps, temp_data, pattern, len(temp_timestamps), machine_type)
                 
                 for k in sensor_config.keys(): # Iterate through expected sensor keys
                     if k in data and k in degraded_part:
                         data[k][degradation_start_index:] = degraded_part[k]
                 health_index[degradation_start_index:] = health_part
            
        elif scenario == "post_maintenance":
            # Simulate recovery - for now, just keep it normal with high health
            health_index[:] = np.random.uniform(0.95, 1.0, size=duration_hours) # Slightly varied high health
        elif scenario == "sudden_sensor_drop":
            # Example: Drop temperature significantly for a short period
            if not sensor_config: # Check if sensor_config is empty
                 print("Warning: No sensors defined for sudden drop scenario.")
            else:
                 sensor_to_affect = list(sensor_config.keys())[0] # Pick first sensor
                 drop_start = duration_hours // 3
                 drop_end = min(drop_start + 12, duration_hours) # 12 hours or end
                 if sensor_to_affect in data and drop_start < drop_end:
                    original_mean = sensor_config[sensor_to_affect].get('mean', np.mean(data[sensor_to_affect]))
                    data[sensor_to_affect][drop_start:drop_end] = original_mean * 0.5 # Drop by 50%
            # Health index ideally stays high if other sensors are normal
            health_index[:] = 1.0 # Keep actual health high for this test case
            
    except Exception as e:
        print(f"Error applying scenario '{scenario}': {e}")
        return pd.DataFrame()
         
    df = pd.DataFrame(data)
    df['timestamp'] = timestamps
    df['machine_id'] = machine_type
    df['health_index'] = health_index # Actual health for reference
    df['simulation_type'] = scenario # Label the scenario
    
    # Ensure correct column order (match training data structure before feature engineering)
    cols_order = ['timestamp', 'machine_id', 'health_index', 'simulation_type'] + list(sensor_config.keys())
    # Add missing sensor columns if any (e.g., if normal gen failed for a sensor)
    for col in sensor_config.keys():
        if col not in df.columns:
            df[col] = np.nan # Add as NaN, feature engineering should handle/drop later
            
    df = df.reindex(columns=cols_order)

    return df

# --- Model Loading --- 

def load_model_and_scaler(machine_type, model_name, results_path):
    """Loads the specified model and associated scaler."""
    model_path = results_path / machine_type
    if not model_path.exists():
        print(f"Error: Results directory not found for {machine_type} at {model_path}")
        return None, None
        
    if model_name in ['RandomForest', 'XGBoost']:
        pipeline_file = model_path / f"{model_name}_pipeline.joblib"
        if not pipeline_file.exists():
             print(f"Error: Pipeline file not found: {pipeline_file}")
             return None, None
        try:
            pipeline = joblib.load(pipeline_file)
            print(f"Loaded pipeline: {pipeline_file}")
            return pipeline, None # Scaler is inside pipeline
        except Exception as e:
             print(f"Error loading pipeline {pipeline_file}: {e}")
             return None, None
             
    elif model_name == 'LSTM':
        scaler_file = model_path / f"{model_name}_scaler.joblib"
        keras_model_file = model_path / f"{model_name}_model.keras"
        if not scaler_file.exists() or not keras_model_file.exists():
             print(f"Error: LSTM scaler ({scaler_file}) or model ({keras_model_file}) not found.")
             return None, None
        try:
            scaler = joblib.load(scaler_file)
            model = tf.keras.models.load_model(keras_model_file)
            print(f"Loaded LSTM scaler: {scaler_file}")
            print(f"Loaded LSTM model: {keras_model_file}")
            return model, scaler
        except Exception as e:
             print(f"Error loading LSTM components for {machine_type}: {e}")
             return None, None
             
    else:
        print(f"Error: Unknown model name '{model_name}'")
        return None, None

# --- Prediction Function --- 

def make_predictions(model, scaler, model_name, df_test_featured):
    """Makes predictions using the loaded model and scaler."""
    # Identify feature columns dynamically based on the engineered dataframe
    machine_type = df_test_featured['machine_id'].iloc[0]
    sensor_cols = list(MACHINE_CONFIGS[machine_type]["sensors"].keys())
    exclude_cols_final = ['timestamp', 'machine_id', 'health_index', 'simulation_type'] + sensor_cols
    feature_cols = [col for col in df_test_featured.columns if col not in exclude_cols_final]
    
    if not feature_cols:
        print("Error: No feature columns found after exclusion.")
        return None
        
    X_test = df_test_featured[feature_cols].astype(np.float32)
    print(f"Prediction feature shape: {X_test.shape}")
    
    predictions = None
    try:
        if model_name in ['RandomForest', 'XGBoost']:
            # Pipeline handles scaling
            if not hasattr(model, 'predict'):
                 print(f"Error: Loaded object for {model_name} is not a valid pipeline/model.")
                 return None
            predictions = model.predict(X_test)
        elif model_name == 'LSTM':
            if scaler is None or not hasattr(model, 'predict'):
                print("Error: LSTM requires an explicit scaler and a loaded Keras model.")
                return None
                
            print("Scaling data for LSTM prediction...")
            X_test_scaled = scaler.transform(X_test)
            
            # Create sequences for LSTM prediction
            print(f"Creating LSTM sequences for prediction (window: {LSTM_WINDOW_SIZE})...")
            num_sequences = len(X_test_scaled) - LSTM_WINDOW_SIZE + 1
            if num_sequences <= 0:
                print("Error: Test data too short to create any LSTM sequences.")
                return None
                
            lstm_sequences = []
            # Efficiently create sequences using array slicing if possible (requires contiguous data)
            # This is a simplified approach; more robust methods exist
            for i in range(num_sequences):
                lstm_sequences.append(X_test_scaled[i:i+LSTM_WINDOW_SIZE, :]) 
            X_test_lstm_seq = np.array(lstm_sequences)
            print(f"LSTM prediction input shape: {X_test_lstm_seq.shape}")
            
            predictions_raw = model.predict(X_test_lstm_seq).flatten()
            
            # Pad predictions to match original test set length for easier comparison
            # Use the first prediction for padding the initial missing values
            pad_value = predictions_raw[0] if len(predictions_raw) > 0 else 0.0
            predictions = np.pad(predictions_raw, (LSTM_WINDOW_SIZE - 1, 0), 'constant', constant_values=pad_value)
            print(f"LSTM raw prediction length: {len(predictions_raw)}, padded length: {len(predictions)}")

        else:
            print(f"Error: Unknown model type for prediction: {model_name}")
            return None
            
    except Exception as e:
        print(f"Error during prediction for {model_name}: {e}")
        # Optionally re-raise or handle specific exceptions
        # import traceback
        # traceback.print_exc()
        return None
        
    # Ensure predictions are numpy array
    if predictions is not None:
        predictions = np.array(predictions).flatten()
        print(f"Generated predictions of shape: {predictions.shape}")
        # Clip predictions to valid health index range [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)
        
    return predictions

# --- Test Execution --- 

def run_test_case(machine_type, model_name, scenario, failure_type=None):
    """Runs a single test case scenario."""
    model, scaler = load_model_and_scaler(machine_type, model_name, RESULTS_DIR)
    if model is None: 
        print(f"Test failed: Model/scaler loading failed for {model_name}.")
        return False 

    # 1. Generate test data
    df_test = generate_test_data(machine_type, scenario, TEST_DURATION_HOURS, failure_type)
    if df_test.empty: 
         print(f"Test failed: Could not generate test data for scenario '{scenario}'.")
         return False
         
    sensor_cols = list(MACHINE_CONFIGS[machine_type]["sensors"].keys())
    
    # 2. Apply Feature Engineering
    print("Applying feature engineering to test data...")
    try:
        # Use the same LAGS and WINDOW_SIZES as in training
        df_test_featured = apply_feature_engineering(df_test.copy(), sensor_cols, LAGS, WINDOW_SIZES)
    except Exception as e:
         print(f"Error during feature engineering for test scenario '{scenario}': {e}")
         return False
         
    if df_test_featured.empty:
         print(f"Test failed: Feature engineering resulted in empty dataframe for scenario '{scenario}' (likely due to NaNs from lags/rolls on short data).")
         return False
         
    # Keep track of actual health index *after* potential NaN drops from feature engineering
    y_actual = df_test_featured['health_index'].values
    
    # 3. Make Predictions
    print("Making predictions...")
    y_pred = make_predictions(model, scaler, model_name, df_test_featured)
    if y_pred is None: 
         print(f"Test failed: Prediction generation failed for {model_name} on scenario '{scenario}'.")
         return False
    
    # Ensure prediction length matches actual length after feature engineering
    if len(y_pred) != len(y_actual):
        print(f"Warning: Prediction length ({len(y_pred)}) mismatch with actual ({len(y_actual)}) for scenario '{scenario}'. Aligning...")
        min_len = min(len(y_pred), len(y_actual))
        if min_len == 0:
            print("Error: Zero length data after alignment. Skipping evaluation.")
            return False
        y_pred = y_pred[:min_len]
        y_actual = y_actual[:min_len]
             
    final_predicted_health = y_pred[-1]
    final_actual_health = y_actual[-1]
    # Use the simpler final health suggestion for automated checks
    maintenance_suggestion = suggest_maintenance_from_final_health(final_predicted_health)

    # 4. Evaluate Results (Simple Checks)
    print("\n--- Test Results --- ")
    print(f"Scenario: {scenario} ({failure_type if failure_type else 'N/A'}) - Model: {model_name}")
    print(f"Final Actual Health: {final_actual_health:.3f}")
    print(f"Final Predicted Health: {final_predicted_health:.3f}")
    print(f"Maintenance Suggestion: {maintenance_suggestion}")
    
    # Define pass criteria based on scenario
    passed = False
    health_threshold_pass = 0.8 # Expect high health for normal/post-maintenance
    health_threshold_fail = 0.5 # Expect low health for failure scenarios
    
    if scenario == "normal":
        passed = final_predicted_health >= health_threshold_pass and "No immediate" in maintenance_suggestion
        if not passed: print(f"Reason: Expected health >= {health_threshold_pass} and no maintenance needed.")
    elif scenario == "early_failure" or scenario == "late_failure":
        passed = final_predicted_health <= health_threshold_fail # Expect significant degradation
        if not passed: print(f"Reason: Expected health <= {health_threshold_fail} due to failure simulation.")
    elif scenario == "post_maintenance":
        passed = final_predicted_health >= health_threshold_pass
        if not passed: print(f"Reason: Expected health >= {health_threshold_pass} after maintenance.")
    elif scenario == "sudden_sensor_drop":
        # Check if health didn't collapse completely and recovered somewhat
        avg_pred_health = np.mean(y_pred)
        passed = avg_pred_health > 0.6 and final_predicted_health > 0.5 # Simple check: doesn't collapse and ends reasonably ok
        print(f"(Average Predicted Health during sensor drop: {avg_pred_health:.3f})")
        if not passed: print(f"Reason: Expected health not to collapse completely (avg > 0.6) and end reasonably (> 0.5) after sensor drop.")

    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print("---------------------")
    return passed

# --- Main Test Runner --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Trained Predictive Maintenance Models")
    parser.add_argument("machine_type", type=str, help="Type of machine to test models for (e.g., siemens_motor)")
    parser.add_argument("--model", type=str, default="all", help="Specific model to test (RandomForest, XGBoost, LSTM) or 'all'")
    args = parser.parse_args()
    machine_type = args.machine_type
    model_to_test = args.model
    
    print(f"\n===== Starting Testing for: {machine_type} =====")
    
    models = []
    if model_to_test.lower() == 'all':
        # Check which models actually exist in results
        model_path = RESULTS_DIR / machine_type
        if (model_path / "RandomForest_pipeline.joblib").exists(): models.append('RandomForest')
        if (model_path / "XGBoost_pipeline.joblib").exists(): models.append('XGBoost')
        if (model_path / "LSTM_model.keras").exists(): models.append('LSTM')
        if not models: 
             print(f"Error: No trained models found for {machine_type} in {model_path}")
             sys.exit(1)
             
    elif model_to_test in ['RandomForest', 'XGBoost', 'LSTM']:
        models = [model_to_test]
    else:
        print(f"Error: Invalid model specified: {model_to_test}. Choose from RandomForest, XGBoost, LSTM, or all.")
        sys.exit(1)
        
    # Get failure types for the machine
    failure_types = list(MACHINE_CONFIGS[machine_type]["failure_patterns"].keys())
    failure_1 = failure_types[0] if len(failure_types) > 0 else None
    failure_2 = failure_types[1] if len(failure_types) > 1 else None
        
    test_scenarios = [
        {"scenario": "normal"},
        {"scenario": "early_failure", "failure_type": failure_1},
        {"scenario": "late_failure", "failure_type": failure_2 if failure_2 else failure_1}, # Use second failure if available
        {"scenario": "post_maintenance"},
        {"scenario": "sudden_sensor_drop"}
    ]
    
    results = {model: {sc["scenario"]: "NOT RUN" for sc in test_scenarios} for model in models}
    overall_passed = True
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for model_name in models:
        print(f"\n===== Testing Model: {model_name} =====")
        model_passed_all = True
        for test in test_scenarios:
            scenario_name = test["scenario"]
            # Skip tests if required failure type is missing
            if "failure_type" in test and test["failure_type"] is None:
                print(f"Skipping scenario '{scenario_name}' for {machine_type} as failure type is not available.")
                results[model_name][scenario_name] = "SKIPPED"
                skipped_count += 1
                continue
            
            # Run the test case
            try:
                passed = run_test_case(machine_type, model_name, **test)
                results[model_name][scenario_name] = "PASS" if passed else "FAIL"
                if passed: 
                    passed_count += 1
                else: 
                    failed_count += 1
                    model_passed_all = False
                    overall_passed = False
            except Exception as e:
                print(f"Error running test case {scenario_name} for {model_name}: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed error
                results[model_name][scenario_name] = "ERROR"
                failed_count += 1
                model_passed_all = False
                overall_passed = False
        
        print(f"\nSummary for {model_name}: {('ALL PASSED' if model_passed_all else 'SOME FAILED/SKIPPED/ERROR') if results[model_name].values() else 'NO TESTS RUN'}")
        
    print("\n===== Overall Test Summary ====")
    for model_name, scenario_results in results.items():
        print(f"--- {model_name} --- ")
        for scenario, result in scenario_results.items():
             print(f"  {scenario:<20}: {result}")
             
    total_tests = passed_count + failed_count + skipped_count
    print(f"\nTotal Tests Run: {passed_count + failed_count}/{total_tests} (Passed: {passed_count}, Failed/Error: {failed_count}, Skipped: {skipped_count})")
    print(f"Overall Result: {('ALL TESTS PASSED' if overall_passed and failed_count == 0 else 'SOME TESTS FAILED/ERROR') if total_tests > 0 else 'NO TESTS EXECUTED'}")
    print(f"===== Testing Complete for: {machine_type} =====") 