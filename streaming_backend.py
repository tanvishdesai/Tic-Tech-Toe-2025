import time
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib # Ensure joblib is available

# --- Configuration (Adapted from dashboard_simulation.py) ---
NUM_MACHINES = 5 # Keep consistent? Or infer from config?
UPDATE_INTERVAL_SECONDS = 2  # How often to generate new data
HISTORY_POINTS = 300        # How many data points to keep per sensor for feature calc/display
MACHINES_CONFIG = { # Same as simulation
    "Siemens Motor": {
        "Temperature": (70, 3, 0.005, "°C"),
        "Vibration": (3, 0.5, 0.002, "mm/s RMS"),
        "Current": (55, 5, -0.001, "A"),
        # Health model needs 'voltage' - add simulation?
        "Voltage": (230, 2, 0.0005, "V") # Adding Voltage simulation
    },
    "ABB Bearing": {
        "Vibration": (1.1, 0.2, 0.003, "mm/s RMS"),
        "Temperature": (50, 4, 0.004, "°C"),
        "Acoustic": (55, 5, 0.006, "dB"),
    },
    "HAAS CNC": {
        "Spindle Load": (40, 8, -0.002, "%"),
        "Vibration": (1.8, 0.3, 0.0025, "mm/s RMS"),
        "Temperature": (55, 3, 0.0035, "°C"),
        # Health model needs 'spindle_speed' - add simulation?
        "Spindle Speed": (1200, 50, 0.1, "RPM") # Adding Spindle Speed simulation
    },
    "Grundfos Pump": {
        "Pressure": (13.5, 2, 0.001, "bar"),
        "Flow Rate": (90, 10, -0.005, "m³/h"),
        "Temperature": (55, 5, 0.0045, "°C"),
         # Health model needs 'power' - add simulation?
        "Power": (5, 0.5, 0.001, "kW") # Adding Power simulation
    },
    "Carrier Chiller": {
        "Refrigerant Pressure": (16.5, 2.5, -0.0015, "bar"),
        "Condenser Temp": (40, 2, 0.003, "°C"),
        "Power Draw/Ton": (0.6, 0.05, 0.001, "kW/ton"), # Map this to 'power' for model?
        # Health model needs 'evaporator_temp', 'power'
        "Evaporator Temp": (5, 0.5, 0.001, "°C"), # Adding Evaporator Temp
        # Map Power Draw/Ton to 'Power' or add separate 'Power' simulation?
        # Let's map 'Power Draw/Ton' to 'power' when feeding the model
    }
}

# --- Add project root for imports ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

try:
    from src.modeling.predict_maintenance import predict_status, HEALTH_MODEL_SENSORS
    # print("INFO: Successfully imported predict_status.", file=sys.stderr)
    print("INFO: Successfully imported predict_status.") # Print to stdout now or just comment out
except ImportError as e:
    print(f"ERROR: Could not import predict_status: {e}", file=sys.stderr)
    print("ERROR: Ensure 'src' is in the Python path or run from project root.", file=sys.stderr)
    sys.exit(1) # Exit if prediction function isn't available

# --- Helper Functions (Adapted from dashboard_simulation.py) ---
def generate_reading(current_base, std_dev, drift, anomaly_prob=0.01):
    """Generates a new sensor reading with noise, drift, and potential anomalies."""
    new_base = current_base + drift
    reading = np.random.normal(new_base, std_dev)
    if np.random.rand() < anomaly_prob:
        anomaly_factor = np.random.uniform(1.5, 3.0) * np.random.choice([-1, 1])
        reading = new_base + anomaly_factor * std_dev * 3
    if new_base >= 0:
         reading = max(0, reading)
    # Simulate sensor failure - very large deviation (less frequent)
    if np.random.rand() < 0.0001:
         reading = new_base * np.random.uniform(5, 10) * np.random.choice([-1, 1])
         # print(f"INFO: Generated EXTREME value: {reading:.2f}", file=sys.stderr)
         print(f"INFO: Generated EXTREME value: {reading:.2f}") # Print to stdout now or comment out
    return reading, new_base

# --- Data Storage ---
# Use dictionaries to store dataframes and current base values
machine_data_history = {}
current_sensor_bases = {}

def initialize_data():
    """Initializes the data storage with some history."""
    # print("INFO: Initializing data history...", file=sys.stderr)
    print("INFO: Initializing data history...") # Print to stdout now or comment out
    for machine_name, sensors in MACHINES_CONFIG.items():
        machine_data_history[machine_name] = {}
        current_sensor_bases[machine_name] = {}
        for sensor_name, (mean, std, drift, unit) in sensors.items():
            initial_timestamps = [datetime.now() - timedelta(seconds=x * UPDATE_INTERVAL_SECONDS) for x in range(HISTORY_POINTS)][::-1]
            initial_values = np.random.normal(mean, std, HISTORY_POINTS)
            df = pd.DataFrame({
                'Timestamp': initial_timestamps,
                'Value': initial_values
            }).set_index('Timestamp')
            machine_data_history[machine_name][sensor_name] = df
            current_sensor_bases[machine_name][sensor_name] = mean # Start drift from the normal mean
    # print("INFO: Data history initialized.", file=sys.stderr)
    print("INFO: Data history initialized.") # Print to stdout now or comment out

# --- Main Loop ---
if __name__ == "__main__":
    initialize_data()
    # print("INFO: Starting simulation loop...", file=sys.stderr)
    print("INFO: Starting simulation loop...") # Print to stdout now or comment out

    while True:
        current_time = datetime.now()
        output_data = {"timestamp": current_time.isoformat(), "machines": {}}

        for machine_name, sensors in MACHINES_CONFIG.items():
            machine_output = {"sensors": {}, "prediction": None}
            all_sensor_dfs_for_machine = [] # To combine for prediction input

            for sensor_name, (mean, std, drift, unit) in sensors.items():
                # Generate new data
                current_base = current_sensor_bases[machine_name][sensor_name]
                new_value, new_base = generate_reading(current_base, std, drift)
                current_sensor_bases[machine_name][sensor_name] = new_base

                # Create new data entry
                new_data = pd.DataFrame([{'Value': new_value}], index=[pd.Timestamp(current_time)])
                new_data.index.name = 'Timestamp'

                # Append new data and keep history limit
                df = machine_data_history[machine_name][sensor_name]
                df = pd.concat([df, new_data])
                df = df.tail(HISTORY_POINTS)
                machine_data_history[machine_name][sensor_name] = df

                # Store latest value and unit for JSON output
                machine_output["sensors"][sensor_name] = {
                    "value": new_value,
                    "unit": unit,
                    # Include recent history for plotting (e.g., last 50 points)
                    "history": df.tail(50).reset_index().to_dict(orient='records')
                }

                # Prepare DataFrame for prediction (rename column to sensor name)
                 # Normalize sensor name for model features (lowercase, underscore)
                normalized_sensor_name = sensor_name.lower().replace(" ", "_").replace("/", "_per_")
                
                # Special mapping for Carrier Chiller Power
                if machine_name == "Carrier Chiller" and sensor_name == "Power Draw/Ton":
                    normalized_sensor_name = "power" # Map to the expected feature name

                # Ensure index is unique before renaming and appending
                # Keep the last entry for any duplicate timestamps
                unique_index_df = df[~df.index.duplicated(keep='last')]
                sensor_df_for_pred = unique_index_df[['Value']].rename(columns={'Value': normalized_sensor_name})
                all_sensor_dfs_for_machine.append(sensor_df_for_pred)


            # Combine sensor data for the current machine for prediction
            if all_sensor_dfs_for_machine:
                try:
                    # Combine horizontally, aligning timestamps
                    # Ensure indexes are unique across the list before concat? Less likely needed now.
                    combined_df = pd.concat(all_sensor_dfs_for_machine, axis=1, join='outer')
                    combined_df = combined_df.sort_index().ffill()

                    # Ensure all required columns exist (fill missing with 0 maybe?)
                    # This might require knowing HEALTH_MODEL_SENSORS beforehand
                    machine_type_norm = machine_name.lower().replace(" ", "_")
                    required_sensors = HEALTH_MODEL_SENSORS.get(machine_type_norm, [])

                    missing_cols = [s for s in required_sensors if s not in combined_df.columns]
                    if missing_cols:
                       # print(f"WARN: Machine '{machine_name}' missing simulated sensors needed for model: {missing_cols}. Filling with 0.", file=sys.stderr)
                        for col in missing_cols:
                             combined_df[col] = 0.0 # Fill missing required sensors with 0

                    # Drop columns not needed by the model to avoid issues
                    cols_to_keep = [s for s in required_sensors if s in combined_df.columns]
                    if cols_to_keep:
                         prediction_input_df = combined_df[cols_to_keep]

                         # Ensure enough data points for feature extraction
                         min_rows_needed = 30 # A guess, predict_status might need more for lags/windows
                         if len(prediction_input_df) >= min_rows_needed:
                             # Call the prediction function
                             prediction_result = predict_status(prediction_input_df, machine_type=machine_type_norm)
                             machine_output["prediction"] = prediction_result
                         else:
                              # Optionally print a message to stderr if needed for real issues
                              # print(f"WARN: ({machine_name}) Need more data ({len(prediction_input_df)}/{min_rows_needed})", file=sys.stderr)
                              machine_output["prediction"] = {
                                 "status": "Initializing",
                                 "reason": f"Need more data ({len(prediction_input_df)}/{min_rows_needed})",
                                 "risk_score": 0.0,
                                 "predicted_health_index": None,
                             }
                    else:
                        machine_output["prediction"] = {
                            "status": "Error",
                            "reason": "No required sensor data available for prediction after filtering.",
                            "risk_score": 0.0,
                            "predicted_health_index": None,
                            }


                except Exception as e:
                    print(f"ERROR: Prediction failed for {machine_name}: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    machine_output["prediction"] = {
                        "status": "Error",
                        "reason": f"Prediction function failed: {e}",
                        "risk_score": 0.0,
                        "predicted_health_index": None,
                    }
            else:
                 # Optionally print a message to stderr if needed for real issues
                 # print(f"ERROR: ({machine_name}) No sensor dataframes generated for prediction.", file=sys.stderr)
                 machine_output["prediction"] = {
                    "status": "Error",
                    "reason": "No sensor dataframes generated for prediction.",
                    "risk_score": 0.0,
                    "predicted_health_index": None,
                 }


            output_data["machines"][machine_name] = machine_output

        # Print the JSON output for the Next.js backend to capture
        # Use separators that are unlikely to appear in the JSON itself for reliable parsing
        print("---JSON_START---")
        print(json.dumps(output_data, default=str)) # Use default=str for datetime etc.
        print("---JSON_END---")
        sys.stdout.flush() # Ensure output is sent immediately

        time.sleep(UPDATE_INTERVAL_SECONDS)
