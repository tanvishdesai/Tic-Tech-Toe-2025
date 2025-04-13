import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

try:
    from redundant.data_generation.machine_configs import MACHINE_CONFIGS
except ImportError:
    print("Error: Could not import MACHINE_CONFIGS. Make sure the script is run from the project root"
          " or src/data_generation_health_index directory, and __init__.py files exist.")
    sys.exit(1)

# --- Configuration ---
MACHINE_TYPE = "abb_bearing" # <-- Changed
TARGET_ROWS = 200_000
SAMPLING_INTERVAL_HOURS = 1
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = OUTPUT_DIR / f"{MACHINE_TYPE}_health_data.csv"
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Helper Functions (Adapted from data_generator.py) ---

def generate_normal_data(timestamps, sensor_config):
    # ... (Function remains the same as in siemens_motor script) ...
    """Generates baseline normal data for all sensors."""
    data = {}
    num_points = len(timestamps)
    for sensor, params in sensor_config.items():
        values = np.random.normal(params["mean"], params["std_dev"], num_points)
        if "min_val" in params:
            values = np.maximum(values, params["min_val"])
        if "max_val" in params:
            values = np.minimum(values, params["max_val"])
        data[sensor] = values
    return data

def apply_cyclical_pattern(timestamps, data, sensor_name, amplitude, period_hours, start_date):
    # ... (Function remains the same as in siemens_motor script) ...
    """Adds a cyclical pattern to a sensor's data."""
    if sensor_name not in data:
        print(f"Warning: Sensor {sensor_name} not found for cyclical pattern. Skipping.")
        return data

    original_values = data[sensor_name]
    avg_value = np.mean(original_values) # Use segment average
    if avg_value == 0: # Avoid division by zero if sensor data is flat zero
         avg_value = 1

    cycle_values = np.zeros(len(timestamps))
    for i, timestamp in enumerate(timestamps):
        hours_elapsed = (timestamp - start_date).total_seconds() / 3600
        cycle_position = (hours_elapsed % period_hours) / period_hours * 2 * np.pi
        cycle_values[i] = np.sin(cycle_position) * amplitude * avg_value

    data[sensor_name] = original_values + cycle_values
    return data

def apply_degradation(timestamps, data, failure_pattern, segment_duration_steps, machine_type): # Added machine_type arg
    # ... (Function body is now identical to the fixed one in siemens_motor script) ...
    """
    Applies degradation to affected sensors based on the failure pattern
    over the segment duration and calculates health index.
    """
    health_index = np.ones(len(timestamps))
    degradation_progress = np.zeros(len(timestamps))

    for i in range(segment_duration_steps):
        progress = i / (segment_duration_steps -1) if segment_duration_steps > 1 else 1.0
        degradation_progress[i] = progress # Linear progress over the segment

        for sensor, details in failure_pattern["degradation"].items():
            if sensor in data:
                severity = details.get("severity", 1.0)
                degradation_factor = progress * severity
                if isinstance(data[sensor], (np.ndarray, list)):
                     if i < len(data[sensor]):
                          data[sensor][i] = data[sensor][i] * (1 + degradation_factor)
                     else:
                          print(f"Warning: Index {i} out of bounds for sensor {sensor} data (length {len(data[sensor])}).")
                else:
                     data[sensor] = data[sensor] * (1 + degradation_factor)

    health_index = 1.0 - degradation_progress

    sensor_config = MACHINE_CONFIGS[machine_type]["sensors"] # Use machine_type
    for sensor in failure_pattern["affected_sensors"]:
         if sensor in data and sensor in sensor_config:
             if isinstance(data[sensor], (np.ndarray, list)):
                 sensor_data_np = np.array(data[sensor]) 
                 if "min_val" in sensor_config[sensor]:
                     sensor_data_np = np.maximum(sensor_data_np, sensor_config[sensor]["min_val"])
                 if "max_val" in sensor_config[sensor]:
                      sensor_data_np = np.minimum(sensor_data_np, sensor_config[sensor]["max_val"])
                 data[sensor] = sensor_data_np 
             else:
                 if "min_val" in sensor_config[sensor]:
                      data[sensor] = max(data[sensor], sensor_config[sensor]["min_val"])
                 if "max_val" in sensor_config[sensor]:
                     data[sensor] = min(data[sensor], sensor_config[sensor]["max_val"])

    return data, health_index

def apply_maintenance_effect(data, health_index, maintenance_indices, sensor_config, effect=0.8):
    # ... (Function body is now identical to the fixed one in siemens_motor script) ...
    """Resets health index and partially reverts sensor values during maintenance."""
    if not maintenance_indices:
        return data, health_index

    for idx in maintenance_indices:
        health_index[idx] = 1.0 # Reset health during maintenance
        for sensor, params in sensor_config.items():
            if sensor in data:
                if isinstance(data[sensor], (np.ndarray, list)) and idx < len(data[sensor]):
                    current_val = data[sensor][idx]
                    normal_mean = params["mean"]
                    data[sensor][idx] = current_val - (current_val - normal_mean) * effect
                    if "min_val" in params:
                        data[sensor][idx] = max(data[sensor][idx], params["min_val"])
                    if "max_val" in params:
                        data[sensor][idx] = min(data[sensor][idx], params["max_val"])
                elif not isinstance(data[sensor], (np.ndarray, list)):
                     pass 
                    
    return data, health_index

# --- Main Generation Logic ---
if __name__ == "__main__":
    print(f"Starting data generation for {MACHINE_TYPE}...")

    machine_config = MACHINE_CONFIGS[MACHINE_TYPE]
    sensor_config = machine_config["sensors"]
    failure_patterns = machine_config["failure_patterns"]
    failure_types = list(failure_patterns.keys())
    maintenance_interval_days = machine_config["maintenance_interval_days"]
    maintenance_duration_hours = 8 # Default or from config if available

    all_data_segments = []
    total_rows_generated = 0
    current_time = datetime.now() - timedelta(days=int(TARGET_ROWS * SAMPLING_INTERVAL_HOURS / 24)+1) # Estimate start date
    global_start_time = current_time # Keep track of the absolute start for cycles

    while total_rows_generated < TARGET_ROWS:
        segment_type = random.choice(['normal'] * 5 + failure_types)
        segment_duration_days = random.randint(15, 90)
        segment_duration_steps = int(segment_duration_days * 24 / SAMPLING_INTERVAL_HOURS)

        timestamps = [current_time + timedelta(hours=i * SAMPLING_INTERVAL_HOURS)
                      for i in range(segment_duration_steps)]

        segment_data = generate_normal_data(timestamps, sensor_config)

        # --- Apply machine-specific cyclical patterns --- # <-- Changed section
        # ABB Bearing: Minor temperature cycle based on ambient temperature
        segment_data = apply_cyclical_pattern(timestamps, segment_data, "temperature",
                                              amplitude=0.1, period_hours=24, start_date=global_start_time)
        # ------------------------------------------------- #

        health_index = np.ones(segment_duration_steps)

        if segment_type != 'normal':
            pattern = failure_patterns[segment_type]
            # Pass MACHINE_TYPE to apply_degradation
            segment_data, health_index = apply_degradation(timestamps, segment_data, pattern, segment_duration_steps, MACHINE_TYPE)
            print(f"  Simulating failure: {segment_type} for {segment_duration_days} days")
        else:
             print(f"  Simulating normal operation for {segment_duration_days} days")

        maintenance_indices = []
        for i, ts in enumerate(timestamps):
             days_since_start = (ts - global_start_time).total_seconds() / (3600 * 24)
             # Use machine-specific maintenance interval
             if days_since_start > 0 and int(days_since_start) % maintenance_interval_days == 0:
                 if ts.hour < maintenance_duration_hours:
                      maintenance_indices.append(i)

        if maintenance_indices:
             print(f"  Applying maintenance effect at indices: {len(maintenance_indices)}")
             segment_data, health_index = apply_maintenance_effect(segment_data, health_index,
                                                                  maintenance_indices, sensor_config)

        df_segment = pd.DataFrame(segment_data)
        df_segment['timestamp'] = timestamps
        df_segment['machine_id'] = MACHINE_TYPE
        df_segment['health_index'] = health_index
        df_segment['simulation_type'] = segment_type

        cols = ['timestamp', 'machine_id', 'health_index', 'simulation_type'] + list(sensor_config.keys())
        # Handle potential missing columns if sensor_config has sensors not generated in segment_data (shouldn't happen here)
        cols = [c for c in cols if c in df_segment.columns or c in sensor_config]
        df_segment = df_segment.reindex(columns=cols) # Ensure consistent column order

        all_data_segments.append(df_segment)
        total_rows_generated += segment_duration_steps
        current_time = timestamps[-1] + timedelta(hours=SAMPLING_INTERVAL_HOURS)

        print(f"  Segment generated. Total rows: {total_rows_generated}/{TARGET_ROWS}")

    final_data = pd.concat(all_data_segments, ignore_index=True)
    final_data = final_data.head(TARGET_ROWS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(OUTPUT_FILENAME, index=False)

    print(f"\nData generation complete for {MACHINE_TYPE}.")
    print(f"Total rows: {len(final_data)}")
    print(f"Data saved to: {OUTPUT_FILENAME}")

    print("\nSample data:")
    print(final_data.head())
    print("\nData distribution:")
    print(final_data['simulation_type'].value_counts(normalize=True))
    print("\nHealth index stats:")
    print(final_data['health_index'].describe()) 