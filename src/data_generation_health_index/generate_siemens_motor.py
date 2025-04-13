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
MACHINE_TYPE = "siemens_motor"
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

def apply_degradation(timestamps, data, failure_pattern, segment_duration_steps, machine_type):
    """
    Applies degradation to affected sensors based on the failure pattern
    over the segment duration and calculates health index.
    
    Args:
        timestamps (list): List of timestamps for the segment.
        data (dict): Dictionary containing sensor data.
        failure_pattern (dict): Dictionary containing failure pattern information.
        segment_duration_steps (int): Number of steps in the segment.
        machine_type (str): The type of machine (used for config lookup).
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
                # Check if data[sensor] is a NumPy array or list/iterable
                if isinstance(data[sensor], (np.ndarray, list)):
                     if i < len(data[sensor]):
                          data[sensor][i] = data[sensor][i] * (1 + degradation_factor)
                     else:
                          print(f"Warning: Index {i} out of bounds for sensor {sensor} data (length {len(data[sensor])}).")
                else: # Handle scalar case if necessary, though less likely here
                     data[sensor] = data[sensor] * (1 + degradation_factor)

    health_index = 1.0 - degradation_progress

    # Use the passed machine_type argument instead of global constant
    sensor_config = MACHINE_CONFIGS[machine_type]["sensors"]
    for sensor in failure_pattern["affected_sensors"]:
         if sensor in data and sensor in sensor_config:
             # Ensure data[sensor] is array-like before using np.maximum/minimum
             if isinstance(data[sensor], (np.ndarray, list)):
                 sensor_data_np = np.array(data[sensor]) # Convert to numpy array if list
                 if "min_val" in sensor_config[sensor]:
                     sensor_data_np = np.maximum(sensor_data_np, sensor_config[sensor]["min_val"])
                 if "max_val" in sensor_config[sensor]:
                      sensor_data_np = np.minimum(sensor_data_np, sensor_config[sensor]["max_val"])
                 data[sensor] = sensor_data_np # Assign back (might change list to array)
             else:
                 # Handle scalar case if needed
                 if "min_val" in sensor_config[sensor]:
                      data[sensor] = max(data[sensor], sensor_config[sensor]["min_val"])
                 if "max_val" in sensor_config[sensor]:
                     data[sensor] = min(data[sensor], sensor_config[sensor]["max_val"])

    return data, health_index

def apply_maintenance_effect(data, health_index, maintenance_indices, sensor_config, effect=0.8):
    """Resets health index and partially reverts sensor values during maintenance."""
    if not maintenance_indices:
        return data, health_index

    for idx in maintenance_indices:
        health_index[idx] = 1.0 # Reset health during maintenance
        for sensor, params in sensor_config.items():
            if sensor in data:
                # Ensure data[sensor] is list/array before indexing
                if isinstance(data[sensor], (np.ndarray, list)) and idx < len(data[sensor]):
                    current_val = data[sensor][idx]
                    normal_mean = params["mean"]
                    # Move value closer to normal mean
                    data[sensor][idx] = current_val - (current_val - normal_mean) * effect
                    # Re-apply limits
                    if "min_val" in params:
                        data[sensor][idx] = max(data[sensor][idx], params["min_val"])
                    if "max_val" in params:
                        data[sensor][idx] = min(data[sensor][idx], params["max_val"])
                elif not isinstance(data[sensor], (np.ndarray, list)):
                     # Handle scalar case if needed (less likely for sensor data)
                     pass # Or apply logic if sensors can be scalar
                    
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
        # Determine segment type and duration
        segment_type = random.choice(['normal'] * 5 + failure_types) # Bias towards normal
        segment_duration_days = random.randint(15, 90) # Simulate a few weeks/months
        segment_duration_steps = int(segment_duration_days * 24 / SAMPLING_INTERVAL_HOURS)

        # Generate timestamps for the segment
        timestamps = [current_time + timedelta(hours=i * SAMPLING_INTERVAL_HOURS)
                      for i in range(segment_duration_steps)]

        # Generate baseline normal data
        segment_data = generate_normal_data(timestamps, sensor_config)

        # Apply base cyclical patterns (e.g., daily load for motor)
        # Siemens motor: daily load cycle on 'current'
        segment_data = apply_cyclical_pattern(timestamps, segment_data, "current",
                                              amplitude=0.15, period_hours=24, start_date=global_start_time)
        # Add others if defined in config

        # Initialize health index
        health_index = np.ones(segment_duration_steps)

        # Apply degradation if it's a failure segment
        if segment_type != 'normal':
            pattern = failure_patterns[segment_type]
            segment_data, health_index = apply_degradation(timestamps, segment_data, pattern, segment_duration_steps, MACHINE_TYPE)
            print(f"  Simulating failure: {segment_type} for {segment_duration_days} days")
        else:
             print(f"  Simulating normal operation for {segment_duration_days} days")


        # Apply maintenance effect periodically
        maintenance_indices = []
        for i, ts in enumerate(timestamps):
             # Check if maintenance is due based on interval from global start
             days_since_start = (ts - global_start_time).total_seconds() / (3600 * 24)
             if days_since_start > 0 and int(days_since_start) % maintenance_interval_days == 0:
                 # Check if within maintenance window (e.g., first 8 hours of the day)
                 if ts.hour < maintenance_duration_hours:
                      maintenance_indices.append(i)

        if maintenance_indices:
             print(f"  Applying maintenance effect at indices: {len(maintenance_indices)}")
             segment_data, health_index = apply_maintenance_effect(segment_data, health_index,
                                                                  maintenance_indices, sensor_config)


        # Create DataFrame for the segment
        df_segment = pd.DataFrame(segment_data)
        df_segment['timestamp'] = timestamps
        df_segment['machine_id'] = MACHINE_TYPE
        df_segment['health_index'] = health_index
        df_segment['simulation_type'] = segment_type # Keep track of the simulation

        # Define column order
        cols = ['timestamp', 'machine_id', 'health_index', 'simulation_type'] + list(sensor_config.keys())
        df_segment = df_segment[cols]


        all_data_segments.append(df_segment)
        total_rows_generated += segment_duration_steps
        current_time = timestamps[-1] + timedelta(hours=SAMPLING_INTERVAL_HOURS)

        print(f"  Segment generated. Total rows: {total_rows_generated}/{TARGET_ROWS}")


    # Concatenate all segments and truncate to target rows
    final_data = pd.concat(all_data_segments, ignore_index=True)
    final_data = final_data.head(TARGET_ROWS)

    # Save the data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(OUTPUT_FILENAME, index=False)

    print(f"Data generation complete for {MACHINE_TYPE}.")
    print(f"Total rows: {len(final_data)}")
    print(f"Data saved to: {OUTPUT_FILENAME}")

    # Display sample data
    print("Sample data:")
    print(final_data.head())
    print("Data distribution:")
    print(final_data['simulation_type'].value_counts(normalize=True))
    print("Health index stats:")
    print(final_data['health_index'].describe()) 