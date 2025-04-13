import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

class MachineDataGenerator:
    """Base class for generating synthetic sensor data for industrial equipment."""
    
    def __init__(self, machine_id, start_date=None, end_date=None, 
                 sampling_interval_minutes=5, random_seed=42):
        """
        Initialize the data generator.
        
        Args:
            machine_id: Unique identifier for the machine
            start_date: Start date for the data generation (default: 60 days ago)
            end_date: End date for the data generation (default: today)
            sampling_interval_minutes: Time between sensor readings in minutes
            random_seed: Seed for reproducibility
        """
        self.machine_id = machine_id
        
        # Set default dates if not provided
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=60)
        else:
            self.start_date = start_date
            
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        self.sampling_interval = timedelta(minutes=sampling_interval_minutes)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Create timestamp range
        self.timestamps = []
        current_time = self.start_date
        while current_time <= self.end_date:
            self.timestamps.append(current_time)
            current_time += self.sampling_interval
        
        # Initialize data storage
        self.data = pd.DataFrame()
        self.data['timestamp'] = self.timestamps
        self.data['machine_id'] = self.machine_id
        
    def add_normal_sensor_reading(self, sensor_name, mean, std_dev, min_val=None, max_val=None):
        """
        Add normally distributed sensor readings.
        
        Args:
            sensor_name: Name of the sensor
            mean: Mean value for normal operation
            std_dev: Standard deviation for normal operation
            min_val: Minimum allowable value (optional)
            max_val: Maximum allowable value (optional)
        """
        values = np.random.normal(mean, std_dev, len(self.timestamps))
        
        # Apply min/max constraints if provided
        if min_val is not None:
            values = np.maximum(values, min_val)
        if max_val is not None:
            values = np.minimum(values, max_val)
            
        self.data[sensor_name] = values
        return self
        
    def add_degradation_pattern(self, sensor_name, start_percent, end_percent, 
                               pattern_type='linear', severity=1.0):
        """
        Add a degradation pattern to a sensor.
        
        Args:
            sensor_name: Name of the sensor to degrade
            start_percent: When to start degradation (0-1, percent of timeline)
            end_percent: When degradation reaches maximum (0-1, percent of timeline)
            pattern_type: Type of degradation ('linear', 'exponential', 'step')
            severity: How severe the degradation is (multiplier)
        """
        if sensor_name not in self.data.columns:
            raise ValueError(f"Sensor {sensor_name} not found in the data.")
            
        # Original values
        original_values = self.data[sensor_name].copy()
        
        # Calculate indices for degradation
        start_idx = int(len(self.timestamps) * start_percent)
        end_idx = int(len(self.timestamps) * end_percent)
        
        # No degradation before start_idx
        degradation_values = original_values.copy()
        
        # Apply degradation pattern
        if pattern_type == 'linear':
            # Linear increase/decrease
            for i in range(start_idx, end_idx):
                progress = (i - start_idx) / (end_idx - start_idx)
                degradation_factor = progress * severity
                degradation_values[i] = original_values[i] * (1 + degradation_factor)
                
        elif pattern_type == 'exponential':
            # Exponential increase/decrease
            for i in range(start_idx, end_idx):
                progress = (i - start_idx) / (end_idx - start_idx)
                degradation_factor = (np.exp(progress) - 1) / (np.e - 1) * severity
                degradation_values[i] = original_values[i] * (1 + degradation_factor)
                
        elif pattern_type == 'step':
            # Step changes
            steps = 5  # Number of degradation steps
            for i in range(start_idx, end_idx):
                progress = (i - start_idx) / (end_idx - start_idx)
                step = int(progress * steps) / steps
                degradation_factor = step * severity
                degradation_values[i] = original_values[i] * (1 + degradation_factor)
        
        # After end_idx, maintain the final degradation level
        final_degradation = degradation_values[end_idx-1] if end_idx > 0 else original_values[0]
        for i in range(end_idx, len(original_values)):
            degradation_values[i] = final_degradation
            
        # Update the data
        self.data[sensor_name] = degradation_values
        return self
        
    def add_sudden_failure(self, sensor_name, failure_start_percent, 
                          duration_hours=2, magnitude=5.0):
        """
        Add a sudden failure event to a sensor.
        
        Args:
            sensor_name: Name of the sensor to affect
            failure_start_percent: When failure occurs (0-1, percent of timeline)
            duration_hours: How long the failure lasts
            magnitude: How severe the failure is (multiplier)
        """
        if sensor_name not in self.data.columns:
            raise ValueError(f"Sensor {sensor_name} not found in the data.")
            
        # Original values
        original_values = self.data[sensor_name].copy()
        
        # Calculate failure indices
        start_idx = int(len(self.timestamps) * failure_start_percent)
        duration_steps = int(duration_hours * 60 / self.sampling_interval.total_seconds() * 60)
        end_idx = min(start_idx + duration_steps, len(self.timestamps))
        
        # Apply failure
        failure_values = original_values.copy()
        for i in range(start_idx, end_idx):
            # Add high magnitude spike or drop
            if random.random() > 0.5:  # Random direction
                failure_values[i] = original_values[i] * (1 + random.random() * magnitude)
            else:
                failure_values[i] = original_values[i] / (1 + random.random() * magnitude)
        
        # Update the data
        self.data[sensor_name] = failure_values
        return self
        
    def add_cyclical_pattern(self, sensor_name, amplitude, period_hours=24):
        """
        Add a cyclical pattern to a sensor (like day/night variations).
        
        Args:
            sensor_name: Name of the sensor
            amplitude: Amplitude of the cycle (as fraction of sensor mean)
            period_hours: Period of the cycle in hours
        """
        if sensor_name not in self.data.columns:
            raise ValueError(f"Sensor {sensor_name} not found in the data.")
            
        # Original values
        original_values = self.data[sensor_name].values
        
        # Calculate average to scale amplitude
        avg_value = np.mean(original_values)
        
        # Create sinusoidal pattern
        cycle_values = np.zeros(len(self.timestamps))
        for i, timestamp in enumerate(self.timestamps):
            # Convert timestamp to hours and calculate position in cycle
            hours_elapsed = (timestamp - self.start_date).total_seconds() / 3600
            cycle_position = (hours_elapsed % period_hours) / period_hours * 2 * np.pi
            cycle_values[i] = np.sin(cycle_position) * amplitude * avg_value
        
        # Add cyclical pattern to original values
        self.data[sensor_name] = original_values + cycle_values
        return self
        
    def add_maintenance_events(self, maintenance_interval_days=30, 
                              maintenance_duration_hours=8, maintenance_effect=0.8):
        """
        Add maintenance events that temporarily improve machine conditions.
        
        Args:
            maintenance_interval_days: Days between scheduled maintenance
            maintenance_duration_hours: How long maintenance takes
            maintenance_effect: How much maintenance improves conditions (0-1 reduction factor)
        """
        # Create maintenance column
        self.data['maintenance'] = 0
        
        # Schedule regular maintenance
        current_date = self.start_date
        while current_date <= self.end_date:
            # Find timestamps during maintenance
            maintenance_start = current_date
            maintenance_end = current_date + timedelta(hours=maintenance_duration_hours)
            
            # Mark maintenance periods
            for i, timestamp in enumerate(self.timestamps):
                if maintenance_start <= timestamp <= maintenance_end:
                    self.data.loc[i, 'maintenance'] = 1
                    
                    # Improve sensor readings temporarily after maintenance
                    for col in self.data.columns:
                        if col not in ['timestamp', 'machine_id', 'maintenance', 'failure']:
                            # Move value closer to normal mean
                            current_val = self.data.loc[i, col]
                            normal_mean = np.mean(self.data[col].iloc[:int(len(self.data)*0.1)])  # Use first 10% as "normal"
                            self.data.loc[i, col] = current_val - (current_val - normal_mean) * maintenance_effect
            
            # Next maintenance
            current_date += timedelta(days=maintenance_interval_days)
        
        return self
        
    def add_anomaly_labels(self, threshold_multiplier=2.0):
        """
        Add anomaly labels based on sensor deviations.
        
        Args:
            threshold_multiplier: How many standard deviations to consider anomalous
        """
        # Initialize anomaly column
        self.data['anomaly'] = 0
        
        # For each sensor, detect values exceeding threshold
        sensor_columns = [col for col in self.data.columns 
                          if col not in ['timestamp', 'machine_id', 'maintenance', 'anomaly', 'failure']]
        
        for col in sensor_columns:
            # Calculate baseline stats from first 20% of data (assumed normal operation)
            baseline_data = self.data[col].iloc[:int(len(self.data)*0.2)]
            mean = np.mean(baseline_data)
            std = np.std(baseline_data)
            
            # Mark anomalies
            upper_threshold = mean + threshold_multiplier * std
            lower_threshold = mean - threshold_multiplier * std
            
            anomalies = (self.data[col] > upper_threshold) | (self.data[col] < lower_threshold)
            self.data.loc[anomalies, 'anomaly'] = 1
        
        return self
        
    def add_failure_labels(self, failure_definition=None):
        """
        Add failure labels based on sensor conditions.
        
        Args:
            failure_definition: Function that takes row of data and returns True for failure
        """
        # Initialize failure column
        self.data['failure'] = 0
        
        if failure_definition is None:
            # Default definition: 3 consecutive anomalies indicate impending failure
            window_size = 3
            for i in range(window_size, len(self.data)):
                if sum(self.data['anomaly'].iloc[i-window_size:i]) == window_size:
                    # Mark the next 24h as "failure impending"
                    future_indices = range(i, min(i + int(24 * 60 / self.sampling_interval.total_seconds() * 60), len(self.data)))
                    self.data.loc[future_indices, 'failure'] = 1
        else:
            # Apply custom failure definition
            for i in range(len(self.data)):
                if failure_definition(self.data.iloc[i]):
                    self.data.loc[i, 'failure'] = 1
        
        return self
    
    def get_data(self):
        """Return the generated data as a pandas DataFrame."""
        return self.data.copy()
    
    def save_data(self, filename):
        """Save the generated data to a CSV file."""
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return self 