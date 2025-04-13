"""
Enhanced script to generate improved synthetic datasets for all machine types.
This version includes better feature engineering to distinguish between operational states,
more failure data, and improved parameters for challenging scenarios.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.data_generation.machine_generators import create_machine_generator
from src.data_generation.machine_configs import MACHINE_CONFIGS

# Output directory
DATA_DIR = "data/improved_synthetic"

def add_engineered_features(df):
    """
    Add engineered features to better distinguish between operational states.
    
    Args:
        df: DataFrame with raw sensor data
    
    Returns:
        DataFrame with added features
    """
    # Keep a copy of the original data
    enhanced_df = df.copy()
    
    # Get sensor columns (excluding metadata columns)
    sensor_cols = [col for col in df.columns if col not in ['timestamp', 'machine_id', 
                                                            'maintenance', 'anomaly', 'failure']]
    
    # 1. Add rolling statistics for better state detection
    window_sizes = [5, 10, 20]
    for window in window_sizes:
        # Rolling mean and std for each sensor
        for col in sensor_cols:
            enhanced_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            enhanced_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    
    # 2. Add rate of change features (first derivative)
    for col in sensor_cols:
        enhanced_df[f'{col}_rate'] = df[col].diff().fillna(0)
        
        # Also add smoothed rate of change (reduces noise)
        enhanced_df[f'{col}_smooth_rate'] = enhanced_df[f'{col}_rate'].rolling(window=5, min_periods=1).mean()
    
    # 3. Add acceleration features (second derivative)
    for col in sensor_cols:
        enhanced_df[f'{col}_acceleration'] = enhanced_df[f'{col}_rate'].diff().fillna(0)
    
    # 4. Calculate entropy for each sensor on rolling windows (randomness indicator)
    def entropy(series):
        # Simplified entropy calculation
        # Binning values into 10 bins and calculating entropy
        try:
            counts = np.histogram(series, bins=10)[0]
            counts = counts[counts > 0]  # Remove zeros
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs))
        except:
            return 0
    
    for col in sensor_cols:
        enhanced_df[f'{col}_entropy'] = df[col].rolling(window=20, min_periods=5).apply(
            entropy, raw=True
        ).fillna(0)
    
    # 5. Add post-maintenance indicators with decay
    if 'maintenance' in df.columns:
        # Create a feature that decays after maintenance
        maintenance_decay = np.zeros(len(df))
        decay_rate = 0.95  # Slower decay to better track post-maintenance state
        
        for i in range(len(df)):
            if df['maintenance'].iloc[i] == 1:
                maintenance_decay[i] = 1.0  # Reset to 1 during maintenance
            elif i > 0:
                # Exponential decay from previous value
                maintenance_decay[i] = maintenance_decay[i-1] * decay_rate
        
        enhanced_df['maintenance_decay'] = maintenance_decay
        
        # Add interaction features between maintenance_decay and key sensors
        for col in sensor_cols:
            enhanced_df[f'{col}_post_maint'] = enhanced_df[col] * enhanced_df['maintenance_decay']
    
    # 6. Add distance from normal operation baseline
    # Calculate baseline from the first 10% of data (assumed to be normal operation)
    baseline_size = int(len(df) * 0.1)
    if baseline_size > 0:
        baselines = {col: df[col].iloc[:baseline_size].mean() for col in sensor_cols}
        stds = {col: df[col].iloc[:baseline_size].std() for col in sensor_cols}
        
        # Calculate normalized distance from baseline (z-score)
        for col in sensor_cols:
            if stds[col] > 0:
                enhanced_df[f'{col}_baseline_dist'] = (df[col] - baselines[col]) / stds[col]
            else:
                enhanced_df[f'{col}_baseline_dist'] = df[col] - baselines[col]
    
    # 7. Add clear failure state indicators
    if 'failure' in df.columns:
        # Add exponential growth indicator near failure points
        failure_growth = np.zeros(len(df))
        growth_rate = 1.2  # Faster growth for clearer failure detection
        
        # Backward pass - start from the end
        for i in range(len(df) - 1, -1, -1):
            if df['failure'].iloc[i] == 1:
                failure_growth[i] = 1.0
            elif i < len(df) - 1:
                # Use backward growth to create a leading indicator
                failure_growth[i] = min(1.0, failure_growth[i+1] / growth_rate)
        
        enhanced_df['failure_proximity'] = failure_growth
    
    return enhanced_df

def generate_machine_datasets(machine_type, num_normal=5, num_failure=8, days=90):
    """
    Generate improved datasets with better separation between operational states.
    
    Args:
        machine_type: Type of machine to generate data for
        num_normal: Number of normal operation datasets to generate
        num_failure: Number of failure datasets to generate per failure type
        days: Number of days of data to generate
    """
    print(f"Generating improved data for {machine_type}...")
    
    # Create output directory
    machine_dir = os.path.join(DATA_DIR, machine_type)
    os.makedirs(machine_dir, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate normal operation datasets with varied conditions
    for i in range(num_normal):
        machine_id = f"{machine_type}_normal_{i+1}"
        print(f"  Generating normal operation data for {machine_id}")
        
        # Create generator with no failure but varied parameters
        # Small random variation in sampling to create more diverse normal data
        sampling_interval = random.choice([10, 15, 20])  # minutes between readings
        
        generator = create_machine_generator(
            machine_type=machine_type,
            machine_id=machine_id,
            failure_type=None,
            start_date=start_date,
            end_date=end_date,
            sampling_interval_minutes=sampling_interval
        )
        
        # Get the raw data
        raw_data = generator.get_data()
        
        # Add engineered features
        enhanced_data = add_engineered_features(raw_data)
        
        # Save enhanced data
        output_file = os.path.join(machine_dir, f"{machine_id}.csv")
        enhanced_data.to_csv(output_file, index=False)
        print(f"    Saved with {len(enhanced_data.columns)} features")
    
    # Generate failure datasets for each failure type with more variation
    failure_types = list(MACHINE_CONFIGS[machine_type]["failure_patterns"].keys())
    for failure_type in failure_types:
        for i in range(num_failure):
            machine_id = f"{machine_type}_{failure_type}_{i+1}"
            print(f"  Generating failure data for {machine_id} ({failure_type})")
            
            # Vary the sampling interval
            sampling_interval = random.choice([10, 15, 20])  # minutes
            
            # Randomize failure timing and progression speed for more diverse scenarios
            start_date_offset = random.randint(-10, 10)  # days
            adjusted_start = start_date + timedelta(days=start_date_offset)
            adjusted_end = end_date + timedelta(days=start_date_offset)
            
            # Create generator with specified failure
            generator = create_machine_generator(
                machine_type=machine_type,
                machine_id=machine_id,
                failure_type=failure_type,
                start_date=adjusted_start,
                end_date=adjusted_end,
                sampling_interval_minutes=sampling_interval
            )
            
            # Get the raw data
            raw_data = generator.get_data()
            
            # Add engineered features
            enhanced_data = add_engineered_features(raw_data)
            
            # Save enhanced data
            output_file = os.path.join(machine_dir, f"{machine_id}.csv")
            enhanced_data.to_csv(output_file, index=False)
            print(f"    Saved with {len(enhanced_data.columns)} features")

def create_labeled_dataset():
    """Create a well-balanced labeled dataset with clear state labels."""
    all_data = []
    
    # Walk through the data directory
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Reading {file_path}")
                try:
                    data = pd.read_csv(file_path)
                    
                    # Extract machine type and condition from filename
                    file_parts = os.path.basename(file_path).replace('.csv', '').split('_')
                    machine_type = file_parts[0]
                    
                    # Add a clear state label column for training
                    data['machine_type'] = machine_type
                    
                    # Define operational state more clearly for training
                    # This creates a single categorical target variable for training
                    conditions = []
                    for _, row in data.iterrows():
                        if row['failure'] == 1:
                            conditions.append('FAILURE')
                        elif row['anomaly'] == 1:
                            conditions.append('ANOMALY')
                        elif row['maintenance'] == 1:
                            conditions.append('MAINTENANCE')
                        elif row.get('maintenance_decay', 0) > 0.3:  # If we have this engineered feature
                            conditions.append('POST_MAINTENANCE')
                        else:
                            conditions.append('NORMAL')
                    
                    data['operational_state'] = conditions
                    
                    # Additional checks for detecting DEGRADING state
                    # Look for consistent degradation pattern in key sensors
                    if 'failure' in data.columns and any(data['failure'] == 1):
                        # Look at the 100 data points before failure
                        failure_idx = data[data['failure'] == 1].index[0]
                        if failure_idx > 100:
                            pre_failure_window = data.iloc[failure_idx-100:failure_idx]
                            
                            # Check if this is a period of degradation
                            for i, row in pre_failure_window.iterrows():
                                if row['operational_state'] == 'NORMAL' and row['anomaly'] == 0:
                                    # Mark as degrading if close to failure but not yet anomalous
                                    data.loc[i, 'operational_state'] = 'DEGRADING'
                    
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Combine all datasets
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create stratified datasets
        combined_file = os.path.join(DATA_DIR, "combined_dataset.csv")
        combined_data.to_csv(combined_file, index=False)
        print(f"Combined dataset saved to {combined_file}")
        
        # Create balanced training data with equal representation of operational states
        states = ['NORMAL', 'POST_MAINTENANCE', 'DEGRADING', 'ANOMALY', 'FAILURE', 'MAINTENANCE']
        min_state_count = min([sum(combined_data['operational_state'] == state) for state in states])
        
        # Ensure minimum of 1000 samples per state, or the maximum available
        samples_per_state = max(1000, min_state_count)
        print(f"Using {samples_per_state} samples per operational state for balanced dataset")
        
        balanced_data = []
        for state in states:
            state_data = combined_data[combined_data['operational_state'] == state]
            if len(state_data) > samples_per_state:
                # Stratified sampling within each operational state
                balanced_data.append(state_data.sample(samples_per_state, random_state=42))
            else:
                # If not enough samples, use all and oversample
                if len(state_data) > 0:
                    oversampled = state_data.sample(samples_per_state, replace=True, random_state=42)
                    balanced_data.append(oversampled)
        
        if balanced_data:
            balanced_combined = pd.concat(balanced_data, ignore_index=True)
            balanced_file = os.path.join(DATA_DIR, "balanced_dataset.csv")
            balanced_combined.to_csv(balanced_file, index=False)
            print(f"Balanced dataset saved to {balanced_file}")
            print(f"  Shape: {balanced_combined.shape}")
            print(f"  State distribution: {balanced_combined['operational_state'].value_counts().to_dict()}")

def main():
    """Main function to generate all improved datasets."""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate datasets for each machine type with improved parameters
    for machine_type in MACHINE_CONFIGS.keys():
        generate_machine_datasets(machine_type)
    
    # Create a combined labeled dataset with improved class balancing
    create_labeled_dataset()
    
    print("Improved data generation complete!")

if __name__ == "__main__":
    main() 