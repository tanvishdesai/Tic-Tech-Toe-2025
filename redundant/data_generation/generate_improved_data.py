"""
Main script to generate improved datasets for machine learning.
This integrates enhanced patterns, feature engineering, and balancing techniques.
"""

import os
import sys
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.data_generation.machine_configs import MACHINE_CONFIGS
from src.data_generation.machine_generators import create_machine_generator
from src.data_generation.enhanced_machine_patterns import enhance_machine_generator
from src.data_generation.generate_improved_datasets import add_engineered_features

# Output directory
DATA_DIR = "data/improved_synthetic"

def generate_enhanced_datasets():
    """Generate the enhanced datasets with better feature engineering and clearer patterns."""
    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Configuration for how much data to generate
    num_normal_samples = 5  # per machine type
    num_failure_samples = 8  # per failure type per machine type
    days_of_data = 90  # 3 months of data
    
    # Generate data for each machine type
    for machine_type in MACHINE_CONFIGS.keys():
        print(f"\n=== Generating enhanced data for {machine_type} ===")
        
        # Create machine-specific directory
        machine_dir = os.path.join(DATA_DIR, machine_type)
        os.makedirs(machine_dir, exist_ok=True)
        
        # Set up date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_of_data)
        
        # 1. Generate normal operation datasets
        for i in range(num_normal_samples):
            machine_id = f"{machine_type}_normal_{i+1}"
            print(f"  Generating normal operation data for {machine_id}")
            
            # Vary sampling interval for diversity
            sampling_interval = random.choice([10, 15, 20])  # minutes
            
            # Create generator
            generator = create_machine_generator(
                machine_type=machine_type,
                machine_id=machine_id,
                failure_type=None,
                start_date=start_date,
                end_date=end_date,
                sampling_interval_minutes=sampling_interval
            )
            
            # Apply enhanced patterns for more distinctive normal operation
            generator = enhance_machine_generator(generator, machine_type)
            
            # Get the raw data
            raw_data = generator.get_data()
            
            # Apply feature engineering
            enhanced_data = add_engineered_features(raw_data)
            
            # Add explicit state labels
            enhanced_data['operational_state'] = 'NORMAL'
            enhanced_data.loc[enhanced_data['maintenance'] == 1, 'operational_state'] = 'MAINTENANCE'
            
            # Detect post-maintenance states using the maintenance_decay feature
            if 'maintenance_decay' in enhanced_data.columns:
                mask = (enhanced_data['maintenance'] == 0) & (enhanced_data['maintenance_decay'] > 0.3)
                enhanced_data.loc[mask, 'operational_state'] = 'POST_MAINTENANCE'
            
            # Save dataset
            output_file = os.path.join(machine_dir, f"{machine_id}.csv")
            enhanced_data.to_csv(output_file, index=False)
            print(f"    Saved with {len(enhanced_data.columns)} features")
        
        # 2. Generate failure datasets with more distinctive patterns
        failure_types = list(MACHINE_CONFIGS[machine_type]["failure_patterns"].keys())
        
        for failure_type in failure_types:
            for i in range(num_failure_samples):
                machine_id = f"{machine_type}_{failure_type}_{i+1}"
                print(f"  Generating failure data for {machine_id} ({failure_type})")
                
                # Vary parameters for more diverse failure scenarios
                sampling_interval = random.choice([10, 15, 20])  # minutes
                
                # Randomize the scenario timeline a bit
                date_offset = random.randint(-15, 15)  # days
                adjusted_start = start_date + timedelta(days=date_offset)
                adjusted_end = end_date + timedelta(days=date_offset)
                
                # Create generator
                generator = create_machine_generator(
                    machine_type=machine_type,
                    machine_id=machine_id,
                    failure_type=failure_type,
                    start_date=adjusted_start,
                    end_date=adjusted_end,
                    sampling_interval_minutes=sampling_interval
                )
                
                # Apply enhanced patterns for more distinctive failure signatures
                generator = enhance_machine_generator(
                    generator, 
                    machine_type,
                    failure_type
                )
                
                # Get the raw data
                raw_data = generator.get_data()
                
                # Apply feature engineering
                enhanced_data = add_engineered_features(raw_data)
                
                # Add explicit operational state labels
                conditions = []
                for idx, row in enhanced_data.iterrows():
                    if row['failure'] == 1:
                        conditions.append('FAILURE')
                    elif row['anomaly'] == 1:
                        conditions.append('ANOMALY')
                    elif row['maintenance'] == 1:
                        conditions.append('MAINTENANCE')
                    elif 'maintenance_decay' in enhanced_data.columns and row['maintenance_decay'] > 0.3:
                        conditions.append('POST_MAINTENANCE')
                    else:
                        conditions.append('NORMAL')
                
                enhanced_data['operational_state'] = conditions
                
                # Mark DEGRADING states - periods before failure but not yet anomalous
                if 'failure' in enhanced_data.columns and enhanced_data['failure'].sum() > 0:
                    first_failure_idx = enhanced_data[enhanced_data['failure'] == 1].index[0]
                    lookback_window = 100  # Number of points to look back
                    
                    if first_failure_idx > lookback_window:
                        # Find points preceding failure but not yet anomalous
                        pre_failure_window = enhanced_data.iloc[first_failure_idx-lookback_window:first_failure_idx]
                        
                        # Add a gradual degrading probability feature
                        degrading_prob = np.linspace(0.1, 0.95, len(pre_failure_window))
                        
                        # Randomly select points to mark as DEGRADING based on increasing probability
                        for i, (idx, row) in enumerate(pre_failure_window.iterrows()):
                            if row['operational_state'] == 'NORMAL' and random.random() < degrading_prob[i]:
                                enhanced_data.loc[idx, 'operational_state'] = 'DEGRADING'
                
                # Save dataset
                output_file = os.path.join(machine_dir, f"{machine_id}.csv")
                enhanced_data.to_csv(output_file, index=False)
                print(f"    Saved with {len(enhanced_data.columns)} features")
                
                # Print state distribution for this dataset
                state_counts = enhanced_data['operational_state'].value_counts()
                print(f"    States: {state_counts.to_dict()}")

def create_balanced_combined_dataset():
    """Combine all datasets and create a balanced version for training."""
    print("\n=== Creating combined and balanced datasets ===")
    
    all_data = []
    
    # Walk through the data directory to collect all CSVs
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv") and not file.startswith("combined") and not file.startswith("balanced"):
                file_path = os.path.join(root, file)
                print(f"Reading {file_path}")
                
                try:
                    data = pd.read_csv(file_path)
                    
                    # Extract machine type from path
                    path_parts = os.path.normpath(root).split(os.sep)
                    if len(path_parts) >= 3:
                        machine_type = path_parts[-1]
                        data['machine_type'] = machine_type
                    
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Combine all datasets
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create the combined dataset
        combined_file = os.path.join(DATA_DIR, "combined_dataset.csv")
        combined_data.to_csv(combined_file, index=False)
        print(f"Combined dataset saved with {len(combined_data)} rows")
        
        # Create a balanced dataset with equal representation of states
        states = ['NORMAL', 'POST_MAINTENANCE', 'DEGRADING', 'ANOMALY', 'FAILURE', 'MAINTENANCE']
        
        # Get count of each state
        state_counts = {state: sum(combined_data['operational_state'] == state) for state in states}
        print("State counts in combined dataset:")
        for state, count in state_counts.items():
            print(f"  {state}: {count}")
        
        # Set samples per state - ensure at least 2000 per state through oversampling
        target_samples = 2000
        
        # Create balanced dataset
        balanced_data = []
        
        for state in states:
            state_data = combined_data[combined_data['operational_state'] == state]
            
            if len(state_data) > 0:
                if len(state_data) >= target_samples:
                    # Downsample
                    balanced_data.append(state_data.sample(target_samples, random_state=42))
                else:
                    # Oversample
                    balanced_data.append(state_data.sample(target_samples, replace=True, random_state=42))
        
        # Combine the balanced data
        if balanced_data:
            balanced_combined = pd.concat(balanced_data, ignore_index=True)
            
            # Shuffle the data
            balanced_combined = balanced_combined.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save balanced dataset
            balanced_file = os.path.join(DATA_DIR, "balanced_dataset.csv")
            balanced_combined.to_csv(balanced_file, index=False)
            
            print(f"Balanced dataset saved to {balanced_file}")
            print(f"  Shape: {balanced_combined.shape}")
            print(f"  State distribution: {balanced_combined['operational_state'].value_counts().to_dict()}")
            
            # Also create train/test split versions
            train_data = balanced_combined.sample(frac=0.8, random_state=42)
            test_data = balanced_combined.drop(train_data.index)
            
            train_file = os.path.join(DATA_DIR, "train_dataset.csv")
            test_file = os.path.join(DATA_DIR, "test_dataset.csv")
            
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)
            
            print(f"Train dataset saved with {len(train_data)} rows")
            print(f"Test dataset saved with {len(test_data)} rows")

def main():
    """Main function to generate all datasets."""
    print("Starting improved dataset generation...")
    
    # Generate enhanced datasets
    generate_enhanced_datasets()
    
    # Create combined and balanced datasets
    create_balanced_combined_dataset()
    
    print("\nImproved data generation complete!")

if __name__ == "__main__":
    main() 