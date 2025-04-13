"""
Main script to generate synthetic datasets for all machine types.
This will create training data for both normal operation and failure conditions.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
from data_generation.machine_generators import create_machine_generator
from data_generation.machine_configs import MACHINE_CONFIGS

# Output directory
DATA_DIR = "data/synthetic"

def generate_machine_datasets(machine_type, num_normal=3, num_failure=2, days=60):
    """
    Generate datasets for a specific machine type.
    
    Args:
        machine_type: Type of machine to generate data for
        num_normal: Number of normal operation datasets to generate
        num_failure: Number of failure datasets to generate per failure type
        days: Number of days of data to generate
    """
    print(f"Generating data for {machine_type}...")
    
    # Create output directory
    machine_dir = os.path.join(DATA_DIR, machine_type)
    os.makedirs(machine_dir, exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate normal operation datasets
    for i in range(num_normal):
        machine_id = f"{machine_type}_normal_{i+1}"
        print(f"  Generating normal operation data for {machine_id}")
        
        # Create generator with no failure
        generator = create_machine_generator(
            machine_type=machine_type,
            machine_id=machine_id,
            failure_type=None,
            start_date=start_date,
            end_date=end_date,
            sampling_interval_minutes=15  # One reading every 15 minutes
        )
        
        # Save data
        output_file = os.path.join(machine_dir, f"{machine_id}.csv")
        generator.save_data(output_file)
    
    # Generate failure datasets for each failure type
    failure_types = list(MACHINE_CONFIGS[machine_type]["failure_patterns"].keys())
    for failure_type in failure_types:
        for i in range(num_failure):
            machine_id = f"{machine_type}_{failure_type}_{i+1}"
            print(f"  Generating failure data for {machine_id} ({failure_type})")
            
            # Create generator with specified failure
            generator = create_machine_generator(
                machine_type=machine_type,
                machine_id=machine_id,
                failure_type=failure_type,
                start_date=start_date,
                end_date=end_date,
                sampling_interval_minutes=15  # One reading every 15 minutes
            )
            
            # Save data
            output_file = os.path.join(machine_dir, f"{machine_id}.csv")
            generator.save_data(output_file)

def merge_datasets():
    """Merge all datasets into a single file for training."""
    all_data = []
    
    # Walk through the data directory
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Reading {file_path}")
                try:
                    data = pd.read_csv(file_path)
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Combine all datasets
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(DATA_DIR, "combined_dataset.csv")
        combined_data.to_csv(combined_file, index=False)
        print(f"Combined dataset saved to {combined_file}")
        
        # Also create a more manageable training set with a subset of data
        training_sample = combined_data.sample(frac=0.3, random_state=42)
        training_file = os.path.join(DATA_DIR, "training_dataset.csv")
        training_sample.to_csv(training_file, index=False)
        print(f"Training sample dataset saved to {training_file}")

def main():
    """Main function to generate all datasets."""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate datasets for each machine type
    for machine_type in MACHINE_CONFIGS.keys():
        generate_machine_datasets(machine_type)
    
    # Merge datasets for easier model training
    merge_datasets()
    
    print("Data generation complete!")

if __name__ == "__main__":
    main() 