import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load training dataset
print("Loading training dataset...")
df = pd.read_csv('data/synthetic/training_dataset.csv')

# Basic information
print(f"Dataset shape: {df.shape}")
print("\nColumns:", df.columns.tolist())

# Check class distribution
print("\nFailure distribution:")
failure_counts = df['failure'].value_counts()
print(failure_counts)
print(f"Failure rate: {failure_counts.get(1, 0) / len(df):.4f}")

# Check machine type distribution
print("\nMachine type distribution:")
df['machine_type'] = df['machine_id'].apply(lambda x: x.split('_')[0])
print(df['machine_type'].value_counts())

# Check the number of unique machines
print(f"\nNumber of unique machines: {df['machine_id'].nunique()}")

# Basic statistics for sensor data by machine type
print("\nBasic statistics by machine type:")
for machine_type in df['machine_type'].unique():
    print(f"\n{machine_type.upper()} statistics:")
    machine_df = df[df['machine_type'] == machine_type]
    
    # Get numeric columns for this machine type
    numeric_cols = machine_df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['failure', 'maintenance', 'anomaly']]
    
    # Print statistics for each sensor
    for col in numeric_cols:
        if machine_df[col].count() > 0:  # Only include columns with data
            print(f"  {col}: mean={machine_df[col].mean():.2f}, std={machine_df[col].std():.2f}, min={machine_df[col].min():.2f}, max={machine_df[col].max():.2f}")
    
    # Check failure rate for this machine type
    failure_rate = machine_df['failure'].mean()
    print(f"  Failure rate: {failure_rate:.4f}")

# Write sample records for each machine type
print("\nSample records for each machine type:")
for machine_type in df['machine_type'].unique():
    samples = df[df['machine_type'] == machine_type].sample(min(3, len(df[df['machine_type'] == machine_type])))
    print(f"\n{machine_type.upper()} samples:")
    print(samples[['machine_id', 'failure'] + 
                 [col for col in samples.columns if col not in ['timestamp', 'machine_id', 'failure', 'maintenance', 'anomaly', 'machine_type'] and not pd.isna(samples[col]).all()]]) 