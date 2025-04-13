"""
Main script to generate synthetic data and create visualizations for the predictive maintenance project.
This is the entry point to create the training datasets.
"""

import os
import sys
import time
from datetime import datetime

# Make sure we can import from the data_generation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.generate_datasets import main as generate_data
from data_generation.visualize_data import visualize_datasets

def main():
    """
    Generate synthetic data and create visualizations.
    """
    start_time = time.time()
    
    print("=" * 80)
    print(f"Predictive Maintenance Data Generation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nStep 1: Generating synthetic data for all machine types...")
    generate_data()
    
    print("\nStep 2: Creating visualizations...")
    visualize_datasets()
    
    end_time = time.time()
    print("\nCompletion Summary:")
    print("-" * 50)
    print(f"Total time elapsed: {(end_time - start_time):.2f} seconds")
    print(f"Data is available in: data/synthetic/")
    print(f"Visualizations are available in: plots/")
    print("-" * 50)
    print("\nNext steps:")
    print("1. Review the generated data and visualizations")
    print("2. Use the combined dataset for training predictive maintenance models")
    print("3. Implement real-time monitoring and anomaly detection")
    print("=" * 80)

if __name__ == "__main__":
    main() 