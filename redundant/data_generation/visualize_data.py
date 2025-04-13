"""
Visualization utilities for the synthetic maintenance data.
This script can plot sensor readings, anomalies, and failure events.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def plot_machine_data(data_file, output_dir="plots"):
    """
    Plot sensor data, anomalies, and failures for a machine.
    
    Args:
        data_file: Path to the CSV file with machine data
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    data = pd.read_csv(data_file)
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Get machine ID
    machine_id = data['machine_id'].iloc[0]
    
    # Get sensor columns (exclude metadata columns)
    metadata_cols = ['timestamp', 'machine_id', 'maintenance', 'anomaly', 'failure']
    sensor_cols = [col for col in data.columns if col not in metadata_cols]
    
    # Create subplots - one for each sensor plus one for maintenance/anomaly/failure
    fig, axes = plt.subplots(len(sensor_cols) + 1, 1, figsize=(15, 3*len(sensor_cols) + 3), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Plot each sensor
    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]
        
        # Plot sensor data
        ax.plot(data['timestamp'], data[sensor], label=sensor, color='royalblue', linewidth=1.5)
        
        # Highlight anomalies
        if 'anomaly' in data.columns:
            anomaly_points = data[data['anomaly'] == 1]
            if not anomaly_points.empty:
                ax.scatter(anomaly_points['timestamp'], anomaly_points[sensor], 
                          color='darkorange', s=30, label='Anomaly', zorder=5)
        
        # Highlight maintenance periods
        if 'maintenance' in data.columns:
            maintenance_periods = data[data['maintenance'] == 1]
            if not maintenance_periods.empty:
                for _, period in maintenance_periods.iterrows():
                    ax.axvline(x=period['timestamp'], color='green', linestyle='--', alpha=0.7)
        
        # Set title and labels
        ax.set_title(f"{sensor.replace('_', ' ').title()}")
        ax.set_ylabel(sensor.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Add legends if there are anomalies
        if 'anomaly' in data.columns and not anomaly_points.empty:
            ax.legend(loc='best')
    
    # Plot maintenance, anomaly, and failure indicators
    ax = axes[-1]
    
    # Create a color map for the status
    status = np.zeros(len(data))
    if 'maintenance' in data.columns:
        status = np.where(data['maintenance'] == 1, 1, status)  # Maintenance
    if 'anomaly' in data.columns:
        status = np.where(data['anomaly'] == 1, 2, status)  # Anomaly
    if 'failure' in data.columns:
        status = np.where(data['failure'] == 1, 3, status)  # Failure
    
    # Plot status as a heatmap
    cmap = ListedColormap(['lightgray', 'lightgreen', 'orange', 'red'])
    ax.imshow(status.reshape(1, -1), cmap=cmap, aspect='auto', 
              extent=[mdates.date2num(data['timestamp'].iloc[0]), 
                      mdates.date2num(data['timestamp'].iloc[-1]), -0.5, 0.5])
    
    # Add status legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', label='Normal'),
        Patch(facecolor='lightgreen', label='Maintenance'),
        Patch(facecolor='orange', label='Anomaly'),
        Patch(facecolor='red', label='Failure')
    ]
    ax.legend(handles=legend_elements, loc='center', ncol=4)
    
    # Set x-axis format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Set title and labels for the status plot
    ax.set_title("Machine Status")
    ax.set_yticks([])
    
    # Set overall title
    fig.suptitle(f"Sensor Data for {machine_id}", fontsize=16, y=1.02)
    
    # Save the plot
    filename = os.path.join(output_dir, f"{os.path.basename(data_file).replace('.csv', '')}_plot.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Plot saved to {filename}")
    
    return filename

def plot_comparison(machine_type, failure_type, normal_file, failure_file, output_dir="plots"):
    """
    Plot a comparison between normal operation and failure data.
    
    Args:
        machine_type: Type of machine
        failure_type: Type of failure
        normal_file: Path to normal operation data
        failure_file: Path to failure data
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    normal_data = pd.read_csv(normal_file)
    failure_data = pd.read_csv(failure_file)
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in normal_data.columns and not pd.api.types.is_datetime64_any_dtype(normal_data['timestamp']):
        normal_data['timestamp'] = pd.to_datetime(normal_data['timestamp'])
        failure_data['timestamp'] = pd.to_datetime(failure_data['timestamp'])
    
    # Get sensor columns (exclude metadata columns)
    metadata_cols = ['timestamp', 'machine_id', 'maintenance', 'anomaly', 'failure']
    sensor_cols = [col for col in normal_data.columns if col not in metadata_cols]
    
    # Create subplots - one for each sensor
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(15, 3*len(sensor_cols)), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    if len(sensor_cols) == 1:
        axes = [axes]  # Make sure axes is a list for single sensor case
    
    # Plot each sensor
    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]
        
        # Plot normal data
        ax.plot(normal_data['timestamp'], normal_data[sensor], 
                label='Normal', color='forestgreen', linewidth=1.5, alpha=0.7)
        
        # Plot failure data
        ax.plot(failure_data['timestamp'], failure_data[sensor], 
                label=failure_type.replace('_', ' ').title(), 
                color='crimson', linewidth=1.5, alpha=0.7)
        
        # Highlight failure periods
        if 'failure' in failure_data.columns:
            failure_periods = failure_data[failure_data['failure'] == 1]
            if not failure_periods.empty:
                ax.fill_between(failure_periods['timestamp'], 
                               ax.get_ylim()[0], ax.get_ylim()[1],
                               color='red', alpha=0.1, label='Failure Period')
        
        # Set title and labels
        ax.set_title(f"{sensor.replace('_', ' ').title()}")
        ax.set_ylabel(sensor.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # Set x-axis format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Set overall title
    fig.suptitle(f"Normal vs {failure_type.replace('_', ' ').title()} - {machine_type.replace('_', ' ').title()}", 
                fontsize=16, y=1.02)
    
    # Save the plot
    filename = os.path.join(output_dir, f"{machine_type}_{failure_type}_comparison.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Comparison plot saved to {filename}")
    
    return filename

def visualize_datasets(data_dir="data/synthetic", output_dir="plots"):
    """
    Visualize all datasets in the data directory.
    
    Args:
        data_dir: Directory containing the data files
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store normal and failure files by machine type
    files_by_machine = {}
    
    # Walk through the data directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv") and "combined" not in file and "training" not in file:
                file_path = os.path.join(root, file)
                
                # Extract machine type and whether it's normal or a failure
                filename = os.path.basename(file_path)
                parts = filename.replace(".csv", "").split("_")
                
                if len(parts) >= 3 and parts[0] in ['siemens', 'abb', 'haas', 'grundfos', 'carrier']:
                    machine_type = f"{parts[0]}_{parts[1]}"
                    
                    # Determine if it's normal or a failure file
                    if "normal" in filename:
                        if machine_type not in files_by_machine:
                            files_by_machine[machine_type] = {"normal": [], "failures": {}}
                        files_by_machine[machine_type]["normal"].append(file_path)
                    else:
                        failure_type = parts[2]
                        if machine_type not in files_by_machine:
                            files_by_machine[machine_type] = {"normal": [], "failures": {}}
                        if failure_type not in files_by_machine[machine_type]["failures"]:
                            files_by_machine[machine_type]["failures"][failure_type] = []
                        files_by_machine[machine_type]["failures"][failure_type].append(file_path)
                
                # Generate individual plots
                print(f"Visualizing {file_path}")
                plot_machine_data(file_path, output_dir)
    
    # Generate comparison plots
    for machine_type, files in files_by_machine.items():
        if files["normal"] and files["failures"]:
            normal_file = files["normal"][0]
            
            for failure_type, failure_files in files["failures"].items():
                if failure_files:
                    failure_file = failure_files[0]
                    print(f"Creating comparison plot for {machine_type} - {failure_type}")
                    plot_comparison(machine_type, failure_type, normal_file, failure_file, output_dir)

if __name__ == "__main__":
    # Visualize all datasets
    visualize_datasets() 