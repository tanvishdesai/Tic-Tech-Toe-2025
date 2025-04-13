"""
Test script for predictive maintenance model inference.
This script creates synthetic test data and tests the model's predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.predictive_maintenance_model import PredictiveMaintenanceModel
import os

# Create a directory for test outputs if it doesn't exist
os.makedirs('test_outputs', exist_ok=True)

def generate_synthetic_test_data(machine_type, num_samples=100):
    """
    Generate synthetic test data for a specific machine type.
    
    Args:
        machine_type: The type of machine to generate data for
        num_samples: Number of test samples to generate
        
    Returns:
        DataFrame containing synthetic test data
    """
    # Create timestamp range
    timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='H')
    
    # Create base dataframe with timestamps
    data = pd.DataFrame({'timestamp': timestamps})
    
    # Add machine ID
    data['machine_id'] = f"{machine_type}_test_001"
    
    # Initialize model to get normal ranges
    model = PredictiveMaintenanceModel()
    
    # Generate data based on machine type
    if machine_type == "siemens_motor":
        # Get normal ranges for this machine type
        ranges = model.normal_ranges[machine_type]
        
        # Generate values within normal range for most samples
        data['temperature'] = np.random.uniform(
            ranges['temperature'][0], 
            ranges['temperature'][1], 
            size=num_samples
        )
        data['vibration'] = np.random.uniform(
            ranges['vibration'][0], 
            ranges['vibration'][1], 
            size=num_samples
        )
        data['current'] = np.random.uniform(
            ranges['current'][0], 
            ranges['current'][1], 
            size=num_samples
        )
        data['voltage'] = np.random.uniform(
            ranges['voltage'][0], 
            ranges['voltage'][1], 
            size=num_samples
        )
        
        # Create a few anomalous samples
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
        
        # For anomalies, increase temperature and vibration
        data.loc[anomaly_indices, 'temperature'] *= 1.2
        data.loc[anomaly_indices, 'vibration'] *= 1.5
        
        # Mark half of the anomalies as failures
        failure_indices = np.random.choice(anomaly_indices, size=len(anomaly_indices)//2, replace=False)
        
        # Initialize failure and anomaly flags
        data['failure'] = 0
        data['anomaly'] = 0
        data['maintenance'] = 0
        
        # Set flags for anomalies and failures
        data.loc[anomaly_indices, 'anomaly'] = 1
        data.loc[failure_indices, 'failure'] = 1
        
    elif machine_type == "abb_bearing":
        # Get normal ranges for this machine type
        ranges = model.normal_ranges[machine_type]
        
        # Generate values within normal range for most samples
        data['vibration'] = np.random.uniform(
            ranges['vibration'][0], 
            ranges['vibration'][1], 
            size=num_samples
        )
        data['temperature'] = np.random.uniform(
            ranges['temperature'][0], 
            ranges['temperature'][1], 
            size=num_samples
        )
        data['acoustic'] = np.random.uniform(
            ranges['acoustic'][0], 
            ranges['acoustic'][1], 
            size=num_samples
        )
        
        # Create a few anomalous samples
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
        
        # For anomalies, increase vibration and acoustic levels
        data.loc[anomaly_indices, 'vibration'] *= 1.7
        data.loc[anomaly_indices, 'acoustic'] *= 1.3
        
        # Mark half of the anomalies as failures
        failure_indices = np.random.choice(anomaly_indices, size=len(anomaly_indices)//2, replace=False)
        
        # Initialize failure and anomaly flags
        data['failure'] = 0
        data['anomaly'] = 0
        data['maintenance'] = 0
        
        # Set flags for anomalies and failures
        data.loc[anomaly_indices, 'anomaly'] = 1
        data.loc[failure_indices, 'failure'] = 1
    
    elif machine_type == "haas_cnc":
        # Get normal ranges for this machine type
        ranges = model.normal_ranges[machine_type]
        
        # Generate values within normal range for most samples
        data['spindle_load'] = np.random.uniform(
            ranges['spindle_load'][0], 
            ranges['spindle_load'][1], 
            size=num_samples
        )
        data['vibration'] = np.random.uniform(
            ranges['vibration'][0], 
            ranges['vibration'][1], 
            size=num_samples
        )
        data['temperature'] = np.random.uniform(
            ranges['temperature'][0], 
            ranges['temperature'][1], 
            size=num_samples
        )
        data['acoustic'] = np.random.uniform(
            ranges['acoustic'][0], 
            ranges['acoustic'][1], 
            size=num_samples
        )
        
        # Create a few anomalous samples
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
        
        # For anomalies, increase spindle load, temperature and vibration
        data.loc[anomaly_indices, 'spindle_load'] *= 1.3
        data.loc[anomaly_indices, 'temperature'] *= 1.2
        data.loc[anomaly_indices, 'vibration'] *= 1.6
        
        # Mark half of the anomalies as failures
        failure_indices = np.random.choice(anomaly_indices, size=len(anomaly_indices)//2, replace=False)
        
        # Initialize failure and anomaly flags
        data['failure'] = 0
        data['anomaly'] = 0
        data['maintenance'] = 0
        
        # Set flags for anomalies and failures
        data.loc[anomaly_indices, 'anomaly'] = 1
        data.loc[failure_indices, 'failure'] = 1
    
    elif machine_type == "grundfos_pump":
        # Get normal ranges for this machine type
        ranges = model.normal_ranges[machine_type]
        
        # Generate values within normal range for most samples
        data['pressure'] = np.random.uniform(
            ranges['pressure'][0], 
            ranges['pressure'][1], 
            size=num_samples
        )
        data['flow_rate'] = np.random.uniform(
            ranges['flow_rate'][0], 
            ranges['flow_rate'][1], 
            size=num_samples
        )
        data['temperature'] = np.random.uniform(
            ranges['temperature'][0], 
            ranges['temperature'][1], 
            size=num_samples
        )
        data['power'] = np.random.uniform(
            ranges['power'][0], 
            ranges['power'][1], 
            size=num_samples
        )
        
        # Create a few anomalous samples
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
        
        # For anomalies, decrease flow rate and increase power consumption
        data.loc[anomaly_indices, 'flow_rate'] *= 0.7
        data.loc[anomaly_indices, 'power'] *= 1.3
        data.loc[anomaly_indices, 'temperature'] *= 1.2
        
        # Mark half of the anomalies as failures
        failure_indices = np.random.choice(anomaly_indices, size=len(anomaly_indices)//2, replace=False)
        
        # Initialize failure and anomaly flags
        data['failure'] = 0
        data['anomaly'] = 0
        data['maintenance'] = 0
        
        # Set flags for anomalies and failures
        data.loc[anomaly_indices, 'anomaly'] = 1
        data.loc[failure_indices, 'failure'] = 1
    
    elif machine_type == "carrier_chiller":
        # Get normal ranges for this machine type
        ranges = model.normal_ranges[machine_type]
        
        # Generate values within normal range for most samples
        data['refrigerant_pressure'] = np.random.uniform(
            ranges['refrigerant_pressure'][0], 
            ranges['refrigerant_pressure'][1], 
            size=num_samples
        )
        data['condenser_temp'] = np.random.uniform(
            ranges['condenser_temp'][0], 
            ranges['condenser_temp'][1], 
            size=num_samples
        )
        data['evaporator_temp'] = np.random.uniform(
            ranges['evaporator_temp'][0], 
            ranges['evaporator_temp'][1], 
            size=num_samples
        )
        data['power'] = np.random.uniform(
            ranges['power'][0], 
            ranges['power'][1], 
            size=num_samples
        )
        
        # Create a few anomalous samples
        anomaly_indices = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
        
        # For anomalies, increase refrigerant pressure and condenser temperature
        data.loc[anomaly_indices, 'refrigerant_pressure'] *= 1.3
        data.loc[anomaly_indices, 'condenser_temp'] *= 1.2
        data.loc[anomaly_indices, 'power'] *= 1.15
        
        # Mark half of the anomalies as failures
        failure_indices = np.random.choice(anomaly_indices, size=len(anomaly_indices)//2, replace=False)
        
        # Initialize failure and anomaly flags
        data['failure'] = 0
        data['anomaly'] = 0
        data['maintenance'] = 0
        
        # Set flags for anomalies and failures
        data.loc[anomaly_indices, 'anomaly'] = 1
        data.loc[failure_indices, 'failure'] = 1
    
    else:
        raise ValueError(f"Unknown machine type: {machine_type}")
    
    return data

def test_model_inference(machine_type):
    """
    Test model inference for a specific machine type.
    
    Args:
        machine_type: The type of machine to test
    """
    print(f"\nTesting predictive maintenance model for {machine_type}...")
    
    # Generate synthetic test data
    test_data = generate_synthetic_test_data(machine_type, num_samples=200)
    
    # Save test data for reference
    test_data.to_csv(f'test_outputs/{machine_type}_test_data.csv', index=False)
    print(f"Generated {len(test_data)} test samples for {machine_type}")
    
    # Create and train a new model
    model = PredictiveMaintenanceModel(model_type='ensemble')
    
    # Train directly on the test data
    # In a real-world scenario, you would load a pre-trained model or train on separate train data
    # Create temporary combined dataset
    train_data_path = "test_outputs/temp_train_data.csv"
    test_data.to_csv(train_data_path, index=False)
    model.data_path = train_data_path
    
    # Train the model
    model.load_data()
    model.preprocess_data()
    eval_results = model.train_and_evaluate(machine_type)
    
    if eval_results:
        print(f"Model training results:")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
        print(f"PR-AUC: {eval_results['pr_auc']:.4f}")
        
        # Get model's predictions on the test data
        # Extract features and target from test data
        machine_data = model.get_machine_specific_data(machine_type)
        X, y_true = model.prepare_features_and_target(machine_data, machine_type)
        
        # Get predictions
        trained_model = eval_results['model']
        y_pred = trained_model.predict(X)
        y_prob = trained_model.predict_proba(X)[:, 1]
        
        # Create a dataframe with results
        results_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'probability': y_prob
        })
        
        # Add some of the original features for reference
        if machine_type == "siemens_motor":
            for col in ['temperature', 'vibration', 'current', 'voltage']:
                if col in machine_data.columns:
                    results_df[col] = machine_data[col].values
        elif machine_type == "abb_bearing":
            for col in ['vibration', 'temperature', 'acoustic']:
                if col in machine_data.columns:
                    results_df[col] = machine_data[col].values
        elif machine_type == "haas_cnc":
            for col in ['spindle_load', 'vibration', 'temperature', 'acoustic']:
                if col in machine_data.columns:
                    results_df[col] = machine_data[col].values
        elif machine_type == "grundfos_pump":
            for col in ['pressure', 'flow_rate', 'temperature', 'power']:
                if col in machine_data.columns:
                    results_df[col] = machine_data[col].values
        elif machine_type == "carrier_chiller":
            for col in ['refrigerant_pressure', 'condenser_temp', 'evaporator_temp', 'power']:
                if col in machine_data.columns:
                    results_df[col] = machine_data[col].values
        
        # Save results
        results_df.to_csv(f'test_outputs/{machine_type}_prediction_results.csv', index=False)
        
        # Print confusion matrix and misclassifications
        print("\nTest data prediction results:")
        tp = ((results_df['predicted'] == 1) & (results_df['actual'] == 1)).sum()
        tn = ((results_df['predicted'] == 0) & (results_df['actual'] == 0)).sum()
        fp = ((results_df['predicted'] == 1) & (results_df['actual'] == 0)).sum()
        fn = ((results_df['predicted'] == 0) & (results_df['actual'] == 1)).sum()
        
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        
        # Find interesting misclassifications
        false_positives = results_df[(results_df['predicted'] == 1) & (results_df['actual'] == 0)]
        false_negatives = results_df[(results_df['predicted'] == 0) & (results_df['actual'] == 1)]
        
        if len(false_positives) > 0:
            print("\nSample False Positives (normal cases predicted as failures):")
            print(false_positives.head(3))
        
        if len(false_negatives) > 0:
            print("\nSample False Negatives (failures predicted as normal):")
            print(false_negatives.head(3))
        
        return eval_results
    else:
        print(f"Failed to train model for {machine_type}")
        return None

if __name__ == "__main__":
    # Test inference for each machine type
    machine_types = [
        "siemens_motor",
        "abb_bearing",
        # Uncomment to test other machine types
        "haas_cnc",
        "grundfos_pump",
        "carrier_chiller"
    ]
    
    results = {}
    for machine_type in machine_types:
        try:
            results[machine_type] = test_model_inference(machine_type)
        except Exception as e:
            print(f"Error testing {machine_type}: {str(e)}")
    
    # Print summary
    print("\nTest Summary:")
    for machine_type, result in results.items():
        if result:
            print(f"{machine_type}: Accuracy = {result['accuracy']:.4f}, PR-AUC = {result['pr_auc']:.4f}")
        else:
            print(f"{machine_type}: Testing failed") 