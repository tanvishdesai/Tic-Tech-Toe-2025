import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import sys
import warnings
from tabulate import tabulate
import json

# Import the model class
from predictive_maintenance_modelV2 import PredictiveMaintenanceModelV2

# Suppress warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceModelTester:
    """
    A comprehensive tester for the PredictiveMaintenanceModelV2 that evaluates model performance
    across various scenarios for each machine type.
    """
    
    def __init__(self, models_dir='models', test_data_path='data/improved_synthetic/test_dataset.csv'):
        """
        Initialize the model tester.
        
        Args:
            models_dir: Directory containing the saved models
            test_data_path: Path to the test dataset
        """
        self.models_dir = models_dir
        self.test_data_path = test_data_path
        self.model = PredictiveMaintenanceModelV2()
        self.test_df = None
        self.results = {}
        self.scenario_results = {}
        
    def load_models(self):
        """Load the trained models."""
        print("Loading trained models...")
        success = self.model.load_models(self.models_dir)
        
        if not success:
            print("Failed to load models. Please ensure models are trained and saved properly.")
            sys.exit(1)
            
        print(f"Successfully loaded models for machine types: {', '.join(self.model.machine_types)}")
        
    def load_test_data(self):
        """Load the test dataset."""
        try:
            print(f"Loading test data from {self.test_data_path}...")
            self.test_df = pd.read_csv(self.test_data_path)
            self.test_df['timestamp'] = pd.to_datetime(self.test_df['timestamp'])
            
            # Extract machine type from machine_id using model's method
            self.test_df['machine_type'] = self.test_df['machine_id'].apply(self.model.get_standardized_machine_type)
            
            print(f"Loaded test data: {len(self.test_df)} rows")
            print(f"Unique machine types in test data: {self.test_df['machine_type'].unique()}")
            
            # If no test data exists, create synthetic test data
            if len(self.test_df) == 0 or self.test_df is None:
                print("No test data found. Creating synthetic test data...")
                self.create_synthetic_test_data()
                
        except FileNotFoundError:
            print(f"Test data file not found at {self.test_data_path}. Creating synthetic test data...")
            self.create_synthetic_test_data()
            
    def create_synthetic_test_data(self):
        """Create synthetic test data for all machine types."""
        print("Generating synthetic test data for all machine types...")
        
        # Get machine types from the model
        machine_types = self.model.machine_types
        
        # Create an empty DataFrame for the test data
        self.test_df = pd.DataFrame()
        
        # Define time range for test data
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        time_range = pd.date_range(start=start_time, end=end_time, freq='15min')
        
        # Generate synthetic data for each machine type
        for machine_type in machine_types:
            print(f"Generating data for {machine_type}...")
            
            # Get the standard machine type if it's a variant
            std_machine_type = machine_type.split('_')[0] if '_' in machine_type else machine_type
            
            # Get relevant sensors for this machine type
            relevant_sensors = self.model.MACHINE_SENSOR_MAPPING.get(std_machine_type, [])
            
            if not relevant_sensors:
                print(f"  No sensor mapping found for {machine_type}, skipping")
                continue
                
            # Create multiple machines of this type with different scenarios
            for scenario in ['normal', 'degrading', 'anomaly', 'failure', 'post_maintenance']:
                machine_id = f"{machine_type}_{scenario}_test"
                
                # Generate data points for this machine
                for timestamp in time_range:
                    # Create a base record
                    record = {
                        'machine_id': machine_id,
                        'timestamp': timestamp,
                        'failure': 0,
                        'anomaly': 0,
                        'maintenance': 0
                    }
                    
                    # Add sensor values based on the scenario
                    for sensor in relevant_sensors:
                        base_value = self._get_base_sensor_value(sensor)
                        variance = base_value * 0.1  # 10% variance
                        
                        # Modify values based on scenario
                        if scenario == 'normal':
                            # Normal operation with slight random variations
                            value = base_value + np.random.normal(0, variance)
                        
                        elif scenario == 'degrading':
                            # Gradually degrading performance over time
                            progress = (timestamp - start_time) / (end_time - start_time)
                            degradation_factor = 1.0 + (progress * 0.5)  # Up to 50% degradation
                            value = base_value * degradation_factor + np.random.normal(0, variance)
                        
                        elif scenario == 'anomaly':
                            # Occasional anomalies
                            if np.random.random() < 0.2:  # 20% chance of anomaly
                                record['anomaly'] = 1
                                value = base_value * (1.5 + np.random.random())  # 50-150% increase
                            else:
                                value = base_value + np.random.normal(0, variance)
                        
                        elif scenario == 'failure':
                            # Leading to failure at the end
                            progress = (timestamp - start_time) / (end_time - start_time)
                            if progress > 0.9:  # Failure in last 10% of timeline
                                record['failure'] = 1
                                value = base_value * (2.0 + np.random.random())  # Severe deviation
                            else:
                                # Gradually worsening
                                degradation_factor = 1.0 + (progress * 1.0)  # Up to 100% degradation
                                value = base_value * degradation_factor + np.random.normal(0, variance)
                        
                        elif scenario == 'post_maintenance':
                            # Maintenance event followed by improved performance
                            mid_point = start_time + (end_time - start_time) / 2
                            
                            if timestamp.date() == mid_point.date():
                                record['maintenance'] = 1
                                value = base_value * 0.9  # Slightly better than base after maintenance
                            elif timestamp < mid_point:
                                # Degrading before maintenance
                                progress = (timestamp - start_time) / (mid_point - start_time)
                                degradation_factor = 1.0 + (progress * 0.4)  # Up to 40% degradation
                                value = base_value * degradation_factor + np.random.normal(0, variance)
                            else:
                                # Improved after maintenance
                                value = base_value * 0.9 + np.random.normal(0, variance * 0.5)
                        
                        record[sensor] = max(0, value)  # Ensure non-negative values
                    
                    # Add record to test_df
                    self.test_df = pd.concat([self.test_df, pd.DataFrame([record])], ignore_index=True)
        
        # Ensure proper column types
        for col in self.test_df.columns:
            if col not in ['machine_id', 'timestamp', 'machine_type']:
                self.test_df[col] = pd.to_numeric(self.test_df[col], errors='coerce')
        
        # Add machine_type column
        self.test_df['machine_type'] = self.test_df['machine_id'].apply(self.model.get_standardized_machine_type)
        
        print(f"Created synthetic test data with {len(self.test_df)} rows")
        
        # Save the synthetic test data
        os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
        self.test_df.to_csv(self.test_data_path, index=False)
        print(f"Saved synthetic test data to {self.test_data_path}")
        
    def _get_base_sensor_value(self, sensor):
        """Helper method to get a reasonable base value for each sensor type."""
        sensor_value_map = {
            'temperature': 70,         # degrees
            'vibration': 0.5,          # mm/s
            'current': 15,             # amperes
            'voltage': 220,            # volts
            'acoustic': 65,            # dB
            'spindle_load': 50,        # percent
            'pressure': 5.5,           # bar
            'flow_rate': 25,           # liters/minute
            'power': 75,               # kW
            'refrigerant_pressure': 8, # bar
            'condenser_temp': 40,      # degrees
            'evaporator_temp': 10      # degrees
        }
        
        return sensor_value_map.get(sensor, 50)  # Default value if sensor not in map
        
    def test_all_machines(self):
        """Test predictions for all machines in the test data."""
        if self.test_df is None:
            print("Test data not loaded. Please load test data first.")
            return
            
        print("\nTesting model predictions for all machines...")
        
        # Create a results dictionary to store metrics for each machine type
        self.results = {
            'machine_type': [],
            'num_samples': [],
            'health_score_mae': [],
            'health_score_rmse': [],
            'health_score_r2': [],
            'days_until_maintenance_mae': [],
            'days_until_maintenance_rmse': [],
            'days_until_maintenance_r2': [],
            'overall_score': []
        }
        
        # Process each machine type
        for machine_type in self.model.machine_types:
            print(f"\nTesting {machine_type}...")
            
            # Filter test data for this machine type
            machine_data = self.test_df[self.test_df['machine_type'] == machine_type]
            
            if len(machine_data) == 0:
                print(f"  No test data found for {machine_type}, skipping")
                continue
                
            # Group by machine_id to test each machine separately
            machine_groups = machine_data.groupby('machine_id')
            
            # Lists to store actual vs predicted values for this machine type
            actual_health = []
            predicted_health = []
            actual_maintenance = []
            predicted_maintenance = []
            
            # Process each machine
            for machine_id, group in tqdm(machine_groups, desc=f"Testing machines of type {machine_type}"):
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Skip machines with too few data points
                if len(group) < 12:  # Need at least a few hours of data
                    continue
                    
                # Get ground truth values for evaluation
                # Set health score as inverse of failure probability
                actual_health_score = 100 - (group['failure'].iloc[-1] * 100)
                
                # Estimate days until maintenance based on failure/anomaly flags
                if group['failure'].iloc[-1] == 1:
                    actual_maintenance_days = 0
                elif group['anomaly'].iloc[-1] == 1:
                    actual_maintenance_days = 7
                else:
                    # Estimate based on simulated degradation
                    if 'normal' in machine_id:
                        actual_maintenance_days = 30
                    elif 'degrading' in machine_id:
                        actual_maintenance_days = 14
                    elif 'post_maintenance' in machine_id:
                        # Check if the latest record is after maintenance
                        if group['maintenance'].sum() > 0:
                            last_maintenance = group[group['maintenance'] == 1].iloc[-1]['timestamp']
                            if group['timestamp'].iloc[-1] > last_maintenance:
                                actual_maintenance_days = 30
                            else:
                                actual_maintenance_days = 10
                        else:
                            actual_maintenance_days = 20
                    else:
                        actual_maintenance_days = 20
                
                # Make prediction using the model
                prediction = self.model.predict_machine_health(group, machine_type)
                
                if prediction:
                    # Store actual vs predicted values
                    actual_health.append(actual_health_score)
                    predicted_health.append(prediction['health_score'])
                    actual_maintenance.append(actual_maintenance_days)
                    predicted_maintenance.append(prediction['days_until_maintenance'])
            
            # Calculate metrics
            if len(actual_health) > 0:
                health_mae = mean_absolute_error(actual_health, predicted_health)
                health_rmse = np.sqrt(mean_squared_error(actual_health, predicted_health))
                health_r2 = r2_score(actual_health, predicted_health) if len(set(actual_health)) > 1 else 0
                
                maintenance_mae = mean_absolute_error(actual_maintenance, predicted_maintenance)
                maintenance_rmse = np.sqrt(mean_squared_error(actual_maintenance, predicted_maintenance))
                maintenance_r2 = r2_score(actual_maintenance, predicted_maintenance) if len(set(actual_maintenance)) > 1 else 0
                
                # Calculate an overall score (lower is better)
                overall_score = (health_mae / 100 + maintenance_mae / 30) / 2
                
                # Store results
                self.results['machine_type'].append(machine_type)
                self.results['num_samples'].append(len(actual_health))
                self.results['health_score_mae'].append(health_mae)
                self.results['health_score_rmse'].append(health_rmse)
                self.results['health_score_r2'].append(health_r2)
                self.results['days_until_maintenance_mae'].append(maintenance_mae)
                self.results['days_until_maintenance_rmse'].append(maintenance_rmse)
                self.results['days_until_maintenance_r2'].append(maintenance_r2)
                self.results['overall_score'].append(overall_score)
                
                print(f"  Processed {len(actual_health)} machines")
                print(f"  Health Score: MAE={health_mae:.2f}, RMSE={health_rmse:.2f}, R²={health_r2:.2f}")
                print(f"  Days Until Maintenance: MAE={maintenance_mae:.2f}, RMSE={maintenance_rmse:.2f}, R²={maintenance_r2:.2f}")
                print(f"  Overall Score: {overall_score:.4f}")
            else:
                print(f"  No valid predictions for {machine_type}")
                
    def test_scenarios(self):
        """Test the model against specific scenarios for each machine type."""
        if self.test_df is None:
            print("Test data not loaded. Please load test data first.")
            return
            
        print("\nTesting model against specific scenarios...")
        
        # Define scenarios to test
        scenarios = ['normal', 'degrading', 'anomaly', 'failure', 'post_maintenance']
        
        # Create results dictionary for scenarios
        self.scenario_results = {
            'machine_type': [],
            'scenario': [],
            'health_score_accuracy': [],
            'maintenance_days_accuracy': [],
            'status_correctness': []
        }
        
        # Process each machine type
        for machine_type in self.model.machine_types:
            print(f"\nTesting scenarios for {machine_type}:")
            
            # Test each scenario
            for scenario in scenarios:
                # Filter test data for this machine type and scenario
                scenario_data = self.test_df[
                    (self.test_df['machine_type'] == machine_type) & 
                    (self.test_df['machine_id'].str.contains(scenario))
                ]
                
                if len(scenario_data) == 0:
                    print(f"  No test data found for {machine_type}, {scenario} scenario, skipping")
                    continue
                
                # Get a representative machine for this scenario
                machine_ids = scenario_data['machine_id'].unique()
                if len(machine_ids) == 0:
                    continue
                    
                machine_id = machine_ids[0]
                machine_data = scenario_data[scenario_data['machine_id'] == machine_id].copy()
                
                # Sort by timestamp
                machine_data = machine_data.sort_values('timestamp')
                
                # Make prediction using the model
                prediction = self.model.predict_machine_health(machine_data, machine_type)
                
                if not prediction:
                    print(f"  Failed to make prediction for {machine_type}, {scenario} scenario")
                    continue
                
                # Evaluate the prediction based on the scenario
                health_score = prediction['health_score']
                days_until_maintenance = prediction['days_until_maintenance']
                status = prediction['status']
                
                # Expected values based on scenario
                expected_health = self._get_expected_health(scenario)
                expected_maintenance = self._get_expected_maintenance(scenario)
                expected_status = self._get_expected_status(scenario)
                
                # Calculate accuracy (0-1 scale, higher is better)
                health_accuracy = 1.0 - min(1.0, abs(health_score - expected_health) / 100)
                maintenance_accuracy = 1.0 - min(1.0, abs(days_until_maintenance - expected_maintenance) / 30)
                status_correct = 1.0 if status == expected_status else 0.0
                
                # Store results
                self.scenario_results['machine_type'].append(machine_type)
                self.scenario_results['scenario'].append(scenario)
                self.scenario_results['health_score_accuracy'].append(health_accuracy)
                self.scenario_results['maintenance_days_accuracy'].append(maintenance_accuracy)
                self.scenario_results['status_correctness'].append(status_correct)
                
                print(f"  {scenario.upper()} scenario:")
                print(f"    Predicted: Health={health_score:.1f}, Days={days_until_maintenance}, Status={status}")
                print(f"    Expected: Health={expected_health}, Days={expected_maintenance}, Status={expected_status}")
                print(f"    Accuracy: Health={health_accuracy:.2f}, Days={maintenance_accuracy:.2f}, Status={status_correct:.1f}")
    
    def _get_expected_health(self, scenario):
        """Get the expected health score for a given scenario."""
        scenario_health = {
            'normal': 95,
            'degrading': 70,
            'anomaly': 60,
            'failure': 20,
            'post_maintenance': 85
        }
        return scenario_health.get(scenario, 50)
    
    def _get_expected_maintenance(self, scenario):
        """Get the expected days until maintenance for a given scenario."""
        scenario_maintenance = {
            'normal': 30,
            'degrading': 14,
            'anomaly': 7,
            'failure': 0,
            'post_maintenance': 25
        }
        return scenario_maintenance.get(scenario, 15)
    
    def _get_expected_status(self, scenario):
        """Get the expected status for a given scenario."""
        scenario_status = {
            'normal': 'Excellent',
            'degrading': 'Good',
            'anomaly': 'Fair',
            'failure': 'Critical',
            'post_maintenance': 'Good'
        }
        return scenario_status.get(scenario, 'Fair')
    
    def test_edge_cases(self):
        """Test the model against edge cases and abnormal inputs."""
        print("\nTesting edge cases and abnormal inputs...")
        
        # Get a sample machine type to test with
        if not self.model.machine_types:
            print("No machine types available. Cannot test edge cases.")
            return
            
        machine_type = self.model.machine_types[0]
        
        # Dictionary to track edge case results
        edge_case_results = {}
        
        # Case 1: Empty DataFrame
        print("Testing with empty DataFrame...")
        empty_df = pd.DataFrame(columns=['machine_id', 'timestamp'])
        try:
            prediction = self.model.predict_machine_health(empty_df, machine_type)
            edge_case_results['empty_df'] = "Failed" if prediction is not None else "Handled gracefully"
        except Exception as e:
            print(f"  Empty DataFrame test raised exception: {str(e)}")
            edge_case_results['empty_df'] = "Exception raised - model needs improvement"
        print(f"  Result: {edge_case_results['empty_df']}")
        
        # Case 2: Missing sensor columns
        print("Testing with missing sensor columns...")
        # Create minimal DataFrame with only ID and timestamp
        timestamp = datetime.now()
        minimal_df = pd.DataFrame({
            'machine_id': [f"{machine_type}_test"],
            'timestamp': [timestamp]
        })
        try:
            prediction = self.model.predict_machine_health(minimal_df, machine_type)
            edge_case_results['missing_sensors'] = "Handled gracefully" if prediction is not None else "Failed"
        except Exception as e:
            print(f"  Missing sensors test raised exception: {str(e)}")
            edge_case_results['missing_sensors'] = "Exception raised - model needs improvement"
        print(f"  Result: {edge_case_results['missing_sensors']}")
        
        # Case 3: Extreme sensor values
        print("Testing with extreme sensor values...")
        # Get relevant sensors for this machine type
        std_machine_type = machine_type.split('_')[0] if '_' in machine_type else machine_type
        relevant_sensors = self.model.MACHINE_SENSOR_MAPPING.get(std_machine_type, [])
        
        if relevant_sensors:
            # Create DataFrame with extreme values
            extreme_data = {
                'machine_id': [f"{machine_type}_extreme"],
                'timestamp': [timestamp]
            }
            
            # Add extreme values for each sensor
            for sensor in relevant_sensors:
                extreme_data[sensor] = [1000]  # Much higher than normal
            
            extreme_df = pd.DataFrame(extreme_data)
            try:
                prediction = self.model.predict_machine_health(extreme_df, machine_type)
                # Check if prediction indicates poor health
                edge_case_results['extreme_values'] = "Correct" if prediction and prediction['health_score'] < 50 else "Incorrect"
                
                if prediction:
                    print(f"    Health Score: {prediction['health_score']:.1f}")
                    print(f"    Days Until Maintenance: {prediction['days_until_maintenance']}")
                    print(f"    Status: {prediction['status']}")
            except Exception as e:
                print(f"  Extreme values test raised exception: {str(e)}")
                edge_case_results['extreme_values'] = "Exception raised - model needs improvement"
            
            print(f"  Result: {edge_case_results['extreme_values']}")
        
        # Case 4: Invalid machine type
        print("Testing with invalid machine type...")
        invalid_type = "invalid_machine_type"
        sample_df = self.test_df.iloc[:10].copy() if self.test_df is not None else minimal_df
        try:
            prediction = self.model.predict_machine_health(sample_df, invalid_type)
            edge_case_results['invalid_type'] = "Handled gracefully" if prediction is None else "Failed"
        except Exception as e:
            print(f"  Invalid machine type test raised exception: {str(e)}")
            edge_case_results['invalid_type'] = "Exception raised - model needs improvement"
        print(f"  Result: {edge_case_results['invalid_type']}")
        
        # Store edge case results
        self.edge_case_results = edge_case_results
        
    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\nGenerating test report...")
        
        # Create report directory if it doesn't exist
        report_dir = "test_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # Convert results to DataFrames
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            # Save results to CSV
            results_df.to_csv(f"{report_dir}/machine_type_results.csv", index=False)
            
            # Create a tabular report
            print("\nMachine Type Performance:")
            print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt='.2f'))
        
        if self.scenario_results:
            scenario_df = pd.DataFrame(self.scenario_results)
            
            # Save scenario results to CSV
            scenario_df.to_csv(f"{report_dir}/scenario_results.csv", index=False)
            
            # Create a tabular report for scenarios
            print("\nScenario Test Results:")
            print(tabulate(scenario_df, headers='keys', tablefmt='grid', floatfmt='.2f'))
        
        # Create a summary report
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'machine_types_tested': len(self.results.get('machine_type', [])),
            'scenarios_tested': len(set(self.scenario_results.get('scenario', []))),
            'edge_cases_tested': len(getattr(self, 'edge_case_results', {})),
            'overall_performance': {}
        }
        
        if self.results:
            # Calculate average metrics
            summary['overall_performance'] = {
                'avg_health_score_mae': np.mean(self.results['health_score_mae']),
                'avg_health_score_rmse': np.mean(self.results['health_score_rmse']),
                'avg_health_score_r2': np.mean(self.results['health_score_r2']),
                'avg_days_until_maintenance_mae': np.mean(self.results['days_until_maintenance_mae']),
                'avg_days_until_maintenance_rmse': np.mean(self.results['days_until_maintenance_rmse']),
                'avg_days_until_maintenance_r2': np.mean(self.results['days_until_maintenance_r2']),
                'avg_overall_score': np.mean(self.results['overall_score'])
            }
            
            # Identify best and worst performing machine types
            best_idx = np.argmin(self.results['overall_score'])
            worst_idx = np.argmax(self.results['overall_score'])
            
            summary['best_machine_type'] = self.results['machine_type'][best_idx]
            summary['worst_machine_type'] = self.results['machine_type'][worst_idx]
        
        if self.scenario_results:
            # Calculate average scenario scores
            summary['scenario_performance'] = {}
            for scenario in set(self.scenario_results['scenario']):
                mask = [s == scenario for s in self.scenario_results['scenario']]
                avg_health_accuracy = np.mean([self.scenario_results['health_score_accuracy'][i] for i, m in enumerate(mask) if m])
                avg_maint_accuracy = np.mean([self.scenario_results['maintenance_days_accuracy'][i] for i, m in enumerate(mask) if m])
                avg_status_correct = np.mean([self.scenario_results['status_correctness'][i] for i, m in enumerate(mask) if m])
                
                summary['scenario_performance'][scenario] = {
                    'health_score_accuracy': avg_health_accuracy,
                    'maintenance_days_accuracy': avg_maint_accuracy,
                    'status_correctness': avg_status_correct,
                    'overall_accuracy': (avg_health_accuracy + avg_maint_accuracy + avg_status_correct) / 3
                }
        
        # Add edge case results
        if hasattr(self, 'edge_case_results'):
            summary['edge_case_results'] = self.edge_case_results
        
        # Save summary to JSON
        with open(f"{report_dir}/test_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Print summary report
        print("\nTest Summary:")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Machine Types Tested: {summary['machine_types_tested']}")
        print(f"Scenarios Tested: {summary['scenarios_tested']}")
        print(f"Edge Cases Tested: {summary['edge_cases_tested']}")
        
        if 'overall_performance' in summary:
            print("\nOverall Performance Metrics:")
            for key, value in summary['overall_performance'].items():
                print(f"  {key}: {value:.4f}")
        
        if 'best_machine_type' in summary:
            print(f"\nBest Performing Machine Type: {summary['best_machine_type']}")
            print(f"Worst Performing Machine Type: {summary['worst_machine_type']}")
        
        if 'scenario_performance' in summary:
            print("\nScenario Performance:")
            for scenario, metrics in summary['scenario_performance'].items():
                print(f"  {scenario.upper()}: Overall Accuracy = {metrics['overall_accuracy']:.4f}")
        
        # Create plots
        if self.results:
            self._create_performance_plots(report_dir)
        
        print(f"\nTest report saved to {report_dir}")
        
    def _create_performance_plots(self, report_dir):
        """Create performance visualization plots."""
        # Create DataFrame from results
        results_df = pd.DataFrame(self.results)
        
        # Plot health score metrics by machine type
        plt.figure(figsize=(12, 6))
        x = np.arange(len(results_df['machine_type']))
        width = 0.25
        
        plt.bar(x - width, results_df['health_score_mae'], width, label='MAE')
        plt.bar(x, results_df['health_score_rmse'], width, label='RMSE')
        plt.bar(x + width, results_df['health_score_r2'] * 100, width, label='R² (×100)')
        
        plt.xlabel('Machine Type')
        plt.ylabel('Metric Value')
        plt.title('Health Score Prediction Performance by Machine Type')
        plt.xticks(x, results_df['machine_type'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{report_dir}/health_score_performance.png")
        
        # Plot maintenance days metrics by machine type
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width, results_df['days_until_maintenance_mae'], width, label='MAE')
        plt.bar(x, results_df['days_until_maintenance_rmse'], width, label='RMSE')
        plt.bar(x + width, results_df['days_until_maintenance_r2'] * 100, width, label='R² (×100)')
        
        plt.xlabel('Machine Type')
        plt.ylabel('Metric Value')
        plt.title('Days Until Maintenance Prediction Performance by Machine Type')
        plt.xticks(x, results_df['machine_type'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{report_dir}/maintenance_days_performance.png")
        
        # Plot overall score by machine type
        plt.figure(figsize=(10, 6))
        plt.bar(results_df['machine_type'], results_df['overall_score'], color='teal')
        plt.xlabel('Machine Type')
        plt.ylabel('Overall Score (lower is better)')
        plt.title('Overall Model Performance by Machine Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{report_dir}/overall_performance.png")
        
        # If we have scenario results, plot those too
        if self.scenario_results:
            scenario_df = pd.DataFrame(self.scenario_results)
            
            # Plot average accuracy for each scenario
            plt.figure(figsize=(12, 8))
            
            # Prepare data
            scenarios = sorted(set(scenario_df['scenario']))
            health_accuracy = []
            maint_accuracy = []
            status_accuracy = []
            
            for scenario in scenarios:
                mask = [s == scenario for s in scenario_df['scenario']]
                health_accuracy.append(np.mean([scenario_df['health_score_accuracy'][i] for i, m in enumerate(mask) if m]))
                maint_accuracy.append(np.mean([scenario_df['maintenance_days_accuracy'][i] for i, m in enumerate(mask) if m]))
                status_accuracy.append(np.mean([scenario_df['status_correctness'][i] for i, m in enumerate(mask) if m]))
            
            # Create bar plot
            x = np.arange(len(scenarios))
            width = 0.25
            
            plt.bar(x - width, health_accuracy, width, label='Health Score Accuracy')
            plt.bar(x, maint_accuracy, width, label='Maintenance Days Accuracy')
            plt.bar(x + width, status_accuracy, width, label='Status Correctness')
            
            plt.xlabel('Scenario')
            plt.ylabel('Accuracy (0-1)')
            plt.title('Model Accuracy by Scenario')
            plt.xticks(x, [s.capitalize() for s in scenarios])
            plt.legend()
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(f"{report_dir}/scenario_accuracy.png")

def main():
    """Main function to run the model tester."""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PredictiveMaintenanceModelV2')
    parser.add_argument('--models-dir', default='models', help='Directory containing the saved models')
    parser.add_argument('--test-data', default='data/synthetic/test_dataset.csv', help='Path to test dataset')
    args = parser.parse_args()
    
    # Create and run the model tester
    tester = PredictiveMaintenanceModelTester(models_dir=args.models_dir, test_data_path=args.test_data)
    
    # Load models
    tester.load_models()
    
    # Load or create test data
    tester.load_test_data()
    
    # Run tests
    tester.test_all_machines()
    tester.test_scenarios()
    tester.test_edge_cases()
    
    # Generate report
    tester.generate_report()
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main() 