import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceModelV2:
    """
    A comprehensive model for predicting machine health and days until maintenance is required
    based on sensor data from industrial equipment.
    """
    
    # Define the mapping of machine types to their relevant sensor columns
    MACHINE_SENSOR_MAPPING = {
        'siemens': ['temperature', 'vibration', 'current', 'voltage'],
        'abb': ['vibration', 'temperature', 'acoustic'],
        'haas': ['spindle_load', 'vibration', 'temperature', 'acoustic'],
        'grundfos': ['pressure', 'flow_rate', 'temperature', 'power'],
        'carrier': ['refrigerant_pressure', 'condenser_temp', 'evaporator_temp', 'power']
    }
    
    # Define a standardized mapping from various machine ID prefixes to standard machine types
    MACHINE_TYPE_MAPPING = {
        'siemens': 'siemens',
        'siemens_motor': 'siemens',
        'abb': 'abb',
        'abb_bearing': 'abb',
        'haas': 'haas',
        'haas_cnc': 'haas',
        'grundfos': 'grundfos',
        'grundfos_pump': 'grundfos',
        'carrier': 'carrier',
        'carrier_chiller': 'carrier'
    }
    
    def __init__(self, data_path='data/improved_synthetic/combined_dataset.csv'):
        """
        Initialize the model with the path to the training data.
        
        Args:
            data_path: Path to the CSV containing the training data
        """
        self.data_path = data_path
        self.df = None
        self.machine_types = None
        self.health_models = {}  # One model per machine type
        self.maintenance_models = {}  # One model per machine type
        self.scalers = {}  # One scaler per machine type
        self.sensor_columns = {}  # Relevant sensor columns per machine type
        
    def get_standardized_machine_type(self, machine_id):
        """
        Extract a standardized machine type from a machine_id.
        
        Args:
            machine_id: The machine identifier string
            
        Returns:
            A standardized machine type string
        """
        # First, extract the base machine type from the id
        parts = machine_id.split('_')
        
        # Try both one and two part prefixes to match potential machine types
        prefix1 = parts[0]
        prefix2 = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else None
        
        # Check if either prefix is in our mapping
        if prefix2 and prefix2 in self.MACHINE_TYPE_MAPPING:
            return self.MACHINE_TYPE_MAPPING[prefix2]
        elif prefix1 in self.MACHINE_TYPE_MAPPING:
            return self.MACHINE_TYPE_MAPPING[prefix1]
        else:
            # Fallback to just the first part if no mapping found
            return prefix1
    
    def load_data(self):
        """Load and perform initial preprocessing on the dataset."""
        print("Loading training dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract machine type from machine_id using a consistent approach
        self.df['machine_type'] = self.df['machine_id'].apply(self.get_standardized_machine_type)
        
        # Store unique machine types
        self.machine_types = self.df['machine_type'].unique()
        
        print(f"Dataset loaded: {self.df.shape} rows and columns")
        print(f"Machine types: {self.machine_types}")
        
    def explore_data(self):
        """Basic exploration of the dataset to understand its characteristics."""
        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        # Basic statistics and information
        print(f"\nDataset contains data for {self.df['machine_id'].nunique()} unique machines")
        
        # Check class distribution
        print("\nFailure distribution:")
        failure_counts = self.df['failure'].value_counts()
        print(failure_counts)
        print(f"Failure rate: {failure_counts.get(1, 0) / len(self.df):.4f}")
        
        # Check machine type distribution
        print("\nMachine type distribution:")
        print(self.df['machine_type'].value_counts())
        
        # Identify relevant sensor columns for each machine type
        for machine_type in self.machine_types:
            print(f"\n{machine_type.upper()} Analysis:")
            
            # Get standardized machine type for the sensor mapping
            std_machine_type = machine_type.split('_')[0] if '_' in machine_type else machine_type
            
            # Get the relevant sensors for this machine type from our mapping
            relevant_sensors = self.MACHINE_SENSOR_MAPPING.get(std_machine_type, [])
            
            if not relevant_sensors:
                print(f"  No sensor mapping found for {machine_type}")
                continue
                
            print(f"  Relevant sensors: {relevant_sensors}")
            
            # Filter data for this machine type
            machine_df = self.df[self.df['machine_type'] == machine_type]
            
            # Get the actual sensor columns in the data that match our relevant sensors
            actual_sensor_cols = []
            for col in machine_df.columns:
                if any(sensor in col for sensor in relevant_sensors) and col not in ['failure', 'maintenance', 'anomaly']:
                    actual_sensor_cols.append(col)
            
            # Store these sensor columns for later use
            self.sensor_columns[machine_type] = actual_sensor_cols
            
            print(f"  Found sensor columns: {actual_sensor_cols}")
            
            # Basic statistics for sensors
            for col in actual_sensor_cols:
                if machine_df[col].count() > 0:
                    print(f"    {col}: mean={machine_df[col].mean():.2f}, std={machine_df[col].std():.2f}, "
                          f"min={machine_df[col].min():.2f}, max={machine_df[col].max():.2f}")
            
            # Check failure rate for this machine type
            failure_rate = machine_df['failure'].mean()
            print(f"  Failure rate: {failure_rate:.4f}")
    
    def engineer_features(self):
        """
        Perform feature engineering to enhance predictive capabilities.
        This includes creating time-based features, rolling statistics, and more.
        """
        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("Engineering features...")
        
        # Process in batches to avoid memory issues
        batch_size = 100000  # Adjust this value based on your available memory
        
        # Get unique machine IDs
        unique_machines = self.df['machine_id'].unique()
        print(f"Processing {len(unique_machines)} unique machines in batches...")
        
        # Lists to store the processed dataframes
        processed_dfs = []
        
        # Process each machine separately to avoid data leakage between machines
        for machine_id in unique_machines:
            # Filter data for just this machine
            group = self.df[self.df['machine_id'] == machine_id].copy()
            
            # Sort by timestamp to ensure correct sequence
            group = group.sort_values('timestamp')
            
            # Extract machine type using our standardized method
            machine_type = self.get_standardized_machine_type(machine_id)
            
            # Get relevant sensor columns for this machine type
            sensor_cols = self.sensor_columns.get(machine_type, [])
            
            if len(sensor_cols) > 0:
                # Create rolling window features (mean, std, min, max)
                for window in [3, 6, 12]:  # Different window sizes (3h, 6h, 12h with 15-minute intervals)
                    for col in sensor_cols:
                        # Only process columns that exist in the data
                        if col in group.columns:
                            # Rolling mean
                            group[f'{col}_rolling_mean_{window}'] = group[col].rolling(window=window, min_periods=1).mean()
                            # Rolling standard deviation
                            group[f'{col}_rolling_std_{window}'] = group[col].rolling(window=window, min_periods=1).std()
                            # Rolling min and max
                            group[f'{col}_rolling_min_{window}'] = group[col].rolling(window=window, min_periods=1).min()
                            group[f'{col}_rolling_max_{window}'] = group[col].rolling(window=window, min_periods=1).max()
                
                # Calculate rate of change for each sensor
                for col in sensor_cols:
                    if col in group.columns:
                        group[f'{col}_rate_of_change'] = group[col].diff() / group['timestamp'].diff().dt.total_seconds()
                
                # Calculate exponentially weighted moving average for smoother trends
                for col in sensor_cols:
                    if col in group.columns:
                        group[f'{col}_ewma'] = group[col].ewm(span=12).mean()  # 3-hour span with 15-minute intervals
                
                # Create features for time since last maintenance
                group['time_since_maintenance'] = 0
                maintenance_indices = group[group['maintenance'] == 1].index
                if len(maintenance_indices) > 0:
                    last_maintenance = maintenance_indices[-1]
                    for idx in group.index:
                        if idx > last_maintenance:
                            time_diff = group.loc[idx, 'timestamp'] - group.loc[last_maintenance, 'timestamp']
                            group.loc[idx, 'time_since_maintenance'] = time_diff.total_seconds() / 3600  # in hours
                
                # NEW: Detect post-maintenance patterns
                group['post_maintenance_state'] = 0  # Initialize if not present
                
                # If post_maintenance_state column already exists in the data, use it
                if 'post_maintenance_state' in group.columns:
                    # Create additional features based on it
                    group['post_maintenance_state_ewma'] = group['post_maintenance_state'].ewm(span=6).mean()
                else:
                    # Create a synthetic post-maintenance state based on maintenance events
                    for maint_idx in maintenance_indices:
                        # Assume the post-maintenance effect diminishes over 5 days
                        effect_window = 5 * 24 * 4  # 5 days of 15-minute intervals
                        for i in range(1, min(effect_window, len(group) - maint_idx)):
                            decay_factor = 1 - (i / effect_window)
                            group.loc[maint_idx + i, 'post_maintenance_state'] = decay_factor
                    
                    group['post_maintenance_state_ewma'] = group['post_maintenance_state'].ewm(span=6).mean()
                
                # NEW: Track sensor value distance from baseline after maintenance
                for col in sensor_cols:
                    if col in group.columns:
                        # Calculate a baseline during normal operation (first 10% if available)
                        baseline_range = min(int(len(group) * 0.1), 100)  # Use at most first 100 readings or 10%
                        if baseline_range > 0:
                            baseline = group[col].iloc[:baseline_range].mean()
                            # Calculate distance from baseline (normalized)
                            std_val = group[col].iloc[:baseline_range].std() or 1.0  # Avoid division by zero
                            group[f'{col}_deviation_from_baseline'] = (group[col] - baseline) / std_val
                            
                            # This feature captures how values change after maintenance
                            group[f'{col}_post_maint_pattern'] = group[f'{col}_deviation_from_baseline'] * group['post_maintenance_state']
                
                # NEW: Create explicit features for anomalous behavior
                if 'anomaly' in group.columns:
                    # Track time since last anomaly
                    group['time_since_anomaly'] = float('inf')  # Initialize with infinity
                    anomaly_indices = group[group['anomaly'] == 1].index
                    if len(anomaly_indices) > 0:
                        for idx in group.index:
                            # Find closest preceding anomaly
                            closest_anomaly = None
                            for anomaly_idx in anomaly_indices:
                                if anomaly_idx < idx and (closest_anomaly is None or anomaly_idx > closest_anomaly):
                                    closest_anomaly = anomaly_idx
                            
                            if closest_anomaly is not None:
                                time_diff = group.loc[idx, 'timestamp'] - group.loc[closest_anomaly, 'timestamp']
                                group.loc[idx, 'time_since_anomaly'] = time_diff.total_seconds() / 3600  # in hours
            
            # Add to the list of processed dataframes
            processed_dfs.append(group)
            
            # Check if we need to combine and process
            if len(processed_dfs) >= 50 or sum(len(df) for df in processed_dfs) > batch_size:
                print(f"Processed {len(processed_dfs)} machines...")
                # Combine and save partial result, then clear memory
                if 'partial_df' not in locals():
                    partial_df = pd.concat(processed_dfs, axis=0)
                else:
                    partial_df = pd.concat([partial_df] + processed_dfs, axis=0)
                processed_dfs = []
                
        # Combine any remaining processed dataframes
        if processed_dfs:
            if 'partial_df' not in locals():
                self.df = pd.concat(processed_dfs, axis=0)
            else:
                self.df = pd.concat([partial_df] + processed_dfs, axis=0)
        elif 'partial_df' in locals():
            self.df = partial_df
        
        # Create health score target (inverse of failure probability)
        # For simplicity, we'll use 100 - (failure*100) as the health score
        self.df['health_score'] = 100 - (self.df['failure'] * 100)
        
        # Create days_until_maintenance target
        # For now, we'll use a simplified approach based on failure flag
        # In a real scenario, this would be derived from actual maintenance schedules
        self.df['days_until_maintenance'] = 30  # Default 30 days
        
        # If failure=1, set days_until_maintenance to 0
        self.df.loc[self.df['failure'] == 1, 'days_until_maintenance'] = 0
        
        # If anomaly=1 but failure=0, set days_until_maintenance to a lower value
        self.df.loc[(self.df['anomaly'] == 1) & (self.df['failure'] == 0), 'days_until_maintenance'] = 7
        
        # For machines in normal operation, we'll simulate a gradual decrease in days_until_maintenance
        # based on time elapsed from the start of the dataset
        print("Calculating days until maintenance...")
        
        # Get unique machine IDs for the final processing
        unique_machines = self.df['machine_id'].unique()
        
        for machine_id in unique_machines:
            # Filter data for this machine
            group = self.df[self.df['machine_id'] == machine_id].copy()
            
            if len(group) > 0:
                # Sort by timestamp
                sorted_group = group.sort_values('timestamp')
                
                # For normal operations (no failure or anomaly)
                normal_mask = (sorted_group['failure'] == 0) & (sorted_group['anomaly'] == 0)
                
                if normal_mask.any():
                    # Get the time range for this machine
                    start_time = sorted_group['timestamp'].min()
                    end_time = sorted_group['timestamp'].max()
                    total_days = (end_time - start_time).total_seconds() / (24 * 3600)
                    
                    # Calculate days elapsed for each timestamp
                    sorted_group.loc[normal_mask, 'days_elapsed'] = (
                        sorted_group.loc[normal_mask, 'timestamp'] - start_time
                    ).dt.total_seconds() / (24 * 3600)
                    
                    # Calculate days until maintenance based on elapsed time
                    # The formula is designed so that days_until_maintenance decreases over time
                    sorted_group.loc[normal_mask, 'days_until_maintenance'] = 30 - (
                        sorted_group.loc[normal_mask, 'days_elapsed'] * 30 / (total_days + 1)
                    )
                    
                    # Update the main dataframe
                    self.df.loc[sorted_group.index, 'days_until_maintenance'] = sorted_group['days_until_maintenance']
        
        # Clean up by rounding days_until_maintenance to integers and ensuring it's non-negative
        self.df['days_until_maintenance'] = np.maximum(0, np.round(self.df['days_until_maintenance'])).astype(int)
        
        print(f"Feature engineering complete. Dataset now has {self.df.shape[1]} columns.")
    
    def prepare_data_for_training(self, machine_type):
        """
        Prepare data for a specific machine type for model training.
        
        Args:
            machine_type: The machine type to prepare data for
            
        Returns:
            X_train, X_test, y_health_train, y_health_test, y_maint_train, y_maint_test
        """
        # Filter data for the specific machine type
        machine_data = self.df[self.df['machine_type'] == machine_type].copy()
        
        if len(machine_data) == 0:
            print(f"No data found for machine type: {machine_type}")
            return None, None, None, None, None, None, None
        
        # Get relevant sensor columns for this machine type
        sensor_cols = self.sensor_columns.get(machine_type, [])
        
        if not sensor_cols:
            print(f"No sensor columns defined for machine type: {machine_type}")
            return None, None, None, None, None, None, None
        
        # Get all engineered feature columns based ONLY on relevant sensors
        feature_cols = []
        
        # Add original sensor columns if they exist
        for col in sensor_cols:
            if col in machine_data.columns:
                feature_cols.append(col)
        
        # Add derived features that were created from these sensors
        for col in machine_data.columns:
            if col not in feature_cols:  # Skip original sensors already added
                # Check if this column is derived from a relevant sensor
                if any(sensor + '_' in col for sensor in sensor_cols):
                    feature_cols.append(col)
        
        # Add time_since_maintenance as a feature
        if 'time_since_maintenance' in machine_data.columns:
            feature_cols.append('time_since_maintenance')
        
        # Ensure we have features to work with
        if len(feature_cols) == 0:
            print(f"No valid features found for machine type: {machine_type}")
            return None, None, None, None, None, None, None
        
        # Remove any columns that are all NaN
        valid_cols = [col for col in feature_cols if not machine_data[col].isna().all()]
        feature_cols = valid_cols
        
        print(f"Training model for {machine_type} with {len(feature_cols)} features")
        
        # Prepare feature matrix X and target vectors y
        X = machine_data[feature_cols]
        y_health = machine_data['health_score']
        y_maint = machine_data['days_until_maintenance']
        
        # Split into training and testing sets with stratification based on failure
        # to ensure both sets have examples of failures
        X_train, X_test, y_health_train, y_health_test, y_maint_train, y_maint_test = train_test_split(
            X, y_health, y_maint, 
            test_size=0.2, 
            random_state=42,
            stratify=machine_data['failure']
        )
        
        # Create and fit scaler for this machine type
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store the scaler for later use
        self.scalers[machine_type] = scaler
        
        # Store feature columns for this machine type for later prediction
        self.sensor_columns[machine_type] = feature_cols
        
        return X_train_scaled, X_test_scaled, y_health_train, y_health_test, y_maint_train, y_maint_test, feature_cols
    
    def train_models(self):
        """Train machine-specific models for health score and days until maintenance."""
        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("Training models for each machine type...")
        
        # Create a directory to save models if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        for machine_type in self.machine_types:
            print(f"\nTraining models for {machine_type}...")
            
            # Prepare data for this machine type
            prepared_data = self.prepare_data_for_training(machine_type)
            
            if prepared_data is None or prepared_data[0] is None:
                print(f"Skipping {machine_type} due to insufficient data")
                continue
                
            X_train_scaled, X_test_scaled, y_health_train, y_health_test, y_maint_train, y_maint_test, feature_cols = prepared_data
            
            # Create and train health score model (regression task)
            print("Training health score model...")
            
            # Simplified hyperparameter search to avoid convergence issues
            health_model_params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'min_child_weight': [3]
            }
            
            # Use LightGBM for health score prediction with early stopping
            health_base_model = lgb.LGBMRegressor(
                objective='regression',
                random_state=42,
                n_jobs=-1,           # Use all cores
                verbose=-1,          # Suppress verbose output
                early_stopping_rounds=10  # Stop if no improvement
            )
            
            try:
                # Use 3-fold cross-validation with fewer combinations
                print("Finding optimal health score model parameters...")
                health_cv = GridSearchCV(
                    health_base_model,
                    health_model_params,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    verbose=0,
                    n_jobs=-1
                )
                
                # Fit the model
                health_cv.fit(X_train_scaled, y_health_train, 
                             eval_set=[(X_test_scaled, y_health_test)],
                             callbacks=[lgb.early_stopping(10, verbose=False)])
                
                # Get best model
                health_model = health_cv.best_estimator_
                
                print(f"Best parameters for health score model: {health_cv.best_params_}")
            except Exception as e:
                print(f"Error during health model training: {str(e)}")
                print("Using default model parameters instead")
                health_model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                health_model.fit(X_train_scaled, y_health_train)
            
            # Evaluate health score model
            health_preds = health_model.predict(X_test_scaled)
            
            # Clamp health scores to the range [0, 100]
            health_preds = np.clip(health_preds, 0, 100)
            
            health_mae = mean_absolute_error(y_health_test, health_preds)
            health_rmse = np.sqrt(mean_squared_error(y_health_test, health_preds))
            health_r2 = r2_score(y_health_test, health_preds)
            
            print(f"Health Score Model Metrics:")
            print(f"  MAE: {health_mae:.2f}")
            print(f"  RMSE: {health_rmse:.2f}")
            print(f"  R²: {health_r2:.2f}")
            
            # Create and train days until maintenance model (regression task)
            print("Training days until maintenance model...")
            
            # Simplified hyperparameter search for maintenance model
            maint_model_params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'min_child_weight': [3]
            }
            
            # Use LightGBM for maintenance prediction with early stopping
            maint_base_model = lgb.LGBMRegressor(
                objective='regression',
                random_state=42,
                n_jobs=-1,           # Use all cores
                verbose=-1,          # Suppress verbose output
                early_stopping_rounds=10  # Stop if no improvement
            )
            
            try:
                # Use 3-fold cross-validation with fewer combinations
                print("Finding optimal maintenance prediction model parameters...")
                maint_cv = GridSearchCV(
                    maint_base_model,
                    maint_model_params,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    verbose=0,
                    n_jobs=-1
                )
                
                # Fit the model
                maint_cv.fit(X_train_scaled, y_maint_train,
                            eval_set=[(X_test_scaled, y_maint_test)],
                            callbacks=[lgb.early_stopping(10, verbose=False)])
                
                # Get best model
                maint_model = maint_cv.best_estimator_
                
                print(f"Best parameters for maintenance model: {maint_cv.best_params_}")
            except Exception as e:
                print(f"Error during maintenance model training: {str(e)}")
                print("Using default model parameters instead")
                maint_model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                maint_model.fit(X_train_scaled, y_maint_train)
            
            # Evaluate maintenance prediction model
            maint_preds = maint_model.predict(X_test_scaled)
            
            # Ensure non-negative maintenance days
            maint_preds = np.maximum(0, maint_preds)
            
            maint_mae = mean_absolute_error(y_maint_test, maint_preds)
            maint_rmse = np.sqrt(mean_squared_error(y_maint_test, maint_preds))
            maint_r2 = r2_score(y_maint_test, maint_preds)
            
            print(f"Maintenance Prediction Model Metrics:")
            print(f"  MAE: {maint_mae:.2f}")
            print(f"  RMSE: {maint_rmse:.2f}")
            print(f"  R²: {maint_r2:.2f}")
            
            # Store models
            self.health_models[machine_type] = health_model
            self.maintenance_models[machine_type] = maint_model
            
            # Save models to disk
            joblib.dump(health_model, f'models/{machine_type}_health_model.pkl')
            joblib.dump(maint_model, f'models/{machine_type}_maintenance_model.pkl')
            joblib.dump(self.scalers[machine_type], f'models/{machine_type}_scaler.pkl')
            joblib.dump(feature_cols, f'models/{machine_type}_feature_cols.pkl')
            
            # Save feature importance plots
            plt.figure(figsize=(12, 10))
            try:
                lgb.plot_importance(health_model, max_num_features=20)
                plt.title(f'Feature Importance for {machine_type} Health Score Model')
                plt.tight_layout()
                plt.savefig(f'models/{machine_type}_health_feature_importance.png')
            except Exception as e:
                print(f"Could not create health feature importance plot: {str(e)}")
            
            plt.figure(figsize=(12, 10))
            try:
                lgb.plot_importance(maint_model, max_num_features=20)
                plt.title(f'Feature Importance for {machine_type} Maintenance Prediction Model')
                plt.tight_layout()
                plt.savefig(f'models/{machine_type}_maintenance_feature_importance.png')
            except Exception as e:
                print(f"Could not create maintenance feature importance plot: {str(e)}")
        
        print("\nModel training complete.")
    
    def predict_machine_health(self, machine_data, machine_type=None):
        """
        Make predictions for a specific machine using the trained models.
        
        Args:
            machine_data: DataFrame with sensor readings for the machine
            machine_type: Type of the machine. If None, it will be extracted from machine_id
            
        Returns:
            Dictionary with health score and days until maintenance predictions
        """
        if not self.health_models or not self.maintenance_models:
            print("Models not trained. Call train_models() first.")
            return None
        
        # Ensure machine_data is a DataFrame
        if isinstance(machine_data, dict):
            machine_data = pd.DataFrame([machine_data])
        
        # Extract machine type if not provided
        if machine_type is None:
            if 'machine_id' in machine_data.columns:
                machine_id = machine_data['machine_id'].iloc[0]
                machine_type = self.get_standardized_machine_type(machine_id)
            else:
                print("Machine type not provided and cannot be extracted from data.")
                return None
        
        # Standardize the machine type
        machine_type = self.MACHINE_TYPE_MAPPING.get(machine_type, machine_type)
        
        # Check if models exist for this machine type
        if machine_type not in self.health_models or machine_type not in self.maintenance_models:
            print(f"No models found for machine type: {machine_type}")
            return None
        
        # Get the relevant feature columns for this machine type
        feature_cols = self.sensor_columns.get(machine_type, [])
        
        if not feature_cols:
            print(f"No feature columns defined for machine type: {machine_type}")
            return None
        
        # Engineer features for prediction data
        pred_data = machine_data.copy()
        
        # Sort by timestamp if available
        if 'timestamp' in pred_data.columns:
            pred_data = pred_data.sort_values('timestamp')
        
        # Get basic sensor columns that exist in the data
        basic_sensor_cols = [col for col in feature_cols if col in pred_data.columns and '_' not in col]
        
        # Create rolling features if we have enough data points
        if len(pred_data) >= 3:
            for window in [3, 6, 12]:
                for col in basic_sensor_cols:
                    # Rolling mean
                    pred_data[f'{col}_rolling_mean_{window}'] = pred_data[col].rolling(window=window, min_periods=1).mean()
                    # Rolling standard deviation
                    pred_data[f'{col}_rolling_std_{window}'] = pred_data[col].rolling(window=window, min_periods=1).std()
                    # Rolling min and max
                    pred_data[f'{col}_rolling_min_{window}'] = pred_data[col].rolling(window=window, min_periods=1).min()
                    pred_data[f'{col}_rolling_max_{window}'] = pred_data[col].rolling(window=window, min_periods=1).max()
            
            # Calculate rate of change for each sensor
            for col in basic_sensor_cols:
                if 'timestamp' in pred_data.columns:
                    pred_data[f'{col}_rate_of_change'] = pred_data[col].diff() / pred_data['timestamp'].diff().dt.total_seconds()
            
            # Calculate exponentially weighted moving average
            for col in basic_sensor_cols:
                pred_data[f'{col}_ewma'] = pred_data[col].ewm(span=12).mean()
        
        # Use the most recent data point for prediction
        latest_data = pred_data.iloc[-1:].copy()
        
        # Create feature matrix with only the features used during training
        X_pred = pd.DataFrame()
        
        for col in feature_cols:
            if col in latest_data.columns:
                X_pred[col] = latest_data[col]
            else:
                # If a required feature is missing, fill with 0
                # A more sophisticated approach would be to impute based on training data
                X_pred[col] = 0
        
        # Apply the same scaling as during training
        X_pred_scaled = self.scalers[machine_type].transform(X_pred)
        # Convert to DataFrame to preserve feature names for LightGBM
        X_pred_scaled = pd.DataFrame(X_pred_scaled, columns=feature_cols)
        
        # Make predictions
        health_score = self.health_models[machine_type].predict(X_pred_scaled)[0]
        days_until_maintenance = self.maintenance_models[machine_type].predict(X_pred_scaled)[0]
        
        # Clamp health score to valid range [0, 100]
        health_score = np.clip(health_score, 0, 100)
        
        # Round days until maintenance to nearest integer and ensure it's non-negative
        days_until_maintenance = max(0, round(days_until_maintenance))
        
        # Create result dictionary
        result = {
            'machine_id': machine_data['machine_id'].iloc[-1] if 'machine_id' in machine_data.columns else 'unknown',
            'machine_type': machine_type,
            'timestamp': machine_data['timestamp'].iloc[-1] if 'timestamp' in machine_data.columns else datetime.now(),
            'health_score': health_score,
            'days_until_maintenance': days_until_maintenance
        }
        
        # Add a machine status based on health score
        if health_score >= 90:
            result['status'] = 'Excellent'
        elif health_score >= 75:
            result['status'] = 'Good'
        elif health_score >= 50:
            result['status'] = 'Fair'
        elif health_score >= 25:
            result['status'] = 'Poor'
        else:
            result['status'] = 'Critical'
        
        return result
    
    def load_models(self, models_dir='models'):
        """
        Load previously trained models from disk.
        
        Args:
            models_dir: Directory containing the saved models
        """
        # Check if models directory exists
        if not os.path.exists(models_dir):
            print(f"Models directory {models_dir} not found.")
            return False
        
        # Get all model files
        model_files = os.listdir(models_dir)
        
        # Load models for each machine type
        machine_types_found = set()
        for file in model_files:
            if file.endswith('_health_model.pkl'):
                machine_type = file.split('_health_model.pkl')[0]
                machine_types_found.add(machine_type)
                
                # Load health model
                health_model_path = os.path.join(models_dir, file)
                if os.path.exists(health_model_path):
                    self.health_models[machine_type] = joblib.load(health_model_path)
                
                # Load maintenance model
                maint_model_path = os.path.join(models_dir, f"{machine_type}_maintenance_model.pkl")
                if os.path.exists(maint_model_path):
                    self.maintenance_models[machine_type] = joblib.load(maint_model_path)
                
                # Load scaler
                scaler_path = os.path.join(models_dir, f"{machine_type}_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scalers[machine_type] = joblib.load(scaler_path)
                    
                # Load feature columns
                feature_cols_path = os.path.join(models_dir, f"{machine_type}_feature_cols.pkl")
                if os.path.exists(feature_cols_path):
                    self.sensor_columns[machine_type] = joblib.load(feature_cols_path)
        
        if machine_types_found:
            self.machine_types = list(machine_types_found)
            print(f"Loaded models for {len(machine_types_found)} machine types: {', '.join(machine_types_found)}")
            return True
        else:
            print("No models found.")
            return False


def demo_prediction(model, machine_type, data_path='data/synthetic/training_dataset.csv'):
    """
    Demonstrate prediction for a specific machine type.
    
    Args:
        model: Trained PredictiveMaintenanceModel
        machine_type: Type of machine to make prediction for
        data_path: Path to dataset containing test data
    """
    # Load a sample of data for the specified machine type
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract and standardize machine type from machine_id
    df['machine_type_extracted'] = df['machine_id'].apply(lambda x: model.get_standardized_machine_type(x))
    
    # Filter for the requested machine type
    machine_data = df[df['machine_type_extracted'] == machine_type].copy()
    
    if len(machine_data) == 0:
        print(f"No data found for machine type: {machine_type}")
        return
    
    # Get a sample machine
    unique_machines = machine_data['machine_id'].unique()
    sample_machine_id = unique_machines[0]
    
    # Get the last 3-4 days of data for this machine
    sample_machine_data = machine_data[machine_data['machine_id'] == sample_machine_id].copy()
    sample_machine_data = sample_machine_data.sort_values('timestamp')
    
    # Get the last 96 readings (4 days with 15-minute intervals)
    if len(sample_machine_data) > 96:
        sample_machine_data = sample_machine_data.iloc[-96:]
    
    # Make prediction
    prediction = model.predict_machine_health(sample_machine_data, machine_type)
    
    if prediction:
        print("\nPrediction Results:")
        print(f"Machine ID: {prediction['machine_id']}")
        print(f"Machine Type: {prediction['machine_type']}")
        print(f"Timestamp: {prediction['timestamp']}")
        print(f"Health Score: {prediction['health_score']:.2f}")
        print(f"Status: {prediction['status']}")
        print(f"Days Until Maintenance: {prediction['days_until_maintenance']}")
    else:
        print("Prediction failed.")


def main():
    """Main function to demonstrate the predictive maintenance workflow."""
    # Create model instance
    model = PredictiveMaintenanceModelV2()
    
    # Load and explore data
    model.load_data()
    model.explore_data()
    
    # Engineer features
    model.engineer_features()
    
    # Train models
    model.train_models()
    
    # Demonstrate predictions for each machine type
    print("\n" + "="*50)
    print("PREDICTION DEMONSTRATION")
    print("="*50)
    
    # Use the standardized machine types for demonstration
    unique_machine_types = set(model.MACHINE_TYPE_MAPPING.values())
    
    for machine_type in unique_machine_types:
        print(f"\nDemonstrating prediction for {machine_type}:")
        demo_prediction(model, machine_type)


if __name__ == "__main__":
    main() 