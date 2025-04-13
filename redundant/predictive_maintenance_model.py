"""
Predictive Maintenance Model for Smart Manufacturing

This module implements advanced machine learning models to predict equipment failures
based on sensor data. It supports both classical ML and deep learning approaches
for more accurate predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PredictiveMaintenanceModel:
    """
    A class to build, train and evaluate predictive maintenance models for different machine types.
    """
    
    def __init__(self, data_path="data/synthetic/combined_dataset.csv", model_type="ensemble"):
        """
        Initialize the model with the path to the dataset.
        
        Args:
            data_path: Path to the combined dataset CSV file
            model_type: Type of model to use - 'ensemble' or 'deep_learning'
        """
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.model_type = model_type
        self.machine_types = [
            "siemens_motor",
            "abb_bearing",
            "haas_cnc",
            "grundfos_pump",
            "carrier_chiller"
        ]
        
        # Define normal operating ranges for each machine type
        self.normal_ranges = {
            "siemens_motor": {
                "temperature": (60, 80),  # °C
                "vibration": (1, 5),      # mm/s RMS
                "current": (10, 100),     # A
                "voltage": (380, 440)     # V
            },
            "abb_bearing": {
                "vibration": (0.2, 2),    # mm/s RMS
                "temperature": (40, 60),  # °C
                "acoustic": (40, 70)      # dB
            },
            "haas_cnc": {
                "spindle_load": (20, 60), # %
                "vibration": (0.5, 3),    # mm/s RMS
                "temperature": (45, 65),  # °C
                "acoustic": (50, 80)      # dB
            },
            "grundfos_pump": {
                "pressure": (2, 25),      # bar
                "flow_rate": (1, 180),    # m³/h
                "temperature": (40, 70),  # °C
                "power": (0.37, 22)       # kW
            },
            "carrier_chiller": {
                "refrigerant_pressure": (8, 25),  # bar
                "condenser_temp": (35, 45),       # °C
                "evaporator_temp": (5, 10),       # °C
                "power": (40, 350)                # kW (scaled as 0.5-0.7 kW/ton for 80-500 ton)
            }
        }
        
    def load_data(self):
        """Load the combined dataset."""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
        print(f"Dataset loaded with {len(self.data)} rows and {len(self.data.columns)} columns")
        
        # Print data info
        print("\nData overview:")
        print(self.data.dtypes)
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
    def preprocess_data(self):
        """Perform data preprocessing on the entire dataset."""
        print("Preprocessing data...")
        
        # Check for and handle missing values
        if self.data.isnull().sum().sum() > 0:
            print(f"Found {self.data.isnull().sum().sum()} missing values")
            # Simple imputation for now - this could be enhanced
            for col in self.data.columns:
                if self.data[col].isnull().sum() > 0:
                    if self.data[col].dtype in ['int64', 'float64']:
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                    else:
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        # Check for and handle outliers
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col not in ['machine_id', 'failure', 'maintenance', 'anomaly']:
                # Use IQR method to detect outliers
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"Found {outliers} outliers in {col}")
                    
                    # Cap outliers instead of removing them
                    self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                    self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
        
        print("Data preprocessing complete")
        
    def get_machine_specific_data(self, machine_type):
        """
        Filter the dataset to only include rows for a specific machine type.
        
        Args:
            machine_type: The type of machine to filter for
            
        Returns:
            DataFrame containing only rows for the specified machine type
        """
        # Filter rows where the machine_id column contains the machine_type
        machine_data = self.data[self.data['machine_id'].str.contains(machine_type, na=False)]
        # Drop columns that are entirely NaN (non-applicable machine-specific fields)
        machine_data = machine_data.dropna(axis=1, how='all')
        return machine_data
    
    def engineer_siemens_motor_features(self, df):
        """
        Create machine-specific features for Siemens SIMOTICS Electric Motors.
        
        Args:
            df: DataFrame containing Siemens motor data
            
        Returns:
            DataFrame with engineered features
        """
        # Check for required columns
        required_cols = ['temperature', 'vibration', 'current', 'voltage']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column {col} missing for Siemens motor")
                df[col] = np.nan
        
        # Create feature dataframe
        features = df.copy()
        
        # Add normalized features based on normal operating ranges
        for col in required_cols:
            if col in features.columns:
                min_val, max_val = self.normal_ranges["siemens_motor"][col]
                features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
                
                # Create deviation from normal range feature
                features[f'{col}_deviation'] = features[col].apply(
                    lambda x: max(0, min_val - x) / min_val if x < min_val 
                    else max(0, x - max_val) / max_val if x > max_val else 0
                )
        
        # Calculate apparent power (S = V * I)
        if 'current' in features.columns and 'voltage' in features.columns:
            features['apparent_power'] = features['voltage'] * features['current']
        
        # Create time-based features
        if 'timestamp' in features.columns:
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
        
        # Create rolling window features for each sensor
        for col in required_cols:
            if col in features.columns:
                features[f'{col}_rolling_mean_12h'] = features[col].rolling(window=12, min_periods=1).mean()
                features[f'{col}_rolling_std_12h'] = features[col].rolling(window=12, min_periods=1).std().fillna(0)
        
        # Temperature-to-vibration ratio can indicate bearing issues
        if 'temperature' in features.columns and 'vibration' in features.columns:
            features['temp_vibration_ratio'] = features['temperature'] / features['vibration'].replace(0, 0.001)
        
        # Remove non-feature columns
        drop_columns = ['timestamp', 'machine_id']
        X = features.drop(columns=drop_columns + ['failure', 'maintenance', 'anomaly'], errors='ignore')
        
        return X
    
    def engineer_abb_bearing_features(self, df):
        """
        Create machine-specific features for ABB Dodge Mounted Bearings.
        
        Args:
            df: DataFrame containing ABB bearing data
            
        Returns:
            DataFrame with engineered features
        """
        # Check for required columns
        required_cols = ['vibration', 'temperature', 'acoustic']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column {col} missing for ABB bearing")
                df[col] = np.nan
        
        # Create feature dataframe
        features = df.copy()
        
        # Add normalized features based on normal operating ranges
        for col in required_cols:
            if col in features.columns:
                min_val, max_val = self.normal_ranges["abb_bearing"][col]
                features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
                
                # Create deviation from normal range feature
                features[f'{col}_deviation'] = features[col].apply(
                    lambda x: max(0, min_val - x) / min_val if x < min_val 
                    else max(0, x - max_val) / max_val if x > max_val else 0
                )
        
        # Create time-based features
        if 'timestamp' in features.columns:
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
        
        # Create rolling window features
        for col in required_cols:
            if col in features.columns:
                features[f'{col}_rolling_mean_12h'] = features[col].rolling(window=12, min_periods=1).mean()
                features[f'{col}_rolling_std_12h'] = features[col].rolling(window=12, min_periods=1).std().fillna(0)
        
        # Create acoustic-to-vibration ratio (useful for bearing monitoring)
        if 'acoustic' in features.columns and 'vibration' in features.columns:
            features['acoustic_vibration_ratio'] = features['acoustic'] / features['vibration'].replace(0, 0.001)
        
        # Remove non-feature columns
        drop_columns = ['timestamp', 'machine_id']
        X = features.drop(columns=drop_columns + ['failure', 'maintenance', 'anomaly'], errors='ignore')
        
        return X
    
    def engineer_haas_cnc_features(self, df):
        """
        Create machine-specific features for HAAS VF-2 CNC Milling Machine.
        
        Args:
            df: DataFrame containing HAAS CNC data
            
        Returns:
            DataFrame with engineered features
        """
        # Check for required columns
        required_cols = ['spindle_load', 'vibration', 'temperature', 'acoustic']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column {col} missing for HAAS CNC")
                df[col] = np.nan
        
        # Create feature dataframe
        features = df.copy()
        
        # Add normalized features based on normal operating ranges
        for col in required_cols:
            if col in features.columns:
                min_val, max_val = self.normal_ranges["haas_cnc"][col]
                features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
                
                # Create deviation from normal range feature
                features[f'{col}_deviation'] = features[col].apply(
                    lambda x: max(0, min_val - x) / min_val if x < min_val 
                    else max(0, x - max_val) / max_val if x > max_val else 0
                )
        
        # Create time-based features
        if 'timestamp' in features.columns:
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
        
        # Create rolling window features
        for col in required_cols:
            if col in features.columns:
                features[f'{col}_rolling_mean_12h'] = features[col].rolling(window=12, min_periods=1).mean()
                features[f'{col}_rolling_std_12h'] = features[col].rolling(window=12, min_periods=1).std().fillna(0)
        
        # Create spindle efficiency indicator - higher temp & vibration at the same load indicates problems
        if all(col in features.columns for col in ['spindle_load', 'temperature', 'vibration']):
            features['spindle_efficiency'] = features['spindle_load'] / (features['temperature'] * features['vibration'])
            
        # Acoustic-to-vibration ratio (changes can indicate issues)
        if 'acoustic' in features.columns and 'vibration' in features.columns:
            features['acoustic_vibration_ratio'] = features['acoustic'] / features['vibration'].replace(0, 0.001)
        
        # Remove non-feature columns
        drop_columns = ['timestamp', 'machine_id']
        X = features.drop(columns=drop_columns + ['failure', 'maintenance', 'anomaly'], errors='ignore')
        
        return X
    
    def engineer_grundfos_pump_features(self, df):
        """
        Create machine-specific features for Grundfos CR Vertical Multistage Pumps.
        
        Args:
            df: DataFrame containing Grundfos pump data
            
        Returns:
            DataFrame with engineered features
        """
        # Check for required columns
        required_cols = ['pressure', 'flow_rate', 'temperature', 'power']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column {col} missing for Grundfos pump")
                df[col] = np.nan
        
        # Create feature dataframe
        features = df.copy()
        
        # Add normalized features based on normal operating ranges
        for col in required_cols:
            if col in features.columns:
                min_val, max_val = self.normal_ranges["grundfos_pump"][col]
                features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
                
                # Create deviation from normal range feature
                features[f'{col}_deviation'] = features[col].apply(
                    lambda x: max(0, min_val - x) / min_val if x < min_val 
                    else max(0, x - max_val) / max_val if x > max_val else 0
                )
        
        # Create time-based features
        if 'timestamp' in features.columns:
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
        
        # Create rolling window features
        for col in required_cols:
            if col in features.columns:
                features[f'{col}_rolling_mean_12h'] = features[col].rolling(window=12, min_periods=1).mean()
                features[f'{col}_rolling_std_12h'] = features[col].rolling(window=12, min_periods=1).std().fillna(0)
        
        # Calculate pump efficiency related metrics
        if 'pressure' in features.columns and 'flow_rate' in features.columns and 'power' in features.columns:
            # Simplified hydraulic power calculation (proportional)
            features['hydraulic_power'] = features['pressure'] * features['flow_rate']
            # Efficiency metric
            features['pump_efficiency'] = features['hydraulic_power'] / features['power'].replace(0, 0.001)
            # Efficiency change over time
            features['efficiency_change'] = features['pump_efficiency'].pct_change().fillna(0)
        
        # Remove non-feature columns
        drop_columns = ['timestamp', 'machine_id']
        X = features.drop(columns=drop_columns + ['failure', 'maintenance', 'anomaly'], errors='ignore')
        
        return X
    
    def engineer_carrier_chiller_features(self, df):
        """
        Create machine-specific features for Carrier 30XA Air-Cooled Chiller.
        
        Args:
            df: DataFrame containing Carrier chiller data
            
        Returns:
            DataFrame with engineered features
        """
        # Check for required columns
        required_cols = ['refrigerant_pressure', 'condenser_temp', 'evaporator_temp', 'power']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column {col} missing for Carrier chiller")
                df[col] = np.nan
        
        # Create feature dataframe
        features = df.copy()
        
        # Add normalized features based on normal operating ranges
        for col in required_cols:
            if col in features.columns:
                min_val, max_val = self.normal_ranges["carrier_chiller"][col]
                features[f'{col}_normalized'] = (features[col] - min_val) / (max_val - min_val)
                
                # Create deviation from normal range feature
                features[f'{col}_deviation'] = features[col].apply(
                    lambda x: max(0, min_val - x) / min_val if x < min_val 
                    else max(0, x - max_val) / max_val if x > max_val else 0
                )
        
        # Create time-based features
        if 'timestamp' in features.columns:
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
        
        # Create rolling window features
        for col in required_cols:
            if col in features.columns:
                features[f'{col}_rolling_mean_12h'] = features[col].rolling(window=12, min_periods=1).mean()
                features[f'{col}_rolling_std_12h'] = features[col].rolling(window=12, min_periods=1).std().fillna(0)
        
        # Calculate temperature differential (critical for chillers)
        if 'condenser_temp' in features.columns and 'evaporator_temp' in features.columns:
            features['temp_differential'] = features['condenser_temp'] - features['evaporator_temp']
            
        # Calculate COP (Coefficient of Performance) - a measure of efficiency
        if 'temp_differential' in features.columns and 'power' in features.columns:
            # Simplified COP calculation
            features['approx_cop'] = features['evaporator_temp'] / features['temp_differential']
            
        # Pressure-temperature relationship (useful for refrigerant issues)
        if 'refrigerant_pressure' in features.columns and 'condenser_temp' in features.columns:
            features['pressure_temp_ratio'] = features['refrigerant_pressure'] / features['condenser_temp']
        
        # Remove non-feature columns
        drop_columns = ['timestamp', 'machine_id']
        X = features.drop(columns=drop_columns + ['failure', 'maintenance', 'anomaly'], errors='ignore')
        
        return X
        
    def prepare_features_and_target(self, machine_data, machine_type):
        """
        Prepare features and target variables for a specific machine type.
        
        Args:
            machine_data: DataFrame containing machine-specific data
            machine_type: The type of machine to create features for
            
        Returns:
            X: Feature DataFrame
            y: Target Series (failure flag)
        """
        # Create a copy to avoid modifying the original data
        df = machine_data.copy()
        
        # Check for and handle missing values specific to this machine type
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col not in ['machine_id', 'failure', 'maintenance', 'anomaly'] and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Get the target variable - the failure flag
        y = df['failure']
        
        # Apply machine-specific feature engineering based on machine type
        if "siemens_motor" in machine_type:
            X = self.engineer_siemens_motor_features(df)
        elif "abb_bearing" in machine_type:
            X = self.engineer_abb_bearing_features(df)
        elif "haas_cnc" in machine_type:
            X = self.engineer_haas_cnc_features(df)
        elif "grundfos_pump" in machine_type:
            X = self.engineer_grundfos_pump_features(df)
        elif "carrier_chiller" in machine_type:
            X = self.engineer_carrier_chiller_features(df)
        else:
            # Fallback for unknown machine types - basic feature extraction
            print(f"Warning: Unknown machine type '{machine_type}'. Using basic feature extraction.")
            # Make a copy to avoid modifying the original data
            X = df.copy()
            
            # Create time-based features if timestamp is available
            if 'timestamp' in X.columns:
                X['hour'] = X['timestamp'].dt.hour
                X['day_of_week'] = X['timestamp'].dt.dayofweek
                
            # Drop non-feature columns
            drop_columns = ['timestamp', 'machine_id']
            X = X.drop(columns=drop_columns + ['failure', 'maintenance', 'anomaly'], errors='ignore')
        
        # Handle NaNs that might have been introduced during feature engineering
        X = X.fillna(0)
        
        print(f"Created {len(X.columns)} machine-specific features for {machine_type}")
        return X, y
    
    def create_deep_learning_model(self, input_dim):
        """
        Create a deep learning model for failure prediction.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Keras model
        """
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_model_pipeline(self, X):
        """
        Build a machine learning pipeline for predictive maintenance.
        
        Args:
            X: Feature DataFrame to determine columns
            
        Returns:
            Pipeline: Scikit-learn pipeline for preprocessing and model
        """
        # Create numeric preprocessing pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])
        
        # Create the pipeline based on selected model type
        if self.model_type == 'deep_learning':
            # Create a Keras model wrapped for scikit-learn
            keras_model = KerasClassifier(
                model=lambda: self.create_deep_learning_model(len(numeric_features)),
                epochs=50,  # Default epochs, will be adjusted with early stopping
                batch_size=32,
                verbose=1
            )
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', keras_model)
            ])
        else:  # Ensemble model (default)
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ))
            ])
        
        return pipeline
    
    def optimize_hyperparameters(self, pipeline, X_train, y_train):
        """
        Perform hyperparameter optimization using GridSearchCV.
        
        Args:
            pipeline: Model pipeline to optimize
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best pipeline found from grid search
        """
        if self.model_type == 'deep_learning':
            # For deep learning, we'll use the default hyperparameters
            # Early stopping will help us find the right number of epochs
            return pipeline
        else:
            # For ensemble models, we'll use GridSearchCV
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=3),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            print("Performing hyperparameter optimization...")
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
    
    def train_and_evaluate(self, machine_type):
        """
        Train and evaluate a model for a specific machine type.
        
        Args:
            machine_type: The type of machine to train a model for
            
        Returns:
            dict: Model evaluation metrics
        """
        print(f"\nTraining model for {machine_type}...")
        
        # Get machine-specific data
        machine_data = self.get_machine_specific_data(machine_type)
        
        if len(machine_data) == 0:
            print(f"No data found for {machine_type}. Skipping...")
            return None
        
        print(f"Found {len(machine_data)} samples for {machine_type}")
        
        # Prepare features and target with machine-specific feature engineering
        X, y = self.prepare_features_and_target(machine_data, machine_type)
        
        # Print class distribution
        print(f"Class distribution: {dict(pd.Series(y).value_counts())}")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model pipeline
        model = self.build_model_pipeline(X)
        
        # Optimize hyperparameters if using ensemble model
        if self.model_type != 'deep_learning':
            model = self.optimize_hyperparameters(model, X_train, y_train)
            model.fit(X_train, y_train)
        else:
            # For deep learning, use Keras callbacks for early stopping
            early_stopping = EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Manually fit the deep learning model with callbacks
            model.fit(
                X_train, y_train,
                callbacks=[early_stopping, reduce_lr],
                validation_split=0.2,
                epochs=100,  # Maximum epochs, early stopping will usually trigger before this
                batch_size=32
            )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC-AUC and PR-AUC
        roc_auc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        # Save the model
        self.models[machine_type] = model
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {machine_type}')
        
        # Create directory if it doesn't exist
        os.makedirs('plots/model_evaluation', exist_ok=True)
        plt.savefig(f'plots/model_evaluation/{machine_type}_confusion_matrix.png')
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'PR-AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {machine_type}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/model_evaluation/{machine_type}_pr_curve.png')
        
        # Plot feature importance if available
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                feature_names = X.columns
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 8))
                plt.title(f'Feature Importance - {machine_type}')
                plt.bar(range(min(20, len(importances))), 
                        [importances[i] for i in indices[:20]], 
                        align='center')
                plt.xticks(range(min(20, len(importances))), 
                          [feature_names[i] for i in indices[:20]], 
                          rotation=90)
                plt.tight_layout()
                plt.savefig(f'plots/model_evaluation/{machine_type}_feature_importance.png')
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'report': report,
            'model': model
        }
    
    def save_model(self, machine_type):
        """
        Save a trained model to disk.
        
        Args:
            machine_type: The type of machine whose model to save
        """
        if machine_type not in self.models:
            print(f"No model found for {machine_type}. Train a model first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Get the feature names this model was trained on
        machine_data = self.get_machine_specific_data(machine_type)
        X, _ = self.prepare_features_and_target(machine_data, machine_type)
        feature_names = X.columns.tolist()
        
        # Save model with the feature names
        model_data = {
            'model': self.models[machine_type],
            'feature_names': feature_names
        }
        
        model_path = f'models/{machine_type}_model.pkl'
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path} with {len(feature_names)} relevant features")
    
    def train_all_models(self):
        """Train models for all machine types."""
        self.load_data()
        self.preprocess_data()
        
        results = {}
        for machine_type in self.machine_types:
            results[machine_type] = self.train_and_evaluate(machine_type)
            if results[machine_type]:
                self.save_model(machine_type)
        
        return results

if __name__ == "__main__":
    # Parse command line arguments for model type
    import argparse
    parser = argparse.ArgumentParser(description='Train predictive maintenance models')
    parser.add_argument('--model', type=str, default='ensemble', 
                        choices=['ensemble', 'deep_learning'],
                        help='Type of model to train: ensemble or deep_learning')
    args = parser.parse_args()
    
    # Create and train models
    predictive_model = PredictiveMaintenanceModel(model_type=args.model)
    results = predictive_model.train_all_models()
    
    # Print summary
    print("\nModel Training Summary:")
    for machine_type, result in results.items():
        if result:
            print(f"{machine_type}: Accuracy = {result['accuracy']:.4f}, PR-AUC = {result['pr_auc']:.4f}")
        else:
            print(f"{machine_type}: No data available") 