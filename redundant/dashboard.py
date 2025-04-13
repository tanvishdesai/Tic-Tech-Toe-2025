import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import math
import pickle
import os
from sklearn.preprocessing import StandardScaler
from src.predictive_maintenance_model import PredictiveMaintenanceModel
import joblib


def simulate_sensor_data(machine_type, start_date, end_date, freq="h", scenario="gradual_degradation", 
                        anomaly_frequency=0.05, degradation_severity=0.3, add_seasonality=False, 
                        failure_event=False, maintenance_effect=False):
    """
    Simulate sensor data for a given machine type and time range with various scenarios.
    
    Args:
        machine_type: Type of machine to simulate data for
        start_date: Start date for simulation
        end_date: End date for simulation
        freq: Data frequency (hourly, etc.)
        scenario: Simulation scenario (gradual_degradation, rapid_degradation, stable, cyclic_load)
        anomaly_frequency: Frequency of anomalies (0.0 to 1.0)
        degradation_severity: Severity of degradation (0.0 to 1.0)
        add_seasonality: Whether to add seasonal patterns to the data
        failure_event: Whether to simulate a catastrophic failure event
        maintenance_effect: Whether to simulate maintenance intervention effect
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    # Define machine-specific sensor parameters with their normal ranges and weights
    sensors = {
        "siemens_motor": {
            "temp": {"normal_range": (40, 60), "weight": 0.3},
            "vibration": {"normal_range": (0.1, 0.5), "weight": 0.3},
            "current": {"normal_range": (10, 20), "weight": 0.2},
            "voltage": {"normal_range": (220, 240), "weight": 0.1},
            "power": {"normal_range": (1000, 1200), "weight": 0.1}
        },
        "abb_bearing": {
            "temp": {"normal_range": (30, 50), "weight": 0.3},
            "vibration": {"normal_range": (0.05, 0.3), "weight": 0.4},
            "acoustic": {"normal_range": (20, 50), "weight": 0.3}
        },
        "haas_cnc": {
            "spindle_speed": {"normal_range": (2000, 5000), "weight": 0.3},
            "spindle_load": {"normal_range": (20, 80), "weight": 0.3},
            "temp": {"normal_range": (50, 80), "weight": 0.2},
            "vibration": {"normal_range": (0.1, 0.4), "weight": 0.2}
        },
        "grundfos_pump": {
            "flow_rate": {"normal_range": (40, 100), "weight": 0.4},
            "pressure": {"normal_range": (20, 50), "weight": 0.3},
            "temp": {"normal_range": (30, 60), "weight": 0.2},
            "power": {"normal_range": (800, 1000), "weight": 0.1}
        },
        "carrier_chiller": {
            "refrigerant_pressure": {"normal_range": (30, 60), "weight": 0.3},
            "evaporator_temp": {"normal_range": (5, 15), "weight": 0.3},
            "condenser_temp": {"normal_range": (30, 45), "weight": 0.2},
            "power": {"normal_range": (5000, 6000), "weight": 0.2}
        }
    }
    
    # Default if machine not found
    if machine_type not in sensors:
        sensors[machine_type] = {
            "temp": {"normal_range": (30, 60), "weight": 0.5},
            "vibration": {"normal_range": (0.1, 0.5), "weight": 0.5}
        }
    
    # Generate data with scenario-specific patterns
    data = {}
    sensor_specs = sensors[machine_type]
    
    # Get simulation parameters
    time_steps = len(date_range)
    
    # Initialize degradation profile based on scenario
    if scenario == "gradual_degradation":
        degradation = np.linspace(0, degradation_severity, time_steps)  # Gradual degradation
    elif scenario == "rapid_degradation":
        # Start with slow degradation, then rapid increase
        mid_point = time_steps // 2
        degradation = np.zeros(time_steps)
        degradation[:mid_point] = np.linspace(0, 0.1, mid_point)
        degradation[mid_point:] = np.linspace(0.1, degradation_severity * 2, time_steps - mid_point)
    elif scenario == "stable":
        # Stable operation with minimal degradation
        degradation = np.linspace(0, 0.05, time_steps)
    elif scenario == "cyclic_load":
        # Cyclic load with periodic stress and recovery
        base = np.linspace(0, degradation_severity * 0.5, time_steps)
        cycle = 0.2 * np.sin(np.linspace(0, 10 * np.pi, time_steps))
        degradation = base + cycle
    else:  # Default behavior
        degradation = np.linspace(0, degradation_severity, time_steps)
    
    # Add anomaly spikes
    anomaly_mask = np.random.random(time_steps) < anomaly_frequency
    
    # If simulating failure event, add catastrophic degradation at the end
    if failure_event:
        # Last 10% of the timeline shows rapid failure
        failure_start = int(time_steps * 0.9)
        degradation[failure_start:] = np.linspace(degradation[failure_start], 1.0, time_steps - failure_start)
        # Add more anomalies during failure
        anomaly_mask[failure_start:] = np.random.random(time_steps - failure_start) < (anomaly_frequency * 3)
    
    # If simulating maintenance effect, add recovery after mid-point
    if maintenance_effect:
        maintenance_point = int(time_steps * 0.6)
        # Temporary improvement after maintenance
        degradation[maintenance_point:maintenance_point+int(time_steps*0.2)] *= 0.3
        # Resume degradation after maintenance effect wears off
    
    for sensor_name, specs in sensor_specs.items():
        low, high = specs["normal_range"]
        
        # Base values within normal range
        base_values = np.random.uniform(low=low, high=high, size=time_steps)
        
        # Apply seasonality if requested (e.g., temperature varies with time of day)
        if add_seasonality and sensor_name in ["temp", "evaporator_temp", "condenser_temp", "power"]:
            # Create a daily seasonal pattern
            if isinstance(date_range[0], pd.Timestamp):
                hours = np.array([d.hour for d in date_range])
                # Daily pattern: peak during working hours
                daily_pattern = 0.1 * np.sin(hours * (2 * np.pi / 24))
                base_values += (high - low) * daily_pattern
        
        # Apply degradation effect based on sensor type
        if sensor_name in ["temp", "evaporator_temp", "condenser_temp"]:
            # Temperature tends to increase with degradation
            values = base_values * (1 + degradation)
        elif sensor_name in ["vibration", "acoustic"]:
            # Vibration increases exponentially with degradation
            values = base_values * (1 + degradation**2 * 3)
        elif sensor_name in ["pressure", "refrigerant_pressure", "flow_rate"]:
            # Pressure and flow might decrease with degradation
            values = base_values * (1 - degradation * 0.5)
        elif sensor_name in ["power", "current"]:
            # Power and current might increase with degradation
            values = base_values * (1 + degradation * 0.8)
        else:
            # Other parameters vary linearly with degradation
            values = base_values * (1 + degradation * 0.5)
        
        # Add anomalies - magnitude varies by scenario
        if scenario == "rapid_degradation":
            anomaly_magnitude = np.random.uniform(1.3, 1.8, size=sum(anomaly_mask))
        elif scenario == "stable":
            anomaly_magnitude = np.random.uniform(1.1, 1.3, size=sum(anomaly_mask))
        else:
            anomaly_magnitude = np.random.uniform(1.2, 1.5, size=sum(anomaly_mask))
            
        values[anomaly_mask] *= anomaly_magnitude
        
        # Add noise - magnitude varies by scenario
        if scenario == "stable":
            noise_level = 0.01  # Minimal noise for stable operation
        elif scenario == "rapid_degradation":
            noise_level = 0.05  # More noise during rapid degradation
        else:
            noise_level = 0.03  # Default noise level
            
        values += np.random.normal(0, (high-low)*noise_level, size=time_steps)
        
        data[sensor_name] = values
        
    data["timestamp"] = date_range
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    
    return df, sensor_specs


def calculate_health_index(data, sensor_specs):
    """Calculate machine health index based on sensor data and specifications."""
    health_index = pd.Series(index=data.index, dtype=float)
    health_index[:] = 100  # Start with perfect health - fixed fill() method
    
    # Calculate health contribution from each sensor
    for sensor_name, specs in sensor_specs.items():
        if sensor_name in data.columns:
            low, high = specs["normal_range"]
            weight = specs["weight"]
            
            # Calculate how much each reading deviates from the normal range
            readings = data[sensor_name]
            
            # Different sensors have different degradation patterns
            if sensor_name in ["temp"]:
                # For temperature, higher is worse
                deviations = np.maximum(0, (readings - high) / (high - low) * 100)
            elif sensor_name in ["vibration"]:
                # For vibration, exponential impact when exceeding threshold
                norm_readings = (readings - low) / (high - low)
                deviations = np.where(norm_readings > 1, 
                                      (norm_readings - 1) * 150,  # Amplify impact
                                      np.maximum(0, (norm_readings - 0.8) * 50))  # Minor penalty near upper range
            else:
                # For other sensors, linear deviation from normal range
                norm_readings = (readings - low) / (high - low)
                deviations = np.where(
                    readings < low, (low - readings) / (high - low) * 100,
                    np.where(readings > high, (readings - high) / (high - low) * 100, 0)
                )
            
            # Apply sensor weight to health reduction
            health_index -= deviations * weight
    
    # Special handling for the "failure_event" scenario - ensure health reaches critical levels
    if health_index.values.size > 0:
        min_health = health_index.min()
        if min_health < 1:
            # Make sure we don't have exactly 0.0 health (lowest should be 0.1)
            health_index = np.maximum(health_index, 0.1)
            
        # Check if we have critical health pattern (values below 20)
        critical_points = (health_index < 20).sum()
        if critical_points > 0 and critical_points < len(health_index) * 0.1:
            # If only a few points are critical, smoothly transition there
            # Find the critical indices
            critical_indices = health_index[health_index < 20].index
            
            # For each critical index, ensure previous points have a reasonable decline
            for idx in critical_indices:
                pos = health_index.index.get_loc(idx)
                if pos > 0:
                    # Get previous 3 points (or fewer if not available)
                    prior_points = min(3, pos)
                    for i in range(1, prior_points + 1):
                        prior_idx = health_index.index[pos - i]
                        # Create a gradual decline to the critical point
                        health_index.loc[prior_idx] = min(
                            health_index.loc[prior_idx],
                            health_index.loc[idx] + (i * 15)  # Add at most 45 points back
                        )
    
    # Ensure health index stays within 0.1-100 range (never exactly 0)
    health_index = np.clip(health_index, 0.1, 100)
    
    return health_index


def predict_failure(health_index, machine_type, data):
    """Predict time to failure and failure probability based on health index and machine type.
    Uses the trained models if available, otherwise falls back to the rule-based approach."""
    
    model_path = os.path.join("models", f"{machine_type}_model.pkl")
    
    # Check if model exists for this machine type
    if os.path.exists(model_path):
        try:
            # Load the model and its feature names
            model_data = joblib.load(model_path)
            
            # Unpack the model and feature names
            if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
                model = model_data['model']
                required_features = model_data['feature_names']
                st.success(f"Using ML model for {machine_type} with {len(required_features)} relevant features")
            else:
                # Backward compatibility with older models
                model = model_data
                # Try to determine features from the model if possible
                if hasattr(model, 'feature_names_in_'):
                    required_features = model.feature_names_in_
                elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    required_features = model.named_steps['classifier'].feature_names_in_
                else:
                    # If we can't determine features, use what we have in the data
                    required_features = data.columns
            
            # Prepare features for prediction
            features = pd.DataFrame()
            
            # Add sensor readings, but only for those features the model was trained on
            for col in data.columns:
                if col != 'timestamp' and col in required_features:
                    features[col] = data[col]
            
            # Add time-based features if needed
            if isinstance(data.index, pd.DatetimeIndex):
                if 'hour' in required_features:
                    features['hour'] = data.index.hour
                if 'day_of_week' in required_features:
                    features['day_of_week'] = data.index.dayofweek
            
            # Add rolling statistics if needed
            for col in data.columns:
                if col != 'timestamp':
                    feature_name = f'{col}_rolling_mean'
                    if feature_name in required_features:
                        features[feature_name] = data[col].rolling(window=3, min_periods=1).mean()
                    
                    feature_name = f'{col}_rolling_std'
                    if feature_name in required_features:
                        features[feature_name] = data[col].rolling(window=3, min_periods=1).std().fillna(0)
            
            # Add health index if needed
            if 'health_index' in required_features:
                features['health_index'] = health_index
            
            # Get latest data point
            latest_features = features.iloc[-1:].copy()
            
            try:
                # Create a DataFrame with exactly the required features
                prediction_features = pd.DataFrame(index=[0])
                
                # Fill in the features we have
                for col in required_features:
                    if col in latest_features.columns:
                        prediction_features[col] = latest_features[col].values
                    else:
                        # Only add missing features if absolutely necessary
                        prediction_features[col] = 0
                
                # Store current health for consistency checks later
                current_health = health_index.iloc[-1]
                
                # Make prediction using the model pipeline
                if hasattr(model, 'predict_proba'):
                    # For classification models
                    prob_prediction = model.predict_proba(prediction_features)
                    failure_probability = prob_prediction[0][1]
                    
                    # Map probability to time ranges
                    if failure_probability >= 0.7:
                        time_to_failure = 2
                    elif failure_probability >= 0.5:
                        time_to_failure = 7
                    elif failure_probability >= 0.3:
                        time_to_failure = 15
                    elif failure_probability >= 0.1:
                        time_to_failure = 30
                    else:
                        time_to_failure = 60
                elif hasattr(model, 'predict'):
                    # For regression models
                    time_to_failure = float(model.predict(prediction_features)[0])
                    failure_probability = math.exp(-0.05 * time_to_failure)
                else:
                    raise AttributeError("Model doesn't have predict or predict_proba method")
                
                # Check for consistency with health index
                current_health = health_index.iloc[-1]
                # If health is excellent (>90) but prediction is bad, adjust prediction
                if current_health > 90 and failure_probability > 0.3:
                    st.warning("Model prediction adjusted for consistency with excellent health index.")
                    # More aggressive scaling down of probability based on health index
                    failure_probability = failure_probability * 0.2  # Much lower probability for excellent health
                    # Adjust time to failure according to new probability
                    if failure_probability >= 0.7:
                        time_to_failure = 2
                    elif failure_probability >= 0.5:
                        time_to_failure = 7
                    elif failure_probability >= 0.3:
                        time_to_failure = 15
                    elif failure_probability >= 0.1:
                        time_to_failure = 30
                    else:
                        time_to_failure = 60
                    
                    # For excellent health, ensure minimum reasonable time to failure
                    time_to_failure = max(time_to_failure, 45)
                
                # If health is poor (<40) but prediction is good, adjust prediction
                elif current_health < 40 and failure_probability < 0.7:
                    st.warning("Model prediction adjusted for consistency with poor health index.")
                    # Scale up probability based on health index
                    failure_probability = failure_probability + (0.95 - failure_probability) * (1 - current_health / 40)
                    # Adjust time to failure according to new probability
                    if failure_probability >= 0.7:
                        time_to_failure = 2
                    elif failure_probability >= 0.5:
                        time_to_failure = 7
                    elif failure_probability >= 0.3:
                        time_to_failure = 15
                    elif failure_probability >= 0.1:
                        time_to_failure = 30
                    else:
                        time_to_failure = 60
                
                # After adjusting probability based on health index, make final consistency check
                # Ensure time_to_failure and failure_probability match logically
                if time_to_failure < 5 and failure_probability < 0.6:
                    # Short TTF should have higher probability
                    failure_probability = max(failure_probability, 0.6)
                elif time_to_failure > 30 and failure_probability > 0.3:
                    # Long TTF should have lower probability
                    failure_probability = min(failure_probability, 0.3)
                elif time_to_failure > 45 and failure_probability > 0.1:
                    # Very long TTF should have very low probability
                    failure_probability = min(failure_probability, 0.1)
                
                # Final health index consistency check
                if current_health > 90:
                    # Excellent health - ensure long TTF and low probability
                    time_to_failure = max(time_to_failure, 45)
                    failure_probability = min(failure_probability, 0.15)
                elif current_health > 75:
                    # Good health - ensure reasonable TTF and probability
                    time_to_failure = max(time_to_failure, 25)
                    failure_probability = min(failure_probability, 0.3)
                elif current_health < 30:
                    # Poor health - ensure short TTF and high probability
                    time_to_failure = min(time_to_failure, 7)
                    failure_probability = max(failure_probability, 0.7)
                elif current_health < 50:
                    # Concerning health - adjust accordingly
                    time_to_failure = min(time_to_failure, 15)
                    failure_probability = max(failure_probability, 0.5)
                
                return round(time_to_failure, 1), round(failure_probability, 2)
                
            except Exception as e:
                st.warning(f"Model prediction error: {str(e)}. Falling back to rule-based approach.")
        except Exception as e:
            st.warning(f"Error loading model {model_path}: {str(e)}. Using rule-based approach instead.")
    
    # Fall back to rule-based approach
    # Get current health and calculate recent trend
    current_health = health_index.iloc[-1]
    
    # Check if health is critical (near zero) - handle this case explicitly
    if current_health <= 10:
        # Health is critically low, failure is imminent
        time_to_failure = 0
        failure_probability = 1.0
        return round(time_to_failure, 1), round(failure_probability, 2)
    
    # Calculate trend over the last 1/3 of the data points
    trend_period = max(1, len(health_index) // 3)
    recent_health = health_index.iloc[-trend_period:]
    
    if len(recent_health) > 1:
        # Calculate linear regression on recent health data
        x = np.arange(len(recent_health))
        y = recent_health.values
        slope, _ = np.polyfit(x, y, 1)
        
        # Daily health degradation rate (convert to daily rate)
        samples_per_day = 24  # Assuming hourly data
        if len(health_index) > 0 and isinstance(health_index.index, pd.DatetimeIndex):
            time_diff = (health_index.index[-1] - health_index.index[0]).total_seconds()
            if time_diff > 0:
                samples = len(health_index)
                seconds_per_day = 24 * 60 * 60
                samples_per_day = samples / (time_diff / seconds_per_day)
        
        daily_degradation = slope * samples_per_day
    else:
        # Default degradation if we can't calculate trend
        daily_degradation = -1.0
    
    # Adjust degradation to ensure it's negative (health decreases over time)
    daily_degradation = min(daily_degradation, -0.1)
    
    # Machine-specific failure thresholds
    failure_thresholds = {
        "siemens_motor": 30,
        "abb_bearing": 25,
        "haas_cnc": 35,
        "grundfos_pump": 20,
        "carrier_chiller": 25
    }
    
    failure_threshold = failure_thresholds.get(machine_type, 30)
    
    # Calculate time to failure
    if current_health <= failure_threshold:
        # Already at or below failure threshold
        time_to_failure = 0
    elif daily_degradation >= 0:
        # If somehow trend is positive or flat, use a default
        time_to_failure = 30
    else:
        # Calculate days until reaching failure threshold
        time_to_failure = (failure_threshold - current_health) / abs(daily_degradation)
        # Add a slight randomization for realism
        time_to_failure *= random.uniform(0.9, 1.1)
        
    # Ensure prediction is within reasonable bounds
    time_to_failure = max(0, min(365, time_to_failure))
    
    # Calculate failure probability using exponential model
    # Low health = high probability, with exponential increase as health approaches failure threshold
    health_margin = max(0, current_health - failure_threshold)
    base_probability = math.exp(-0.05 * health_margin)
    
    # Adjust probability based on degradation rate
    trend_factor = min(1.0, abs(daily_degradation) / 2)
    failure_probability = min(1.0, base_probability * (1 + trend_factor))
    
    # Ensure consistency between probability and time to failure
    # If health is very low, probability should be high
    if current_health < 20:
        failure_probability = max(failure_probability, 0.9)
    elif current_health < 40:
        failure_probability = max(failure_probability, 0.7)
        
    # If time to failure is very low, probability should be high
    if time_to_failure < 1:
        failure_probability = max(failure_probability, 0.95)
    elif time_to_failure < 5:
        failure_probability = max(failure_probability, 0.8)
    
    # Ensure high health index means low failure probability
    if current_health > 90:
        # Maximum probability with excellent health should be very low
        failure_probability = min(failure_probability, 0.1)
        # Minimum time to failure with excellent health should be high
        time_to_failure = max(time_to_failure, 45)
    elif current_health > 80:
        failure_probability = min(failure_probability, 0.2)
        time_to_failure = max(time_to_failure, 30)
    
    # Final consistency check between time to failure and probability
    if time_to_failure < 5 and failure_probability < 0.6:
        # Short TTF should have higher probability
        failure_probability = max(failure_probability, 0.6)
    elif time_to_failure > 30 and failure_probability > 0.3:
        # Long TTF should have lower probability
        failure_probability = min(failure_probability, 0.3)
    elif time_to_failure > 45 and failure_probability > 0.1:
        # Very long TTF should have very low probability
        failure_probability = min(failure_probability, 0.1)
    
    # Final health index consistency check
    if current_health > 90:
        # Excellent health - ensure long TTF and low probability
        time_to_failure = max(time_to_failure, 45)
        failure_probability = min(failure_probability, 0.15)
    elif current_health > 75:
        # Good health - ensure reasonable TTF and probability
        time_to_failure = max(time_to_failure, 25)
        failure_probability = min(failure_probability, 0.3)
    elif current_health < 30:
        # Poor health - ensure short TTF and high probability
        time_to_failure = min(time_to_failure, 7)
        failure_probability = max(failure_probability, 0.7)
    elif current_health < 50:
        # Concerning health - adjust accordingly
        time_to_failure = min(time_to_failure, 15)
        failure_probability = max(failure_probability, 0.5)
    
    return round(time_to_failure, 1), round(failure_probability, 2)


def recommend_maintenance(time_to_failure, failure_probability, machine_type):
    """Recommend maintenance schedule based on time to failure, probability and machine type."""
    # Machine criticality (scale of 1-10, higher means more critical)
    criticality = {
        "siemens_motor": 8,
        "abb_bearing": 7,
        "haas_cnc": 9,
        "grundfos_pump": 8,
        "carrier_chiller": 6
    }.get(machine_type, 7)
    
    # Maintenance lead time by machine type (days needed to prepare)
    lead_times = {
        "siemens_motor": 2,
        "abb_bearing": 1,
        "haas_cnc": 3,
        "grundfos_pump": 2,
        "carrier_chiller": 2
    }.get(machine_type, 2)
    
    # Critical case - immediate maintenance needed
    if time_to_failure == 0 or failure_probability >= 0.95:
        urgency = "Critical"
        offset = 0
        maintenance_date = datetime.now()
        return urgency, maintenance_date, offset
    
    # Determine maintenance urgency
    if time_to_failure <= 3 or failure_probability >= 0.8:
        urgency = "Urgent"
        offset = max(0, min(time_to_failure - 1, lead_times))
    elif time_to_failure <= 10 or failure_probability >= 0.5:
        urgency = "High"
        offset = min(time_to_failure * 0.3, lead_times + 2)
    elif time_to_failure <= 20 or failure_probability >= 0.3:
        urgency = "Medium"
        offset = min(time_to_failure * 0.5, 7)
    else:
        urgency = "Low"
        offset = min(time_to_failure * 0.7, 14)
    
    # Adjust for machine criticality
    if criticality >= 8:
        offset = max(1, offset * 0.8)  # Schedule maintenance earlier for critical machines
    
    # Recommended maintenance date
    maintenance_date = datetime.now() + timedelta(days=max(1, round(offset)))
    
    return urgency, maintenance_date, round(offset)


def main():
    st.title("Predictive Maintenance Dashboard")

    st.sidebar.header("Machine Selection & Simulation Settings")

    machine_types = ["siemens_motor", "abb_bearing", "haas_cnc", "grundfos_pump", "carrier_chiller"]
    selected_machine = st.sidebar.selectbox("Select Machine", machine_types)

    # Date input for simulation time frame
    today = datetime.now()
    default_start = today - timedelta(days=3)
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)

    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        return

    freq = st.sidebar.selectbox("Data Frequency", options=["h", "30T", "15T"], index=0,
                                format_func=lambda x: {"h": "Hourly", "30T": "Every 30 Minutes", "15T": "Every 15 Minutes"}[x])
    
    # Add simulation scenario options
    st.sidebar.header("Simulation Scenarios")
    scenario = st.sidebar.selectbox(
        "Select Scenario",
        options=["gradual_degradation", "rapid_degradation", "stable", "cyclic_load"],
        format_func=lambda x: {
            "gradual_degradation": "Gradual Degradation", 
            "rapid_degradation": "Rapid Degradation",
            "stable": "Stable Operation",
            "cyclic_load": "Cyclic Load Pattern"
        }[x]
    )
    
    # Add more simulation parameters
    anomaly_frequency = st.sidebar.slider("Anomaly Frequency", 0.0, 0.5, 0.05, 0.01, 
                                         help="Frequency of anomalies in the data (0.05 = 5%)")
    
    degradation_severity = st.sidebar.slider("Degradation Severity", 0.0, 1.0, 0.3, 0.1,
                                            help="How severe the degradation becomes (higher = faster failure)")
    
    add_seasonality = st.sidebar.checkbox("Add Seasonal Patterns", False,
                                         help="Add daily/hourly seasonal patterns to the data")
    
    failure_event = st.sidebar.checkbox("Simulate Failure Event", False,
                                       help="Add a catastrophic failure at the end of the timeline")
    
    maintenance_effect = st.sidebar.checkbox("Simulate Maintenance Effect", False,
                                           help="Simulate the effect of maintenance intervention")

    st.write(f"Simulated sensor data for **{selected_machine}** from {start_date} to {end_date} (Frequency: {freq})")
    st.write(f"Scenario: **{scenario.replace('_', ' ').title()}**")

    # Simulate sensor data with the chosen scenario
    data, sensor_specs = simulate_sensor_data(
        selected_machine, 
        start_date, 
        end_date, 
        freq,
        scenario=scenario,
        anomaly_frequency=anomaly_frequency,
        degradation_severity=degradation_severity,
        add_seasonality=add_seasonality,
        failure_event=failure_event,
        maintenance_effect=maintenance_effect
    )
    
    st.dataframe(data)

    # Time Series Plots
    st.subheader("Time Series Plots")
    for column in data.columns:
        st.line_chart(data[[column]])

    # Calculate health index
    health_index = calculate_health_index(data, sensor_specs)
    
    # Plot health index
    st.subheader("Machine Health Index")
    st.line_chart(health_index)
    
    # Current health status
    current_health = health_index.iloc[-1]
    health_status = "Good" if current_health >= 80 else "Fair" if current_health >= 60 else "Poor" if current_health >= 40 else "Critical"
    st.metric(label="Current Health Index", value=f"{current_health:.1f}/100", delta=f"{health_status}")

    # Predictive Indicators
    st.subheader("Predictive Indicators")
    
    # Calculate time to failure and failure probability
    time_to_failure, failure_probability = predict_failure(health_index, selected_machine, data)
    
    # Display if using model or rule-based approach
    prediction_method = "Machine Learning Model" if os.path.exists(f"models/{selected_machine}_model.pkl") else "Rule-Based Calculation"
    st.info(f"Prediction Method: {prediction_method}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Time to Failure (days)", value=f"{time_to_failure}")
    with col2:
        st.metric(label="Failure Probability", value=f"{failure_probability:.2f}")

    # Maintenance Scheduling
    st.subheader("Maintenance Scheduling")
    if st.button("Generate Maintenance Recommendation"):
        urgency, maintenance_date, days_offset = recommend_maintenance(time_to_failure, failure_probability, selected_machine)
        
        if urgency == "Critical":
            st.error(f"🚨 **{urgency}** - IMMEDIATE MAINTENANCE REQUIRED! Machine has failed or failure is imminent!")
        elif urgency == "Urgent":
            st.error(f"⚠️ **{urgency}** maintenance required! Schedule within {days_offset} days (by {maintenance_date.strftime('%Y-%m-%d')})")
        elif urgency == "High":
            st.warning(f"⚠️ **{urgency}** priority maintenance recommended in {days_offset} days (by {maintenance_date.strftime('%Y-%m-%d')})")
        elif urgency == "Medium":
            st.info(f"ℹ️ **{urgency}** priority maintenance suggested in {days_offset} days (by {maintenance_date.strftime('%Y-%m-%d')})")
        else:
            st.success(f"✓ **{urgency}** priority maintenance can be scheduled in {days_offset} days (by {maintenance_date.strftime('%Y-%m-%d')})")

    # Anomaly Detection
    st.subheader("Anomaly Detection")
    threshold_percentile = st.sidebar.slider("Anomaly Threshold Percentile", min_value=90, max_value=99, value=95)
    
    # Detect anomalies for each sensor based on percentile threshold
    anomalies = pd.DataFrame(index=data.index)
    has_anomalies = False
    
    for column in data.columns:
        threshold = np.percentile(data[column], threshold_percentile)
        if column in ["temp", "vibration"]:
            # For these sensors, high values are anomalies
            anomalies[column] = data[column] > threshold
        elif column in ["pressure", "flow_rate"]:
            # For these sensors, low values might be anomalies
            lower_threshold = np.percentile(data[column], 100 - threshold_percentile)
            anomalies[column] = data[column] < lower_threshold
        else:
            # For other sensors, detect values outside normal range
            upper_threshold = np.percentile(data[column], threshold_percentile)
            lower_threshold = np.percentile(data[column], 100 - threshold_percentile)
            anomalies[column] = (data[column] > upper_threshold) | (data[column] < lower_threshold)
        
        if anomalies[column].any():
            has_anomalies = True
    
    if not has_anomalies:
        st.info("No anomalies detected based on the current threshold.")
    else:
        st.warning("Anomalies detected:")
        # Display timestamps with anomalies and the corresponding sensor values
        anomaly_points = anomalies.any(axis=1)
        anomaly_dates = anomalies.index[anomaly_points]
        
        # Create empty DataFrame for anomalies
        anomaly_data = pd.DataFrame(columns=['timestamp', 'anomalous_sensors'])
        
        # Fill anomaly data using concat instead of append
        anomaly_rows = []
        for date in anomaly_dates:
            row_anomalies = anomalies.loc[date]
            anomalous_sensors = row_anomalies[row_anomalies].index.tolist()
            
            # Only record the values for the anomalous sensors
            values = {}
            for sensor in anomalous_sensors:
                values[sensor] = data.loc[date, sensor]
            
            anomaly_rows.append({
                'timestamp': date,
                'anomalous_sensors': ", ".join(anomalous_sensors),
                **values
            })
        
        if anomaly_rows:
            anomaly_data = pd.concat([anomaly_data, pd.DataFrame(anomaly_rows)], ignore_index=True)
            st.dataframe(anomaly_data)

    # Model Training Section
    st.subheader("Model Training")
    if st.button("Train New Models"):
        try:
            with st.spinner("Training models with machine-specific feature selection... This may take a few minutes."):
                model = PredictiveMaintenanceModel()
                results = model.train_all_models()
                
                # Display training results
                st.success("Model training complete! Models now use only machine-specific relevant features.")
                for machine_type, result in results.items():
                    if result:
                        st.write(f"**{machine_type}**:")
                        st.write(f"- Accuracy: {result['accuracy']:.4f}")
                        st.write(f"- ROC-AUC: {result['roc_auc']:.4f}")
                        st.write(f"- PR-AUC: {result['pr_auc']:.4f}")
                        
                        # Display feature importance plot if available
                        plot_path = f'plots/model_evaluation/{machine_type}_feature_importance.png'
                        if os.path.exists(plot_path):
                            st.image(plot_path, caption=f"Feature Importance - {machine_type}")
        except Exception as e:
            st.error(f"Error during model training: {e}")


if __name__ == "__main__":
    main() 