import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import math
import os
import joblib
from predictive_maintenance_modelV2 import PredictiveMaintenanceModelV2

# Set page configuration
st.set_page_config(
    page_title="Industrial Equipment Health Monitor",
    page_icon="🔧",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    model = PredictiveMaintenanceModelV2()
    
    # Try to load pre-trained models if available
    if os.path.exists('models') and os.listdir('models'):
        model.load_models('models')
    else:
        # Load data and train models if no pre-trained models are available
        model.load_data()
        model.engineer_features()
        model.train_models()
    
    return model

model = load_model()

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
        "siemens": {
            "temperature": {"normal_range": (40, 60), "weight": 0.3},
            "vibration": {"normal_range": (0.1, 0.5), "weight": 0.3},
            "current": {"normal_range": (10, 20), "weight": 0.2},
            "voltage": {"normal_range": (220, 240), "weight": 0.1},
            "power": {"normal_range": (1000, 1200), "weight": 0.1}
        },
        "abb": {
            "temperature": {"normal_range": (30, 50), "weight": 0.3},
            "vibration": {"normal_range": (0.05, 0.3), "weight": 0.4},
            "acoustic": {"normal_range": (20, 50), "weight": 0.3}
        },
        "haas": {
            "spindle_load": {"normal_range": (20, 80), "weight": 0.3},
            "vibration": {"normal_range": (0.1, 0.4), "weight": 0.2},
            "temperature": {"normal_range": (50, 80), "weight": 0.2},
            "acoustic": {"normal_range": (40, 70), "weight": 0.3}
        },
        "grundfos": {
            "flow_rate": {"normal_range": (40, 100), "weight": 0.4},
            "pressure": {"normal_range": (20, 50), "weight": 0.3},
            "temperature": {"normal_range": (30, 60), "weight": 0.2},
            "power": {"normal_range": (800, 1000), "weight": 0.1}
        },
        "carrier": {
            "refrigerant_pressure": {"normal_range": (30, 60), "weight": 0.3},
            "evaporator_temp": {"normal_range": (5, 15), "weight": 0.3},
            "condenser_temp": {"normal_range": (30, 45), "weight": 0.2},
            "power": {"normal_range": (5000, 6000), "weight": 0.2}
        }
    }
    
    # Default if machine not found
    if machine_type not in sensors:
        sensors[machine_type] = {
            "temperature": {"normal_range": (30, 60), "weight": 0.5},
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
        if add_seasonality and sensor_name in ["temperature", "evaporator_temp", "condenser_temp", "power"]:
            # Create a daily seasonal pattern
            if isinstance(date_range[0], pd.Timestamp):
                hours = np.array([d.hour for d in date_range])
                # Daily pattern: peak during working hours
                daily_pattern = 0.1 * np.sin(hours * (2 * np.pi / 24))
                base_values += (high - low) * daily_pattern
        
        # Apply degradation effect based on sensor type
        if sensor_name in ["temperature", "evaporator_temp", "condenser_temp"]:
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
    
    # Add machine_id
    machine_id = f"{machine_type}_001"
    data["machine_id"] = [machine_id] * time_steps
    
    data["timestamp"] = date_range
    df = pd.DataFrame(data)
    
    return df, sensor_specs

def calculate_health_index(data, sensor_specs):
    """Calculate machine health index based on sensor data and specifications."""
    health_index = pd.Series(index=data.index, dtype=float)
    health_index[:] = 100  # Start with perfect health
    
    # Calculate health contribution from each sensor
    for sensor_name, specs in sensor_specs.items():
        if sensor_name in data.columns:
            low, high = specs["normal_range"]
            weight = specs["weight"]
            
            # Calculate how much each reading deviates from the normal range
            readings = data[sensor_name]
            
            # Different sensors have different degradation patterns
            if sensor_name in ["temperature"]:
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

def create_visualizations(data, health_index, prediction):
    """Create visualizations of sensor data and predictions"""
    figures = {}
    
    # 1. Create time series plot of key sensors
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Get sensor columns that aren't metadata
    sensor_cols = [col for col in data.columns if col not in ['timestamp', 'machine_id']]
    
    # Limit to max 5 sensors for clarity
    if len(sensor_cols) > 5:
        sensor_cols = sensor_cols[:5]
    
    # Plot each sensor
    for sensor in sensor_cols:
        ax1.plot(data['timestamp'], data[sensor], label=sensor)
    
    ax1.set_title('Sensor Readings')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sensor Value')
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    figures['sensor_plot'] = fig1
    
    # 2. Create health index plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data['timestamp'], health_index, 'b-', linewidth=2)
    ax2.set_title('Machine Health Index')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Health Score')
    ax2.set_ylim(0, 105)
    ax2.grid(True)
    plt.tight_layout()
    
    figures['health_plot'] = fig2
    
    # 3. Create health score visualization (gauge chart)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    # Create a gauge-like visualization
    health_score = prediction['health_score']
    
    # Create a simple gauge chart
    gauge_colors = ['#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#008000']
    gauge_labels = ['Critical', 'Poor', 'Fair', 'Good', 'Excellent']
    gauge_ranges = [0, 25, 50, 75, 90, 100]
    
    # Determine color index based on health score
    color_idx = 0
    for i, threshold in enumerate(gauge_ranges[1:]):
        if health_score < threshold:
            color_idx = i
            break
        color_idx = len(gauge_ranges) - 2  # Default to last color
    
    # Create a horizontal bar chart to simulate a gauge
    ax3.barh(['Health'], [100], color='lightgray', height=0.5)
    ax3.barh(['Health'], [health_score], color=gauge_colors[color_idx], height=0.5)
    
    # Add text with the score
    ax3.text(health_score, 0, f' {health_score:.1f}', va='center', fontweight='bold')
    
    # Add a title and clean up the axes
    ax3.set_title(f'Health Score: {prediction["status"]}')
    ax3.set_xlim(0, 100)
    ax3.set_xticks(gauge_ranges)
    ax3.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # Add color bands for reference
    for i in range(len(gauge_ranges) - 1):
        ax3.axvspan(gauge_ranges[i], gauge_ranges[i+1], alpha=0.2, color=gauge_colors[i])
        ax3.text((gauge_ranges[i] + gauge_ranges[i+1])/2, -0.15, gauge_labels[i], 
                 ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    
    figures['gauge_plot'] = fig3
    
    # 4. Create maintenance timeline visualization
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    
    days = prediction['days_until_maintenance']
    
    # Create a timeline
    if days <= 0:
        color = 'red'
        status = 'MAINTENANCE REQUIRED NOW'
    elif days < 7:
        color = 'orange'
        status = 'MAINTENANCE SOON'
    else:
        color = 'green'
        status = 'MAINTENANCE SCHEDULED'
    
    # Create a horizontal bar to show days until maintenance
    days_max = max(30, days + 5)  # Set max to at least 30 days or slightly more than predicted days
    ax4.barh(['Timeline'], [days_max], color='lightgray', height=0.3)
    ax4.barh(['Timeline'], [days], color=color, height=0.3)
    
    # Add markers for important thresholds
    ax4.axvline(x=7, color='orange', linestyle='--', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add text labels
    ax4.text(days, 0, f' {days} days', va='center', fontweight='bold')
    ax4.text(0, 0.2, 'Now', ha='center', va='bottom', fontsize=8)
    ax4.text(7, 0.2, '1 Week', ha='center', va='bottom', fontsize=8)
    
    # Add a title and clean up the axes
    ax4.set_title(f'Maintenance Timeline: {status}')
    ax4.set_xlim(-1, days_max)
    ax4.set_xticks(range(0, int(days_max)+1, 7))
    ax4.set_xlabel('Days Until Maintenance')
    ax4.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    plt.tight_layout()
    
    figures['maintenance_plot'] = fig4
    
    return figures

def main():
    st.title("Industrial Equipment Health Monitor")
    st.write("Simulate sensor data and predict equipment health using machine learning models")
    
    # Define sidebar for inputs
    st.sidebar.header("Machine & Simulation Settings")
    
    # Get list of machine types from model
    if not model.machine_types:
        machine_types = ["No machines available"]
    else:
        # Use the standardized machine types from the mapping
        machine_types = sorted(list(set(model.MACHINE_TYPE_MAPPING.values())))
    
    # Machine selection
    selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)
    
    # Date input for simulation time frame
    today = datetime.now()
    default_start = today - timedelta(days=3)
    
    st.sidebar.subheader("Simulation Time Frame")
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)
    
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        return

    # Data frequency
    freq = st.sidebar.selectbox("Data Frequency", 
                                options=["h", "30T", "15T"], 
                                format_func=lambda x: {"h": "Hourly", "30T": "Every 30 Minutes", "15T": "Every 15 Minutes"}[x],
                                index=0)
    
    # Simulation scenario settings
    st.sidebar.subheader("Simulation Scenarios")
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
    
    # Advanced simulation parameters
    st.sidebar.subheader("Advanced Parameters")
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
    
    # Main content
    st.subheader(f"Simulated Sensor Data for {selected_machine.capitalize()}")
    st.write(f"Scenario: **{scenario.replace('_', ' ').title()}**")
    
    # Simulate data and make predictions when user clicks button
    if st.button("Run Simulation & Analyze"):
        with st.spinner("Simulating sensor data and analyzing..."):
            # Simulate sensor data
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
            
            # Calculate health index
            health_index = calculate_health_index(data, sensor_specs)
            
            # Make prediction using the model
            prediction = model.predict_machine_health(data, selected_machine)
            
            if prediction is None:
                st.error("Failed to make prediction. Try a different machine type or scenario.")
                return
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Data & Plots", "Health Report", "Download"])
            
            with tab1:
                st.subheader("Simulated Sensor Data")
                st.dataframe(data)
                
                # Create and display visualizations
                figures = create_visualizations(data, health_index, prediction)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(figures['sensor_plot'])
                with col2:
                    st.pyplot(figures['health_plot'])
            
            with tab2:
                # Display health metrics and prediction
                st.subheader("Equipment Health Report")
                
                # Create a card-like layout for the health status
                health_card, maintenance_card = st.columns(2)
                
                with health_card:
                    st.markdown(f"### Health Status: {prediction['status']}")
                    
                    # Display gauge chart
                    st.pyplot(figures['gauge_plot'])
                    
                    # Color-coded health score
                    health_score = prediction['health_score']
                    if health_score >= 90:
                        st.success(f"Health Score: {health_score:.1f}/100")
                    elif health_score >= 75:
                        st.success(f"Health Score: {health_score:.1f}/100")
                    elif health_score >= 50:
                        st.warning(f"Health Score: {health_score:.1f}/100")
                    elif health_score >= 25:
                        st.warning(f"Health Score: {health_score:.1f}/100")
                    else:
                        st.error(f"Health Score: {health_score:.1f}/100")
                    
                with maintenance_card:
                    st.markdown("### Maintenance Recommendation")
                    
                    # Display maintenance timeline
                    st.pyplot(figures['maintenance_plot'])
                    
                    # Get days until maintenance
                    days = prediction['days_until_maintenance']
                    
                    # Display color-coded maintenance recommendation
                    if days <= 0:
                        st.error("**IMMEDIATE MAINTENANCE REQUIRED!**")
                    elif days < 7:
                        st.warning(f"**Maintenance Required Soon:** Schedule within {days} days")
                    elif days < 15:
                        st.info(f"**Maintenance Recommended:** Schedule within {days} days")
                    else:
                        st.success(f"**Regular Maintenance:** Schedule in {days} days")
            
            with tab3:
                st.subheader("Download Simulated Data")
                
                # Convert DataFrame to CSV for download
                csv = data.to_csv(index=False)
                
                # Create a download button
                st.download_button(
                    label="Download Sensor Data (CSV)",
                    data=csv,
                    file_name=f"{selected_machine}_simulation_{datetime.now().strftime('%Y%m%d%H%M')}.csv",
                    mime="text/csv"
                )
                
                # Description of the CSV
                st.write("""
                The downloaded CSV contains simulated sensor data that can be used for:
                - Testing predictive maintenance algorithms
                - Training machine learning models
                - Benchmarking your maintenance systems
                """)

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)
    
    main() 