import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import altair as alt # Using Altair for more control over charts
import os # Added to potentially help with pathing if needed
import sys # Added for path manipulation
import uuid
from pathlib import Path

from src.modeling.feature_engineering import extract_base_features # Added for path manipulation

# Add project root to sys.path to allow importing custom modules
project_root = Path(__file__).resolve().parent # Assuming dashboard is in root
sys.path.append(str(project_root))

# --- Configuration ---
NUM_MACHINES = 5
UPDATE_INTERVAL_SECONDS = 2  # How often to generate new data (in seconds)
HISTORY_POINTS = 300        # How many data points to keep and display per sensor
LOOKBACK_POINTS_FOR_ANALYSIS = 50 # How many recent points to analyze for maintenance status
ANOMALY_THRESHOLD_STD_DEV = 2.5 # How many std devs away to consider a point potentially anomalous
TREND_THRESHOLD_PERCENT = 0.10  # % change in mean over lookback period to indicate a trend
SUSTAINED_ANOMALY_COUNT = 5     # How many consecutive anomalies trigger a warning

# --- Machine Definitions (Simplified Sensor Simulation) ---
# Structure: Machine Name: { Sensor Name: (Normal Mean, Normal StdDev, Drift per update, Unit) }
MACHINES_CONFIG = {
    "Siemens Motor": {
        "Temperature": (70, 3, 0.005, "°C"),       # Normal 60-80
        "Vibration": (3, 0.5, 0.002, "mm/s RMS"),  # Normal 1-5
        "Current": (55, 5, -0.001, "A"),           # Normal 10-100 (example)
    },
    "ABB Bearing": {
        "Vibration": (1.1, 0.2, 0.003, "mm/s RMS"), # Normal 0.2-2
        "Temperature": (50, 4, 0.004, "°C"),        # Normal 40-60
        "Acoustic": (55, 5, 0.006, "dB"),          # Normal 40-70
    },
    "HAAS CNC": {
        "Spindle Load": (40, 8, -0.002, "%"),       # Normal 20-60
        "Vibration": (1.8, 0.3, 0.0025, "mm/s RMS"),# Normal 0.5-3
        "Temperature": (55, 3, 0.0035, "°C"),      # Normal 45-65
    },
    "Grundfos Pump": {
        "Pressure": (13.5, 2, 0.001, "bar"),      # Normal 2-25 (example mid-range)
        "Flow Rate": (90, 10, -0.005, "m³/h"),     # Normal 1-180 (example mid-range)
        "Temperature": (55, 5, 0.0045, "°C"),      # Normal 40-70
    },
    "Carrier Chiller": {
        "Refrigerant Pressure": (16.5, 2.5, -0.0015, "bar"), # Normal 8-25
        "Condenser Temp": (40, 2, 0.003, "°C"),       # Normal 35-45
        "Power Draw/Ton": (0.6, 0.05, 0.001, "kW/ton"),# Normal 0.5-0.7
    }
}

# --- Try importing the prediction function ---
try:
    from src.modeling.predict_maintenance import predict_status, HEALTH_MODEL_SENSORS
    PREDICT_FUNC_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import predict_status: {e}")
    print("Dashboard will use the fallback analyze_sensor_data function.")
    PREDICT_FUNC_AVAILABLE = False

# --- Helper Functions ---

def generate_reading(current_base, std_dev, drift, anomaly_prob=0.015):
    """Generates a new sensor reading with noise, drift, and potential anomalies."""
    # Apply drift
    new_base = current_base + drift
    # Generate normal fluctuation
    reading = np.random.normal(new_base, std_dev)
    # Introduce random anomalies (spikes/dips)
    is_generated_anomaly = False
    if np.random.rand() < anomaly_prob:
        anomaly_factor = np.random.uniform(1.5, 3.0) * np.random.choice([-1, 1])
        reading = new_base + anomaly_factor * std_dev * 3 # Make anomalies significant
        # print(f"Generated anomaly: {reading:.2f} (base: {new_base:.2f})") # Debug
        is_generated_anomaly = True # Flag if we forced an anomaly
    # Ensure non-negative values where applicable (e.g., vibration, load)
    if new_base >= 0:
         reading = max(0, reading)
    # Simulate sensor failure - very large deviation
    if np.random.rand() < 0.0005: # Very rare chance of extreme value
         reading = new_base * np.random.uniform(3, 5) * np.random.choice([-1, 1])
         print(f"Generated EXTREME value: {reading:.2f}") # Debug
         is_generated_anomaly = True # Treat extreme as anomaly

    return reading, new_base, is_generated_anomaly

def analyze_sensor_data(data: pd.Series, normal_mean, normal_std):
    """ 
    *** FALLBACK ANALYSIS - Used if predict_status cannot be imported ***
    Analyzes recent sensor data to determine status, anomalies, and trends.
    """
    if len(data) < LOOKBACK_POINTS_FOR_ANALYSIS:
        return {
            "status": "Initializing", "reason": "Need more data...", "risk_score": 0,
            "is_anomaly_now": False, "anomalies_indices": [], "trend_detected": False,
            "upper_bound": normal_mean + ANOMALY_THRESHOLD_STD_DEV * normal_std,
            "lower_bound": normal_mean - ANOMALY_THRESHOLD_STD_DEV * normal_std
        }

    recent_data = data.tail(LOOKBACK_POINTS_FOR_ANALYSIS)
    current_value = recent_data.iloc[-1]
    current_timestamp = recent_data.index[-1]

    # Define bounds for anomaly detection
    upper_bound = normal_mean + ANOMALY_THRESHOLD_STD_DEV * normal_std
    lower_bound = normal_mean - ANOMALY_THRESHOLD_STD_DEV * normal_std
    # Ensure lower bound is not negative if the mean is positive
    if normal_mean >= 0:
        lower_bound = max(0, lower_bound)

    # 1. Identify Anomalies (Points outside bounds)
    anomaly_mask = (recent_data > upper_bound) | (recent_data < lower_bound)
    anomalies_indices = recent_data[anomaly_mask].index.tolist()
    is_anomaly_now = anomaly_mask.iloc[-1]
    num_anomalies = len(anomalies_indices)

    # 2. Check for Sustained Anomalies
    consecutive_anomalies = 0
    if len(recent_data) >= SUSTAINED_ANOMALY_COUNT:
        last_n_mask = anomaly_mask.tail(SUSTAINED_ANOMALY_COUNT)
        if last_n_mask.all():
             consecutive_anomalies = SUSTAINED_ANOMALY_COUNT

    # 3. Check for Significant Trend (Drift)
    mean_recent = recent_data.mean()
    mean_change_ratio = abs(mean_recent - normal_mean) / normal_mean if normal_mean != 0 else 0
    trend_detected = mean_change_ratio > TREND_THRESHOLD_PERCENT

    # 4. Determine Status & Risk Score (Simplified Logic)
    status = "Healthy"
    reason = f"Latest: {current_value:.2f}. Within normal range [{lower_bound:.1f}-{upper_bound:.1f}]."
    risk_score = 0 # 0: Healthy, 1: Investigate, 2: Warning, 3: Critical

    if is_anomaly_now:
        status = "Investigate"
        reason = f"Single significant deviation ({current_value:.2f}) outside normal range [{lower_bound:.1f}-{upper_bound:.1f}]."
        risk_score = 1

    if num_anomalies > LOOKBACK_POINTS_FOR_ANALYSIS * 0.25: # High rate recently
        status = "Investigate"
        reason = f"High anomaly rate ({num_anomalies}/{LOOKBACK_POINTS_FOR_ANALYSIS} points outside normal range). Latest: {current_value:.2f}."
        risk_score = max(risk_score, 1) # Keep investigate if it was already anomaly_now

    if trend_detected and num_anomalies > LOOKBACK_POINTS_FOR_ANALYSIS * 0.1: # Trend + some anomalies
        status = "Warning"
        reason = f"Potential drift. Mean ({mean_recent:.2f}) deviates >{TREND_THRESHOLD_PERCENT*100:.0f}% from normal ({normal_mean:.2f}) with anomalies present ({num_anomalies}). Latest: {current_value:.2f}."
        risk_score = 2

    if consecutive_anomalies >= SUSTAINED_ANOMALY_COUNT:
        status = "Warning"
        reason = f"Sustained anomalies detected ({consecutive_anomalies} consecutive points outside normal range). Latest: {current_value:.2f}"
        risk_score = 2

    # Simple logic override for critical failure simulation
    critical_upper = normal_mean + 5 * normal_std
    critical_lower = normal_mean - 5 * normal_std
    if normal_mean >= 0: critical_lower = max(0, critical_lower) # Avoid negative critical low if mean is positive

    if current_value > critical_upper or current_value < critical_lower :
         status = "CRITICAL"
         reason = f"Extreme deviation detected! Value: {current_value:.2f}. Normal range: [{lower_bound:.1f}-{upper_bound:.1f}]"
         risk_score = 3

    return {
        "status": status,
        "reason": reason,
        "risk_score": risk_score,
        "is_anomaly_now": is_anomaly_now,
        "anomalies_indices": anomalies_indices, # Timestamps of anomalies in lookback
        "trend_detected": trend_detected,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound
    }

def plot_sensor_data(df: pd.DataFrame, analysis_results: dict, sensor_name: str, unit: str):
    """Creates an Altair chart visualizing sensor data, bounds, and anomalies."""
    df_reset = df.reset_index()  # Need Timestamp as a column for Altair
    if 'Timestamp' not in df_reset.columns and 'index' in df_reset.columns:
        df_reset.rename(columns={'index': 'Timestamp'}, inplace=True)

    base = alt.Chart(df_reset).encode(
        x=alt.X('Timestamp', axis=alt.Axis(title=None, format="%H:%M:%S"))
    )

    # Base line for sensor values
    line = base.mark_line(point=False, color='cornflowerblue').encode(
        y=alt.Y('Value', axis=alt.Axis(title=f"{sensor_name} ({unit})")),
        tooltip=['Timestamp', alt.Tooltip('Value', format='.2f')]
    )

    # Add Bounds as rule lines
    upper_bound_line = alt.Chart(pd.DataFrame({'bound': [analysis_results['upper_bound']]})).mark_rule(color='orange', strokeDash=[3,3]).encode(y='bound')
    lower_bound_line = alt.Chart(pd.DataFrame({'bound': [analysis_results['lower_bound']]})).mark_rule(color='orange', strokeDash=[3,3]).encode(y='bound')

    # Create DataFrame for anomalies to plot as points
    anomaly_df = df_reset[df_reset['Timestamp'].isin(analysis_results['anomalies_indices'])]
    anomaly_points = alt.Chart(anomaly_df).mark_point(
        color='red', size=60, opacity=0.8, filled=True
    ).encode(
        x='Timestamp',
        y='Value',
        tooltip=['Timestamp', alt.Tooltip('Value', format='.2f')]
    )

    # --- Add Rolling Mean and Std Dev Lines ---
    # We need to calculate these features for the *entire history* in df
    rolling_features = extract_base_features(df['Value'], window=LOOKBACK_POINTS_FOR_ANALYSIS) # Use analysis window
    # Rename columns for clarity in plot
    rolling_features.columns = ['Rolling Mean', 'Rolling Std Dev']
    # Merge with the reset index DataFrame
    df_with_features = pd.concat([df_reset, rolling_features], axis=1)

    # Rolling Mean Line
    rolling_mean_line = alt.Chart(df_with_features).mark_line(
        color='purple', point=False, strokeDash=[1,1] # Dashed purple line
    ).encode(
        x='Timestamp',
        y=alt.Y('Rolling Mean'),
        tooltip=['Timestamp', alt.Tooltip('Rolling Mean', format='.2f')]
    )

    # Rolling Std Dev (Plotted relative to mean, potentially on secondary axis or as area)
    # Option 1: Area band around the mean
    # Define the upper and lower bounds for the std dev area
    df_with_features['Mean + Std'] = df_with_features['Rolling Mean'] + df_with_features['Rolling Std Dev']
    df_with_features['Mean - Std'] = df_with_features['Rolling Mean'] - df_with_features['Rolling Std Dev']

    rolling_std_area = alt.Chart(df_with_features).mark_area(
        opacity=0.3,
        color='gray'
    ).encode(
        x='Timestamp',
        y=alt.Y('Mean - Std', axis=alt.Axis(title=f"{sensor_name} ({unit})")), # Use same axis as main value
        y2='Mean + Std',
        tooltip=[
            'Timestamp',
            alt.Tooltip('Rolling Mean', format='.2f'),
            alt.Tooltip('Rolling Std Dev', format='.2f')
        ]
    )

    # Combine the charts (Layer order matters)
    chart = alt.layer(
        rolling_std_area, # Area first (background)
        line, 
        rolling_mean_line, 
        upper_bound_line, 
        lower_bound_line, 
        anomaly_points # Points last (foreground)
    ).interactive() # Make chart interactive (zoom/pan)

    return chart

def plot_health_index(health_data: pd.DataFrame):
    """Creates an Altair chart visualizing predicted health index over time."""
    if health_data.empty:
        return alt.Chart(pd.DataFrame({"Timestamp": [], "Health Index": []})).mark_line() # Return empty chart
        
    base = alt.Chart(health_data.reset_index()).encode(
        x=alt.X('Timestamp', axis=alt.Axis(title=None, format="%H:%M:%S"))
    )

    line = base.mark_line(point=alt.OverlayMarkDef(size=10), color='green').encode(
        y=alt.Y('Health Index', axis=alt.Axis(title="Health Index"), scale=alt.Scale(domain=[0, 1])), # Force Y-axis 0-1
        tooltip=['Timestamp', alt.Tooltip('Health Index', format='.3f')]
    )
    
    # Add threshold line
    DEFAULT_HEALTH_THRESHOLD = 0.2  # Define threshold constant if not imported
    threshold_line = alt.Chart(pd.DataFrame({'threshold': [DEFAULT_HEALTH_THRESHOLD]}))\
        .mark_rule(color='red', strokeDash=[3,3])\
        .encode(y='threshold')

    chart = alt.layer(line, threshold_line).properties(
        # title="Predicted Health Index"
    ).interactive()

    return chart

# --- Maintenance Log Functions ---
def add_maintenance_log(machine, sensor, reason, timestamp):
    # Avoid adding duplicate pending logs for the same machine/sensor issue
    if "maintenance_log" in st.session_state:
        for log in st.session_state.maintenance_log:
            if log["machine"] == machine and log["sensor"] == sensor and log["status"] == "Pending":
                return
    else:
        st.session_state.maintenance_log = []
    log_entry = {
        "id": f"{machine}-{sensor}-{uuid.uuid4()}",
        "timestamp": timestamp,
        "machine": machine,
        "sensor": sensor,
        "reason": reason,
        "status": "Pending"
    }
    st.session_state.maintenance_log.append(log_entry)

def update_log_status(log_id, new_status):
     for log in st.session_state.maintenance_log:
          if log["id"] == log_id:
               log["status"] = new_status
               break
     # Force rerun to update the display immediately after button click
     # st.experimental_rerun() # REMOVED - Streamlit should rerun automatically

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Predictive Maintenance Simulation")

st.title("🔧 Predictive Maintenance Dashboard Simulation")
st.caption(f"Simulating real-time sensor data & simplified predictive analysis. Updates every {UPDATE_INTERVAL_SECONDS} seconds.")
st.info("**Note:** The 'Status' and 'Risk Score' below simulate the output of a complex predictive model based on simplified rules (thresholds, trends). A real system uses trained ML models.", icon="💡")

# --- Display a warning if the actual prediction function isn't loaded ---
if not PREDICT_FUNC_AVAILABLE:
    st.warning("Could not load `predict_status` from `src.modeling.predict_maintenance`. Using fallback simulation logic for status analysis.", icon="⚠️")

# Initialize session state for data storage and simulation parameters
if 'machine_data' not in st.session_state:
    st.session_state.machine_data = {}
    st.session_state.current_bases = {}
    st.session_state.maintenance_log = []  # Initialize maintenance log
    st.session_state.health_history = {}  # Initialize health index history
    for machine_name, sensors in MACHINES_CONFIG.items():
        st.session_state.machine_data[machine_name] = {}
        st.session_state.current_bases[machine_name] = {}
        st.session_state.health_history[machine_name] = pd.DataFrame(columns=['Health Index'])  # Store machine-level health
        for sensor_name, (mean, std, drift, unit) in sensors.items():
            # Initialize with some starting history
            initial_timestamps = [datetime.now() - timedelta(seconds=x*UPDATE_INTERVAL_SECONDS) for x in range(HISTORY_POINTS)][::-1]
            initial_values = np.random.normal(mean, std, HISTORY_POINTS)
            st.session_state.machine_data[machine_name][sensor_name] = pd.DataFrame({
                'Timestamp': initial_timestamps,
                'Value': initial_values
            }).set_index('Timestamp')
            st.session_state.current_bases[machine_name][sensor_name] = mean  # Start drift from the normal mean
else:
    # Even if session state exists, clear the maintenance log to avoid legacy entries causing duplicate keys
    st.session_state.maintenance_log = []

# Create placeholders for dynamic content
machine_cols = st.columns(NUM_MACHINES)
placeholders = {}
for i, machine_name in enumerate(MACHINES_CONFIG.keys()):
    with machine_cols[i]:
        placeholders[machine_name] = {}
        st.header(machine_name)
        for sensor_name in MACHINES_CONFIG[machine_name].keys():
            placeholders[machine_name][sensor_name] = {
                "metric": st.empty(),
                "status": st.empty(),
                "chart": st.empty()
            }

# --- Maintenance Log Display Area ---
st.divider()
st.header("🚨 Maintenance Action Log")

# --- Main Simulation Loop ---
placeholder = st.empty()

while True:
    with placeholder.container():
        # Create columns for layout
        num_columns = 3 # Adjust as needed
        cols = st.columns(num_columns)
        col_idx = 0

        current_time = datetime.now()

        # Update and display data for each machine
        for machine_name, sensors in MACHINES_CONFIG.items():
            machine_col = cols[col_idx % num_columns]
            col_idx += 1

            with machine_col: # Ensure this context manager covers the whole machine's UI
                st.subheader(f"⚙️ {machine_name}")
                overall_machine_risk = 0
                latest_machine_health = None

                sensor_tab_names = list(sensors.keys())
                if len(sensor_tab_names) > 0:
                    sensor_tabs = st.tabs(sensor_tab_names)
                else:
                    st.write("No sensors configured for this machine.")
                    continue

                for i, sensor_name in enumerate(sensor_tab_names):
                    mean, std, drift, unit = sensors[sensor_name]

                    # All code related to a single sensor tab goes INSIDE this 'with' block
                    with sensor_tabs[i]:
                        # Initialize data structures
                        if sensor_name not in st.session_state.machine_data[machine_name]:
                            st.session_state.machine_data[machine_name][sensor_name] = pd.DataFrame(columns=['Value'])
                            st.session_state.current_bases[machine_name][sensor_name] = mean

                        # Generate new data
                        current_base = st.session_state.current_bases[machine_name][sensor_name]
                        new_value, new_base, _ = generate_reading(current_base, std, drift)
                        st.session_state.current_bases[machine_name][sensor_name] = new_base

                        # Create new data entry
                        new_data = pd.DataFrame([{'Value': new_value}], index=[pd.Timestamp(current_time)])
                        new_data.index.name = 'Timestamp'
                        # Append new data and keep history limit
                        df = st.session_state.machine_data[machine_name][sensor_name]
                        df = pd.concat([df, new_data])
                        df = df.tail(HISTORY_POINTS)
                        st.session_state.machine_data[machine_name][sensor_name] = df

                        # -- Perform Analysis --
                        analysis_results = analyze_sensor_data(df['Value'], mean, std)

                        # --- Display Sensor Info & Chart ---
                        status = analysis_results.get("status", "Error")
                        reason = analysis_results.get("reason", "Analysis failed")
                        risk_score = analysis_results.get("risk_score", 0)
                        health_index = analysis_results.get("predicted_health_index", None)

                        if health_index is not None:
                            latest_machine_health = health_index

                        # Status Badge Colors
                        if status == "Healthy": badge_color = "green"
                        elif status == "Investigate": badge_color = "blue" # Changed Investigate to blue for distinction
                        elif status == "Warning" or status == "Anomaly Detected": badge_color = "orange"
                        elif status == "Critical": badge_color = "red"
                        else: badge_color = "grey"

                        st.markdown(f"**{sensor_name}:** <span style='color:{badge_color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
                        health_display = f"Health: {health_index:.2f}" if health_index is not None else "Health: N/A"
                        if PREDICT_FUNC_AVAILABLE and not pd.isna(health_index):
                            st.caption(f"_{reason}_ | Risk: {risk_score:.2f} | {health_display} | Roll Mean: {health_index:.2f}")
                        else:
                            st.caption(f"_{reason}_ | Risk: {risk_score:.2f} | {health_display}")

                        overall_machine_risk = max(overall_machine_risk, risk_score)

                        # Plot sensor data
                        if not df.empty:
                            chart = plot_sensor_data(df, analysis_results, sensor_name, unit)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.write("Gathering initial data...")

                        # --- Trigger Maintenance Log ---
                        if risk_score >= 0.6:
                            add_maintenance_log(machine_name, sensor_name, reason, pd.Timestamp(current_time))

                # --- Machine-Level Prediction using all sensor data ---
                machine_type_norm = machine_name.lower().replace(" ", "_")
                combined_dfs = []
                for sensor_key, sensor_df in st.session_state.machine_data[machine_name].items():
                    normalized_key = sensor_key.lower().replace(" ", "_")
                    combined_dfs.append(sensor_df.rename(columns={'Value': normalized_key}))
                if combined_dfs:
                    machine_df = pd.concat(combined_dfs, axis=1)
                else:
                    machine_df = pd.DataFrame()
                required_sensors = HEALTH_MODEL_SENSORS.get(machine_type_norm, [])
                for sensor in required_sensors:
                    if sensor not in machine_df.columns:
                        machine_df[sensor] = 0.0
                prediction_result = predict_status(machine_df, machine_type=machine_type_norm)
                overall_status = prediction_result.get("status", "Unknown")
                overall_risk = prediction_result.get("risk_score", 0.0)
                overall_health = prediction_result.get("predicted_health_index", None)
                st.markdown(f"**Overall Machine Status: {overall_status}** | Risk: {overall_risk:.2f} | Health: {overall_health if overall_health is not None else 'N/A'}")
                latest_machine_health = overall_health

                # --- Machine-Level Health Index Plot (Still inside 'with machine_col:', but after the sensor loop) ---
                if latest_machine_health is not None:
                    new_health_entry = pd.DataFrame([{'Health Index': latest_machine_health}], index=[pd.Timestamp(current_time)])
                    health_hist_df = st.session_state.health_history[machine_name]
                    health_hist_df = pd.concat([health_hist_df, new_health_entry])
                    health_hist_df = health_hist_df.tail(HISTORY_POINTS)
                    st.session_state.health_history[machine_name] = health_hist_df
                else:
                    health_hist_df = st.session_state.health_history[machine_name]

                if PREDICT_FUNC_AVAILABLE and not health_hist_df.empty:
                    st.markdown("**Overall Predicted Health Trend**")
                    health_chart = plot_health_index(health_hist_df)
                    st.altair_chart(health_chart, use_container_width=True)
                elif PREDICT_FUNC_AVAILABLE:
                    st.caption("_Health index data not yet available._")

        # --- Display Maintenance Log (Inside the loop and placeholder) ---
        st.divider()
        if not st.session_state.maintenance_log:
                st.write("No maintenance actions currently recommended.")
        else:
            log_df = pd.DataFrame(st.session_state.maintenance_log).sort_values(by="timestamp", ascending=False)

            # Display Pending Logs with buttons
            st.subheader("Pending Actions")
            pending_logs = log_df[log_df['status'] == 'Pending'].drop_duplicates(subset=['id']).reset_index(drop=True)
            if not pending_logs.empty:
                for index, row in pending_logs.iterrows():
                        log_id = row['id']
                        cols_log = st.columns([0.15, 0.15, 0.4, 0.15, 0.15])
                        cols_log[0].write(f"**{row['machine']}**")
                        cols_log[1].write(f"{row['sensor']}")
                        cols_log[2].caption(f"{row['reason']}")
                        # Use unique keys incorporating UUID
                        cols_log[3].button("Acknowledge", key=f"ack_{log_id}_{uuid.uuid4()}", on_click=update_log_status, args=(log_id, "Acknowledged"))
                        cols_log[4].button("Schedule", key=f"sch_{log_id}_{uuid.uuid4()}", on_click=update_log_status, args=(log_id, "Scheduled"))
            else:
                st.write("No pending actions.")

            # Display Acknowledged/Scheduled Logs (Optional)
            st.subheader("Completed/Scheduled Actions")
            completed_logs = log_df[log_df['status'] != 'Pending']
            if not completed_logs.empty:
                    st.dataframe(completed_logs[['timestamp', 'machine', 'sensor', 'status', 'reason']], use_container_width=True)
            else:
                    st.write("No completed or scheduled actions logged yet.")

    # Wait before the next update cycle
    time.sleep(UPDATE_INTERVAL_SECONDS)
