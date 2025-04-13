"""Functions for feature engineering on time-series sensor data."""

import pandas as pd
import numpy as np

# Define constants for feature engineering
ROLLING_WINDOW_SIZE = 10 # Example window size, adjust as needed

def create_lag_features(df, columns, lags):
    """Creates lagged features for specified columns."""
    df_lagged = df.copy()
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    return df_lagged

def create_rolling_features(df, columns, window_sizes):
    """Creates rolling window statistics for specified columns."""
    df_rolled = df.copy()
    for col in columns:
        for window in window_sizes:
            rolling_window = df_rolled[col].rolling(window=window)
            df_rolled[f'{col}_roll_mean_{window}'] = rolling_window.mean()
            df_rolled[f'{col}_roll_std_{window}'] = rolling_window.std()
            df_rolled[f'{col}_roll_min_{window}'] = rolling_window.min()
            df_rolled[f'{col}_roll_max_{window}'] = rolling_window.max()
            # Calculate rolling slope (trend) - using linear regression on the window
            df_rolled[f'{col}_roll_slope_{window}'] = rolling_window.apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
    return df_rolled

def create_expanding_features(df, columns):
    """Creates expanding window statistics (e.g., overall std dev up to that point)."""
    df_expanded = df.copy()
    for col in columns:
        expanding_window = df_expanded[col].expanding(min_periods=2) # Need at least 2 points for std dev
        df_expanded[f'{col}_expand_std'] = expanding_window.std()
        # Can add other expanding features like mean if needed
        # df_expanded[f'{col}_expand_mean'] = expanding_window.mean()
    return df_expanded

def time_since_last_maintenance(df, maintenance_col='simulation_type', maintenance_label='normal'):
    """Calculates time since the last 'normal' period ended (simulating maintenance)."""
    df_time = df.copy()
    # Assuming maintenance happens implicitly when simulation switches back to 'normal'
    maintenance_events = df_time[df_time[maintenance_col] == maintenance_label].index
    
    # Find groups of consecutive non-maintenance periods
    df_time['group'] = (df_time[maintenance_col] != maintenance_label).cumsum()
    df_time['time_since_maintenance'] = df_time.groupby('group').cumcount()
    
    # Set time to 0 during maintenance periods
    df_time.loc[df_time[maintenance_col] == maintenance_label, 'time_since_maintenance'] = 0
    
    df_time = df_time.drop(columns=['group'])
    return df_time

def apply_feature_engineering(df, sensor_columns, lags, window_sizes):
    """Applies all feature engineering steps."""
    print(f"Original shape: {df.shape}")
    df = create_lag_features(df, sensor_columns, lags)
    print(f"Shape after lags: {df.shape}")
    df = create_rolling_features(df, sensor_columns, window_sizes)
    print(f"Shape after rolling features: {df.shape}")
    df = create_expanding_features(df, sensor_columns)
    print(f"Shape after expanding features: {df.shape}")
    # df = time_since_last_maintenance(df) # Optional: Can be complex to get right
    # print(f"Shape after time since maintenance: {df.shape}")

    # Add time-based features
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['dayofyear'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        print(f"Shape after time features from timestamp: {df.shape}")
    elif hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
        # If timestamp column not found but we have a DatetimeIndex, use that
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        print(f"Shape after time features from index: {df.shape}")
    else:
        print("Warning: No datetime information found. Skipping time features.")
        
    # Drop rows with NaNs created by lags/rolling windows
    initial_rows = len(df)
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape}. Rows removed: {initial_rows - len(df)}")

    return df 

def calculate_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Calculates the rolling mean."""
    return series.rolling(window=window, min_periods=1).mean()

def calculate_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Calculates the rolling standard deviation."""
    return series.rolling(window=window, min_periods=1).std()

def extract_base_features(series: pd.Series, window: int = ROLLING_WINDOW_SIZE) -> pd.DataFrame:
    """
    Compute rolling mean and rolling standard deviation for the given series.
    
    Parameters:
         series: pd.Series of sensor values.
         window: Window size for computing rolling statistics.
    
    Returns:
         DataFrame with two columns: 'Rolling Mean' and 'Rolling Std Dev'.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0)
    return pd.DataFrame({'Rolling Mean': rolling_mean, 'Rolling Std Dev': rolling_std})

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Create sample data
    dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01',
                            '2023-01-01 00:00:02', '2023-01-01 00:00:03',
                            '2023-01-01 00:00:04', '2023-01-01 00:00:05',
                            '2023-01-01 00:00:06', '2023-01-01 00:00:07',
                            '2023-01-01 00:00:08', '2023-01-01 00:00:09',
                            '2023-01-01 00:00:10', '2023-01-01 00:00:11'])
    values = [10, 11, 10.5, 11.5, 12, 11.8, 12.2, 15, 14.5, 14.8, 15.2, 12] # Example includes an anomaly
    ts = pd.Series(values, index=dates)

    print("Original Series:")
    print(ts)
    print("---")

    # Extract features
    base_features = extract_base_features(ts, window=5)
    print(f"Extracted Features (Window=5):")
    print(base_features)

    # Example of getting the latest features
    latest_features = base_features.iloc[-1:] # Get the last row as a DataFrame
    print("---")
    print("Latest Features:")
    print(latest_features) 