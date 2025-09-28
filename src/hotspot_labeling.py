import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from spatial_utils import assign_h3_cells, aggregate_by_h3

def generate_hotspot_labels(df, spatial_col='h3_cell', time_col='Start_Time', 
                          window_days=30, threshold=10, severity_weight=True):
    """
    Generate hotspot labels based on accident frequency and severity.

    Parameters:
    - df: DataFrame with accident data
    - spatial_col: Column containing spatial identifiers (e.g., H3 cells)
    - time_col: Column containing datetime information
    - window_days: Time window for recent accidents (days)
    - threshold: Minimum number of accidents to be considered a hotspot
    - severity_weight: Whether to weight accidents by severity
    """
    print(f"Generating hotspot labels with {window_days}-day window and threshold {threshold}...")

    # Ensure time column is datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Define recent time window
    max_date = df[time_col].max()
    cutoff_date = max_date - timedelta(days=window_days)
    recent_df = df[df[time_col] > cutoff_date].copy()

    print(f"Using {len(recent_df):,} recent accidents (after {cutoff_date.date()})")

    if severity_weight and 'Severity' in recent_df.columns:
        # Weight accidents by severity (4=fatal gets highest weight)
        severity_weights = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}
        recent_df['weight'] = recent_df['Severity'].map(severity_weights)
        hotspot_counts = recent_df.groupby(spatial_col)['weight'].sum()
    else:
        # Simple count
        hotspot_counts = recent_df[spatial_col].value_counts()

    # Identify hotspots
    hotspots = hotspot_counts[hotspot_counts >= threshold].index.tolist()

    # Add hotspot labels to original dataframe
    df['hotspot'] = df[spatial_col].isin(hotspots).astype(int)
    df['hotspot_score'] = df[spatial_col].map(hotspot_counts).fillna(0)

    print(f"Identified {len(hotspots)} hotspot locations")
    print(f"Hotspot coverage: {df['hotspot'].mean()*100:.1f}% of accidents")

    return df, hotspots

def create_temporal_hotspots(df, spatial_col='h3_cell', time_col='Start_Time'):
    """Create time-aware hotspot labels."""
    print("Creating temporal hotspot patterns...")

    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek

    # Create time-space combinations
    time_periods = [
        ('morning_rush', [7, 8, 9]),
        ('evening_rush', [17, 18, 19]),
        ('night', [22, 23, 0, 1, 2, 3, 4, 5, 6]),
        ('weekend', None)  # Special case for weekends
    ]

    hotspot_patterns = {}

    for period_name, hours in time_periods:
        if period_name == 'weekend':
            period_df = df[df['dayofweek'].isin([5, 6])]  # Saturday, Sunday
        else:
            period_df = df[df['hour'].isin(hours)]

        if len(period_df) > 100:  # Minimum data requirement
            counts = period_df[spatial_col].value_counts()
            # Use 90th percentile as threshold for temporal hotspots
            threshold = counts.quantile(0.9)
            temporal_hotspots = counts[counts >= threshold].index.tolist()

            # Add temporal hotspot flag
            df[f'hotspot_{period_name}'] = df[spatial_col].isin(temporal_hotspots).astype(int)
            hotspot_patterns[period_name] = temporal_hotspots

            print(f"{period_name}: {len(temporal_hotspots)} hotspots")

    return df, hotspot_patterns

def create_severity_hotspots(df, spatial_col='h3_cell'):
    """Create hotspot labels based on accident severity."""
    print("Creating severity-based hotspots...")

    if 'Severity' not in df.columns:
        print("No severity column found, skipping severity hotspots")
        return df, {}

    # High severity hotspots (severity 3-4)
    high_severity_df = df[df['Severity'] >= 3]
    if len(high_severity_df) > 0:
        severity_counts = high_severity_df[spatial_col].value_counts()
        # Use cells with 2+ high severity accidents
        high_severity_hotspots = severity_counts[severity_counts >= 2].index.tolist()
        df['hotspot_high_severity'] = df[spatial_col].isin(high_severity_hotspots).astype(int)
        print(f"High severity hotspots: {len(high_severity_hotspots)}")
    else:
        df['hotspot_high_severity'] = 0
        high_severity_hotspots = []

    # Fatal accident hotspots (severity 4)
    fatal_df = df[df['Severity'] == 4]
    if len(fatal_df) > 0:
        fatal_counts = fatal_df[spatial_col].value_counts()
        fatal_hotspots = fatal_counts[fatal_counts >= 1].index.tolist()  # Any fatal accident
        df['hotspot_fatal'] = df[spatial_col].isin(fatal_hotspots).astype(int)
        print(f"Fatal accident hotspots: {len(fatal_hotspots)}")
    else:
        df['hotspot_fatal'] = 0
        fatal_hotspots = []

    return df, {
        'high_severity': high_severity_hotspots,
        'fatal': fatal_hotspots
    }

def create_weather_conditional_hotspots(df, spatial_col='h3_cell'):
    """Create hotspots that are conditional on weather conditions."""
    print("Creating weather-conditional hotspots...")

    weather_conditions = []

    # Rain/precipitation hotspots
    if 'Precipitation(in)' in df.columns:
        rain_df = df[df['Precipitation(in)'] > 0.1]  # Significant precipitation
        if len(rain_df) > 100:
            rain_counts = rain_df[spatial_col].value_counts()
            threshold = rain_counts.quantile(0.85)
            rain_hotspots = rain_counts[rain_counts >= threshold].index.tolist()
            df['hotspot_rain'] = df[spatial_col].isin(rain_hotspots).astype(int)
            weather_conditions.append(('rain', rain_hotspots))

    # Low visibility hotspots
    if 'Visibility(mi)' in df.columns:
        low_vis_df = df[df['Visibility(mi)'] < 2]  # Poor visibility
        if len(low_vis_df) > 100:
            vis_counts = low_vis_df[spatial_col].value_counts()
            threshold = vis_counts.quantile(0.85)
            vis_hotspots = vis_counts[vis_counts >= threshold].index.tolist()
            df['hotspot_low_visibility'] = df[spatial_col].isin(vis_hotspots).astype(int)
            weather_conditions.append(('low_visibility', vis_hotspots))

    # High wind hotspots
    if 'Wind_Speed(mph)' in df.columns:
        wind_df = df[df['Wind_Speed(mph)'] > 20]  # High wind
        if len(wind_df) > 100:
            wind_counts = wind_df[spatial_col].value_counts()
            threshold = wind_counts.quantile(0.85)
            wind_hotspots = wind_counts[wind_counts >= threshold].index.tolist()
            df['hotspot_high_wind'] = df[spatial_col].isin(wind_hotspots).astype(int)
            weather_conditions.append(('high_wind', wind_hotspots))

    print(f"Created {len(weather_conditions)} weather-conditional hotspot types")
    return df, dict(weather_conditions)

def create_comprehensive_hotspot_score(df):
    """Create a comprehensive hotspot risk score."""
    print("Creating comprehensive hotspot risk score...")

    # Collect all hotspot columns
    hotspot_cols = [col for col in df.columns if col.startswith('hotspot') and col != 'hotspot_score']

    if not hotspot_cols:
        print("No hotspot columns found for comprehensive scoring")
        return df

    # Weight different types of hotspots
    weights = {
        'hotspot': 1.0,  # Base hotspot
        'hotspot_high_severity': 2.0,
        'hotspot_fatal': 3.0,
        'hotspot_morning_rush': 1.5,
        'hotspot_evening_rush': 1.5,
        'hotspot_night': 1.2,
        'hotspot_weekend': 1.3,
        'hotspot_rain': 1.4,
        'hotspot_low_visibility': 1.6,
        'hotspot_high_wind': 1.3
    }

    # Calculate weighted risk score
    df['risk_score'] = 0
    for col in hotspot_cols:
        if col in df.columns:
            weight = weights.get(col, 1.0)
            df['risk_score'] += df[col] * weight

    # Normalize to 0-1 scale
    if df['risk_score'].max() > 0:
        df['risk_score'] = df['risk_score'] / df['risk_score'].max()

    # Create categorical risk levels
    df['risk_level'] = pd.cut(df['risk_score'], 
                            bins=[0, 0.2, 0.5, 0.8, 1.0],
                            labels=['Low', 'Medium', 'High', 'Critical'],
                            include_lowest=True)

    print("Comprehensive risk scoring completed")
    return df

def hotspot_labeling_pipeline(filepath, output_path, h3_level=7):
    """Complete hotspot labeling pipeline."""
    print("Starting hotspot labeling pipeline...")
    print("=" * 50)

    # Load data
    df = pd.read_csv(filepath, parse_dates=['Start_Time'])
    print(f"Loaded {len(df):,} records")

    # Assign H3 cells if not already present
    if 'h3_cell' not in df.columns:
        df = assign_h3_cells(df, h3_level=h3_level)
        print(f"Assigned H3 cells (level {h3_level})")

    # Generate different types of hotspots
    df, base_hotspots = generate_hotspot_labels(df)
    df, temporal_patterns = create_temporal_hotspots(df)
    df, severity_hotspots = create_severity_hotspots(df)
    df, weather_hotspots = create_weather_conditional_hotspots(df)

    # Create comprehensive risk score
    df = create_comprehensive_hotspot_score(df)

    # Summary statistics
    print("\nHotspot Labeling Summary:")
    print("=" * 30)
    print(f"Base hotspots: {len(base_hotspots)}")
    print(f"High severity hotspots: {len(severity_hotspots.get('high_severity', []))}")
    print(f"Fatal accident hotspots: {len(severity_hotspots.get('fatal', []))}")
    print(f"Weather-conditional types: {len(weather_hotspots)}")

    risk_distribution = df['risk_level'].value_counts()
    print(f"\nRisk Level Distribution:")
    for level, count in risk_distribution.items():
        print(f"  {level}: {count:,} ({count/len(df)*100:.1f}%)")

    # Save labeled data
    df.to_csv(output_path, index=False)
    print(f"\nLabeled data saved to {output_path}")

    return df

if __name__ == "__main__":
    hotspot_labeling_pipeline('../data/features_accidents.csv', '../data/hotspot_accidents.csv')