import pandas as pd
import numpy as np
import h3
from datetime import datetime

def add_time_features(df, time_col='Start_Time'):
    """Add temporal features from datetime column."""
    df[time_col] = pd.to_datetime(df[time_col])

    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['month'] = df[time_col].dt.month
    df['year'] = df[time_col].dt.year
    df['day'] = df[time_col].dt.day
    df['quarter'] = df[time_col].dt.quarter

    # Derived time features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)

    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def add_spatial_features(df, lat_col='Start_Lat', lng_col='Start_Lng', h3_level=7):
    """Add spatial features including H3 cells."""
    df['h3_cell'] = df.apply(lambda row: h3.geo_to_h3(row[lat_col], row[lng_col], h3_level), axis=1)

    # Calculate spatial derivatives
    df['lat_rounded'] = df[lat_col].round(3)
    df['lng_rounded'] = df[lng_col].round(3)

    return df

def add_weather_features(df):
    """Engineer weather-related features."""
    # Handle missing values
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                    'Wind_Speed(mph)', 'Precipitation(in)', 'Wind_Chill(F)']

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Create weather categories
    if 'Temperature(F)' in df.columns:
        df['temp_category'] = pd.cut(df['Temperature(F)'], 
                                   bins=[-float('inf'), 32, 50, 70, 90, float('inf')],
                                   labels=['freezing', 'cold', 'mild', 'warm', 'hot'])

    if 'Wind_Speed(mph)' in df.columns:
        df['wind_category'] = pd.cut(df['Wind_Speed(mph)'],
                                   bins=[-1, 5, 15, 25, float('inf')],
                                   labels=['calm', 'light', 'moderate', 'strong'])

    if 'Visibility(mi)' in df.columns:
        df['visibility_category'] = pd.cut(df['Visibility(mi)'],
                                         bins=[-1, 1, 5, 10, float('inf')],
                                         labels=['poor', 'limited', 'good', 'excellent'])

    # Weather interaction features
    if 'Temperature(F)' in df.columns and 'Humidity(%)' in df.columns:
        df['heat_index'] = df['Temperature(F)'] + 0.5 * df['Humidity(%)']

    if 'Wind_Speed(mph)' in df.columns and 'Temperature(F)' in df.columns:
        df['wind_chill_calc'] = 35.74 + 0.6215 * df['Temperature(F)'] - 35.75 * (df['Wind_Speed(mph)'] ** 0.16) + 0.4275 * df['Temperature(F)'] * (df['Wind_Speed(mph)'] ** 0.16)

    # One-hot encode Weather_Condition (top N only)
    if 'Weather_Condition' in df.columns:
        top_weather = df['Weather_Condition'].value_counts().nlargest(10).index.tolist()
        for i, cond in enumerate(top_weather):
            df[f'weather_{i}'] = (df['Weather_Condition'] == cond).astype(int)
    return df        
def add_road_features(df):
    """Add road infrastructure features."""
    # Boolean road feature columns
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
                     'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 
                     'Traffic_Signal', 'Turning_Loop']

    for feature in road_features:
        if feature in df.columns:
            df[feature] = df[feature].astype(bool).astype(int)

    # Create road complexity score
    available_features = [f for f in road_features if f in df.columns]
    if available_features:
        df['road_complexity'] = df[available_features].sum(axis=1)
        df['has_traffic_control'] = ((df.get('Traffic_Signal', 0) == 1) | 
                                   (df.get('Stop', 0) == 1) | 
                                   (df.get('Traffic_Calming', 0) == 1)).astype(int)

    return df

def add_severity_features(df):
    """Add features related to accident severity."""
    if 'Severity' in df.columns:
        df['is_severe'] = (df['Severity'] >= 3).astype(int)
        df['is_fatal'] = (df['Severity'] == 4).astype(int)

    return df

def create_interaction_features(df):
    """Create interaction features between different categories."""
    interactions = []

    # Time-weather interactions
    if 'is_rush_hour' in df.columns and 'temp_category' in df.columns:
        df['rush_hour_temp'] = df['is_rush_hour'].astype(str) + '_' + df['temp_category'].astype(str)

    if 'is_weekend' in df.columns and 'is_night' in df.columns:
        df['weekend_night'] = df['is_weekend'] * df['is_night']

    # Weather-road interactions
    if 'visibility_category' in df.columns and 'road_complexity' in df.columns:
        df['visibility_complexity'] = df['visibility_category'].astype(str) + '_' + df['road_complexity'].astype(str)

    return df

def engineer_features(filepath, output_path=None):
    """Main feature engineering pipeline."""
    df = pd.read_csv(filepath, parse_dates=['Start_Time'])

    print("Starting feature engineering...")

    # Apply all feature engineering steps
    df = add_time_features(df)
    print("✓ Time features added")

    df = add_spatial_features(df)
    print("✓ Spatial features added")

    df = add_weather_features(df)
    print("✓ Weather features added")

    df = add_road_features(df)
    print("✓ Road features added")

    df = add_severity_features(df)
    print("✓ Severity features added")

    df = create_interaction_features(df)
    print("✓ Interaction features added")

    print(f"Feature engineering complete. Dataset shape: {df.shape}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Engineered features saved to {output_path}")

    return df

if __name__ == "__main__":
    engineer_features('../data/processed_accidents.csv', '../data/features_accidents.csv')            