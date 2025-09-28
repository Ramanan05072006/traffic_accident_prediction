
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """Load raw accident data from CSV."""
    print(f"Loading data from {filepath}...")

    # Read with proper date parsing
    date_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
    df = pd.read_csv(filepath, parse_dates=date_columns, low_memory=False)

    print(f"Loaded {len(df):,} records")
    print(f"Columns: {list(df.columns)}")

    return df

def clean_essential_columns(df):
    """Clean and validate essential columns."""
    print("Cleaning essential columns...")

    initial_count = len(df)

    # Drop rows with missing essential information
    essential_cols = ['Start_Lat', 'Start_Lng', 'Severity', 'Start_Time']
    for col in essential_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Validate coordinate ranges
    df = df[(df['Start_Lat'].between(-90, 90)) & (df['Start_Lng'].between(-180, 180))]

    # Validate severity values
    if 'Severity' in df.columns:
        df = df[df['Severity'].isin([1, 2, 3, 4])]

    # Remove duplicate records
    df = df.drop_duplicates()

    print(f"Removed {initial_count - len(df):,} invalid records")
    print(f"Remaining records: {len(df):,}")

    return df

def clean_weather_features(df):
    """Clean and impute weather-related features."""
    print("Cleaning weather features...")

    weather_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    for col in weather_cols:
        if col in df.columns:
            # Remove extreme outliers
            if col == 'Temperature(F)':
                df[col] = df[col].clip(-50, 130)
            elif col == 'Humidity(%)':
                df[col] = df[col].clip(0, 100)
            elif col == 'Wind_Speed(mph)':
                df[col] = df[col].clip(0, 200)
            elif col == 'Visibility(mi)':
                df[col] = df[col].clip(0, 50)
            elif col == 'Pressure(in)':
                df[col] = df[col].clip(20, 35)
            elif col == 'Precipitation(in)':
                df[col] = df[col].clip(0, 20)

            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())

    # Clean categorical weather columns
    categorical_weather = ['Weather_Condition', 'Wind_Direction', 'Sunrise_Sunset']
    for col in categorical_weather:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    print("Weather features cleaned")
    return df

def clean_location_features(df):
    """Clean location-related features."""
    print("Cleaning location features...")

    location_cols = ['Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone']

    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            # Clean string columns
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.title()

    # Clean distance
    if 'Distance(mi)' in df.columns:
        df['Distance(mi)'] = df['Distance(mi)'].fillna(0)
        df['Distance(mi)'] = df['Distance(mi)'].clip(0, 1000)  # Remove extreme outliers

    print("Location features cleaned")
    return df

def clean_road_features(df):
    """Clean road infrastructure features."""
    print("Cleaning road features...")

    # Boolean road features
    bool_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
                 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 
                 'Traffic_Signal', 'Turning_Loop']

    for col in bool_cols:
        if col in df.columns:
            # Convert to boolean, treating NaN as False
            df[col] = df[col].fillna(False)
            df[col] = df[col].astype(bool)

    print("Road features cleaned")
    return df

def add_data_quality_flags(df):
    """Add flags indicating data quality issues."""
    print("Adding data quality flags...")

    # Flag for missing weather data
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Wind_Speed(mph)', 'Visibility(mi)']
    df['missing_weather'] = df[weather_cols].isnull().any(axis=1).astype(int)

    # Flag for missing location details
    location_cols = ['Street', 'City', 'State']
    df['missing_location'] = df[location_cols].isnull().any(axis=1).astype(int)

    # Flag for end coordinates vs start coordinates
    if all(col in df.columns for col in ['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng']):
        df['coordinates_match'] = ((df['Start_Lat'] == df['End_Lat']) & 
                                 (df['Start_Lng'] == df['End_Lng'])).astype(int)

    print("Data quality flags added")
    return df

def create_derived_features(df):
    """Create basic derived features during preprocessing."""
    print("Creating derived features...")

    # Duration of accident (if End_Time available)
    if 'End_Time' in df.columns and 'Start_Time' in df.columns:
        df['duration_hours'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600
        df['duration_hours'] = df['duration_hours'].clip(0, 24)  # Cap at 24 hours

    # Create year-month for time series analysis
    if 'Start_Time' in df.columns:
        df['year_month'] = df['Start_Time'].dt.to_period('M')

    print("Derived features created")
    return df

def preprocess_pipeline(filepath, output_path):
    """Complete preprocessing pipeline."""
    print("Starting preprocessing pipeline...")
    start_time = datetime.now()

    # Load data
    df = load_data(filepath)

    # Clean data step by step
    df = clean_essential_columns(df)
    df = clean_weather_features(df)
    df = clean_location_features(df)
    df = clean_road_features(df)
    df = add_data_quality_flags(df)
    df = create_derived_features(df)

    # Final validation
    print("\nFinal data summary:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values per column:")
    missing_summary = df.isnull().sum()
    for col, missing in missing_summary[missing_summary > 0].items():
        print(f"  {col}: {missing:,} ({missing/len(df)*100:.1f}%)")

    # Save processed data
    df.to_csv(output_path, index=False)

    duration = datetime.now() - start_time
    print(f"\nPreprocessing complete in {duration.total_seconds():.1f} seconds")
    print(f"Processed data saved to {output_path}")

    return df

if __name__ == "__main__":
    preprocess_pipeline('../data/us_accidents.csv', '../data/processed_accidents.csv')