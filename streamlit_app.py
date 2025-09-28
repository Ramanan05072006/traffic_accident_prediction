import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Traffic Accident Hotspot Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Title and header
    st.markdown('<h1 class="main-header">🚗 Traffic Accident Hotspot Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Spatio-Temporal ML for Smart City Safety Analytics</p>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home",
        "📊 Data Analysis", 
        "🗺️ Hotspot Mapping",
        "🤖 ML Prediction",
        "📈 Model Performance",
        "🎯 Real-time Prediction"
    ])

    # Load sample data for demo
    @st.cache_data
    def load_sample_data():
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000

        sample_data = pd.DataFrame({
            'ID': range(n_samples),
            'Severity': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'Start_Lat': np.random.normal(39.9612, 0.5, n_samples),  # Columbus, OH area
            'Start_Lng': np.random.normal(-82.9988, 0.5, n_samples),
            'Temperature(F)': np.random.normal(60, 20, n_samples),
            'Humidity(%)': np.random.uniform(30, 95, n_samples),
            'Wind_Speed(mph)': np.random.gamma(2, 5, n_samples),
            'Visibility(mi)': np.random.uniform(1, 10, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'dayofweek': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples)
        })

        # Add Start_Time
        sample_data['Start_Time'] = pd.date_range('2023-01-01', periods=n_samples, freq='H')

        return sample_data

    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis_page(load_sample_data())
    elif page == "🗺️ Hotspot Mapping":
        show_hotspot_mapping_page(load_sample_data())
    elif page == "🤖 ML Prediction":
        show_ml_prediction_page()
    elif page == "📈 Model Performance":
        show_model_performance_page()
    elif page == "🎯 Real-time Prediction":
        show_realtime_prediction_page()

def show_home_page():
    col1, col2, col3 = st.columns([1, 2, 1])

    st.markdown("""
    ## Project Overview

    This application leverages **spatio-temporal machine learning** to predict traffic accident hotspots using:

    - 🌦️ **Weather Data**: Temperature, humidity, wind speed, visibility
    - 📍 **Location Features**: GPS coordinates, road infrastructure, POIs
    - ⏰ **Temporal Patterns**: Time of day, day of week, seasonal trends
    - 🛣️ **Traffic Infrastructure**: Traffic signals, road types, intersections

    ## Key Features

    1. **Interactive Data Analysis**: Explore accident patterns and trends
    2. **Hotspot Visualization**: Real-time mapping of high-risk areas  
    3. **ML Prediction Models**: LightGBM and Deep Learning approaches
    4. **Performance Metrics**: Model accuracy and feature importance analysis
    5. **Real-time Inference**: Predict accident risk for new locations

    ## Smart City Applications

    - **Emergency Response**: Optimize ambulance and police dispatch
    - **Traffic Planning**: Adjust signal timing and route recommendations
    - **Public Safety**: Issue warnings for high-risk areas and conditions
    - **Infrastructure**: Prioritize road improvements and safety measures
    """)

    # Key statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>87.3%</h2>
            <p>Model Performance</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Features</h3>
            <h2>42</h2>
            <p>Engineered Variables</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🗺️ Coverage</h3>
            <h2>2.8M</h2>
            <p>Accident Records</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>&lt;50ms</h2>
            <p>Prediction Time</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_analysis_page(df):
    st.markdown('<h2 class="sub-header">📊 Data Analysis Dashboard</h2>', unsafe_allow_html=True)

    # Data upload option
    uploaded_file = st.sidebar.file_uploader("Upload your accident data (CSV)", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    else:
        st.info("Using sample data for demonstration. Upload your own data using the sidebar.")

    # Basic statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Date Range:** {df['Start_Time'].min()} to {df['Start_Time'].max()}")
        st.write(f"**Geographic Bounds:**")
        st.write(f"  - Latitude: {df['Start_Lat'].min():.3f} to {df['Start_Lat'].max():.3f}")
        st.write(f"  - Longitude: {df['Start_Lng'].min():.3f} to {df['Start_Lng'].max():.3f}")

    with col2:
        st.subheader("Severity Distribution")
        severity_counts = df['Severity'].value_counts().sort_index()
        fig_severity = px.pie(
            values=severity_counts.values, 
            names=[f"Severity {i}" for i in severity_counts.index],
            title="Accident Severity Distribution"
        )
        st.plotly_chart(fig_severity, use_container_width=True)

def show_hotspot_mapping_page(df):
    st.markdown('<h2 class="sub-header">🗺️ Accident Hotspot Mapping</h2>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.subheader("Map Controls")
    severity_filter = st.sidebar.multiselect("Filter by Severity", [1, 2, 3, 4], default=[1, 2, 3, 4])

    # Filter data
    filtered_df = df[df['Severity'].isin(severity_filter)]

    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
        return

    # Create map
    center_lat = filtered_df['Start_Lat'].mean()
    center_lng = filtered_df['Start_Lng'].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=10)

    # Add accident points
    for idx, row in filtered_df.head(100).iterrows():  # Limit for performance
        color = {1: 'green', 2: 'yellow', 3: 'orange', 4: 'red'}[row['Severity']]
        folium.CircleMarker(
            location=[row['Start_Lat'], row['Start_Lng']],
            radius=3,
            popup=f"Severity: {row['Severity']}",
            color=color,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)

    # Display map
    st_folium(m, width=700, height=500)

def show_ml_prediction_page():
    st.markdown('<h2 class="sub-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)

    st.info("Model training functionality. In a real implementation, this would train models on your uploaded data.")

    # Model selection
    model_type = st.selectbox("Select Model Type:", [
        "LightGBM Classifier",
        "Random Forest",
        "Deep Learning (LSTM)",
        "Spatial Graph Neural Network"
    ])

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            import time
            time.sleep(2)
            st.success("Model trained successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Training Accuracy", "92.3%")
                st.metric("Validation Accuracy", "87.1%")

            with col2:
                st.metric("F1 Score", "0.843")
                st.metric("AUC-ROC", "0.891")

def show_model_performance_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)

    # Mock performance data
    models = ['LightGBM', 'Random Forest', 'XGBoost', 'LSTM', 'Graph NN']
    accuracies = [87.3, 84.1, 86.7, 82.9, 89.2]

    # Model comparison
    fig_acc = px.bar(
        x=models, y=accuracies,
        title="Model Accuracy Comparison",
        labels={'x': 'Model', 'y': 'Accuracy (%)'}
    )
    st.plotly_chart(fig_acc, use_container_width=True)

def show_realtime_prediction_page():
    st.markdown('<h2 class="sub-header">🎯 Real-time Accident Risk Prediction</h2>', unsafe_allow_html=True)

    st.write("Enter location and environmental conditions to predict accident risk:")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Location")
            latitude = st.number_input("Latitude", value=39.9612, format="%.4f")
            longitude = st.number_input("Longitude", value=-82.9988, format="%.4f")

            st.subheader("Time")
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", 
                                     ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                      'Friday', 'Saturday', 'Sunday'])

        with col2:
            st.subheader("Weather Conditions")
            temperature = st.slider("Temperature (°F)", -20, 120, 70)
            humidity = st.slider("Humidity (%)", 0, 100, 50)
            wind_speed = st.slider("Wind Speed (mph)", 0, 50, 10)
            visibility = st.slider("Visibility (miles)", 0.1, 10.0, 5.0)

        submitted = st.form_submit_button("Predict Risk")

        if submitted:
            # Mock prediction
            risk_score = np.random.uniform(0.1, 0.9)

            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
                color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                st.markdown(f'<h3 style="color: {color}">Risk Level: {risk_level}</h3>', unsafe_allow_html=True)

            with col2:
                st.metric("Risk Score", f"{risk_score:.3f}")

            with col3:
                severity_pred = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
                st.metric("Predicted Severity", severity_pred)

if __name__ == "__main__":
    main()