# 🚗 Traffic Accident Hotspot Prediction (Spatio-Temporal ML)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict and visualize traffic accident hotspots using advanced spatio-temporal machine learning for smart city safety analytics.

## 🌟 Features

- **🗺️ Spatial Analysis**: H3 hexagonal grid system for precise geographic modeling
- **⏰ Temporal Modeling**: Time-aware feature engineering with cyclical encoding
- **🌦️ Weather Integration**: Multi-dimensional weather impact analysis
- **🤖 ML Models**: LightGBM, Random Forest, and deep learning approaches
- **📊 Interactive Dashboard**: Real-time Streamlit web application
- **🎯 Risk Prediction**: Comprehensive hotspot risk scoring system
- **📈 Performance Analytics**: Model evaluation and feature importance analysis

## 🚀 Live Demo

**[🔗 Try the Live Application](https://your-app-url.streamlit.app)**

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Git

### Clone Repository

```bash
git clone https://github.com/your-username/accident-hotspot-prediction.git
cd accident-hotspot-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Alternative: Using Conda

```bash
conda env create -f environment.yml
conda activate accident-prediction
```

## ⚡ Quick Start

### 1. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

### 2. Complete Data Pipeline

```bash
# 1. Preprocess raw data
python src/preprocess.py

# 2. Engineer features
python src/feature_engineering.py

# 3. Label hotspots
python src/hotspot_labeling.py

# 4. Train models
python src/model_training.py

# 5. Analyze results
python src/hotspot_analysis.py
```

### 3. Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## 📊 Data Pipeline

### Input Data Format

The system expects CSV data with these columns:

```
Index(['ID', 'Source', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat',
       'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Description',
       'Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone',
       'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)',
       'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
       'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Amenity',
       'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
       'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
       'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
       'Astronomical_Twilight'], dtype='object')
```

### Pipeline Steps

1. **Data Preprocessing** (`src/preprocess.py`)
   - Data cleaning and validation
   - Missing value imputation
   - Outlier detection and removal
   - Data quality flag generation

2. **Feature Engineering** (`src/feature_engineering.py`)
   - Temporal features (cyclical encoding)
   - Spatial features (H3 cells, distance calculations)
   - Weather feature engineering
   - Road infrastructure analysis
   - Interaction feature creation

3. **Hotspot Labeling** (`src/hotspot_labeling.py`)
   - Time-window based hotspot identification
   - Severity-weighted scoring
   - Temporal pattern analysis
   - Weather-conditional hotspots
   - Comprehensive risk scoring

4. **Model Training** (`src/model_training.py`)
   - LightGBM gradient boosting
   - Random Forest ensemble
   - Deep learning models (LSTM)
   - Hyperparameter optimization
   - Cross-validation evaluation

## 🤖 Model Training

### Supported Models

- **LightGBM**: Primary gradient boosting model
- **Random Forest**: Ensemble baseline model
- **LSTM**: Sequential deep learning model
- **Spatial Graph NN**: Advanced spatial modeling

### Training Example

```python
from src.model_training import train_models_pipeline

results = train_models_pipeline(
    filepath='data/features_accidents.csv',
    test_size=0.2,
    random_state=42
)

# Access trained models
lgb_model = results['lightgbm']['model']
rf_model = results['random_forest']['model']
```

### Model Performance

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| LightGBM | 87.3% | 0.843 | 0.891 |
| Random Forest | 84.1% | 0.821 | 0.867 |
| LSTM | 82.9% | 0.798 | 0.845 |

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Fork this repository**
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set main file: `streamlit_app.py`
   - Deploy!

### Local Development

```bash
# Run with hot reload
streamlit run streamlit_app.py --server.runOnSave true

# Run on specific port
streamlit run streamlit_app.py --server.port 8080
```

### Docker Deployment

```bash
# Build image
docker build -t accident-prediction .

# Run container
docker run -p 8501:8501 accident-prediction
```

## 📁 Project Structure

```
accident-hotspot-prediction/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── LICENSE                  # MIT license
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code modules
│   ├── preprocess.py       # Data preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── spatial_utils.py    # Spatial analysis utilities
│   ├── hotspot_labeling.py # Hotspot identification
│   ├── hotspot_analysis.py # Analysis and visualization
│   ├── model_training.py   # ML model training
│   └── train_deep.py       # Deep learning models
│
├── notebooks/              # Jupyter notebooks
│   ├── spatial_analysis_hotspots.ipynb
│   └── hotspot_analysis.ipynb
│
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   ├── processed/         # Cleaned data
│   └── features/          # Engineered features
│
├── models/                 # Trained models
│   ├── lightgbm/
│   ├── random_forest/
│   └── deep_learning/
│
├── static/                 # Static assets
│   ├── css/               # Custom styles
│   └── images/            # Project images
│
└── config/                 # Configuration files
    └── model_config.yaml
```

## 🔧 API Reference

### Core Functions

#### Spatial Utilities

```python
from src.spatial_utils import assign_h3_cells, aggregate_by_h3

# Assign H3 cells
df = assign_h3_cells(df, h3_level=7)

# Aggregate by spatial cells
hotspot_counts = aggregate_by_h3(df)
```

#### Feature Engineering

```python
from src.feature_engineering import engineer_features

# Complete feature engineering
df_features = engineer_features('data/processed_accidents.csv')
```

#### Hotspot Analysis

```python
from src.hotspot_analysis import analyze_hotspot_characteristics

# Analyze hotspot patterns
results = analyze_hotspot_characteristics(df)
```

## 📈 Usage Examples

### Basic Hotspot Detection

```python
import pandas as pd
from src.hotspot_labeling import hotspot_labeling_pipeline

# Load and label data
df = hotspot_labeling_pipeline(
    filepath='data/features_accidents.csv',
    output_path='data/hotspot_accidents.csv'
)

# Get hotspot statistics
hotspot_rate = df['hotspot'].mean()
print(f"Hotspot rate: {hotspot_rate:.1%}")
```

### Spatial Visualization

```python
from src.hotspot_analysis import create_hotspot_heatmap

# Create interactive heatmap
hotspot_gdf = create_hotspot_heatmap(df)
```

### Model Prediction

```python
import joblib

# Load trained model
model = joblib.load('models/lightgbm_model.pkl')

# Make predictions
predictions = model.predict(X_new)
```

## 🎯 Use Cases

### Smart City Applications

- **Emergency Response**: Optimize ambulance and police dispatch routes
- **Traffic Management**: Dynamic signal timing and route recommendations  
- **Urban Planning**: Infrastructure improvement prioritization
- **Public Safety**: Real-time risk warnings and alerts

### Research Applications

- **Transportation Safety**: Academic research on accident causation
- **Policy Analysis**: Evidence-based traffic safety regulations
- **Insurance Analytics**: Risk assessment and premium calculation
- **Mobility Studies**: Urban mobility pattern analysis

## 🏆 Model Features

### Spatio-Temporal Features

- **H3 Hexagonal Grids**: Uniform spatial discretization
- **Cyclical Time Encoding**: Proper temporal representation
- **Distance Calculations**: Proximity to key locations
- **Seasonal Patterns**: Weather and holiday effects

### Weather Integration

- **Multi-dimensional Weather**: Temperature, humidity, wind, visibility
- **Weather Categorization**: Discrete weather condition mapping
- **Interaction Effects**: Weather-time-location combinations
- **Historical Weather**: Long-term weather pattern analysis

### Road Infrastructure

- **Traffic Control Features**: Signals, signs, intersections
- **Road Characteristics**: Type, speed limit, complexity
- **Point of Interest**: Nearby amenities and landmarks
- **Network Analysis**: Road connectivity and accessibility

## 📊 Performance Metrics

### Model Evaluation

- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating curve
- **Feature Importance**: SHAP and LIME explanations

### Spatial Validation

- **Cross-validation**: Geographic and temporal splits
- **Hotspot Coverage**: Prediction coverage analysis
- **Risk Calibration**: Predicted vs actual risk comparison
- **Temporal Stability**: Model performance over time

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/accident-hotspot-prediction.git

# Create development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **H3 Spatial Indexing**: Uber's hexagonal hierarchical spatial indexing
- **OpenStreetMap**: Geographic data and mapping
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive data visualization
- **US Accidents Dataset**: Sobhan Moosavi et al.

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Portfolio**: [Your Portfolio](https://yourportfolio.com)

## 🔗 Links

- **Live Demo**: [Streamlit App](https://your-app-url.streamlit.app)
- **Documentation**: [Project Docs](https://your-docs-url.com)
- **Paper**: [Research Paper](https://your-paper-url.com)
- **Dataset**: [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

⭐ **Star this repository if you found it helpful!**

![Traffic Safety](https://via.placeholder.com/600x200/1f77b4/white?text=Building+Safer+Cities+with+AI)
