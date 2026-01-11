# Data Mining Project: Fire Prediction

## Overview

This project applies multiple machine learning algorithms to predict forest fires using environmental and geographical data. It includes data preprocessing, exploratory data analysis, and comprehensive model evaluation with various resampling strategies.

## Project Structure

```
├── code/                          # Core utilities and models
│   ├── data_loader.py            # Data loading utilities
│   ├── file_tree.py              # File structure utilities
│   ├── Metrics/                  # Custom evaluation metrics
│   └── myModels/                 # Custom model implementations
│       ├── clarans.py
│       ├── dbscan.py
│       ├── DecisionTree.py
│       ├── kmeans.py
│       ├── knn.py
│       └── RandomForest.py
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── EDA/                      # Exploratory data analysis
│   │   └── [climate, elevation, fire, landcover, soil analyses]
│   ├── processing_1-3/           # Data preprocessing pipelines
│   ├── modeling/                 # Model training & evaluation
│   │   ├── CLARANS/
│   │   ├── DBSCAN/
│   │   ├── DT/                   # Decision Tree
│   │   ├── kMeans/
│   │   ├── knn/                  # K-Nearest Neighbors
│   │   └── RF/                   # Random Forest
│   └── Merge/                    # Data merging operations
│
└── models/                       # Trained model artifacts

```

## Dataset & Features

The project analyzes multiple feature categories:

- **Climate**: Temperature, humidity, wind patterns
- **Elevation**: Topographic data
- **Fire**: Fire occurrence and characteristics
- **Landcover**: Vegetation and land use types
- **Soil**: Soil properties and composition

## Methods

### Clustering Algorithms

- K-Means
- CLARANS
- DBSCAN

### Classification Algorithms

- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

### Data Balancing Strategies

- SMOTE + Tomek Links
- K-Means SMOTE
- NearMiss
- Original (unbalanced)

## Key Features

✅ Comprehensive EDA on individual features  
✅ Multiple preprocessing pipelines  
✅ Hyperparameter optimization (search notebooks)  
✅ Bayesian evaluation metrics  
✅ Comparative model analysis  
✅ Custom metric implementations

## Usage

1. **Data Preparation**: Run notebooks in `notebooks/processing_*` folders
2. **Exploration**: Check `notebooks/EDA/` for data insights
3. **Modeling**: Execute model notebooks in `notebooks/modeling/{Algorithm}/`
4. **Results**: Model outputs and predictions stored in respective algorithm folders

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn
- matplotlib, seaborn (visualization)
- imbalanced-learn (SMOTE, NearMiss)

## Author Notes

- Results include Bayesian evaluation metrics
- Search notebooks contain hyperparameter tuning configurations
- Models are trained on multiple data balancing strategies for comparison
