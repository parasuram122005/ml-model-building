# ML Model Building

A complete Machine Learning pipeline — from raw data to trained model — using Python and Scikit-learn.

## Features
- Data loading and preprocessing
- Feature engineering and selection
- Train/test split with cross-validation
- Multiple ML models: Linear Regression, Decision Tree, Random Forest
- Model evaluation metrics (Accuracy, RMSE, R², F1-Score)
- Confusion matrix and classification report
- Save and load trained models
- Prediction on new data

## Tech Stack
- Python 3
- Scikit-learn — ML models
- Pandas — data handling
- NumPy — numerical ops
- Matplotlib / Seaborn — result visualization
- Joblib — model persistence

## Setup
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

## How to Run
```bash
# Train all models
python train_model.py

# Evaluate models
python evaluate_model.py

# Predict new data
python predict.py
```

## Project Structure
```
ml-model-building/
├── train_model.py        # Training pipeline
├── evaluate_model.py     # Metrics and evaluation
├── predict.py            # Predict on new input
├── preprocessor.py       # Data preprocessing
├── sample_dataset.csv    # Sample training data
├── requirements.txt
└── README.md
```

## Model Results (Sample)
```
Model              Accuracy    RMSE     R²
Linear Regression  -           1.23    0.87
Decision Tree      91.3%       -       -
Random Forest      94.7%       -       -
```
