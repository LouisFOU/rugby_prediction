"""
Module: train.py
Description: Trains a Logistic Regression model to predict Top 14 match outcomes.
Includes data preprocessing (Label Encoding) and model evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Constants
INPUT_FILE = "data/processed/dataset_modeling.csv"
MODEL_DIR = "models"

def load_and_preprocess(file_path: str):
    """
    Loads data and encodes categorical variables (Team Names) into integers.
    Returns X (features), y (target), and the encoders (to decode later).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Initialize LabelEncoder
    # Note: We use a single encoder for both columns to ensure consistency
    # (e.g., 'Toulouse' must have the same ID in home_team and away_team)
    encoder = LabelEncoder()
    
    # Fit encoder on all unique teams found in both columns
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    encoder.fit(all_teams)
    
    # Transform strings to numbers
    df['home_team_encoded'] = encoder.transform(df['home_team'])
    df['away_team_encoded'] = encoder.transform(df['away_team'])
    
    # Select Features (X) and Target (y)
    features = ['home_team_encoded', 'away_team_encoded']
    target = 'home_win'
    
    return df[features], df[target], encoder

def train_model(X, y):
    """
    Splits data and trains a Logistic Regression model.
    """
    # Split: 80% for training, 20% for testing
    # random_state=42 ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Training set size: {len(X_train)} matches")
    print(f"[INFO] Test set size:     {len(X_test)} matches")
    
    # Initialize and train classifier
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Generate predictions on the test set
    predictions = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print("-" * 30)
    print(f"MODEL ACCURACY: {accuracy:.2%}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    return model

def save_artifacts(model, encoder):
    """
    Saves the trained model and encoder to disk for later use.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "team_encoder.pkl"))
    print(f"[SUCCESS] Model and encoder saved to {MODEL_DIR}/")

if __name__ == "__main__":
    try:
        print("[INFO] Starting training pipeline...")
        
        # 1. Prepare Data
        X, y, encoder = load_and_preprocess(INPUT_FILE)
        
        # 2. Train and Evaluate
        model = train_model(X, y)
        
        # 3. Save for Production
        save_artifacts(model, encoder)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        