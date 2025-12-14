"""
Module: train_xgb.py
Description: Entraînement XGBoost corrigé (Sauvegarde via Joblib).
"""
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Chemins
INPUT_FILE = "data/processed/dataset_enriched.csv"
MODEL_DIR = "models"
# CHANGEMENT : On passe en .pkl comme le random forest
MODEL_PATH = os.path.join(MODEL_DIR, "rugby_xgb.pkl") 
ENCODER_PATH = os.path.join(MODEL_DIR, "team_encoder_xgb.pkl")

# --- FEATURE ENGINEERING ---
PLAYOFF_EXPERIENCE = {
    'Toulouse': 10, 'La Rochelle': 8, 'Clermont': 7, 'Racing 92': 9,
    'Toulon': 7, 'Castres': 6, 'Bordeaux': 6, 'Montpellier': 5,
    'Lyon': 4, 'Paris': 4, 'Pau': 0, 'Bayonne': 0, 'Perpignan': 0,
    'Oyonnax': 0, 'Agen': 0, 'Brive': 0, 'Grenoble': 0, 'Vannes': 0, 'Biarritz': 0,
    'Montauban': 0, 'Mt.Marsan': 0
}

def get_playoff_exp(team_name):
    for key, val in PLAYOFF_EXPERIENCE.items():
        if key in str(team_name): return val
    return 0

def train():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    print("[INFO] Chargement des données...")
    df = pd.read_csv(INPUT_FILE)

    # Features
    df['home_exp'] = df['home_team'].apply(get_playoff_exp)
    df['away_exp'] = df['away_team'].apply(get_playoff_exp)

    # Encodage
    le = LabelEncoder()
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    le.fit(all_teams)
    df['home_team_code'] = le.transform(df['home_team'])
    df['away_team_code'] = le.transform(df['away_team'])

    # Poids Temporel
    min_season = df['season'].min()
    weights = (df['season'] - min_season + 1) ** 2

    features = [
        'home_team_code', 'away_team_code', 
        'home_budget', 'away_budget', 
        'stadium_capacity', 'is_international_window',
        'home_exp', 'away_exp'
    ]
    X = df[features]
    y = df['home_win']

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    print("[INFO] Entraînement XGBoost...")
    # CORRECTION : Suppression de use_label_encoder=False (obsolète)
    model = XGBClassifier(
        n_estimators=200,      
        learning_rate=0.05,    
        max_depth=4,          
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- RÉSULTATS XGBOOST ---")
    print(f"Précision globale : {acc:.2%}")
    
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("Importance des variables :")
    print(importances)

    # CORRECTION CRITIQUE : Utilisation de joblib au lieu de save_model
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[SUCCESS] Modèle sauvegardé sous {MODEL_PATH}")

if __name__ == "__main__":
    train()