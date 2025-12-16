"""
Module: train_xgb.py
Description: Entraînement XGBoost avec 'Squad Strength' + 'Struggle Score' (Historique Bas de Tableau).
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
MODEL_PATH = os.path.join(MODEL_DIR, "rugby_xgb.pkl") 
ENCODER_PATH = os.path.join(MODEL_DIR, "team_encoder_xgb.pkl")

# --- FEATURE 1 : EXPERIENCE PHASES FINALES ---
PLAYOFF_EXPERIENCE = {
    'Toulouse': 10, 'La Rochelle': 8, 'Clermont': 7, 'Racing 92': 9,
    'Toulon': 7, 'Castres': 6, 'Bordeaux': 6, 'Montpellier': 5,
    'Lyon': 4, 'Paris': 4, 'Pau': 0, 'Bayonne': 0, 'Perpignan': 0,
    'Oyonnax': 0, 'Agen': 0, 'Brive': 0, 'Grenoble': 0, 'Vannes': 0,
    'Montauban': 0
}

# --- FEATURE 2 : SQUAD STRENGTH (EFFECTIF 2025) ---
SQUAD_STRENGTH = {
    'Toulouse': 10, 'La Rochelle': 9, 'Bordeaux': 9, 'Racing 92': 8,
    'Toulon': 8, 'Paris': 7, 'Lyon': 6, 'Clermont': 7,
    'Montpellier': 6, 'Castres': 5, 'Pau': 4, 'Bayonne': 5,
    'Perpignan': 3, 'Oyonnax': 2, 'Vannes': 2, 'Montauban': 1,
    'Agen': 1, 'Brive': 2
}

# --- FEATURE 3 : RECENT STRUGGLES (Historique Bas de Tableau pondéré) ---
# Score élevé = A souvent fini dans les 3 derniers récemment.
# Calcul mental rapide : (13e ou 14e l'an passé = 5 pts), (il y a 2 ans = 3 pts)...
RECENT_STRUGGLES = {
    'Montpellier': 5, # A failli descendre en 2024
    'Perpignan': 8,   # Habitue du barrage
    'Oyonnax': 9,     # Yo-yo
    'Brive': 7,
    'Agen': 8,
    'Pau': 4,         # Souvent limite
    'Lyon': 1,
    'Clermont': 2,    # A eu des frayeurs
    'Paris': 1,
    'Castres': 0,
    'Bayonne': 1,     # A ete en ProD2 mais solide depuis
    'Toulouse': 0, 'La Rochelle': 0, 'Bordeaux': 0, 'Racing 92': 0, 'Toulon': 0,
    'Vannes': 10,     # Promu = Risque Max
    'Montauban': 10   # Promu = Risque Max
}

def get_meta_value(team_name, dictionary, default_val=0):
    for key, val in dictionary.items():
        if key in str(team_name): return val
    return default_val

def train():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    print("[INFO] Chargement des données...")
    df = pd.read_csv(INPUT_FILE)

    # Création des Features
    print("[INFO] Engineering : Expérience, Stars & Struggle...")
    
    # 1. Experience
    df['home_exp'] = df['home_team'].apply(lambda x: get_meta_value(x, PLAYOFF_EXPERIENCE))
    df['away_exp'] = df['away_team'].apply(lambda x: get_meta_value(x, PLAYOFF_EXPERIENCE))
    
    # 2. Stars
    df['home_stars'] = df['home_team'].apply(lambda x: get_meta_value(x, SQUAD_STRENGTH, 2))
    df['away_stars'] = df['away_team'].apply(lambda x: get_meta_value(x, SQUAD_STRENGTH, 2))

    # 3. Struggle (Nouveau !)
    df['home_struggle'] = df['home_team'].apply(lambda x: get_meta_value(x, RECENT_STRUGGLES, 5))
    df['away_struggle'] = df['away_team'].apply(lambda x: get_meta_value(x, RECENT_STRUGGLES, 5))

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
        'home_exp', 'away_exp',
        'home_stars', 'away_stars',
        'home_struggle', 'away_struggle' # <--- AJOUT ICI
    ]
    X = df[features]
    y = df['home_win']

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    print("[INFO] Entraînement XGBoost...")
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, eval_metric='logloss')
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- RÉSULTATS XGBOOST ---")
    print(f"Précision globale : {acc:.2%}")
    
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("Importance des variables :")
    print(importances)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[SUCCESS] Modèle sauvegardé.")

if __name__ == "__main__":
    train()