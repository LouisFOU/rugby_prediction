"""
Module: predict.py
Description: CLI tool to predict the winner between two Top 14 teams
using the trained Logistic Regression model.
"""

import joblib
import os
import sys
import numpy as np
import warnings

# Suppress scikit-learn warnings regarding feature names
warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "models/logistic_regression.pkl"
ENCODER_PATH = "models/team_encoder.pkl"

def load_artifacts():
    """
    Loads the trained model and the label encoder from disk.
    Raises FileNotFoundError if artifacts are missing.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Artifacts not found. Run 'src/train.py' first.")
        
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

def predict_match(home_team: str, away_team: str):
    """
    Encodes team names and outputs the probability of a home win.
    """
    try:
        model, encoder = load_artifacts()
        
        # Verify teams exist in the encoder's vocabulary
        known_teams = encoder.classes_
        
        if home_team not in known_teams:
            print(f"[ERROR] Unknown team: '{home_team}'")
            print(f"Available teams: {', '.join(known_teams)}")
            return

        if away_team not in known_teams:
            print(f"[ERROR] Unknown team: '{away_team}'")
            return

        # Encode inputs using the loaded encoder
        # reshape(1, -1) is required for single sample prediction
        home_encoded = encoder.transform([home_team])[0]
        away_encoded = encoder.transform([away_team])[0]
        
        features = np.array([[home_encoded, away_encoded]])
        
        # Get probability (Class 1 = Home Win)
        # predict_proba returns [[prob_loss, prob_win]]
        probabilities = model.predict_proba(features)[0]
        home_win_prob = probabilities[1]
        
        # Determine predicted winner based on probability
        if home_win_prob > 0.5:
            winner = home_team
            confidence = home_win_prob
        else:
            winner = away_team
            confidence = 1 - home_win_prob
        
        print("-" * 40)
        print(f"PREDICTION: {home_team} (Home) vs {away_team} (Away)")
        print("-" * 40)
        print(f"Predicted Winner : {winner}")
        print(f"Confidence       : {confidence:.2%}")
        print("-" * 40)
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")

if __name__ == "__main__":
    print("--- TOP 14 PREDICTOR ---")
    
    # Interactive input
    try:
        h_team = input("Enter Home Team: ").strip()
        a_team = input("Enter Away Team: ").strip()
        
        if h_team and a_team:
            predict_match(h_team, a_team)
        else:
            print("[ERROR] Inputs cannot be empty.")
            
    except KeyboardInterrupt:
        print("\n[INFO] Operation cancelled by user.")
        sys.exit(0)