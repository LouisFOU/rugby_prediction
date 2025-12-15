"""
Module: predict_2026_xgb.py
Description: Simulation XGBoost prenant en compte la force de l'effectif (Stars).
"""
import pandas as pd
import joblib
import os
import random

# Chemins
MODEL_PATH = "models/rugby_xgb.pkl"
ENCODER_PATH = "models/team_encoder_xgb.pkl"

# Métadonnées complètes pour 2026
# Ajout de la clé 'stars' (1-10)
CLUB_METADATA = {
    'Toulouse':    {'budget': 3, 'capacity': 19000, 'exp': 10, 'stars': 10},
    'La Rochelle': {'budget': 3, 'capacity': 16000, 'exp': 8, 'stars': 9},
    'Paris':       {'budget': 3, 'capacity': 20000, 'exp': 4, 'stars': 7},
    'Toulon':      {'budget': 3, 'capacity': 18000, 'exp': 7, 'stars': 8},
    'Racing 92':   {'budget': 3, 'capacity': 30000, 'exp': 9, 'stars': 8},
    'Bordeaux':    {'budget': 3, 'capacity': 33000, 'exp': 6, 'stars': 9},
    'Lyon':        {'budget': 3, 'capacity': 25000, 'exp': 4, 'stars': 6},
    'Clermont':    {'budget': 3, 'capacity': 19000, 'exp': 7, 'stars': 7},
    'Montpellier': {'budget': 3, 'capacity': 15600, 'exp': 5, 'stars': 6},
    'Castres':     {'budget': 2, 'capacity': 12500, 'exp': 6, 'stars': 5},
    'Pau':         {'budget': 2, 'capacity': 18000, 'exp': 0, 'stars': 4},
    'Bayonne':     {'budget': 2, 'capacity': 16900, 'exp': 0, 'stars': 5},
    'Perpignan':   {'budget': 1, 'capacity': 14500, 'exp': 0, 'stars': 3},
    'Montauban':   {'budget': 1, 'capacity': 11000, 'exp': 0, 'stars': 2, 'proxy_model': 'Vannes'}
}

def simulate_match_points(proba_home_win):
    noise = random.uniform(-0.04, 0.04)
    adjusted_proba = proba_home_win + noise

    if adjusted_proba > 0.75: return 5, 0
    elif adjusted_proba > 0.55: return 4, 0
    elif adjusted_proba > 0.45: return 4, 1
    elif adjusted_proba > 0.25: return 0, 4
    else: return 0, 5

def simulate_season():
    print("[INFO] Chargement du moteur XGBoost (Version Squad Strength)...")
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle introuvable : {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    teams = list(CLUB_METADATA.keys())
    standings = {team: 0 for team in teams}
    
    print("[INFO] Simulation Saison 2026 en cours...")

    for home in teams:
        for away in teams:
            if home == away: continue

            home_name_for_model = CLUB_METADATA[home].get('proxy_model', home)
            away_name_for_model = CLUB_METADATA[away].get('proxy_model', away)

            try:
                home_code = encoder.transform([home_name_for_model])[0]
                away_code = encoder.transform([away_name_for_model])[0]
            except: continue

            home_meta = CLUB_METADATA[home]
            away_meta = CLUB_METADATA[away]

            match_data = pd.DataFrame([{
                'home_team_code': home_code,
                'away_team_code': away_code,
                'home_budget': home_meta['budget'],
                'away_budget': away_meta['budget'],
                'stadium_capacity': home_meta['capacity'],
                'is_international_window': 0,
                'home_exp': home_meta['exp'],
                'away_exp': away_meta['exp'],
                'home_stars': home_meta['stars'],
                'away_stars': away_meta['stars']
            }])
            
            probs = model.predict_proba(match_data)
            proba_home_win = probs[0][1] 

            pts_home, pts_away = simulate_match_points(proba_home_win)
            standings[home] += pts_home
            standings[away] += pts_away

    print("\n" + "="*45)
    print(" 🚀 CLASSEMENT FINAL 2026 (EFFECTIFS INCLUS) 🚀")
    print("="*45)
    
    sorted_standings = sorted(standings.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (team, pts) in enumerate(sorted_standings, 1):
        prefix = "🆕 " if team == "Montauban" else f"{rank}. "
        qualif = ""
        if rank <= 2: qualif = " (1/2 Finale)"
        elif rank <= 6: qualif = " (Barrage)"
        elif rank == 13: qualif = " (Barrage Acces)"
        elif rank == 14: qualif = " (Relégation)"
        
        print(f"{prefix}{team:<15} : {pts} pts{qualif}")
    print("="*45)

if __name__ == "__main__":
    simulate_season()