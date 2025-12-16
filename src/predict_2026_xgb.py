"""
Module: predict_2026_xgb.py
Description: Simulation MONTE CARLO avec feature 'Struggle' (Historique négatif).
"""
import pandas as pd
import joblib
import os
import random
import numpy as np

# Chemins
MODEL_PATH = "models/rugby_xgb.pkl"
ENCODER_PATH = "models/team_encoder_xgb.pkl"

# Nombre de simulations
N_SIMULATIONS = 200 

# Métadonnées complètes (Avec Stars + Struggle)
# Struggle : 0 = Serein, 10 = En danger critique (Promu ou habitué du fond)
CLUB_METADATA = {
    'Toulouse':    {'budget': 3, 'capacity': 19000, 'exp': 10, 'stars': 10, 'struggle': 0},
    'La Rochelle': {'budget': 3, 'capacity': 16000, 'exp': 8, 'stars': 9, 'struggle': 0},
    'Paris':       {'budget': 3, 'capacity': 20000, 'exp': 4, 'stars': 7, 'struggle': 1},
    'Toulon':      {'budget': 3, 'capacity': 18000, 'exp': 7, 'stars': 8, 'struggle': 0},
    'Racing 92':   {'budget': 3, 'capacity': 30000, 'exp': 9, 'stars': 8, 'struggle': 0},
    'Bordeaux':    {'budget': 3, 'capacity': 33000, 'exp': 6, 'stars': 9, 'struggle': 0},
    'Lyon':        {'budget': 3, 'capacity': 25000, 'exp': 4, 'stars': 6, 'struggle': 1},
    'Clermont':    {'budget': 3, 'capacity': 19000, 'exp': 7, 'stars': 7, 'struggle': 2},
    'Montpellier': {'budget': 3, 'capacity': 15600, 'exp': 5, 'stars': 6, 'struggle': 5}, # <-- Poids lourd ici
    'Castres':     {'budget': 2, 'capacity': 12500, 'exp': 6, 'stars': 5, 'struggle': 0},
    'Pau':         {'budget': 2, 'capacity': 18000, 'exp': 0, 'stars': 4, 'struggle': 4},
    'Bayonne':     {'budget': 2, 'capacity': 16900, 'exp': 0, 'stars': 5, 'struggle': 1},
    'Perpignan':   {'budget': 1, 'capacity': 14500, 'exp': 0, 'stars': 3, 'struggle': 8},
    'Montauban':   {'budget': 1, 'capacity': 11000, 'exp': 0, 'stars': 2, 'struggle': 10, 'proxy_model': 'Vannes'}
}

def simulate_match_points(proba_home_win):
    noise = random.uniform(-0.05, 0.05)
    adjusted_proba = proba_home_win + noise
    if adjusted_proba > 0.75: return 5, 0
    elif adjusted_proba > 0.55: return 4, 0
    elif adjusted_proba > 0.45: return 4, 1
    elif adjusted_proba > 0.25: return 0, 4
    else: return 0, 5

def simulate_season():
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle introuvable : {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    teams = list(CLUB_METADATA.keys())
    
    stats = {team: {'points': 0, 'titles': 0, 'top2': 0, 'top6': 0, 'relegation': 0} for team in teams}
    
    print(f"[INFO] Simulation (Avec facteur 'Struggle')...")

    for i in range(N_SIMULATIONS):
        current_standings = {team: 0 for team in teams}
        for home in teams:
            for away in teams:
                if home == away: continue
                
                home_name = CLUB_METADATA[home].get('proxy_model', home)
                away_name = CLUB_METADATA[away].get('proxy_model', away)
                try:
                    home_code = encoder.transform([home_name])[0]
                    away_code = encoder.transform([away_name])[0]
                except: continue
                
                hm = CLUB_METADATA[home]
                am = CLUB_METADATA[away]
                
                match_data = pd.DataFrame([{
                    'home_team_code': home_code, 'away_team_code': away_code,
                    'home_budget': hm['budget'], 'away_budget': am['budget'],
                    'stadium_capacity': hm['capacity'], 'is_international_window': 0,
                    'home_exp': hm['exp'], 'away_exp': am['exp'],
                    'home_stars': hm['stars'], 'away_stars': am['stars'],
                    'home_struggle': hm['struggle'], 'away_struggle': am['struggle'] # <-- Nouveau
                }])
                
                probs = model.predict_proba(match_data)
                pts_h, pts_a = simulate_match_points(probs[0][1])
                current_standings[home] += pts_h
                current_standings[away] += pts_a

        sorted_season = sorted(current_standings.items(), key=lambda x: x[1], reverse=True)
        for rank, (team, pts) in enumerate(sorted_season, 1):
            stats[team]['points'] += pts
            if rank == 1: stats[team]['titles'] += 1
            if rank <= 2: stats[team]['top2'] += 1
            if rank <= 6: stats[team]['top6'] += 1
            if rank >= 13: stats[team]['relegation'] += 1

    print("\n" + "="*85)
    print(f" 📊 CLASSEMENT FINAL (Facteur 'Galère' inclus) 📊")
    print("="*85)
    print(f"{'Rg':<3} {'Équipe':<15} {'Pts':<6} | {'Titre':<7} {'Top 2':<7} {'Top 6':<7} | {'Reléguable':<10}")
    print("-" * 85)

    avg_points = {team: data['points'] / N_SIMULATIONS for team, data in stats.items()}
    final_ranking = sorted(avg_points.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (team, avg_pts) in enumerate(final_ranking, 1):
        d = stats[team]
        p_title, p_top2 = (d['titles']/N_SIMULATIONS)*100, (d['top2']/N_SIMULATIONS)*100
        p_top6, p_releg = (d['top6']/N_SIMULATIONS)*100, (d['relegation']/N_SIMULATIONS)*100
        
        s_title = f"{p_title:>5.1f}%" if p_title > 0 else "   -  "
        s_top2  = f"{p_top2:>5.1f}%" if p_top2 > 0 else "   -  "
        s_top6  = f"{p_top6:>5.1f}%" if p_top6 > 0 else "   -  "
        s_releg = f"{p_releg:>5.1f}%" if p_releg > 0 else "   -  "
        
        prefix = "🆕 " if team == "Montauban" else f"{rank:<2}. "
        print(f"{prefix}{team:<15} {avg_pts:<6.1f} | {s_title} {s_top2} {s_top6} | {s_releg}")
    print("="*85)

if __name__ == "__main__":
    simulate_season()