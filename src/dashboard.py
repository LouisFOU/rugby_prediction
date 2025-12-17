"""
Module: dashboard.py
Description: Web Interface (Streamlit) for Top 14 prediction.
Language: English
Style: Professional/Minimalist
"""
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Rugby Top 14 Predictor",
    layout="wide"
)

# --- 1. METADATA ---
CLUB_METADATA = {
    'Toulouse':    {'budget': 3, 'capacity': 19000, 'exp': 10, 'stars': 10, 'struggle': 0},
    'La Rochelle': {'budget': 3, 'capacity': 16000, 'exp': 8, 'stars': 9, 'struggle': 0},
    'Bordeaux':    {'budget': 3, 'capacity': 33000, 'exp': 6, 'stars': 9, 'struggle': 0},
    'Racing 92':   {'budget': 3, 'capacity': 30000, 'exp': 9, 'stars': 8, 'struggle': 0},
    'Toulon':      {'budget': 3, 'capacity': 18000, 'exp': 7, 'stars': 8, 'struggle': 0},
    'Paris':       {'budget': 3, 'capacity': 20000, 'exp': 4, 'stars': 7, 'struggle': 1},
    'Clermont':    {'budget': 3, 'capacity': 19000, 'exp': 7, 'stars': 7, 'struggle': 2},
    'Lyon':        {'budget': 3, 'capacity': 25000, 'exp': 4, 'stars': 6, 'struggle': 1},
    'Montpellier': {'budget': 3, 'capacity': 15600, 'exp': 5, 'stars': 6, 'struggle': 5},
    'Castres':     {'budget': 2, 'capacity': 12500, 'exp': 6, 'stars': 5, 'struggle': 0},
    'Bayonne':     {'budget': 2, 'capacity': 16900, 'exp': 0, 'stars': 5, 'struggle': 1},
    'Pau':         {'budget': 2, 'capacity': 18000, 'exp': 0, 'stars': 4, 'struggle': 4},
    'Perpignan':   {'budget': 1, 'capacity': 14500, 'exp': 0, 'stars': 3, 'struggle': 8},
    'Montauban':   {'budget': 1, 'capacity': 11000, 'exp': 0, 'stars': 2, 'struggle': 10, 'proxy_model': 'Vannes'}
}

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    possible_paths = [
        "models/rugby_xgb.pkl",
        "../models/rugby_xgb.pkl",
        "rugby_prediction/models/rugby_xgb.pkl"
    ]
    
    model_path = None
    encoder_path = None

    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            encoder_path = p.replace("rugby_xgb.pkl", "team_encoder_xgb.pkl")
            break
    
    if model_path is None:
        return None, None
        
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# --- 3. SIMULATION ENGINE ---
def run_simulation(n_simulations, chaos_factor):
    model, encoder = load_resources()
    
    if model is None:
        st.error("Model file not found. Please ensure 'rugby_xgb.pkl' is in the models directory.")
        st.stop()

    teams = list(CLUB_METADATA.keys())
    
    # Match Construction
    matches = []
    for home in teams:
        for away in teams:
            if home == away: continue
            
            hm = CLUB_METADATA[home]
            am = CLUB_METADATA[away]
            
            h_name = hm.get('proxy_model', home)
            a_name = am.get('proxy_model', away)
            
            try:
                h_code = encoder.transform([h_name])[0]
                a_code = encoder.transform([a_name])[0]
            except: 
                continue

            matches.append({
                'home': home, 'away': away,
                'home_team_code': h_code, 'away_team_code': a_code,
                'home_budget': hm['budget'], 'away_budget': am['budget'],
                'stadium_capacity': hm['capacity'], 'is_international_window': 0,
                'home_exp': hm['exp'], 'away_exp': am['exp'],
                'home_stars': hm['stars'], 'away_stars': am['stars'],
                'home_struggle': hm['struggle'], 'away_struggle': am['struggle']
            })

    df_season = pd.DataFrame(matches)
    
    features_cols = ['home_team_code', 'away_team_code', 'home_budget', 'away_budget', 
                     'stadium_capacity', 'is_international_window', 'home_exp', 'away_exp', 
                     'home_stars', 'away_stars', 'home_struggle', 'away_struggle']
    
    base_probas = model.predict_proba(df_season[features_cols])[:, 1]
    
    stats = {t: {'points': 0, 'titles': 0, 'top6': 0, 'releg': 0} for t in teams}

    progress_bar = st.progress(0)
    
    for i in range(n_simulations):
        if i % 10 == 0: progress_bar.progress(int((i / n_simulations) * 100))
        
        noise = np.random.uniform(-chaos_factor, chaos_factor, size=len(base_probas))
        sim_probas = base_probas + noise
        
        season_pts = {t: 0 for t in teams}
        
        for idx, p in enumerate(sim_probas):
            h, a = matches[idx]['home'], matches[idx]['away']
            if p > 0.75:   season_pts[h] += 5
            elif p > 0.55: season_pts[h] += 4
            elif p > 0.45: season_pts[h] += 4; season_pts[a] += 1
            elif p > 0.25: season_pts[a] += 4
            else:          season_pts[a] += 5
            
        ranking = sorted(season_pts.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (team, pts) in enumerate(ranking, 1):
            stats[team]['points'] += pts
            if rank == 1: stats[team]['titles'] += 1
            if rank <= 6: stats[team]['top6'] += 1
            if rank >= 13: stats[team]['releg'] += 1
            
    progress_bar.empty()
    
    results = []
    for team, data in stats.items():
        results.append({
            'Team': team,
            'Avg Points': int(round(data['points'] / n_simulations)),
            'Title Prob': int(round(data['titles'] / n_simulations * 100)),
            'Top 6 Prob': (data['top6'] / n_simulations * 100),
            'Relegation Prob': (data['releg'] / n_simulations * 100)
        })
        
    return pd.DataFrame(results).sort_values('Avg Points', ascending=False)

# --- 4. USER INTERFACE (UI) ---

st.title("Top 14 Performance Model (2026 Forecast)")
st.markdown("""
**Predictive Analytics Dashboard** based on XGBoost & Monte Carlo Simulations.
This tool simulates the upcoming season to forecast final standings, championship probabilities, and relegation risks using historical data and squad metrics.
""")

# Sidebar
with st.sidebar:
    st.header("Simulation Settings")
    n_sim = st.slider("Monte Carlo Iterations", 10, 10000, 200, step=10)
    chaos = st.slider("Uncertainty Factor (Noise)", 0.0, 0.9, 0.05, step=0.01)
    
    launch_btn = st.button("Run Analysis", type="primary")

# Execution Logic
if launch_btn:
    st.divider()
    with st.spinner('Processing simulations...'):
        df_results = run_simulation(n_sim, chaos)
    
    # KPI Display
    champion = df_results.iloc[0]['Team']
    proba_champ = df_results.iloc[0]['Title Prob']
    relegue = df_results.iloc[-1]['Team']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Champion", champion)
    col2.metric("Title Probability", f"{proba_champ}%")
    col3.metric("High Relegation Risk", relegue)

    st.subheader("Forecasted Standings & Probabilities")
    
    # Data Table
    st.dataframe(
        df_results,
        column_config={
            "Team": st.column_config.TextColumn("Club", width="medium"),
            "Avg Points": st.column_config.NumberColumn("Avg Points", format="%.1f"),
            "Title Prob": st.column_config.ProgressColumn(
                "Title Chance", format="%.1f%%", min_value=0, max_value=1
            ),
            "Top 6 Prob": st.column_config.ProgressColumn(
                "Playoff Chance", format="%.1f%%", min_value=0, max_value=1
            ),
            "Relegation Prob": st.column_config.ProgressColumn(
                "Relegation Risk", format="%.1f%%", min_value=0, max_value=1
            ),
        },
        hide_index=True,
        use_container_width=True,
        height=600
    )

else:
    st.info("Configure simulation parameters in the sidebar and click **Run Analysis** to proceed.")