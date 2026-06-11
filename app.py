import os
import joblib
import pandas as pd
import numpy as np
import datetime
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stApp {
        background: radial-gradient(circle at top right, #1a1b26, #0d0e15);
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    .reportview-container {
        background: #0f1116;
    }
    .card {
        background-color: #161925;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d3142;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .win-badge {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .loss-badge {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .draw-badge {
        background-color: #6b7280;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Class for missing keys (same as used in training)
class MissingDict(dict):
    def __missing__(self, key):
        return key

# Load model and helper mappings
@st.cache_resource
def load_ml_resources():
    model_path = "model.pkl"
    mappings_path = "mappings.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(mappings_path):
        return None, None
        
    model = joblib.load(model_path)
    mappings = joblib.load(mappings_path)
    return model, mappings

model, mappings = load_ml_resources()

# App Header
st.title("⚽ Premier League Match Winner Predictor")
st.markdown("Predict the outcomes of English Premier League matches using a Random Forest model trained on historical match data, shooting metrics, and rolling team forms.")

if model is None or mappings is None:
    st.error("⚠️ Model files not found! Please run `python3 train.py` first to train the machine learning model.")
    st.stop()

# Extract mappings data
all_teams = mappings["all_teams"]
venue_categories = mappings["venue_categories"]
opponent_categories = mappings["opponent_categories"]
latest_team_matches = mappings["latest_team_matches"]
accuracy = mappings["accuracy"]
precision = mappings["precision"]

# Sidebar information
with st.sidebar:
    st.markdown("### Model Performance")
    st.metric("Accuracy", f"{accuracy:.1%}")
    st.metric("Precision", f"{precision:.1%}")
    
    st.markdown("---")
    st.markdown("### About Project")
    st.markdown("""
    This is a machine learning project to predict the winner of Premier League matches.
    It uses a Random Forest model trained on matches.csv.
    
    **Features used:**
    - 3-game rolling averages (goals, shots, shots on target, distance, etc.)
    - Venue (Home/Away)
    - Opponent team
    - Kick-off time & Day of the week
    """)
    st.write("Created by: Aryan")

# Setup layout
col1 = st.container()

with col1:
    st.subheader("Match Selection")
    
    # Selection inputs
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Select Home Team:", all_teams, index=all_teams.index("Man City") if "Man City" in all_teams else 0)
    with c2:
        # Default Away team to Liverpool or Arsenal if available
        default_away_idx = all_teams.index("Liverpool") if "Liverpool" in all_teams else (all_teams.index("Arsenal") if "Arsenal" in all_teams else 1)
        away_team = st.selectbox("Select Away Team:", all_teams, index=default_away_idx)
        
    if home_team == away_team:
        st.warning("⚠️ Home and Away team must be different.")
        
    match_date = st.date_input("Match Date:", datetime.date.today())

    # Predict button
    predict_btn = st.button("🔮 Run Predictor", use_container_width=True)

    if predict_btn and home_team != away_team:
        st.subheader("📊 Match Prediction Results")
        
        # Prepare feature values
        # 1. Date details
        day_code = match_date.weekday() # Monday=0, Sunday=6
        
        # 2. Get latest matches of Home and Away teams to compute rolling stats
        # We need the last 3 matches
        home_history = latest_team_matches[latest_team_matches["team"] == home_team].tail(3)
        away_history = latest_team_matches[latest_team_matches["team"] == away_team].tail(3)
        
        if len(home_history) < 3 or len(away_history) < 3:
            st.error("Insufficient historical form data to make predictions for one of these teams.")
        else:
            cols_to_roll = mappings["cols_to_roll"]
            
            # Compute rolling averages
            home_rolling = home_history[cols_to_roll].mean().to_dict()
            away_rolling = away_history[cols_to_roll].mean().to_dict()
            
            # Get venue and opponent codes
            # Venue: Home is encoded as 1 (or whatever code matches in categories)
            # We look up in venue_categories which index represents "Home" and "Away"
            try:
                home_venue_code = venue_categories.index("Home")
                away_venue_code = venue_categories.index("Away")
            except ValueError:
                # Fallbacks in case categories are different
                home_venue_code = 1
                away_venue_code = 0
                
            # Opponent: map team names to opponent column versions
            opp_name_for_home = away_team
            opp_name_for_away = home_team
            
            # Look up codes
            try:
                home_opp_code = opponent_categories.index(opp_name_for_home)
            except ValueError:
                home_opp_code = 0 # Fallback
                
            try:
                away_opp_code = opponent_categories.index(opp_name_for_away)
            except ValueError:
                away_opp_code = 0
                
            # Create feature dicts matching predictors order
            # The order in mappings["predictors"] must be strictly followed
            # e.g., ["venue_code", "opp_code", "day_code", "gf_rolling", ...]
            
            home_features_dict = {
                "venue_code": home_venue_code,
                "opp_code": home_opp_code,
                "day_code": day_code
            }
            for col in cols_to_roll:
                home_features_dict[f"{col}_rolling"] = home_rolling[col]
                
            away_features_dict = {
                "venue_code": away_venue_code,
                "opp_code": away_opp_code,
                "day_code": day_code
            }
            for col in cols_to_roll:
                away_features_dict[f"{col}_rolling"] = away_rolling[col]
                
            # Convert to DataFrame matching predictors order
            predictors = mappings["predictors"]
            home_df = pd.DataFrame([home_features_dict])[predictors]
            away_df = pd.DataFrame([away_features_dict])[predictors]
            
            # Run Predictor
            # Predict probability of target=1 (Win)
            # classes_ are [0, 1] so index 1 represents probability of Win
            home_win_prob = model.predict_proba(home_df)[0][1]
            away_win_prob = model.predict_proba(away_df)[0][1]
            
            # Displays
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">{home_team} (Home)</div>
                    <div class="metric-label">Likelihood of Win</div>
                    <div class="metric-value">{home_win_prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">{away_team} (Away)</div>
                    <div class="metric-label">Likelihood of Win</div>
                    <div class="metric-value">{away_win_prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Summary Prediction
            st.markdown("### Model Verdict")
            diff = home_win_prob - away_win_prob
            
            if home_win_prob > 0.40 and diff > 0.12:
                st.success(f"🏆 **Model Prediction**: **{home_team}** is favored to win at home (probability: {home_win_prob:.1%}).")
            elif away_win_prob > 0.35 and diff < -0.12:
                st.success(f"🏆 **Model Prediction**: **{away_team}** is favored to win away from home (probability: {away_win_prob:.1%}).")
            elif abs(diff) <= 0.10:
                st.info(f"⚖️ **Model Prediction**: **Draw or Close Match**. Both teams have a very close win likelihood (Diff: {abs(diff):.1%}). A stalemate is highly probable.")
            else:
                st.info(f"⚖️ **Model Prediction**: **Draw or Low Confidence**. Neither team exhibits a clear win probability. A draw or tight match is expected.")
            
            # Done prediction block




