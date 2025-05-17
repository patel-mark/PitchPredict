import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
from difflib import get_close_matches

# -------------------- Load Model Artifacts -------------------- #
artifacts = joblib.load("xgb_artifacts.pkl")
team_stats = artifacts["team_stats"]

model = joblib.load("xgb_model_fold1.pkl")
clf = model["classifier"]
reg_home = model["regressor_home"]
reg_away = model["regressor_away"]

# -------------------- Standard Team Names -------------------- #
STANDARD_TEAM_NAMES = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester Utd',
    'Newcastle Utd', "Nott'ham Forest", 'Southampton', 'Tottenham',
    'West Ham', 'Wolves', 'Atalanta', 'Bologna', 'Cagliari', 'Como',
    'Empoli', 'Fiorentina', 'Genoa', 'Hellas Verona', 'Inter',
    'Juventus', 'Lazio', 'Lecce', 'Milan', 'Monza', 'Napoli', 'Parma',
    'Roma', 'Torino', 'Udinese', 'Venezia', 'Augsburg',
    'Bayern Munich', 'Bochum', 'Dortmund', 'Eint Frankfurt',
    'Freiburg', 'Gladbach', 'Heidenheim', 'Hoffenheim',
    'Holstein Kiel', 'Leverkusen', 'Mainz 05', 'RB Leipzig',
    'St. Pauli', 'Stuttgart', 'Union Berlin', 'Werder Bremen',
    'Wolfsburg', 'Angers', 'Auxerre', 'Brest', 'Le Havre', 'Lens',
    'Lille', 'Lyon', 'Marseille', 'Monaco', 'Montpellier', 'Nantes',
    'Nice', 'Paris S-G', 'Reims', 'Rennes', 'Saint-Étienne',
    'Strasbourg', 'Toulouse', 'Alavés', 'Athletic Club',
    'Atlético Madrid', 'Barcelona', 'Betis', 'Celta Vigo', 'Espanyol',
    'Getafe', 'Girona', 'Las Palmas', 'Leganés', 'Mallorca', 'Osasuna',
    'Rayo Vallecano', 'Real Madrid', 'Real Sociedad', 'Sevilla',
    'Valencia', 'Valladolid', 'Villarreal', 'Blackburn',
    'Bristol City', 'Burnley', 'Cardiff City', 'Coventry City',
    'Derby County', 'Hull City', 'Leeds United', 'Luton Town',
    'Middlesbrough', 'Millwall', 'Norwich City', 'Oxford United',
    'Plymouth Argyle', 'Portsmouth', 'Preston', 'QPR', 'Sheffield Utd',
    'Sheffield Weds', 'Stoke City', 'Sunderland', 'Swansea City',
    'Watford', 'West Brom', 'Bari', 'Brescia', 'Carrarese',
    'Catanzaro', 'Cesena', 'Cittadella', 'Cosenza', 'Cremonese',
    'Frosinone', 'Juve Stabia', 'Mantova', 'Modena', 'Palermo', 'Pisa',
    'Reggiana', 'Salernitana', 'Sampdoria', 'Sassuolo', 'Spezia',
    'Südtirol'
]

# -------------------- Standardization Function -------------------- #
def standardize_team_names(df, column):
    standardized = []
    for name in df[column]:
        matches = get_close_matches(name.strip(), STANDARD_TEAM_NAMES, n=1, cutoff=0.8)
        if matches:
            standardized.append(matches[0])
        else:
            st.warning(f"⚠️ Could not confidently match team name: '{name}'")
            standardized.append(name.strip())  # fallback
    df[column] = standardized
    return df

# -------------------- Match Parser -------------------- #
def parse_match_input(user_input):
    match = re.search(r"(.+?)\s+vs\.?\s+(.+)", user_input, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

# -------------------- Prediction Function -------------------- #
def predict_single_match(home_team, away_team):
    data = pd.DataFrame([{
        "Home_Team": home_team,
        "Away_Team": away_team
    }])
    data = standardize_team_names(data, "Home_Team")
    data = standardize_team_names(data, "Away_Team")
    
    data["features"] = data.apply(
        lambda row: (
            list(team_stats.get(row["Home_Team"], {}).values()) +
            list(team_stats.get(row["Away_Team"], {}).values())
        ), axis=1
    )
    X_new = np.array(data["features"].tolist())

    class_probs = clf.predict_proba(X_new)
    home_xg = reg_home.predict(X_new)[0]
    away_xg = reg_away.predict(X_new)[0]

    return {
        "home_team": data["Home_Team"].iloc[0],
        "away_team": data["Away_Team"].iloc[0],
        "home_win_prob": round(class_probs[0][2] * 100, 1),
        "draw_prob": round(class_probs[0][1] * 100, 1),
        "away_win_prob": round(class_probs[0][0] * 100, 1),
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2)
    }

# -------------------- Streamlit UI -------------------- #
st.set_page_config(page_title="Football Match Predictor", page_icon="⚽")
st.title("⚽ PitchPredict")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Type your match query (e.g. 'Brighton vs Arsenal'):")

if user_input:
    st.session_state.chat_history = []  # Clear chat for each new input
    st.session_state.chat_history.append({"user": user_input})

    home, away = parse_match_input(user_input)
    if home and away:
        try:
            pred = predict_single_match(home, away)
            response = (
                f"🏟️ {pred['home_team']} vs {pred['away_team']}\n\n"
                f"🔮 **Win Probabilities:**\n"
                f"- 🏠 {pred['home_team']} Win: {pred['home_win_prob']}%\n"
                f"- 🤝 Draw: {pred['draw_prob']}%\n"
                f"- 🚌 {pred['away_team']} Win: {pred['away_win_prob']}%\n\n"
                f"📊 **Expected Goals (xG):**\n"
                f"- {pred['home_team']}: {pred['home_xg']} xG\n"
                f"- {pred['away_team']}: {pred['away_xg']} xG"
            )
        except Exception as e:
            response = f"⚠️ Error making prediction: {str(e)}"
    else:
        response = "❌ Please enter a valid format like 'Team A vs Team B'"

    st.session_state.chat_history.append({"bot": response})


# Display chat history
for msg in st.session_state.chat_history:
    if "user" in msg:
        st.markdown(f"**🧑 You:** {msg['user']}")
    if "bot" in msg:
        st.markdown(f"**🤖 Bot:** {msg['bot']}")
