import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from teams import valid_teams
import matplotlib.pyplot as plt

# ---- Your function: standardize team names (placeholder) ---- #
def standardize_team_names(df, column):
    # Your logic (if applicable)
    return df

# ---- Prediction Function ---- #
def predict_fixtures_from_text(home_team, away_team):
    artifacts = joblib.load("xgb_artifacts.pkl")
    team_stats = artifacts["team_stats"]

    model = joblib.load("xgb_model_fold1.pkl")
    clf = model["classifier"]
    reg_home = model["regressor_home"]
    reg_away = model["regressor_away"]

    # Defensive check
    if home_team not in team_stats:
        raise ValueError(f"Home team '{home_team}' not found in team stats.")
    if away_team not in team_stats:
        raise ValueError(f"Away team '{away_team}' not found in team stats.")

    # Create features
    features = list(team_stats[home_team].values()) + list(team_stats[away_team].values())

    if len(features) != 16:
        raise ValueError(f"Feature shape mismatch. Got {len(features)} instead of 16.")

    X_new = np.array([features])
    class_probs = clf.predict_proba(X_new)
    home_xg = reg_home.predict(X_new)
    away_xg = reg_away.predict(X_new)

    return {
        "Home_Team": home_team,
        "Away_Team": away_team,
        "Home_Win_Prob": class_probs[0][2],
        "Draw_Prob": class_probs[0][1],
        "Away_Win_Prob": class_probs[0][0],
        "Home_xG": home_xg[0],
        "Away_xG": away_xg[0]
    }

# ---- Streamlit UI ---- #
st.set_page_config(page_title="⚽ Match Predictor Chat", layout="centered")
st.title("⚽ Football Match Predictor")

user_input = st.text_input("Type a fixture like:", "Newcastle United vs Nottingham Forest")

if " vs " in user_input:
    home_team, away_team = map(str.strip, user_input.split(" vs "))
    if st.button("Predict"):
        try:
            prediction = predict_fixtures_from_text(home_team, away_team)

            st.success(f"Match: {prediction['Home_Team']} vs {prediction['Away_Team']}")
            st.write(f"**Expected Goals (xG)**:")
            st.write(f"🏠 {prediction['Home_Team']}: `{prediction['Home_xG']:.2f}`")
            st.write(f"🛫 {prediction['Away_Team']}: `{prediction['Away_xG']:.2f}`")

            # ---- Bar Chart for Probabilities ---- #
            labels = ["Away Win", "Draw", "Home Win"]
            probs = [
                prediction["Away_Win_Prob"],
                prediction["Draw_Prob"],
                prediction["Home_Win_Prob"]
            ]
            colors = ['#e74c3c', '#f1c40f', '#2ecc71']  # red, yellow, green

            fig, ax = plt.subplots()
            bars = ax.barh(labels, [p * 100 for p in probs], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%', va='center')

            st.pyplot(fig)

        except ValueError as e:
            st.error(str(e))

# -------------------- Standardize Helper -------------------- #
def standardize_team_names(df, column):
    df[column] = df[column].str.strip()
    return df

# -------------------- CSV Batch Prediction Function -------------------- #
def predict_fixtures(fixtures_csv_path):
    artifacts = joblib.load('xgb_artifacts.pkl')
    team_stats = artifacts['team_stats']

    models = [joblib.load(f'xgb_model_fold{i+1}.pkl') for i in range(5)]

    new_fixtures = pd.read_csv(fixtures_csv_path)
    new_fixtures = standardize_team_names(new_fixtures, 'Home_Team')
    new_fixtures = standardize_team_names(new_fixtures, 'Away_Team')

    new_fixtures['features'] = new_fixtures.apply(
        lambda row: (
            list(team_stats.get(row['Home_Team'], {}).values()) +
            list(team_stats.get(row['Away_Team'], {}).values())
        ), axis=1
    )

    # Filter out rows with missing features
    valid_rows = new_fixtures['features'].apply(lambda x: len(x) == 16)
    new_fixtures = new_fixtures[valid_rows]
    X_new = np.array(new_fixtures['features'].tolist())

    class_probs = np.zeros((X_new.shape[0], 3))
    home_xg = np.zeros(X_new.shape[0])
    away_xg = np.zeros(X_new.shape[0])

    for model in models:
        class_probs += model['classifier'].predict_proba(X_new)
        home_xg += model['regressor_home'].predict(X_new)
        away_xg += model['regressor_away'].predict(X_new)

    df_preds = pd.DataFrame({
        'Home_Team': new_fixtures['Home_Team'],
        'Away_Team': new_fixtures['Away_Team'],
        'Home_Win_Prob': class_probs[:, 2] / len(models),
        'Draw_Prob': class_probs[:, 1] / len(models),
        'Away_Win_Prob': class_probs[:, 0] / len(models),
        'Predicted_Home_xG': home_xg / len(models),
        'Predicted_Away_xG': away_xg / len(models)
    })

    df_preds['Prob_Diff'] = abs(df_preds['Home_Win_Prob'] - df_preds['Away_Win_Prob'])
    df_sorted = df_preds.sort_values('Prob_Diff', ascending=False)
    return df_sorted

# -------------------- Streamlit App Section -------------------- #

st.title("📂 Batch Match Predictor")

st.markdown("### Upload a CSV file with your fixtures to get predictions for multiple matches at once.")

uploaded_file = st.file_uploader("Upload your fixtures CSV", type=["csv"])

st.markdown("### 📊 Predictions sorted by Probability Gap")
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with open("temp_fixtures.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        df_predictions = predict_fixtures("temp_fixtures.csv")

        st.success("✅ Predictions generated successfully!")
        st.dataframe(df_predictions, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
