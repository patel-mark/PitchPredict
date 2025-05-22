from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
import io

app = FastAPI(
    title="PitchPredict API",
    description="Predict football match outcomes from team matchups or CSV uploads.",
    version="1.0",
)

# Allow access from any origin (good for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and artifacts
with open("xgb_model_fold1.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

team_stats = artifacts["team_stats"]
feature_columns = artifacts["feature_columns"]


@app.post("/predict")
def predict(matchup: str = Form(...)):
    """
    Predict match outcome from a matchup string like 'Chelsea vs Arsenal'
    """
    try:
        # Parse input
        home_team, away_team = [t.strip() for t in matchup.split("vs")]

        # Validate
        if home_team not in team_stats["Home_Team"] or away_team not in team_stats["Away_Team"]:
            raise ValueError("One or both teams not found in team_stats.")

        # Compute feature difference
        home_features = pd.Series(team_stats["Home_Team"][home_team])
        away_features = pd.Series(team_stats["Away_Team"][away_team])
        features = home_features - away_features

        X = features[feature_columns].values.reshape(1, -1)
        probabilities = model.predict_proba(X)[0]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": {
                "home_win": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "away_win": float(probabilities[2]),
            },
        }

    except Exception as e:
        return {"detail": str(e)}



@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with two columns: 'home_team' and 'away_team'
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if not {"home_team", "away_team"}.issubset(df.columns):
            raise ValueError("CSV must contain 'home_team' and 'away_team' columns.")

        predictions = []

        for _, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            if home_team in team_stats and away_team in team_stats:
                home_features = pd.Series(team_stats[home_team])
                away_features = pd.Series(team_stats[away_team])
                features = home_features - away_features
                X = features[feature_columns].values.reshape(1, -1)
                probabilities = model.predict_proba(X)[0]

                predictions.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_win": float(probabilities[0]),
                    "draw": float(probabilities[1]),
                    "away_win": float(probabilities[2]),
                })
            else:
                predictions.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "error": "One or both teams not found in team_stats."
                })

        return {"predictions": predictions}

    except Exception as e:
        return {"detail": str(e)}
