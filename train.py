import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Configuration
DATA_PATH = "matches.csv"
MODEL_PATH = "model.pkl"
MAPPINGS_PATH = "mappings.pkl"
SPLIT_DATE = "2025-08-01"  # Train on 2000-2025, test on the 2025/26 season

def compute_rolling_averages(group, cols, new_cols):
    """Computes the 3-game rolling average of goals scored & conceded for form."""
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
        
    print(f"Loading merged dataset {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    
    # Create target label (1 for Win, 0 for Loss/Draw)
    df["target"] = (df["result"] == "W").astype(int)
    
    # Convert categorical to codes
    df["venue"] = df["venue"].astype("category")
    df["venue_code"] = df["venue"].cat.codes
    
    df["opponent"] = df["opponent"].astype("category")
    df["opp_code"] = df["opponent"].cat.codes
    
    # Day of week (0=Monday, 6=Sunday)
    df["day_code"] = df["date"].dt.dayofweek
    
    # Features for rolling forms
    cols_to_roll = ["gf", "ga"]
    new_cols = [f"{col}_rolling" for col in cols_to_roll]
    
    print("Computing 3-game rolling forms (goals scored & conceded)...")
    df_rolling = df.groupby("team", group_keys=False).apply(lambda x: compute_rolling_averages(x, cols_to_roll, new_cols))
    df_rolling = df_rolling.reset_index(drop=True)
    
    # Define features for training
    base_predictors = ["venue_code", "opp_code", "day_code"]
    all_predictors = base_predictors + new_cols
    
    # Split chronologically: Train on matches before 25/26, test on the active 25/26 season!
    print(f"Splitting data into train (before {SPLIT_DATE}) and test (25/26 season)...")
    train = df_rolling[df_rolling["date"] < SPLIT_DATE]
    test = df_rolling[df_rolling["date"] >= SPLIT_DATE]
    
    print(f"Train matches: {train.shape[0]} | Test matches: {test.shape[0]}")
    
    # Train RandomForestClassifier
    print("Training RandomForest model...")
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
    rf.fit(train[all_predictors], train["target"])
    
    # Evaluate model performance
    preds = rf.predict(test[all_predictors])
    accuracy = accuracy_score(test["target"], preds)
    precision = precision_score(test["target"], preds)
    
    print(f"\nModel Performance on Test Set (2025/26 Season):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Save the mappings and category levels for prediction use
    venue_categories = df["venue"].cat.categories.tolist()
    opponent_categories = df["opponent"].cat.categories.tolist()
    
    # Keep the latest played matches for each team to calculate rolling averages in the UI
    latest_team_matches = df[df["result"].notna()].sort_values("date").groupby("team").tail(5)
    
    mappings = {
        "venue_categories": venue_categories,
        "opponent_categories": opponent_categories,
        "predictors": all_predictors,
        "base_predictors": base_predictors,
        "rolling_predictors": new_cols,
        "cols_to_roll": cols_to_roll,
        "latest_team_matches": latest_team_matches,
        "all_teams": sorted(df["team"].unique().tolist()),
        "accuracy": accuracy,
        "precision": precision
    }
    
    # Save model and mappings using joblib
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(mappings, MAPPINGS_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    print(f"Saved helper mappings to {MAPPINGS_PATH}")
    print("Training process finished successfully!")

if __name__ == "__main__":
    train_model()
