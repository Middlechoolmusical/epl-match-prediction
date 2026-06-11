# Premier League Match Predictor ⚽

This is a simple machine learning project I built to predict the outcomes of English Premier League matches. It uses historical data, calculates team forms, and runs a prediction model through a clean web interface using Streamlit.

---

## 📂 Project Files

* **`matches.csv`**: The combined dataset containing Premier League match stats for 26 seasons (from 2000 to 2026).
* **`train.py`**: A python script that cleans the raw data, prepares the features (like 3-game rolling averages for goals, shots, corners, fouls, cards), and trains a Random Forest model.
* **`app.py`**: The Streamlit user interface where you can pick any Home and Away team to run a prediction.
* **`requirements.txt`**: A file listing the python packages needed to run the project.
* **`.gitignore`**: Excludes large files (like model weights) and temp files from being pushed to GitHub.

---

## ⚡ How to Setup & Run

### 1. Install packages
Run this command in your terminal to install the necessary libraries:
```bash
pip install pandas scikit-learn streamlit joblib numpy
```

### 2. Train the model
Run the training script to process the data, evaluate the model on the 2025/26 season, and save the model file:
```bash
python3 train.py
```

### 3. Launch the Web UI
Run the Streamlit app to open the prediction dashboard in your browser:
```bash
streamlit run app.py
```

---

## 🧠 How it Works

1. **Data Preparation**: We use `matches.csv`, which holds team-by-team match records from 2000 to 2026.
2. **Team Form Calculation**: We calculate a 3-game rolling average of goals scored and goals conceded. This captures the offensive and defensive form of each team.
3. **Training**: We train a Random Forest Classifier using scikit-learn.
4. **Validation**: We train the model on matches from 2000 to 2025, and test its accuracy on the matches from the current 2025/26 season. This shows us how well the model predicts future matches.
