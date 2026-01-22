# PUBG Game Winner Prediction Using Machine Learning

This project predicts the winning probability of players or teams in PUBG
(PlayerUnknown’s Battlegrounds) matches using Machine Learning techniques.
The model analyzes in-game statistics to estimate the chances of winning.

---

## Problem Statement
PUBG is a battle royale game where multiple players compete to be the last
survivor. Predicting the winner based on gameplay statistics helps in
performance analysis and strategic decision-making.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Flask (optional – for web interface)

---

## Dataset
- PUBG Match Statistics Dataset
- Features include:
  - Kills
  - Assists
  - Damage dealt
  - Walk distance
  - Ride distance
  - Weapons acquired
  - Match duration

Dataset source:
- Kaggle PUBG Finish Placement Prediction Dataset

---

## Machine Learning Model
- Data preprocessing and feature scaling
- Train-test split
- Models used:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting (optional)
- Evaluation using R² score and RMSE

---

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
