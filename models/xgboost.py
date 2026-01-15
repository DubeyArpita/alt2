import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/dataset.csv")
print(df.head())
for col in ["employment_type", "income_range"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

print(df.head())
# -----------------------------
# 2. Separate Features & Target
# -----------------------------
X = df.drop(columns=["user_id", "alt_credit_score"])
y = df["alt_credit_score"]

# -----------------------------
# 3. Column Types
# -----------------------------
categorical_cols = [
    "employment_type",
    "income_range",
    
]

numeric_cols = [
    "bank_account_age_months",
    "num_bank_accounts",
    "monthly_income",
    "rent_paid_on_time",
    "utility_delay_days",
    "upi_txn_count",
    "avg_month_end_balance",
    "overdraft_event",
    "city_tier"
]

# -----------------------------
# 4. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# -----------------------------
# 5. XGBoost Regressor
# -----------------------------
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# -----------------------------
# 6. Pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 8. Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 9. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 100)

# -----------------------------
# 10. Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("XGBoost Regressor Performance")
print(f"MAE  : {mae:.2f}")  #mean absolute error
print(f"RMSE : {rmse:.2f}") #root mean squared error
print(f"R²   : {r2:.3f}")   # coefficient of determination

# -----------------------------
# 11. Sample Prediction
# -----------------------------
sample_user = X_test.iloc[[4]]
predicted_score = model.predict(sample_user)
predicted_score = np.clip(predicted_score, 0, 100)

print("\nPredicted Credit Score:", round(predicted_score[0], 2))
print("Actual Credit Score   :", y_test.iloc[4])

# =============================
# 12. TEST WITH NEW USER INPUT
# =============================

new_user = pd.DataFrame([{
    "employment_type": "gig",
    "income_range": "1000-3000",
    "city_tier": 3,
    "bank_account_age_months": 2,
    "num_bank_accounts": 1,
    "monthly_income": 2000,
    "rent_paid_on_time": 0.2,
    "utility_delay_days": 10,
    "upi_txn_count": 34,
    "avg_month_end_balance": 200,
    "overdraft_event": 0
}])

# Predict credit score
new_score = model.predict(new_user)
new_score = np.clip(new_score, 0, 100)[0]

# Credit label
def credit_label(score):
    if score < 40:
        return "High"
    elif score < 70:
        return "Medium"
    else:
        return "Low"
   

print("\n🧪 NEW USER CREDIT ASSESSMENT")
print("Predicted Credit Score :", round(new_score, 2))
print("Credit Risk Category   :", credit_label(new_score))
