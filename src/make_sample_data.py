import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 5000

tenure = rng.integers(0, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
payment = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
)
monthly = np.round(
    20 + 30 * (internet == "Fiber optic") + 15 * (internet == "DSL") + rng.normal(0, 8, n), 2
).clip(18, 120)
total = np.round(monthly * tenure + rng.normal(0, 20, n), 2).clip(0)

def yes_no(p):
    return rng.choice(["Yes", "No"], n, p=[p, 1 - p])

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(n)],
    "gender": rng.choice(["Male", "Female"], n),
    "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
    "Partner": yes_no(0.48),
    "Dependents": yes_no(0.30),
    "tenure": tenure,
    "PhoneService": yes_no(0.90),
    "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
    "InternetService": internet,
    "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
    "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
    "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
    "Contract": contract,
    "PaperlessBilling": yes_no(0.6),
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
})

# churn probability driven by realistic factors
logit = (
    -1.0
    + 1.4 * (df.Contract == "Month-to-month")
    - 1.2 * (df.Contract == "Two year")
    + 0.7 * (df.InternetService == "Fiber optic")
    + 0.6 * (df.PaymentMethod == "Electronic check")
    - 0.04 * df.tenure
    + 0.012 * (df.MonthlyCharges - 60)
    + 0.4 * (df.TechSupport == "No")
)
p = 1 / (1 + np.exp(-logit))
df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")

# mimic the real dataset's quirk: blank TotalCharges for brand-new customers
df["TotalCharges"] = df["TotalCharges"].astype(object)
df.loc[df.tenure == 0, "TotalCharges"] = " "

out = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}  | churn rate: {(df.Churn == 'Yes').mean():.1%}")
