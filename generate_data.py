"""
generate_data.py
Generates a realistic synthetic SaaS customer churn dataset.
Run once: python data/generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

tenure         = np.random.randint(1, 73, N)                          # months 1–72
monthly_charges= np.round(np.random.uniform(20, 120, N), 2)
total_charges  = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.0, N), 2)
num_products   = np.random.choice([1, 2, 3, 4], N, p=[0.3, 0.35, 0.25, 0.1])
support_tickets= np.random.poisson(1.5, N)
login_frequency= np.random.randint(0, 31, N)                          # logins/month
contract_type  = np.random.choice(["Month-to-month", "One year", "Two year"],
                                   N, p=[0.5, 0.3, 0.2])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    N, p=[0.35, 0.2, 0.25, 0.2]
)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
tech_support   = np.random.choice(["Yes", "No"], N, p=[0.4, 0.6])
online_backup  = np.random.choice(["Yes", "No"], N, p=[0.45, 0.55])

# Churn probability — driven by real-world signals
churn_prob = (
    0.05
    + 0.30 * (contract_type == "Month-to-month")
    + 0.15 * (support_tickets > 3)
    + 0.10 * (login_frequency < 5)
    + 0.10 * (tenure < 6)
    - 0.10 * (num_products >= 3)
    - 0.08 * (tech_support == "Yes")
    + 0.05 * senior_citizen
    + 0.05 * (payment_method == "Electronic check")
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn      = (np.random.rand(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "tenure":           tenure,
    "monthly_charges":  monthly_charges,
    "total_charges":    total_charges,
    "num_products":     num_products,
    "support_tickets":  support_tickets,
    "login_frequency":  login_frequency,
    "contract_type":    contract_type,
    "payment_method":   payment_method,
    "senior_citizen":   senior_citizen,
    "tech_support":     tech_support,
    "online_backup":    online_backup,
    "churn":            churn,
})

df.to_csv("data/churn_data.csv", index=False)
print(f"Dataset saved → data/churn_data.csv  |  {N} rows  |  Churn rate: {churn.mean():.1%}")
