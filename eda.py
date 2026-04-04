"""
eda.py  —  Exploratory Data Analysis for Churn Dataset
Run   : python notebooks/eda.py
Outputs plots to notebooks/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("notebooks", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("data/churn_data.csv")
print("=== Dataset Overview ===")
print(df.shape)
print(df.dtypes)
print("\nChurn distribution:\n", df["churn"].value_counts())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe().round(2))

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Customer Churn — EDA Dashboard", fontsize=16, fontweight="bold")

# 1. Churn distribution
df["churn"].map({0:"No Churn",1:"Churn"}).value_counts().plot(
    kind="bar", ax=axes[0,0], color=["#27ae60","#e74c3c"], rot=0
)
axes[0,0].set_title("Churn Distribution"); axes[0,0].set_ylabel("Count")

# 2. Tenure vs churn
df.groupby("churn")["tenure"].plot(kind="kde", ax=axes[0,1], legend=True)
axes[0,1].set_title("Tenure Distribution by Churn"); axes[0,1].set_xlabel("Months")
axes[0,1].legend(["No Churn","Churn"])

# 3. Monthly charges vs churn
sns.boxplot(data=df, x="churn", y="monthly_charges", ax=axes[0,2],
            palette=["#27ae60","#e74c3c"])
axes[0,2].set_title("Monthly Charges vs Churn")
axes[0,2].set_xticklabels(["No Churn","Churn"])

# 4. Contract type
ct = df.groupby(["contract_type","churn"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=axes[0,3], color=["#27ae60","#e74c3c"], rot=15)
axes[0,3].set_title("Contract Type vs Churn"); axes[0,3].legend(["No Churn","Churn"])

# 5. Support tickets
sns.histplot(data=df, x="support_tickets", hue="churn", multiple="stack",
             ax=axes[1,0], palette=["#27ae60","#e74c3c"], bins=10)
axes[1,0].set_title("Support Tickets vs Churn")

# 6. Login frequency
sns.boxplot(data=df, x="churn", y="login_frequency", ax=axes[1,1],
            palette=["#27ae60","#e74c3c"])
axes[1,1].set_title("Login Frequency vs Churn")
axes[1,1].set_xticklabels(["No Churn","Churn"])

# 7. Correlation heatmap
num_df = df[["tenure","monthly_charges","total_charges","num_products",
             "support_tickets","login_frequency","senior_citizen","churn"]]
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
            ax=axes[1,2], linewidths=0.5)
axes[1,2].set_title("Correlation Heatmap")

# 8. Products vs churn
pp = df.groupby(["num_products","churn"]).size().unstack(fill_value=0)
pp.plot(kind="bar", ax=axes[1,3], color=["#27ae60","#e74c3c"], rot=0)
axes[1,3].set_title("# Products vs Churn"); axes[1,3].legend(["No Churn","Churn"])

plt.tight_layout()
plt.savefig("notebooks/eda_dashboard.png", dpi=150, bbox_inches="tight")
print("\n✅ EDA dashboard saved → notebooks/eda_dashboard.png")
