# feat: baseline trainer
# Trains a simple model, saves figures + metrics
import json, os
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42
Path("figures").mkdir(exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)

# 1) Load data
df = pd.read_csv("data/synthetic_readmission.csv")
y = df["readmitted"].values
X = df[["age","sex","num_chronic","days_in_hosp","prior_admits"]].copy()

# One-hot encode 'sex' to numbers
X = pd.get_dummies(X, columns=["sex"], drop_first=True)  # makes 'sex_M' column

# 2) 70/15/15 split (train/valid/test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

# 3) Train baseline model
clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# 4) Evaluate (use test set)
p_test = clf.predict_proba(X_test)[:,1]
metrics = {
    "AUROC": float(roc_auc_score(y_test, p_test)),
    "AUPRC": float(average_precision_score(y_test, p_test)),
    "Brier": float(brier_score_loss(y_test, p_test))
}
print("Test metrics:", metrics)

# 5) Save ROC curve
fpr, tpr, _ = roc_curve(y_test, p_test)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
plt.tight_layout(); plt.savefig("figures/roc.png", dpi=160)

# 6) Save PR curve
prec, rec, _ = precision_recall_curve(y_test, p_test)
plt.figure(); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
plt.tight_layout(); plt.savefig("figures/pr.png", dpi=160)

# 7) Save Calibration plot
frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10)
plt.figure(); plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0,1],[0,1],'--')
plt.xlabel("Mean predicted"); plt.ylabel("Observed fraction"); plt.title("Calibration")
plt.tight_layout(); plt.savefig("figures/calibration.png", dpi=160)

# 8) Save metrics.json
with open("artifacts/metrics.json","w") as f:
    json.dump({"split":"70/15/15","model":"LogReg(balanced)","test":metrics}, f, indent=2)

print("Saved: figures/roc.png, figures/pr.png, figures/calibration.png, artifacts/metrics.json")


# Extra fairness prep (group by sex)
df_test = pd.concat([X_test.reset_index(drop=True),
                     pd.Series(y_test, name="readmitted")], axis=1)
results = df_test.groupby("sex_M")["readmitted"].mean().to_dict()
with open("artifacts/fairness_preview.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved fairness_preview.json" , "Saved artifacts/fairness_preview.json", results)

