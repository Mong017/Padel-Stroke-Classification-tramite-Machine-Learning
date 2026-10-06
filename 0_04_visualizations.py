"""
SCRIPT 4 - Visualizations for the report. Generates the main charts starting 
from the same pipeline of the previous script. Picks the model with the highest 
balanced accuracy on leave-one-session-out.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  # only used for the [A] random-split reference
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Config
PROJECT_FOLDER = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\secondo semestre\stat machine learning\Project"
CLEAN_PATH = os.path.join(PROJECT_FOLDER, "padel_dataset_2026_mixed_sessions_clean.csv")
CHANNELS = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]
RANDOM_STATE = 5

df_long = pd.read_csv(CLEAN_PATH)
if "session" not in df_long.columns:
    def extract_session_from_stroke_id(sid):
        m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", str(sid))
        return m.group(0) if m else str(sid)
    df_long["session"] = df_long["stroke_id"].apply(extract_session_from_stroke_id)
if "block_id" not in df_long.columns:
    df_long["block_id"] = df_long["session"]
if "person" not in df_long.columns:
    df_long["person"] = "unknown"

# Quick pipeline reconstruction (feature engineering)
def compute_engineered_features(g):
    feat = {}
    for channel in CHANNELS:
        x = g[channel].values
        feat[f"{channel}_std"] = np.std(x)
        feat[f"{channel}_range"] = np.max(x) - np.min(x)
        feat[f"{channel}_energy"] = np.mean((x - np.mean(x)) ** 2)
        feat[f"{channel}_mean"] = np.mean(x)
    if "acc_mag" in g.columns:
        acc_mag = g["acc_mag"].values
        feat["acc_mag_max"] = np.max(acc_mag)
        feat["acc_mag_std"] = np.std(acc_mag)
        feat["acc_mag_energy"] = np.mean((acc_mag - np.mean(acc_mag)) ** 2)
        feat["peak_pos_rel"] = int(np.argmax(acc_mag)) / len(acc_mag)
    feat["label"] = g["label"].iloc[0]
    feat["session"] = g["session"].iloc[0]
    feat["block_id"] = g["block_id"].iloc[0]
    feat["person"] = g["person"].iloc[0]
    return pd.Series(feat)

stroke_features = df_long.groupby("stroke_id", sort=False).apply(
    compute_engineered_features, include_groups=False
).reset_index()

feature_cols = [c for c in stroke_features.columns
                 if c not in ("stroke_id", "label", "session", "block_id", "person")]
X = stroke_features[feature_cols]
y = stroke_features["label"]
sessions = stroke_features["session"]
classes = sorted(y.unique())
n_sessions = sessions.nunique()
chance_level = 100.0 / len(classes)

# Only SVM (already known to be the best model from previous script) 
gkf = GroupKFold(n_splits=n_sessions)

best_model_name = "SVM"
svm_estimator = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
svm_param_grid = {"svc__C": [0.1, 1, 10, 100], "svc__kernel": ["rbf", "linear"],
                   "svc__gamma": ["scale", "auto"]}

print("Fitting SVM only (best model from Script 3)...")
grid = GridSearchCV(svm_estimator, svm_param_grid, scoring="balanced_accuracy",
                     cv=gkf.split(X, y, sessions))
grid.fit(X, y)
print(f"  Best parameters: {grid.best_params_}")

all_y_true, all_y_pred = [], []
for train_idx, test_idx in gkf.split(X, y, sessions):
    model = grid.best_estimator_
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    all_y_pred.extend(model.predict(X.iloc[test_idx]).tolist())
    all_y_true.extend(y.iloc[test_idx].tolist())

best_model = grid.best_estimator_
loso_acc = accuracy_score(all_y_true, all_y_pred) * 100
loso_bal_acc = balanced_accuracy_score(all_y_true, all_y_pred) * 100

print(f"SVM leave-one-session-out accuracy: {loso_acc:.1f}%  (balanced: {loso_bal_acc:.1f}%)")
print("-" * 70)

# Random split reference
def build_wide(df):
    rows = []
    for stroke_id, g in df.groupby("stroke_id", sort=False):
        g = g.reset_index(drop=True)
        row = {}
        for channel in CHANNELS:
            for t, v in enumerate(g[channel].values):
                row[f"{channel}_{t}"] = v
        row["stroke_type"] = g["label"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)

wide_data = build_wide(df_long)
wide_feature_cols = [c for c in wide_data.columns if any(c.startswith(ch + "_") for ch in CHANNELS)]
X_tr, X_te, y_tr, y_te = train_test_split(
    wide_data[wide_feature_cols], wide_data["stroke_type"], test_size=0.3,
    stratify=wide_data["stroke_type"], random_state=RANDOM_STATE
)
knn_ref = KNeighborsClassifier(n_neighbors=1, p=1).fit(X_tr, y_tr)
ref_acc = accuracy_score(y_te, knn_ref.predict(X_te)) * 100

print("Pipeline reconstructed. Generating charts...")
cmap = plt.get_cmap("tab10")

# CHART 1: Comparison between validation methods
methods = ["Random split\n(not reliable)", f"Leave-one-\nsession-out\n({best_model_name})", "Chance\nlevel"]
values = [ref_acc, loso_acc, chance_level]
colors = ["#d62728", "#2ca02c", "#7f7f7f"]

fig, ax = plt.subplots(figsize=(2.2 * len(methods), 5.5))
bars = ax.bar(methods, values, color=colors, edgecolor="black")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Comparison between validation methods")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "chart1_validation_comparison.png"), dpi=150)
plt.close()
print("Saved: chart1_validation_comparison.png")

# CHART 2: Confusion matrix 
cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix - leave-one-session-out, {best_model_name} ({loso_acc:.1f}%)")
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "chart2_confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: chart2_confusion_matrix.png")

# CHART 3 - Precision / Recall / F1 per class
prec, rec, f1, supp = precision_recall_fscore_support(all_y_true, all_y_pred, labels=classes, zero_division=0)
x_pos = np.arange(len(classes)); width = 0.25
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x_pos - width, prec, width, label="Precision", color="#1f77b4")
ax.bar(x_pos, rec, width, label="Recall", color="#ff7f0e")
ax.bar(x_pos + width, f1, width, label="F1-score", color="#2ca02c")
ax.set_xticks(x_pos); ax.set_xticklabels(classes, rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title(f"Precision / Recall / F1 per class (leave-one-session-out, {best_model_name})")
ax.axhline(chance_level/100, color="gray", linestyle="--", linewidth=1, label="Chance level")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "chart3_metrics_per_class.png"), dpi=150)
plt.close()
print("Saved: chart3_metrics_per_class.png")

# CHART 4: 2D PCA projection by class
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(8, 7))
for i, cls in enumerate(classes):
    mask = (y == cls).values
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls, color=cmap(i),
               alpha=0.7, s=40, edgecolor="white", linewidth=0.5)
ax.set_xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA projection of the feature space")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "chart4_pca_feature_space.png"), dpi=150)
plt.close()
print("Saved: chart4_pca_feature_space.png")

# CHART 5: Example raw signals (acc_mag) for each class
fig, axes = plt.subplots(len(classes), 1, figsize=(9, 2 * len(classes)), sharex=True)
for i, cls in enumerate(classes):
    example_id = df_long.loc[df_long["label"] == cls, "stroke_id"].iloc[0]
    signal = df_long.loc[df_long["stroke_id"] == example_id, "acc_mag"].values
    axes[i].plot(signal, color=cmap(i))
    axes[i].set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=9)
    axes[i].axvline(np.argmax(signal), color="red", linestyle="--", linewidth=0.8)
axes[-1].set_xlabel("Sample within the window (centered on the peak)")
fig.suptitle("Example acc_mag signal for each stroke type", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "chart5_example_signals.png"), dpi=150)
plt.close()
print("Saved: chart5_example_signals.png")