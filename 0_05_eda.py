"""
SCRIPT 5 - Exploratory Data Analysis (EDA).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
PROJECT_FOLDER = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\secondo semestre\stat machine learning\Project"
CLEAN_PATH = os.path.join(PROJECT_FOLDER, "padel_dataset_2026_mixed_sessions_clean.csv")
CHANNELS = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]

df_long = pd.read_csv(CLEAN_PATH)
if "session" not in df_long.columns:
    def extract_session_from_stroke_id(sid):
        m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", str(sid))
        return m.group(0) if m else str(sid)
    df_long["session"] = df_long["stroke_id"].apply(extract_session_from_stroke_id)
if "person" not in df_long.columns:
    df_long["person"] = "unknown"

classes = sorted(df_long["label"].unique())
people = sorted(df_long["person"].unique())
cmap = plt.get_cmap("tab10")

# Per-stroke features
def compute_features(g):
    feat = {}
    for channel in CHANNELS:
        x = g[channel].values
        feat[f"{channel}_std"] = np.std(x)
        feat[f"{channel}_range"] = np.max(x) - np.min(x)
        feat[f"{channel}_energy"] = np.mean((x - np.mean(x)) ** 2)
        feat[f"{channel}_mean"] = np.mean(x)
    acc_mag = g["acc_mag"].values
    feat["acc_mag_max"] = np.max(acc_mag)
    feat["acc_mag_std"] = np.std(acc_mag)
    feat["acc_mag_energy"] = np.mean((acc_mag - np.mean(acc_mag)) ** 2)
    feat["label"] = g["label"].iloc[0]
    feat["person"] = g["person"].iloc[0]
    return pd.Series(feat)

stroke_features = df_long.groupby("stroke_id", sort=False).apply(compute_features, include_groups=False).reset_index()

print("Generating EDA charts...")

# PLOT A: Class balance
counts = stroke_features["label"].value_counts().reindex(classes)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values, color=[cmap(i) for i in range(len(classes))], edgecolor="black")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Number of strokes")
ax.set_title("Stroke distribution per class")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "eda_A_class_balance.png"), dpi=150)
plt.close()
print("Saved: eda_A_class_balance.png")

# PLOT B: Boxplot of key features per class
key_features = ["acc_mag_max", "acc_mag_energy", "x_gyro_range"]
fig, axes = plt.subplots(1, len(key_features), figsize=(15, 5))
for ax, feat in zip(axes, key_features):
    data_per_class = [stroke_features.loc[stroke_features["label"] == c, feat].values for c in classes]
    bp = ax.boxplot(data_per_class, tick_labels=classes, patch_artist=True)
    for patch, i in zip(bp["boxes"], range(len(classes))):
        patch.set_facecolor(cmap(i)); patch.set_alpha(0.7)
    ax.set_title(feat)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "eda_B_key_feature_boxplots.png"), dpi=150)
plt.close()
print("Saved: eda_B_key_feature_boxplots.png")

# PLOT C: Correlation matrix between features
feature_cols = [c for c in stroke_features.columns if c not in ("stroke_id", "label", "person")]
corr = stroke_features[feature_cols].corr()
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_cols))); ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=90, fontsize=7)
ax.set_yticklabels(feature_cols, fontsize=7)
ax.set_title("Correlation between engineered features")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "eda_C_feature_correlation.png"), dpi=150)
plt.close()
print("Saved: eda_C_feature_correlation.png")

# PLOT D: Intra-class variability: multiple overlaid signals
N_EXAMPLES = 8
n_classes = len(classes)
n_col = 3
n_row = int(np.ceil(n_classes / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(14, 2.5 * n_row), sharex=True)
axes = np.atleast_1d(axes).flatten()
for i, cls in enumerate(classes):
    class_strokes = df_long.loc[df_long["label"] == cls, "stroke_id"].unique()
    for sid in class_strokes[:N_EXAMPLES]:
        signal = df_long.loc[df_long["stroke_id"] == sid, "acc_mag"].values
        axes[i].plot(signal, color=cmap(i), alpha=0.5, linewidth=1)
    axes[i].set_title(cls)
    axes[i].set_ylabel("acc_mag")
for ax in axes[len(classes):]:
    ax.axis("off")
fig.suptitle(f"Intra-class variability: {N_EXAMPLES} overlaid strokes per type", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "eda_D_intra_class_variability.png"), dpi=150)
plt.close()
print("Saved: eda_D_intra_class_variability.png")

# PLOT E: 2D scatter of two interpretable features
fig, ax = plt.subplots(figsize=(8, 7))
for i, cls in enumerate(classes):
    mask = stroke_features["label"] == cls
    ax.scatter(stroke_features.loc[mask, "acc_mag_max"], stroke_features.loc[mask, "acc_mag_energy"],
               label=cls, color=cmap(i), alpha=0.7, s=40, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Peak acc_mag")
ax.set_ylabel("acc_mag energy")
ax.set_title("Class separation: peak vs energy of the movement")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_FOLDER, "eda_E_peak_vs_energy_scatter.png"), dpi=150)
plt.close()
print("Saved: eda_E_peak_vs_energy_scatter.png")

# PLOT F: Stroke distribution per person per class
if len(people) >= 2:
    table = pd.crosstab(stroke_features["person"], stroke_features["label"]).reindex(columns=classes)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(len(classes))
    width = 0.8 / len(people)
    for i, person in enumerate(people):
        values = table.loc[person].values if person in table.index else np.zeros(len(classes))
        ax.bar(x_pos + i * width, values, width, label=person, color=cmap(i))
    ax.set_xticks(x_pos + width * (len(people) - 1) / 2)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Number of strokes")
    ax.set_title("Stroke distribution per person and class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_FOLDER, "eda_F_distribution_per_person.png"), dpi=150)
    plt.close()
    print("Saved: eda_F_distribution_per_person.png")
else:
    print("Error, only one person")