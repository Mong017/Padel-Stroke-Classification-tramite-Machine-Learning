"""
SCRIPT 3 - Final modeling and validation.

Compares 4 algorithms (KNN, Random Forest, SVM, MLP) at 1 validation
level:
  [B] Leave-one-session-out  (generalization across different mountings)

Also includes a dedicated sub-task: smash_in vs smash_out, evaluated
with the same 4 algorithms.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# Config
PROJECT_FOLDER = r"C:\Users\giann\Desktop\universita\magistrale\esami da fare\SML\Project_SML"
CLEAN_PATH = os.path.join(PROJECT_FOLDER, "padel_dataset_2026_mixed_sessions_clean.csv")
CHANNELS = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]
RANDOM_STATE = 5

if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(f"Cannot find '{CLEAN_PATH}'. Run Script 2 first!")

# Load the cleaned master padel dataset from the CSV file.
df_long = pd.read_csv(CLEAN_PATH)

# Safety check: If the 'session' column is missing, recreate it dynamically
if "session" not in df_long.columns:
    # Define a helper function to extract the session timestamp using regular expressions
    def extract_session_from_stroke_id(sid):
        # Search for a standard timestamp pattern (YYYY-MM-DD_HH-MM-SS) inside the unique stroke ID
        m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", str(sid))
        # If the timestamp pattern is found, return it; otherwise, fallback to the raw stroke ID string
        return m.group(0) if m else str(sid)
    
    # Apply the extraction function to the 'stroke_id' column to populate the missing 'session' column
    df_long["session"] = df_long["stroke_id"].apply(extract_session_from_stroke_id)

# Safety check: If 'block_id' is missing, fallback to using the session identifier as a temporary block group
if "block_id" not in df_long.columns:
    df_long["block_id"] = df_long["session"]

# Safety check: If the 'person' column is missing, label the records as "unknown"
if "person" not in df_long.columns:
    df_long["person"] = "unknown"

# Print data summary statistics to verify the loaded structure before cross-validation
print(f"Available sessions: {df_long['session'].nunique()}")  # Number of unique recording sessions
print(f"Classes: {sorted(df_long['label'].unique())}")        # List of sorted shot labels
print("-" * 90)

# stratified random split (literature) --> risk of residual leakege, we use it only as a reference

# this function allows to convert the time series sensor data from long format to wide format
# this because machine learning algorithms cannot read 3D shapes, they expect a standard 2D tabular layout
#where each row represents one instance and each column represents one statistic feature. 
def build_wide(df):
    rows = [] # empty list that will store dictionaries; each dictionary will represents one flattened padel shot
    for stroke_id, g in df.groupby("stroke_id", sort=False): # this splits the dataset into small chunks g that contains all consecutive time samples belonging to one specific shot
        g = g.reset_index(drop=True) #resets the row index numbers inside the current shot group
        row = {} #initializes an empty dictionary
        for channel in CHANNELS: # iterates throigh each sensor stream (x_acc, y_acc,..)
            for t, v in enumerate(g[channel].values): # walk through every sequential data point in that channel; t is the time index and v the sensor value
                row[f"{channel}_{t}"] = v # creates a unique column key dictionary combining the channel name and the time step
        # appending the data
        row["stroke_type"] = g["label"].iloc[0] #extraxt the label 
        row["stroke_id"] = stroke_id #save the stroke id 
        rows.append(row) # the full dictionary is appended to the main rows list
    return pd.DataFrame(rows)

print("=== 1. Stratified random split (comparison with the literature) ===")
wide_data = build_wide(df_long) # collapsing each 50 sample shot into a single row, trasforming our time series data into flat rows comtaining 50samples * 6sensor_channel
# isolating features X and target labels Y: 
# wide_feature_cols --> keeps only the sensor data columns that start with our channel prefixes, ignoring identification metadata columns
# X_full --> feature matrix containing raw data shapes
#Y_full --> our ground-truth target vector containing the shot classifications
wide_feature_cols = [c for c in wide_data.columns if any(c.startswith(ch + "_") for ch in CHANNELS)]
X_full = wide_data[wide_feature_cols]
y_full = wide_data["stroke_type"]
# the following step splits the shots randomly, allocating 70% for training the model and 30% for testing it
# stratify = y_full --> ensures that both our training and test sets maintain the exact same proportions of shot types
# random_state = 5 --> locks the random number generator seed: every time we run the code, the random shuffle slices data in the exact same way, for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.3, stratify=y_full, random_state=RANDOM_STATE
)
# TRAINING AND EVALUATING 
knn_ref = KNeighborsClassifier(n_neighbors=1, p=1).fit(X_train, y_train) # 1 Nearest Neighbor classifier, with p=1 Manhattan Distance formula to measure similarity between shots
# make predictions on the test set compares them against the real labels and calculates the overall percentage of correct guesses
ref_acc = accuracy_score(y_test, knn_ref.predict(X_test)) 
print(f"Random split accuracy: {ref_acc*100:.1f}%")
print("-" * 70)

# Feature engineering: instead of feeding the raw time series signal, this function condenses the complex window
# into a few highly descriptive statistical summaries. These features describe the shape, intensity, speed of movement 
# and intensity distribution, making much easier for the algorithm that we will use to learn the actual physics of a 
# padel stroke 
def compute_engineered_features(g):
    feat = {} # empty dictionary to collect the extracted features for the current stroke
    # loop through each of our 6 inertial data (x_acc, y_acc, z_acc, x_gyro, ...)
    for channel in CHANNELS:
        x = g[channel].values # extract the raw numpy array containing the 50 sequential measurement for that specific channel
        feat[f"{channel}_std"] = np.std(x) # standard deviation (dispersion or variability of the signal): a violent stroke will have a much higher standard deviation
        feat[f"{channel}_range"] = np.max(x) - np.min(x) # range (absolute span between the peak positive and peak negative forces in the movement)
        feat[f"{channel}_energy"] = np.mean((x - np.mean(x)) ** 2) # energy (computes the variance of the zero centered signal)
        feat[f"{channel}_mean"] = np.mean(x) # mean (computes the average value of the signal)
    if "acc_mag" in g.columns: # acceleration magnitude features
        acc_mag = g["acc_mag"].values # extract the array of values 
        feat["acc_mag_max"] = np.max(acc_mag) # peak acceleration (exact moment of ball impact)
        feat["acc_mag_std"] = np.std(acc_mag) # dispersion
        feat["acc_mag_energy"] = np.mean((acc_mag - np.mean(acc_mag)) ** 2) # energy
        feat["peak_pos_rel"] = int(np.argmax(acc_mag)) / len(acc_mag) # relative peak position (where the max impact occurred relative to the window length)
    # appending group metadata and output
    feat["label"] = g["label"].iloc[0]
    feat["session"] = g["session"].iloc[0]
    feat["block_id"] = g["block_id"].iloc[0]
    feat["person"] = g["person"].iloc[0]
    return pd.Series(feat)

# our stroke features becomes a highly compressed table where each row represents exactly one single padel shot
# instead of 50 raw values per channel, each row contains our summaries alongside the target labels and split grouping tags
stroke_features = df_long.groupby("stroke_id", sort=False).apply(
    compute_engineered_features, include_groups=False
).reset_index()

# creating a list of column names, excluding the identification strings; it leeaves us with the clean metrics
engineered_feature_cols = [c for c in stroke_features.columns
                            if c not in ("stroke_id", "label", "session", "block_id", "person")]

X = stroke_features[engineered_feature_cols] # feature matrix (independent variable)
y = stroke_features["label"] # target vector (holds the categorical class for each shot)
# extracting the categorical columns used for grouping our data during cross validation
sessions = stroke_features["session"] 
blocks = stroke_features["block_id"]
people = stroke_features["person"]
classes = sorted(y.unique()) # finds all unique shot types and sorting them
chance_level = 100.0 / len(classes) # baseline accuracy (100/7) 

print(f"Number of features per stroke: {len(engineered_feature_cols)}")
print("-" * 70)

# COMPARISON of 4 models: KNN, Random Forest, SVM, MLP
# Grid search + evaluation with leave-one-session-out
n_sessions = sessions.nunique() # number of session 
gkf_session = GroupKFold(n_splits=n_sessions) # this initialized a Leave One Session Out (LOSO) validation scheme
# LOSO: instad of splitting shots at random, GroupKFold guarantees that when the data is split into train and test sets,
# one entire session is held out exclusively for testing, while the model trains on the remaining sessions. It repeats this
# loop until every session has been used as the test set exactly once.

print(f"=== 2. Model comparison - Leave-one-session-out (n={n_sessions} sessions) ===")

# dictionary that defines our 4ml estimators we want to benchmark. For each model it provides the base algorithm and a grid of
# hyperparameters to tune: 

# - KNN --> classifies a new shot based on how close its features are to the shots it already memorized in the training phase; 
# it has 2 parameters: n_neighbors (number of close neighbors to vote on the class) and p (distance metric, if p = 1 uses Manhattan distance, 
# while p=2 uses Euclidean distance) 

# - Random Forest --> is an ensemble method that builds an array of independent decision trees and averages their votes to pick
# the shot type. The parameters tuned are: n_estimators (number of decision trees in the forest and max_depth, the max depth
# allowed for each tree; None lets them expand fully)

# - SVM --> finds an optimal boundary hyperplane that maximizes the spatial margin between different shot classes. It uses a 
# pipeline to normalize our feature using StandardScaler right before passing them to hte classifier. The parameters tuned are
# svc__C (controls the error penalty trade-off) and svc__kernel (mathematical projection function)
 
# - MLP --> A feedforward artificial neural network consisting of interconnected layers of artifical neurons that optimize weights to 
# learn non linear patterns. It also uses a Pipeline with a standard sclaer. The parameter tuned are mlp__hidden_layer_sizes (the internal 
# layout), mlp__alpha (the L2 regularization parameter used to penalize overly large weights and prevent overfitting) and mlp__activation 
# (the activation mathematical formula, relu or tanh).
models_config = {
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5, 7], "p": [1, 2]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [None, 10]},
    ),
    "SVM": (
        Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
        {"svc__C": [0.1, 1, 10, 100], "svc__kernel": ["rbf", "linear"],
         "svc__gamma": ["scale", "auto"]},
    ),
    "MLP (neural network)": (
        Pipeline([("scaler", StandardScaler()),
                  ("mlp", MLPClassifier(random_state=RANDOM_STATE, max_iter=2000))]),
        {"mlp__hidden_layer_sizes": [(32,), (64,), (32, 16)],
         "mlp__alpha": [0.0001, 0.001, 0.01],
         "mlp__activation": ["relu", "tanh"]},
    ),
}

# estimating the hyperparameter tuning via Grid Search and compile out of fold predictions using GroupKFold 
# to evaluate a model's true capability
def evaluate_model_groupkfold(estimator, param_grid, X, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits) # instantietes the GroupKFold with our requested num of splits
    # Grid Search: this matches every combination of hyperparameters defined in our configuration dictionary   
    # we use balanced accuracy because some shot classes might have fewer examples than others, giving equal weight to every single padel shot 
    grid = GridSearchCV(estimator, param_grid, scoring="balanced_accuracy",
                         cv=gkf.split(X, y, groups))
    grid.fit(X, y) # run the search across all group folds to discover the single best parameter setup

    y_true_m, y_pred_m = [], [] # empty lists for true values and predictions
    # loop through each cross valdiation fold. Then isolate the hyperparameter combination that proved to be the best
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = grid.best_estimator_
        model.fit(X.iloc[train_idx], y.iloc[train_idx]) # re-train that optimal model version strictly on the training indices for that specific fold
        y_pred_m.extend(model.predict(X.iloc[test_idx]).tolist()) # making prediction on the held out test fold
        y_true_m.extend(y.iloc[test_idx].tolist())
    # Computing the final metrics: accuracy and balanced accuracy based on predictions made when those sessions were hidden from the training cycle
    return {
        "estimator": grid.best_estimator_,
        "best_params": grid.best_params_,
        "accuracy": accuracy_score(y_true_m, y_pred_m),
        "balanced_accuracy": balanced_accuracy_score(y_true_m, y_pred_m),
        "y_true": y_true_m, "y_pred": y_pred_m,
    }


session_results = {}
# this loop iterates through the models_config dictionary; for each ML algorithm, it runs the evaluate_model_groupkfold function
# this means every model undergoes its own Grid Search hyperparameter tuning and corss validation on out 6 sessions
#It saves all evaluation metrics inside the session results dictionary 
for name, (estimator, param_grid) in models_config.items():
    res = evaluate_model_groupkfold(estimator, param_grid, X, y, sessions, n_sessions)
    session_results[name] = res
    print(f"  {name:20s} -> accuracy: {res['accuracy']*100:5.1f}%   "
          f"balanced accuracy: {res['balanced_accuracy']*100:5.1f}%   "
          f"(best parameters: {res['best_params']})")

print("-" * 70)
#Selecting the winning model
best_model_name = max(session_results, key=lambda k: session_results[k]["balanced_accuracy"])
final_model = session_results[best_model_name]["estimator"]
all_y_true = session_results[best_model_name]["y_true"]
all_y_pred = session_results[best_model_name]["y_pred"]
loso_acc = session_results[best_model_name]["accuracy"]

print(f"Model chosen for subsequent analyses: {best_model_name} "
      f"(highest balanced accuracy: {session_results[best_model_name]['balanced_accuracy']*100:.1f}%)")
print(f"Chance level ({len(classes)} classes): {chance_level:.1f}%")
print(f"Leave-one-SESSION-out accuracy ({best_model_name}): {loso_acc*100:.1f}%")
print(classification_report(all_y_true, all_y_pred, labels=classes, digits=3))
print("-" * 70)

# smash_in vs smash_out (binary)
# Same 4-model comparison
print("=== 3. Dedicated sub-task: ONLY smash in vs smash out (binary outcome) ===")

smash_mask = y.isin(["smash_in", "smash_out"]) # boolean filter to isolate rows that are labeled as either a valid smash or an unforced smash
X_smash = X[smash_mask] # subset our feature matrix x
y_smash = y[smash_mask] # subset our target vector y
# cross validation sessions tracking lists to only contain data for these two specific shot types
sessions_smash = sessions[smash_mask]
n_sessions_smash = sessions_smash.nunique() # counts how many unique recording sessions actually contains smash data

print(f"Strokes in the sub-task: {len(X_smash)}  |  Sessions: {n_sessions_smash}")
print(y_smash.value_counts())
print()

smash_results = {}
# we take the 4ML models along with their tuning parameter grids and run them against our new isolated dataset of smash
for name, (estimator, param_grid) in models_config.items():
    res = evaluate_model_groupkfold(estimator, param_grid, X_smash, y_smash, sessions_smash, n_sessions_smash)
    smash_results[name] = res
    print(f"  {name:20s} -> accuracy: {res['accuracy']*100:5.1f}%   "
          f"balanced accuracy: {res['balanced_accuracy']*100:5.1f}%   "
          f"(best parameters: {res['best_params']})")

print("-" * 70)
best_smash_model_name = max(smash_results, key=lambda k: smash_results[k]["balanced_accuracy"])
smash_acc = smash_results[best_smash_model_name]["accuracy"] * 100

# Selecting the Smash winner algorithm
print(f"Best model for smash_in vs smash_out: {best_smash_model_name}")
print(f"\nChance level (binary): 50.0%")
print(f"smash_in vs smash_out accuracy: {smash_acc:.1f}%")
print(classification_report(
    smash_results[best_smash_model_name]["y_true"],
    smash_results[best_smash_model_name]["y_pred"],
    digits=3,
))

# Final summary
print("-" * 70)
print("=== SUMMARY ===")
print(f"Chance level ({len(classes)} classes):              {chance_level:.1f}%")
print(f"Random split (literature reference):         {ref_acc*100:.1f}%")
print(f"Leave-one-session-out (n={n_sessions}, main):           {loso_acc*100:.1f}%  [{best_model_name}]")
print(f"smash_in vs smash_out sub-task (binary):     {smash_acc:.1f}%  [{best_smash_model_name}]")