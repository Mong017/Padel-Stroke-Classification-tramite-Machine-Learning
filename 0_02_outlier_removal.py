"""
SCRIPT 2 - Outlier removal and fixed-window trimming.

Works on the mixed sessions (padel_dataset_2026_mixed_sessions.csv),
now multi-person.
"""

import os
import numpy as np
import pandas as pd


PROJECT_FOLDER = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\secondo semestre\stat machine learning\Project"
INPUT_PATH = os.path.join(PROJECT_FOLDER, "padel_dataset_2026_mixed_sessions.csv")
OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "padel_dataset_2026_mixed_sessions_clean.csv")

FINAL_N = 50 #exact length of the final time-series window for each shot
MAD_THRESHOLD = 3.5 #strictenss statistical boundary used to flag a shot as an outlier

df = pd.read_csv(INPUT_PATH)

#  Compute a few summary features for each single stroke: before removing outliers, we characterize the
# shape and intensity of each individual stroke. We group the raw time-series data by its unique stroke_id
# and process each group g
def compute_features(g):
    gyro_mag = np.sqrt(g['x_gyro']**2 + g['y_gyro']**2 + g['z_gyro']**2) # computing the absolute rotational force vector (Euclidean Norm) of the gyroscope
    peak_pos = int(g['acc_mag'].values.argmax()) # find the index location where the peak acceleration occurs within the temporary 150 sample window
    # return a flattened metadata DataFrame where each row represents exactly one single stroke
    return pd.Series({
        'label': g['label'].iloc[0],
        'person': g['person'].iloc[0] if 'person' in g.columns else 'unknown',
        'acc_mag_max': g['acc_mag'].max(),
        'acc_mag_mean': g['acc_mag'].mean(),
        'acc_mag_std': g['acc_mag'].std(),
        'gyro_mag_max': gyro_mag.max(),
        'peak_pos': peak_pos,
        'peak_dev': abs(peak_pos - 50) # compute how far away the actual peak is from the theoretical center (index 50) of the initial extraction window
    })

stroke_features = df.groupby('stroke_id', sort=False).apply(compute_features).reset_index()

# Outlier detection per label, using MAD-based robust z-score: flags abnormal data points using statistical variance
# We use robust z score because standard z scores rely on the mean and standard deviation, which are heavily skewed by outliers. 
feature_cols = ['acc_mag_max', 'acc_mag_mean', 'acc_mag_std', 'gyro_mag_max', 'peak_dev']

# This function implements MAD (Median Absolute Deviation)
def robust_z(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    mad = mad if mad > 1e-6 else 1e-6
    return 0.6745 * (x - median) / mad # the constant 0.6745 scales the MAD to make it comparable to standard normal distributions

# outlier detection is done per shot class because for example a smash will inherently have much higher acceleration
#  than a forehand volley; checking them globally would accidentally label every smash as an outlier.
outlier_flag = pd.Series(False, index=stroke_features.index)
# we loop over all key feature columns
for col in feature_cols:
    z = stroke_features.groupby('label')[col].transform(robust_z)
    outlier_flag |= (z.abs() > MAD_THRESHOLD) # |= --> if a stroke is flagged as anomalous in even just one metric, it's marked as a true outlier overall

stroke_features['outlier'] = outlier_flag

print("Total strokes:", len(stroke_features))
print("Strokes discarded as outliers:", outlier_flag.sum())
print(stroke_features.groupby('label')['outlier'].sum())
print("-" * 60)

valid_strokes = stroke_features.loc[~stroke_features['outlier'], 'stroke_id']

# Trim to a fixed window of FINAL_N rows, centered on the real peak: our initial window was 150 samples wide. 
# This function trims the signal down to a strict size of 50 samples (FINAL_N), dynamically centered on the
# exact point of ball impact (acc_mag.argmax())
def trim_window(g, n=FINAL_N):
    stroke_id = g.name
    g = g.reset_index(drop=True)
    peak = int(g['acc_mag'].values.argmax())
    half = n // 2

    start = peak - half
    end = start + n
    # if a peak occurs too close to the beginning or end of our data segment, these bounds shift 
    #the window inward so the script never attempts to slice outside the array limits
    if start < 0:
        end -= start
        start = 0
    if end > len(g):
        start -= (end - len(g))
        end = len(g)
    start = max(start, 0)

    window = g.iloc[start:end].copy()
    window['stroke_id'] = stroke_id
    return window

# we filter the raw data, preserving only the stroke_id keys that were confirmed as valid in the previous step. 
clean_dataset = (
    df[df['stroke_id'].isin(valid_strokes)]
    .groupby('stroke_id', sort=False, group_keys=False)
    .apply(trim_window)
)

# we apply the window slicing function to the filtered data. Then if any stroke fails to provide exactlu 50 rows, it is dropped entirely. 
lengths = clean_dataset.groupby('stroke_id').size()
short_strokes = lengths[lengths != FINAL_N].index
if len(short_strokes) > 0:
    print(f"Discarding {len(short_strokes)} strokes too close to the edge of the original window")
    clean_dataset = clean_dataset[~clean_dataset['stroke_id'].isin(short_strokes)]

# Report and saving
print("-" * 60)
print("Final strokes in the clean dataset:", clean_dataset['stroke_id'].nunique())
print(clean_dataset.groupby('label')['stroke_id'].nunique())
print("Total rows:", len(clean_dataset), "(should be final_strokes (1220) *", FINAL_N, " = 61000)")

if 'person' in clean_dataset.columns:
    print("\nPerson x class distribution (clean dataset):")
    print(pd.crosstab(
        clean_dataset.groupby('stroke_id')['person'].first(),
        clean_dataset.groupby('stroke_id')['label'].first(),
    ))

clean_dataset.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")