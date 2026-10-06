"""
SCRIPT 1 - Data Preprocessing
Extraction of MIXED sessions, with AUTOMATIC detection of
block boundaries at runtime.
"""

import os
import zipfile
import itertools
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PROJECT_FOLDER = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\secondo semestre\stat machine learning\Project"
extraction_folder = os.path.join(PROJECT_FOLDER, "extracted_padel_folders")
os.makedirs(extraction_folder, exist_ok=True)

# Base sequence of stroke types, cycled to label automatically detected
# blocks (works both for sessions with a single lap and with multiple
# laps of the sequence)
BASE_SEQUENCE = ["forehand", "backhand", "forehand_volley", "backhand_volley",
                  "smash_in", "smash_out", "lob"]

# Session configuration. Each session has:
#   - person: who performed the recording
#   - blocks: explicit list (start_time, end_time, label) IF already
#     verified by hand; otherwise None to use automatic detection
#   - pause_threshold: threshold in seconds for automatic boundary
#     detection (used only if blocks=None). Default 5.0, increase if
#     detection produces spurious blocks due to short internal pauses.

SESSIONS = {
    "all_shots_-2026-07-08_08-42-51.zip": {
        "person": "massi",
        "blocks": [
            (0.0, 33.0, "forehand"), (33.0, 68.1, "backhand"), (68.1, 103.4, "forehand_volley"),
            (103.4, 138.8, "backhand_volley"), (138.8, 173.4, "smash_in"), (173.4, 206.9, "smash_out"),
            (206.9, 243.4, "lob"), (243.4, 280.3, "forehand"), (280.3, 317.0, "backhand"),
            (317.0, 351.9, "forehand_volley"), (351.9, 392.3, "backhand_volley"), (392.3, 431.4, "smash_in"),
            (431.4, 473.2, "smash_out"), (473.2, 507.8, "lob"),
        ],
        "pause_threshold": None,
    },
    "all_shots_2-2026-07-08_10-32-40.zip": {
        "person": "massi",
        "blocks": [
            (0.0, 32.5, "forehand"), (32.5, 67.0, "backhand"), (67.0, 103.1, "forehand_volley"),
            (103.1, 142.8, "backhand_volley"), (142.8, 180.8, "smash_in"), (180.8, 218.4, "smash_out"),
            (218.4, 261.5, "lob"), (261.5, 303.8, "forehand"), (303.8, 344.6, "backhand"),
            (344.6, 383.7, "forehand_volley"), (383.7, 425.4, "backhand_volley"), (425.4, 467.5, "smash_in"),
            (467.5, 508.2, "smash_out"), (508.2, 543.3, "lob"),
        ],
        "pause_threshold": None,
    },
    # edo's sessions
    "Registration_1_edo-2026-07-08_18-16-34.zip": {
        "person": "edo",
        "blocks": [
            (0.0, 32.8, "forehand"), (32.8, 77.2, "backhand"), (77.2, 114.3, "forehand_volley"),
            (114.3, 149.1, "backhand_volley"), (149.1, 184.7, "smash_in"), (184.7, 219.5, "smash_out"),
            (219.5, 270.8, "lob"), (270.8, 321.1, "forehand"), (321.1, 363.5, "backhand"),
            (363.5, 396.6, "forehand_volley"), (396.6, 431.2, "backhand_volley"), (431.2, 467.3, "smash_in"),
            (467.3, 502.6, "smash_out"), (502.6, 540.1, "lob"),
        ],
        "pause_threshold": None,
    },
    "Registration_2_edo-2026-07-08_18-37-08.zip": {
        "person": "edo",
        "blocks": [
            (0.0, 31.0, "forehand"), (31.0, 74.9, "backhand"), (74.9, 112.7, "forehand_volley"),
            (112.7, 151.9, "backhand_volley"), (151.9, 200.8, "smash_in"), (200.8, 248.0, "smash_out"),
            (248.0, 306.7, "lob"), (306.7, 356.8, "forehand"), (356.8, 401.7, "backhand"),
            (401.7, 442.9, "forehand_volley"), (442.9, 490.5, "backhand_volley"), (490.5, 540.1, "smash_in"),
            (540.1, 584.1, "smash_out"), (584.1, 626.3, "lob"),
        ],
        "pause_threshold": None,
    },
    "Registration_3_edo-2026-07-08_19-04-50.zip": {
        "person": "edo",
        "blocks": [
            (0.0, 35.1, "forehand"), (35.1, 76.3, "backhand"), (76.3, 115.5, "forehand_volley"),
            (115.5, 161.3, "backhand_volley"), (161.3, 206.9, "smash_in"), (206.9, 260.5, "smash_out"),
            (260.5, 319.0, "lob"), (319.0, 369.0, "forehand"), (369.0, 416.2, "backhand"),
            (416.2, 463.2, "forehand_volley"), (463.2, 514.4, "backhand_volley"), (514.4, 561.0, "smash_in"),
            (561.0, 608.5, "smash_out"), (608.5, 658.6, "lob"),
        ],
        "pause_threshold": None,
    },

    #     massi's third session, CUSTOM sequence (different from the
    #     first two, to break any potential confound between block
    #     order and class)
    "all_shots_massi_3-2026-07-09_09-11-33.zip": {
        "person": "massi",
        "blocks": [
            (0.0, 29.7, "backhand_volley"), (29.7, 62.5, "smash_in"), (62.5, 99.3, "smash_out"),
            (99.3, 134.9, "lob"), (134.9, 174.0, "forehand"), (174.0, 214.7, "backhand"),
            (214.7, 261.3, "forehand_volley"), (261.3, 304.4, "smash_out"), (304.4, 339.2, "lob"),
            (339.2, 377.9, "forehand"), (377.9, 419.8, "backhand"), (419.8, 460.4, "forehand_volley"),
            (460.4, 501.0, "backhand_volley"), (501.0, 536.3, "smash_in"),
        ],
        "pause_threshold": None,
    },
}

# this function runs a peak-finding algorithm on the accelaration magnitude
# to identify every single physical strike. It filters out noise by ensuring
# a peal must have an acceleration value of at least 12 and must be separated
# from any other peak by at least 100 data points. It then grabs the exact 
# timestamps when these shots occurred and stores them in peak_times.
def auto_detect_blocks(df_sync, base_sequence, pause_threshold):
    """
    Detects block boundaries by looking for pauses (time gaps between
    consecutive strokes) longer than pause_threshold seconds. Labels are
    assigned by cycling through the declared sequence, whatever the
    number of detected blocks.
    """
    peaks, _ = find_peaks(df_sync["acc_mag"], height=12.0, distance=100)
    peak_times = df_sync["seconds_elapsed"].iloc[peaks].values
    # if the recording contains fewer than 2 shots, calculating a gap between
    # shots is impossible
    if len(peak_times) < 2:
        return [], peaks
    gap = np.diff(peak_times) # computing the elapsed time between every consecutive shot
    pause_indices = np.where(gap > pause_threshold)[0] # checks where these time gaps are larger than our pause_threshold

    # For every major pause detected, it calculates the mathematical midpoint between the last shot
    # of the old block and the first shot of the new block. This midpoint acts as the clean 
    # cutting line between drills.
    time_boundaries = [0.0]
    for idx in pause_indices:
        midpoint = (peak_times[idx] + peak_times[idx + 1]) / 2
        time_boundaries.append(midpoint)
    # Here, it appends the last timestamp of the recording to close off the final block
    time_boundaries.append(df_sync["seconds_elapsed"].iloc[-1])

    # counting how many block were created, by taking our predefined order of shots and
    # cycling through it indefinitely
    n_blocks = len(time_boundaries) - 1
    cyclic_sequence = list(itertools.islice(itertools.cycle(base_sequence), n_blocks))

    # packaging into a list of tuples and returning both segmentend blocks and the raw peak indices
    blocks = [(time_boundaries[i], time_boundaries[i + 1], cyclic_sequence[i])
              for i in range(n_blocks)]
    return blocks, peaks

# Helper function that takes the exact timestamp of asingle detected shot and figure out 
# which labeled drilling block it belongs to
def assign_block_from_time(time_seconds, blocks):
    for i, (start, end, label) in enumerate(blocks):
        if start <= time_seconds < end:
            return label, i
    return None, None

# This function handles the entire cycle of a single recording session: unzipping the 
# data, syncing different sensors, finding the shots, cutting them into windows and
# tagging them with metadata
def extract_session(zip_filename, config):
    # File extraction
    zip_path = os.path.join(PROJECT_FOLDER, zip_filename)
    if not os.path.exists(zip_path):
        print(f"WARNING: '{zip_filename}' not found, skipping it.")
        return None

    session_folder = os.path.join(extraction_folder, zip_filename.replace(".zip", ""))
    os.makedirs(session_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(session_folder)

    acc_file, gyro_file = None, None
    for root, dirs, files in os.walk(session_folder):
        for file in files:
            if file == "Accelerometer.csv":
                acc_file = os.path.join(root, file)
            elif file == "Gyroscope.csv":
                gyro_file = os.path.join(root, file)

    acc_df = pd.read_csv(acc_file).sort_values("time").reset_index(drop=True)
    gyro_df = pd.read_csv(gyro_file).sort_values("time").reset_index(drop=True)
    # Joining the data: since accelerometer and gyroscope sample data at slightly different intervals, it uses
    # an asymmetric merge to align the gyroscope data to the closest matching accelerometer timestamp.
    df_sync = pd.merge_asof(acc_df, gyro_df, on="time", direction="nearest", suffixes=("_acc", "_gyro"))
    # Calculating acceleration magnitude
    df_sync["acc_mag"] = np.sqrt(df_sync["x_acc"] ** 2 + df_sync["y_acc"] ** 2 + df_sync["z_acc"] ** 2)
    df_sync["seconds_elapsed"] = df_sync["seconds_elapsed_acc"]

    # if we have the exact time blocks in our config dictionary, we use those boundaries
    # and simply calls find_peaks to locate all the shots. If blocks is set to None. automatically
    # triggers the auto_detect_blocks function to divide the timeline based on quiet pauses, printing
    # out a summary table
    explicit_blocks = config.get("blocks")
    if explicit_blocks is not None:
        blocks = explicit_blocks
        peaks, _ = find_peaks(df_sync["acc_mag"], height=12.0, distance=100)
        print(f"\n{zip_filename}")
    else:
        threshold = config.get("pause_threshold", 5.0)
        blocks, peaks = auto_detect_blocks(df_sync, BASE_SEQUENCE, threshold)
        print(f"\n{zip_filename}  [automatically detected boundaries, threshold={threshold}s]")
        print(f"  Blocks detected: {len(blocks)}")
        for i, (t_start, t_end, lab) in enumerate(blocks):
            print(f"    block {i+1}: {lab:16s}  [{t_start:.1f}s - {t_end:.1f}s]  duration={t_end-t_start:.1f}s")

    person = config["person"]
    session_name = zip_filename.replace(".zip", "")
    session_dataset = []
    discarded = 0
    # then, we iterate throigh every single shot (peak) we found: we check the peak's timestamp
    # against the active blocks using assign_block_from_time. If it occurred during a break or 
    # outside a block, it's skipped. For a valid shot, it extracts a 150 sample window around
    # the impact point, 50 data point before the peak and 100 data point after it. It stamps the 
    # window with metadata: person (who hit it), session (file it came from), block_id (exercise 
    # block it bleongs to) and stroke_id (a globally unique name for that exact shot)
    for idx, p in enumerate(peaks):
        peak_time = df_sync["seconds_elapsed"].iloc[p]
        label, block_idx = assign_block_from_time(peak_time, blocks)
        if label is None:
            discarded += 1
            continue
        start = max(0, p - 50)
        end = min(len(df_sync), p + 100)
        window = df_sync.iloc[start:end].copy()
        window["label"] = label
        window["person"] = person
        window["session"] = session_name
        window["block_id"] = f"{session_name}_block{block_idx}"
        window["stroke_id"] = f"{label}_{session_name}_{idx+1}"
        session_dataset.append(window)

    print(f"  Strokes detected: {len(peaks)}  |  Discarded: {discarded}  |  Labeled: {len(session_dataset)}")

    # This final aggregations consists in concatenating all individual 150 sample into one single data frame.
    #Before returning the data frame, we print the class distribution
    if not session_dataset:
        return None
    df_out = pd.concat(session_dataset, ignore_index=True)
    print("  Class distribution:")
    counts = df_out.groupby("stroke_id")["label"].first().value_counts()
    for lab, cnt in counts.items():
        print(f"    {lab:16s}: {cnt}")
    return df_out


# Extraction of all configured sessions
# This final block is the exporter of this script: it loop extracts all the individual session,
# combines them into a single data frame, saved it as a CSV file and print a final data summary 
print("-" * 70)
print("Extraction of mixed sessions (multi-person)")
print("-" * 70)


all_sessions = [] # empty list to store the individual DataFrames
# iterating through our Sessions dictionary, passing each ZIP filename and its configuration
# settings to the extract_session function and appending the resulting processed DF to the list
for zip_filename, config in SESSIONS.items():
    session_df = extract_session(zip_filename, config)
    if session_df is not None:
        all_sessions.append(session_df)

# If the list is empty (no files) the script stops
if not all_sessions:
    raise RuntimeError("No session extracted. Check the zip paths.")
# if data exists, take all the separate session dataframes and stacks them vertically into one large
# df called mixed_session_df
mixed_sessions_df = pd.concat(all_sessions, ignore_index=True)

# this define the exact order and names of the columns we want to keep. The list comprehension 
# present_columns ensures it only tries to filter columns that actually exists in the data,
# preventing potential crashes. 
final_columns = ["time", "x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro",
                  "acc_mag", "label", "person", "session", "block_id", "stroke_id"]
present_columns = [c for c in final_columns if c in mixed_sessions_df.columns]
mixed_sessions_df = mixed_sessions_df[present_columns]
# CSV saving
output_path = os.path.join(PROJECT_FOLDER, "padel_dataset_2026_mixed_sessions.csv")
mixed_sessions_df.to_csv(output_path, index=False)

# Summary
print("\n" + "=" * 150)
print(f"Saved: {output_path}")
print(f"Total sessions: {mixed_sessions_df['session'].nunique()}")
print(f"Total people: {mixed_sessions_df['person'].nunique()} "
      f"({sorted(mixed_sessions_df['person'].unique())})")
print(f"Total strokes: {mixed_sessions_df['stroke_id'].nunique()}")
print("\nPerson x class distribution:")
print(pd.crosstab(
    mixed_sessions_df.groupby("stroke_id")["person"].first(),
    mixed_sessions_df.groupby("stroke_id")["label"].first(),
))

# Now we can run the next script (2) in order to remove outlier on  padel_dataset_2026_mixed_sessions.csv