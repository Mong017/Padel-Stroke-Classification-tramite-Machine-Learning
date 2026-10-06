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

# BLOCK DISCOVERY TOOL
#
# We run this script, on any new zip file, to get the "blocks" list
# for the SESSIONS dictionary # in the next script. 
# This is what generated every "blocks" list currently hardcoded
# in SESSIONS 
# The logic, in three steps:
#   1. Every possible pause threshold is tried (the midpoint between
#      each pair of consecutive, sorted, inter-stroke time gaps is a
#      candidate cut point).
#   2. Only thresholds whose resulting number of blocks is a whole
#      multiple of len(BASE_SEQUENCE) are kept. This is not arbitrary:
#      the recording protocol always consists of complete laps of the
#      7-stroke base sequence, so a valid segmentation MUST produce a
#      block count that is a multiple of 7.
#   3. Among the valid candidates, the threshold that produces the most
#      UNIFORM block sizes is selected (lowest coefficient of variation
#      of strokes-per-block), since every drill block should contain
#      roughly the same number of repetitions.

def discover_session_blocks(zip_filename, base_sequence=BASE_SEQUENCE):
    """
    Unzips the given file (if not already extracted), detects strokes,
    searches for the best pause threshold, and returns the resulting
    (start_time, end_time, label) list, together with the threshold
    and the block-size coefficient of variation used to select it.
    """
    zip_path = os.path.join(PROJECT_FOLDER, zip_filename)
    if not os.path.exists(zip_path):
        print(f"WARNING: '{zip_filename}' not found, skipping discovery.")
        return None, None, None

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
    df_sync = pd.merge_asof(acc_df, gyro_df, on="time", direction="nearest", suffixes=("_acc", "_gyro"))
    df_sync["acc_mag"] = np.sqrt(df_sync["x_acc"] ** 2 + df_sync["y_acc"] ** 2 + df_sync["z_acc"] ** 2)
    df_sync["seconds_elapsed"] = df_sync["seconds_elapsed_acc"]

    peaks, _ = find_peaks(df_sync["acc_mag"], height=12.0, distance=100)
    peak_times = df_sync["seconds_elapsed"].iloc[peaks].values
    gaps = np.diff(peak_times)

    # every midpoint between two consecutive sorted gap values is a
    # candidate pause threshold worth testing
    sorted_gaps = np.sort(np.unique(gaps))
    candidates = [(sorted_gaps[i] + sorted_gaps[i + 1]) / 2
                  for i in range(len(sorted_gaps) - 1)]

    n_types = len(base_sequence)
    best = None  # will hold (coefficient_of_variation, threshold)
    for thr in candidates:
        pause_idx = np.where(gaps > thr)[0]
        boundaries_idx = [0] + list(pause_idx + 1) + [len(peak_times)]
        block_sizes = np.diff(boundaries_idx)
        n_blocks = len(block_sizes)
        # reject thresholds that do not produce a whole number of laps
        # of the base sequence
        if n_blocks < n_types or n_blocks % n_types != 0:
            continue
        cv = np.std(block_sizes) / np.mean(block_sizes)
        if best is None or cv < best[0]:
            best = (cv, thr)

    if best is None:
        print(f"  Could not find a threshold producing a block count "
              f"that is a multiple of {n_types} for '{zip_filename}'.")
        return None, None, None

    cv, threshold = best
    pause_idx = np.where(gaps > threshold)[0]
    time_boundaries = [0.0]
    for idx in pause_idx:
        midpoint = (peak_times[idx] + peak_times[idx + 1]) / 2
        time_boundaries.append(round(midpoint, 1))
    time_boundaries.append(round(df_sync["seconds_elapsed"].iloc[-1], 1))

    n_blocks = len(time_boundaries) - 1
    cyclic_sequence = list(itertools.islice(itertools.cycle(base_sequence), n_blocks))
    blocks = [(time_boundaries[i], time_boundaries[i + 1], cyclic_sequence[i])
              for i in range(n_blocks)]
    return blocks, threshold, cv


def print_sessions_snippet(zip_filename, blocks, person="TODO"):
    """
    Prints the SESSIONS dictionary entry for this file, ready to copy
    and paste. 'person' defaults to a TODO placeholder since it cannot
    be inferred from the signal - fill it in by hand after pasting.
    """
    print(f'    "{zip_filename}": {{')
    print(f'        "person": "{person}",')
    print(f'        "blocks": [')
    for i in range(0, len(blocks), 3):
        chunk = blocks[i:i + 3]
        line = ", ".join(f'({s}, {e}, "{lab}")' for s, e, lab in chunk)
        print(f'            {line},')
    print(f'        ],')
    print(f'        "pause_threshold": None,')
    print(f'    }},')


SESSIONS_TO_DISCOVER = [
    "all_shots_-2026-07-08_08-42-51.zip",
    "all_shots_2-2026-07-08_10-32-40.zip",
    "Registration_1_edo-2026-07-08_18-16-34.zip",
    "Registration_2_edo-2026-07-08_18-37-08.zip",
    "Registration_3_edo-2026-07-08_19-04-50.zip",
    "all_shots_massi_3-2026-07-09_09-11-33.zip",
]

print("=" * 70)
print("BLOCK DISCOVERY - copy the printed entries below into SESSIONS")
print("=" * 70)
for zip_filename in SESSIONS_TO_DISCOVER:
    blocks, threshold, cv = discover_session_blocks(zip_filename)
    if blocks is None:
        continue
    print(f"\n# {zip_filename}  (auto-selected threshold={threshold:.2f}s, block-size CV={cv:.3f})")
    print_sessions_snippet(zip_filename, blocks)
print("\n" + "=" * 70)
print("END OF DISCOVERY OUTPUT")
print("=" * 70 + "\n")
