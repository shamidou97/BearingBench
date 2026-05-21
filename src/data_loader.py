"""
BearingBench — Data Loader
XJTU-SY Rolling Element Bearing Dataset

Loads CSV vibration signals, applies sliding window,
and prepares inputs for CNN, LSTM, and Transformer models.

Signals per CSV:
    Channel 0: Horizontal_vibration_signals
    Channel 1: Vertical_vibration_signals

Sampling:
    Frequency : 25,600 Hz (fixed across all conditions)
    Duration  : 1.28 seconds per CSV
    Samples   : 32,768 per CSV

Window:
    size    = 1024 points = 40ms @ 25.6kHz (~1.5 shaft revolutions)
    stride  = 512  points (50% overlap)
    windows = floor((32768 - 1024) / 512) + 1 = 63 per CSV

Bearings used (14 of 15 — 1 skipped due to ambiguous combined fault):
    WC1 (2100 RPM / 12kN): Bearing1_1, 1_2, 1_3, 1_4           [4 bearings]
    WC2 (2250 RPM / 11kN): Bearing2_1, 2_2, 2_3, 2_4, 2_5      [5 bearings]
    WC3 (2400 RPM / 10kN): Bearing3_1, 3_2, 3_3, 3_4, 3_5      [5 bearings]

Skipped (only 52 files, combined fault):
    Bearing1_5 — Inner Race + Outer Race (combined, ambiguous)

Kept with dominant label:
    Bearing3_2 — labeled as Outer Race (dominant fault, 2496 files)

Fault labels (4 classes):
    0 — Normal     : first 80% of CSV files per bearing (all bearings)
    1 — Inner Race : Bearing2_1, Bearing3_3, Bearing3_4
    2 — Outer Race : Bearing1_1, 1_2, 1_3, Bearing2_2, 2_4, 2_5,
                     Bearing3_1, 3_2, 3_5
    3 — Cage       : Bearing1_4, Bearing2_3

Dataset stats (Strategy 5 — cap Normal + keep all fault windows):
    CSV files used      : 9,164  (9,216 minus 52 from Bearing1_5)
    MAX_NORMAL_WINDOWS  : 10 per CSV  → Normal ~73,310 windows
    MAX_FAULT_WINDOWS   : 63 per CSV  → keep all fault windows
    Approx distribution :
        Normal     : ~73,310  (42%)
        Inner Race : ~29,925  (17%)
        Outer Race : ~77,238  (44%)
        Cage       :  ~8,253   (5%)
        Total      : ~188,726 samples

Run: python src/data_loader.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
CACHE_DIR   = os.path.join(DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

SEQ_CACHE   = os.path.join(CACHE_DIR, 'sequences.pkl')
IMG_CACHE   = os.path.join(CACHE_DIR, 'images.pkl')

# ── Config ────────────────────────────────────────────────────
WINDOW_SIZE = 256 
STRIDE      = 128 
IMG_H = 16
IMG_W = 16
SEED                 = 42

# ── Class imbalance strategy 5 ───────────────────────────────
# Cap Normal windows per CSV to reduce 80/20 dominance
# Keep all fault windows to preserve rare fault information
MAX_NORMAL_WINDOWS   = 10    # per CSV → Normal ~73,310 total
MAX_FAULT_WINDOWS    = 63    # per CSV → keep all fault windows

# Recommended class weights for trainer
# (higher = model penalizes misclassification more)
CLASS_WEIGHTS = {
    0: 1.0,    # Normal      — most common after capping
    1: 4.0,    # Inner Race  — moderately rare
    2: 2.0,    # Outer Race  — moderate
    3: 15.0,   # Cage        — rarest class
}

# ── Fault label map ───────────────────────────────────────────
# Bearing1_5 skipped — combined Inner+Outer fault (only 52 files)
# Bearing3_2 kept — labeled as Outer Race (dominant fault, 2496 files)
SKIP_BEARINGS = ['Bearing1_5']

FAULT_MAP = {
    # WC1
    'Bearing1_1': 2,   # Outer Race
    'Bearing1_2': 2,   # Outer Race
    'Bearing1_3': 2,   # Outer Race
    'Bearing1_4': 3,   # Cage
    # Bearing1_5 skipped — combined Inner+Outer fault

    # WC2
    'Bearing2_1': 1,   # Inner Race
    'Bearing2_2': 2,   # Outer Race
    'Bearing2_3': 3,   # Cage
    'Bearing2_4': 2,   # Outer Race
    'Bearing2_5': 2,   # Outer Race

    # WC3
    'Bearing3_1': 2,   # Outer Race
    'Bearing3_2': 2,   # Outer Race (dominant — Inner+Ball+Cage+Outer combined)
    'Bearing3_3': 1,   # Inner Race
    'Bearing3_4': 1,   # Inner Race
    'Bearing3_5': 2,   # Outer Race
}

FAULT_NAMES = {
    0: 'Normal',
    1: 'Inner Race',
    2: 'Outer Race',
    3: 'Cage'
}

BEARINGS = {
    'WC1': ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4'],
             # Bearing1_5 skipped — combined Inner+Outer fault
    'WC2': ['Bearing2_1','Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5'],
    'WC3': ['Bearing3_1','Bearing3_2','Bearing3_3','Bearing3_4','Bearing3_5'],
             # Bearing3_2 kept — labeled as Outer Race (dominant fault)
}

# ── Normalize ─────────────────────────────────────────────────
def normalize(signal):
    """Z-score normalization per window."""
    sig  = np.array(signal, dtype=np.float32)
    mean = sig.mean()
    std  = sig.std()
    if std > 1e-8:
        return (sig - mean) / std
    return sig - mean

# ── Sliding window ────────────────────────────────────────────
def sliding_windows(horiz, vert, label,
                    window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Apply sliding window to one CSV.
    Returns list of (window_array, label).
    window_array shape: (window_size, 2)

    windows = floor((32768 - 1024) / 512) + 1 = 63 per CSV
    """
    total   = len(horiz)
    windows = []
    start   = 0
    while start + window_size <= total:
        h_win = normalize(horiz[start:start + window_size])
        v_win = normalize(vert[start:start + window_size])
        win   = np.stack([h_win, v_win], axis=-1)  # (1024, 2)
        windows.append((win, label))
        start += stride
    return windows

# ── Load one CSV ──────────────────────────────────────────────
def load_csv(path):
    df    = pd.read_csv(path)
    horiz = df['Horizontal_vibration_signals'].values
    vert  = df['Vertical_vibration_signals'].values
    return horiz, vert

# ── Get sorted CSV files ──────────────────────────────────────
def get_csv_files(bearing_path):
    files = [f for f in os.listdir(bearing_path) if f.endswith('.csv')]
    files.sort(key=lambda x: int(x.replace('.csv', '')))
    return files

# ── Build full dataset ────────────────────────────────────────
def build_dataset(max_windows_per_csv=None):
    """
    Build X_seq (N,1024,2), X_img (N,32,32,2), y (N,), meta.

    Args:
        max_windows_per_csv: limit windows per CSV to reduce dataset size
                             None = use all 63 windows per CSV
    """
    X_seq, X_img, y, meta = [], [], [], []
    total_bearings = sum(len(v) for v in BEARINGS.values())
    processed = 0

    for wc, bearing_list in BEARINGS.items():
        for bearing in bearing_list:
            bearing_path = os.path.join(DATA_DIR, wc, bearing)
            if not os.path.exists(bearing_path):
                print(f'  WARNING: {bearing_path} not found — skipping')
                continue

            files       = get_csv_files(bearing_path)
            n_files     = len(files)
            fault_label = FAULT_MAP[bearing]
            cutoff      = int(n_files * 0.8)

            normal_count = 0
            fault_count  = 0

            # Boundary zone: files between 75-85% of lifetime
            # These show early degradation but are labeled Normal
            # → excluded to remove ambiguous label-signal pairs
            boundary_start = int(n_files * 0.75)
            boundary_end   = int(n_files * 0.85)

            for i, fname in enumerate(files):
                # Skip ambiguous boundary files
                if boundary_start <= i < boundary_end:
                    continue
                label = 0 if i < cutoff else fault_label
                path  = os.path.join(bearing_path, fname)

                try:
                    horiz, vert = load_csv(path)
                except Exception as e:
                    print(f'  ERROR: {path}: {e}')
                    continue

                windows = sliding_windows(horiz, vert, label)

                # ── Apply class-specific window caps ──────────
                # Normal  : cap at MAX_NORMAL_WINDOWS to reduce dominance
                # Fault   : keep all windows to preserve rare fault info
                if label == 0:
                    cap = max_windows_per_csv if max_windows_per_csv                           else MAX_NORMAL_WINDOWS
                else:
                    cap = MAX_FAULT_WINDOWS

                if len(windows) > cap:
                    idx     = np.random.choice(
                        len(windows), cap, replace=False)
                    windows = [windows[j] for j in sorted(idx)]

                for win, lbl in windows:
                    X_seq.append(win)
                    X_img.append(win.reshape(IMG_H, IMG_W, 2))
                    y.append(lbl)
                    meta.append({
                        'wc'     : wc,
                        'bearing': bearing,
                        'file'   : fname,
                        'label'  : lbl,
                        'fault'  : FAULT_NAMES[lbl],
                    })

                if label == 0:
                    normal_count += len(windows)
                else:
                    fault_count  += len(windows)

            processed += 1
            print(f'  [{processed:02d}/{total_bearings}] '
                  f'{wc}/{bearing}: {n_files} files | '
                  f'Normal={normal_count:,} | '
                  f'{FAULT_NAMES[fault_label]}={fault_count:,}')

    X_seq = np.array(X_seq, dtype=np.float32)
    X_img = np.array(X_img, dtype=np.float32)
    y     = np.array(y,     dtype=np.int32)

    print(f'\n  Dataset  : X_seq={X_seq.shape}  X_img={X_img.shape}')
    print(f'  Labels   : Normal={np.sum(y==0):,}  '
          f'Inner={np.sum(y==1):,}  '
          f'Outer={np.sum(y==2):,}  '
          f'Cage={np.sum(y==3):,}')

    return X_seq, X_img, y, meta

# ── Load with cache ───────────────────────────────────────────
def load_dataset(force_rebuild=False, max_windows_per_csv=None):
    """
    Load dataset with caching.
    First run builds from CSVs, subsequent runs load from cache.
    """
    if (os.path.exists(SEQ_CACHE) and
        os.path.exists(IMG_CACHE) and
        not force_rebuild):
        print(f'  Loading sequence cache ...')
        with open(SEQ_CACHE, 'rb') as f:
            X_seq, y, meta = pickle.load(f)
        print(f'  Loading image cache    ...')
        with open(IMG_CACHE, 'rb') as f:
            X_img = pickle.load(f)
        print(f'  Loaded {len(X_seq):,} samples  shape={X_seq.shape}')
        return X_seq, X_img, y, meta

    print('  Building from CSV files...')
    X_seq, X_img, y, meta = build_dataset(max_windows_per_csv)

    with open(SEQ_CACHE, 'wb') as f:
        pickle.dump((X_seq, y, meta), f)
    with open(IMG_CACHE, 'wb') as f:
        pickle.dump(X_img, f)
    print(f'  Cached to: {CACHE_DIR}')

    return X_seq, X_img, y, meta

# ── Train / Val / Test split ──────────────────────────────────
def get_splits(X_seq, X_img, y,
               val_size=0.15, test_size=0.15, seed=SEED):
    """
    Stratified split returning both sequence and image formats.
    Default: 70% train / 15% val / 15% test
    """
    idx = np.arange(len(y))

    idx_tmp, idx_test = train_test_split(
        idx, test_size=test_size,
        random_state=seed, stratify=y)

    idx_train, idx_val = train_test_split(
        idx_tmp,
        test_size=val_size / (1 - test_size),
        random_state=seed, stratify=y[idx_tmp])

    return {
        'seq': {
            'train': (X_seq[idx_train], y[idx_train]),
            'val'  : (X_seq[idx_val],   y[idx_val]),
            'test' : (X_seq[idx_test],  y[idx_test]),
        },
        'img': {
            'train': (X_img[idx_train], y[idx_train]),
            'val'  : (X_img[idx_val],   y[idx_val]),
            'test' : (X_img[idx_test],  y[idx_test]),
        }
    }

# ── Summary ───────────────────────────────────────────────────
def print_stats(y, splits):
    print('\n── Class distribution ───────────────────────────')
    total = len(y)
    for lbl, name in FAULT_NAMES.items():
        count = np.sum(y == lbl)
        pct   = count / total * 100
        bar   = 'X' * int(pct / 2)
        print(f'  {lbl} {name:<12}: {count:>8,} ({pct:5.1f}%)  {bar}')
    print(f'  {"Total":<14}: {total:>8,}')

    print('\n── Split sizes ──────────────────────────────────')
    for split, (X, ys) in splits['seq'].items():
        s0 = np.sum(ys==0); s1 = np.sum(ys==1)
        s2 = np.sum(ys==2); s3 = np.sum(ys==3)
        print(f'  {split:<6}: {len(X):>8,} | '
              f'Normal={s0:,} Inner={s1:,} '
              f'Outer={s2:,} Cage={s3:,}')

    print('\n── Input shapes ─────────────────────────────────')
    X_seq_tr, _ = splits['seq']['train']
    X_img_tr, _ = splits['img']['train']
    print(f'  LSTM/Transformer : {X_seq_tr.shape}   '
          f'(samples, timesteps, channels)')
    print(f'  CNN              : {X_img_tr.shape}  '
          f'(samples, H, W, channels)')

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nBearingBench — Data Loader')
    print('=' * 55)

    # Window size info
    n_windows = (32768 - WINDOW_SIZE) // STRIDE + 1
    print(f'\n  Sampling freq : 25,600 Hz')
    print(f'  Window size   : {WINDOW_SIZE} points = 40ms')
    print(f'  Stride        : {STRIDE} points (50% overlap)')
    print(f'  Windows/CSV   : {n_windows}')
    # 9216 total - 52 (Bearing1_5 skipped) = 9164 used
    used_csvs = 9216 - 52
    print(f'  Total CSVs    : 9,216 (9,164 used — Bearing1_5 skipped)')
    print(f'  Bearing3_2    : kept as Outer Race (dominant fault)')
    print(f'  Max windows   : {used_csvs:,} x {n_windows} = '
          f'{used_csvs * n_windows:,} (before capping)')
    print(f'')
    print(f'  Class imbalance strategy:')
    print(f'  MAX_NORMAL_WINDOWS : {MAX_NORMAL_WINDOWS} per CSV')
    print(f'  MAX_FAULT_WINDOWS  : {MAX_FAULT_WINDOWS} per CSV (all)')
    print(f'  CLASS_WEIGHTS      : {CLASS_WEIGHTS}')
    print(f'  Est. total after cap: ~188,726 samples')

    print('\n  Fault label map:')
    for lbl, name in FAULT_NAMES.items():
        if lbl == 0:
            print(f'  {lbl} {name:<12}: all bearings (first 80% of files)')
        else:
            blist = [b for b, l in FAULT_MAP.items() if l == lbl]
            print(f'  {lbl} {name:<12}: {", ".join(blist)}')

    print('\nLoading dataset...')
    print('  (using max_windows_per_csv=5 for quick test)')
    print('  (set to None for full 580,608 sample dataset)')

    X_seq, X_img, y, meta = load_dataset(max_windows_per_csv=5)

    print('\nSplitting...')
    splits = get_splits(X_seq, X_img, y)
    print_stats(y, splits)

    # Preview one sample
    X_tr, y_tr = splits['seq']['train']
    print(f'\n  Sample window shape : {X_tr[0].shape}')
    print(f'  First 3 timesteps:')
    print(f'  {"t":<8} {"Horizontal":>12} {"Vertical":>12}')
    print(f'  {"-"*34}')
    for i in range(3):
        print(f'  {i+1:<8} {X_tr[0][i][0]:>12.4f} '
              f'{X_tr[0][i][1]:>12.4f}')

    print('\nData loader ready!')
    print('  load_dataset()  -> X_seq, X_img, y, meta')
    print('  get_splits()    -> train/val/test for seq + img\n')
