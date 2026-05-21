import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
BearingBench — Flask Backend v1.0
Serves real XJTU-SY bearing data from MySQL
+ live fault prediction from uploaded CSV files

Endpoints:
    GET  /                        → dashboard
    GET  /api/summary             → WC conditions + stats
    GET  /api/bearings            → all 15 bearings metadata
    GET  /api/bearing/<name>      → per-file lifetime data
    GET  /api/distribution        → fault class file counts
    GET  /api/models              → benchmark results from MySQL
    GET  /api/available_models    → which models are loaded
    POST /api/predict             → upload CSV → fault prediction

Run  : python src/app.py
Opens: http://127.0.0.1:5000
"""

import sys
import numpy as np
import pandas as pd
import tempfile

from flask import Flask, jsonify, request, render_template

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TMPL_DIR   = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

sys.path.insert(0, SRC_DIR)
from db import get_conn
from data_loader import FAULT_NAMES

app = Flask(__name__,
            template_folder=TMPL_DIR,
            static_folder=STATIC_DIR)

# ── Model paths ───────────────────────────────────────────────
MODEL_PATHS = {
    'CNN'        : os.path.join(MODELS_DIR, 'cnn_best.h5'),
    'LSTM'       : os.path.join(MODELS_DIR, 'lstm_best.h5'),
    'BiLSTM'     : os.path.join(MODELS_DIR, 'bilstm_best.h5'),
    'Transformer': os.path.join(MODELS_DIR, 'transformer_best.h5'),
}

# Window config — must match what models were trained with
WINDOW_SIZE = 1024   # CNN and LSTM used 1024
IMG_H       = 32     # CNN image grid (32x32 = 1024)
IMG_W       = 32

# ── Load all models at startup ────────────────────────────────
print('\nBearingBench — Loading models...')
MODELS = {}

try:
    import tensorflow as tf

    # Suppress GPU memory allocation issues
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    # Load CNN, LSTM, BiLSTM via standard Keras loader
    for name in ['CNN', 'LSTM', 'BiLSTM']:
        path = MODEL_PATHS[name]
        if os.path.exists(path):
            try:
                m = tf.keras.models.load_model(path)
                MODELS[name] = m
                print(f'  {name:<14}: loaded  '
                      f'({m.count_params():,} params)')
            except Exception as e:
                print(f'  {name:<14}: failed — {e}')
        else:
            print(f'  {name:<14}: not found at {path}')

    # Load Transformer with custom PositionalEncoding layer
    from trainer import build_transformer, PositionalEncoding
    path = MODEL_PATHS['Transformer']
    if os.path.exists(path):
        try:
            m = build_transformer()
            m.load_weights(path)
            MODELS['Transformer'] = m
            print(f'  {"Transformer":<14}: loaded  '
                  f'({m.count_params():,} params)')
        except Exception as e:
            print(f'  {"Transformer":<14}: failed — {e}')
    else:
        print(f'  {"Transformer":<14}: not found at {path}')

except ImportError as e:
    print(f'  TensorFlow not available: {e}')
except Exception as e:
    print(f'  Model loading error: {e}')

print(f'  Models ready: {list(MODELS.keys())}')

# ── Signal preprocessing ──────────────────────────────────────
def normalize_window(signal):
    """Z-score normalization per window."""
    sig  = np.array(signal, dtype=np.float32)
    mean = sig.mean()
    std  = sig.std()
    if std > 1e-8:
        return (sig - mean) / std
    return sig - mean

def csv_to_windows(filepath, stride=512):
    """
    Read bearing CSV and extract sliding windows.
    Returns array of shape (N_windows, WINDOW_SIZE, 2)
    """
    df    = pd.read_csv(filepath)
    horiz = df['Horizontal_vibration_signals'].values
    vert  = df['Vertical_vibration_signals'].values
    total = len(horiz)

    windows = []
    start   = 0
    while start + WINDOW_SIZE <= total:
        h = normalize_window(horiz[start:start + WINDOW_SIZE])
        v = normalize_window(vert[start:start + WINDOW_SIZE])
        windows.append(np.stack([h, v], axis=-1))  # (1024, 2)
        start += stride

    return np.array(windows, dtype=np.float32)

def prepare_input(window, model_name, model):
    """
    Prepare window for specific model input shape.
    window shape: (WINDOW_SIZE, 2) = (1024, 2)
    """
    if model_name == 'CNN':
        # Reshape 1024 points → (32, 32, 2) image
        return window.reshape(1, IMG_H, IMG_W, 2)
    else:
        # LSTM / BiLSTM / Transformer — use sequence
        # Get expected sequence length from model
        seq_len = model.input_shape[1]  # e.g. 256 or 1024

        if window.shape[0] >= seq_len:
            seq = window[:seq_len]
        else:
            # Pad with zeros if window shorter than expected
            pad = np.zeros((seq_len - window.shape[0], 2),
                           dtype=np.float32)
            seq = np.vstack([window, pad])

        return seq[np.newaxis, :, :]  # (1, seq_len, 2)

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('dashboard.html')

# ── Working conditions summary ────────────────────────────────
@app.route('/api/summary')
def summary():
    try:
        conn   = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                wc.name              AS wc,
                wc.rpm,
                wc.load_kn,
                wc.severity_index    AS si,
                wc.description,
                COUNT(b.id)          AS bearings,
                SUM(b.total_files)   AS total_files,
                SUM(b.is_skipped)    AS skipped
            FROM working_conditions wc
            JOIN bearings b ON b.wc_id = wc.id
            GROUP BY wc.id
            ORDER BY wc.name
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── All bearings metadata ─────────────────────────────────────
@app.route('/api/bearings')
def bearings():
    try:
        conn   = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                b.name,
                wc.name          AS wc,
                b.total_files,
                b.lifetime_min,
                b.fault_type,
                b.fault_label,
                b.is_skipped,
                b.skip_reason,
                wc.rpm,
                wc.load_kn,
                wc.severity_index
            FROM bearings b
            JOIN working_conditions wc ON wc.id = b.wc_id
            ORDER BY wc.name, b.name
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Per-bearing file timeline ─────────────────────────────────
@app.route('/api/bearing/<name>')
def bearing_detail(name):
    try:
        conn   = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                f.file_number,
                f.lifetime_pct,
                f.fault_label,
                f.fault_state
            FROM files f
            JOIN bearings b ON b.id = f.bearing_id
            WHERE b.name = %s
            ORDER BY f.file_number
        """, (name,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Fault class distribution ──────────────────────────────────
@app.route('/api/distribution')
def distribution():
    try:
        conn   = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT fault_state,
                   COUNT(*) AS count
            FROM files
            WHERE fault_state != 'Inner Race + Outer Race'
            GROUP BY fault_state
            ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Model benchmark results ───────────────────────────────────
@app.route('/api/models')
def model_results():
    try:
        conn   = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                model_name,
                input_shape,
                params,
                ROUND(accuracy * 100, 1)       AS accuracy,
                ROUND(macro_f1, 3)             AS macro_f1,
                ROUND(recall_normal, 3)        AS recall_normal,
                ROUND(recall_inner,  3)        AS recall_inner,
                ROUND(recall_outer,  3)        AS recall_outer,
                ROUND(recall_cage,   3)        AS recall_cage,
                ROUND(auc_normal, 3)           AS auc_normal,
                ROUND(auc_inner,  3)           AS auc_inner,
                ROUND(auc_outer,  3)           AS auc_outer,
                ROUND(auc_cage,   3)           AS auc_cage,
                ROUND(training_time_sec/3600,2) AS training_hrs,
                epochs_trained
            FROM model_results
            ORDER BY accuracy DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Available loaded models ───────────────────────────────────
@app.route('/api/available_models')
def available_models():
    return jsonify(list(MODELS.keys()))

# ── Live fault prediction ─────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Upload a bearing CSV file → return fault prediction.

    Form data:
        file   : CSV file (Horizontal + Vertical vibration)
        model  : 'CNN' | 'LSTM' | 'BiLSTM' | 'Transformer'
        window : integer window index (0-based)
    """
    if not MODELS:
        return jsonify({
            'error': 'No models loaded. Train models first.'
        }), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file in request'}), 400

    file       = request.files['file']
    model_name = request.form.get('model', 'LSTM')
    win_idx    = int(request.form.get('window', 0))

    if model_name not in MODELS:
        return jsonify({
            'error': f'{model_name} not loaded. '
                     f'Available: {list(MODELS.keys())}'
        }), 400

    # Save uploaded file temporarily
    tmp_path = None
    with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        windows = csv_to_windows(tmp_path)
        os.unlink(tmp_path)
        tmp_path = None

        if len(windows) == 0:
            return jsonify({
                'error': 'No windows extracted from CSV. '
                         'Check file format.'
            }), 400

        # Clamp window index to valid range
        win_idx = max(0, min(win_idx, len(windows) - 1))
        window  = windows[win_idx]

        # Prepare input for specific model
        model = MODELS[model_name]
        X     = prepare_input(window, model_name, model)

        # Run inference
        probs = model.predict(X, verbose=0)[0]
        pred  = int(np.argmax(probs))

        return jsonify({
            'prediction'    : pred,
            'fault_name'    : FAULT_NAMES[pred],
            'confidence'    : [round(float(p), 4) for p in probs],
            'model'         : model_name,
            'window_index'  : win_idx,
            'total_windows' : len(windows),
            'fault_names'   : list(FAULT_NAMES.values()),
        })

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({'error': str(e)}), 500

# ── Health check ──────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status'       : 'ok',
        'models_loaded': list(MODELS.keys()),
        'db'           : 'bearingbench',
    })

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'\n  Dashboard : http://127.0.0.1:5000')
    print(f'  API       : http://127.0.0.1:5000/api/health')
    print(f'  Models    : {list(MODELS.keys())}\n')
    app.run(debug=True, port=5000, use_reloader=False)
