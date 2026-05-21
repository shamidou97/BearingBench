import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
BearingBench — Unified Trainer
Trains CNN, LSTM, BiLSTM, and Transformer on XJTU-SY bearing data.

Models:
    CNN         : (batch, 32, 32, 2) — 2D image input
    LSTM        : (batch, 1024, 2)   — sequential input
    BiLSTM      : (batch, 1024, 2)   — bidirectional sequential
    Transformer : (batch, 1024, 2)   — attention-based input

Fault classes:
    0 — Normal
    1 — Inner Race
    2 — Outer Race
    3 — Cage

Run  : python src/trainer.py 2>/dev/null
Saves: models/*.h5
       results/*_history.png
       results/*_confusion_matrix.png
       results/*_roc_curve.png
       results/benchmark_report.txt
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# ── Path setup ────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)
from db import get_conn
from data_loader import (load_dataset, get_splits,
                         CLASS_WEIGHTS, FAULT_NAMES)

# ── Config ────────────────────────────────────────────────────
N_CLASSES  = 4
BATCH_SIZE = 128 
EPOCHS     = 50
SEED       = 42

# CNN
IMG_H, IMG_W = 16, 16

# LSTM / Transformer
SEQ_LEN    = 256 
N_FEATURES = 2

# Transformer
D_MODEL    = 64
N_HEADS    = 4
D_FF       = 128
N_BLOCKS   = 3

# ── GPU ───────────────────────────────────────────────────────
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f'  GPU: {gpus[0].name}')
    else:
        print('  No GPU — using CPU')

# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════

# ── CNN ───────────────────────────────────────────────────────
def build_cnn():
    """
    Lightweight CNN for bearing fault detection.
    Input  : (batch, 32, 32, 2) — vibration signal as 2D image
    Output : (batch, 4)         — Normal/Inner/Outer/Cage
    """
    inp = Input(shape=(IMG_H, IMG_W, N_FEATURES), name='image_input')

    # Block 1
    x = layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # FCN head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return Model(inp, out, name='CNN_BearingBench')

# ── LSTM ──────────────────────────────────────────────────────
def build_lstm():
    """
    LSTM with BatchNormalization.
    Input  : (batch, 1024, 2)
    Output : (batch, 4)
    """
    inp = Input(shape=(SEQ_LEN, N_FEATURES), name='sequence_input')

    x = layers.LSTM(128, return_sequences=True, name='lstm1')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64, return_sequences=True, name='lstm2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(32, return_sequences=False, name='lstm3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return Model(inp, out, name='LSTM_BearingBench')

# ── BiLSTM ────────────────────────────────────────────────────
def build_bilstm():
    """
    Bidirectional LSTM — reads sequence forward AND backward.
    Input  : (batch, 1024, 2)
    Output : (batch, 4)
    """
    inp = Input(shape=(SEQ_LEN, N_FEATURES), name='sequence_input')

    x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True),
            name='bilstm1')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True),
            name='bilstm2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False),
            name='bilstm3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return Model(inp, out, name='BiLSTM_BearingBench')

# ── Positional Encoding ───────────────────────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        positions = np.arange(max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(
            10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_enc = tf.cast(
            angles[np.newaxis, :, :], tf.float32)

    def call(self, x):
        return x + self.pos_enc[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'max_len': self.max_len, 'd_model': self.d_model})
        return cfg

# ── Transformer encoder block ─────────────────────────────────
def transformer_block(x, d_model, n_heads, d_ff, dropout):
    attn = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        dropout=dropout)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x    = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ffn  = layers.Dense(d_ff, activation='relu')(x)
    ffn  = layers.Dropout(dropout)(ffn)
    ffn  = layers.Dense(d_model)(ffn)
    ffn  = layers.Dropout(dropout)(ffn)
    x    = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# ── Transformer ───────────────────────────────────────────────
def build_transformer():
    """
    Transformer encoder — attends to ALL 1024 timesteps simultaneously.
    Input  : (batch, 1024, 2)
    Output : (batch, 4)
    """
    inp = Input(shape=(SEQ_LEN, N_FEATURES), name='sequence_input')

    x = layers.Dense(D_MODEL, name='input_projection')(inp)
    x = PositionalEncoding(SEQ_LEN, D_MODEL, name='pos_encoding')(x)
    x = layers.Dropout(0.2)(x)

    for i in range(N_BLOCKS):
        x = transformer_block(x, D_MODEL, N_HEADS, D_FF, 0.2)

    x = layers.GlobalAveragePooling1D(name='gap')(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return Model(inp, out, name='Transformer_BearingBench')

# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_history(history, model_name):
    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    epochs     = range(1, len(train_acc) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BearingBench — {model_name} Training History\n'
                 f'XJTU-SY Bearing Fault Detection',
                 fontsize=13, fontweight='bold')
    style = dict(linewidth=2, marker='o', markersize=3)

    ax = axes[0, 0]
    ax.plot(epochs, train_acc, color='#1a7abf',
            label='Training accuracy', **style)
    ax.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    ax.annotate(f'Final: {train_acc[-1]:.3f}',
                xy=(len(epochs), train_acc[-1]),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=9, color='#1a7abf')

    ax = axes[0, 1]
    ax.plot(epochs, val_acc, color='#27ae60',
            label='Validation accuracy', linestyle='--', **style)
    best_ep  = int(np.argmax(val_acc)) + 1
    best_val = max(val_acc)
    ax.scatter([best_ep], [best_val], color='#27ae60', s=80, zorder=5)
    ax.annotate(f'Best: {best_val:.3f} (ep {best_ep})',
                xy=(best_ep, best_val),
                xytext=(8, -15), textcoords='offset points',
                fontsize=9, color='#27ae60')
    ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, train_loss, color='#c0392b',
            label='Training loss', **style)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, val_loss, color='#e67e22',
            label='Validation loss', linestyle='--', **style)
    best_loss_ep  = int(np.argmin(val_loss)) + 1
    best_loss_val = min(val_loss)
    ax.scatter([best_loss_ep], [best_loss_val],
               color='#e67e22', s=80, zorder=5)
    ax.annotate(f'Best: {best_loss_val:.3f} (ep {best_loss_ep})',
                xy=(best_loss_ep, best_loss_val),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='#e67e22')
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{name}_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ['Normal', 'Inner Race', 'Outer Race', 'Cage']
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'BearingBench — {model_name}\nConfusion Matrix')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black',
                    fontsize=12, fontweight='bold')
    plt.tight_layout()
    name = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{name}_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

def plot_roc(y_true, y_prob, model_name):
    colors = ['#1a7abf', '#27ae60', '#c0392b', '#9b59b6']
    labels = ['Normal', 'Inner Race', 'Outer Race', 'Cage']
    y_bin  = tf.keras.utils.to_categorical(y_true, N_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{labels[i]} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'BearingBench — {model_name} ROC Curves')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    name = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{name}_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')
    return aucs

# ══════════════════════════════════════════════════════════════
# MYSQL SAVE
# ══════════════════════════════════════════════════════════════

def save_to_mysql(model_name, model, report_dict,
                  aucs, training_time, epochs_trained,
                  model_path, input_shape):
    try:
        conn   = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_results (
                model_name, input_shape, params, accuracy, macro_f1,
                precision_normal, recall_normal, f1_normal,
                precision_inner,  recall_inner,  f1_inner,
                precision_outer,  recall_outer,  f1_outer,
                precision_cage,   recall_cage,   f1_cage,
                auc_normal, auc_inner, auc_outer, auc_cage,
                training_time_sec, epochs_trained,
                batch_size, learning_rate, model_path
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            model_name, input_shape, model.count_params(),
            round(report_dict['accuracy'], 4),
            round(report_dict['macro avg']['f1-score'], 4),
            round(report_dict['Normal']['precision'], 4),
            round(report_dict['Normal']['recall'], 4),
            round(report_dict['Normal']['f1-score'], 4),
            round(report_dict['Inner Race']['precision'], 4),
            round(report_dict['Inner Race']['recall'], 4),
            round(report_dict['Inner Race']['f1-score'], 4),
            round(report_dict['Outer Race']['precision'], 4),
            round(report_dict['Outer Race']['recall'], 4),
            round(report_dict['Outer Race']['f1-score'], 4),
            round(report_dict['Cage']['precision'], 4),
            round(report_dict['Cage']['recall'], 4),
            round(report_dict['Cage']['f1-score'], 4),
            round(aucs[0], 4), round(aucs[1], 4),
            round(aucs[2], 4), round(aucs[3], 4),
            round(training_time, 1), epochs_trained,
            BATCH_SIZE, 1e-3, model_path
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f'  Results saved to MySQL')
    except Exception as e:
        print(f'  MySQL save skipped: {e}')

# ══════════════════════════════════════════════════════════════
# TRAIN ONE MODEL
# ══════════════════════════════════════════════════════════════

def train_model(model, model_name, model_path,
                X_train, y_train, X_val, y_val,
                X_test, y_test, input_shape):

    print(f'\n{"="*55}')
    print(f'  Training {model_name}')
    print(f'  Params  : {model.count_params():,}')
    print(f'  Input   : {input_shape}')
    print(f'{"="*55}')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
    ]

    start   = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks
    )
    training_time  = time.time() - start
    epochs_trained = len(history.history['accuracy'])

    print(f'\n  Training time : {training_time:.1f}s '
          f'({training_time/60:.1f} min)')
    print(f'  Epochs        : {epochs_trained}')

    # Load best weights
    model.load_weights(model_path)

    # Evaluate
    print(f'\nEvaluating {model_name}...')
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_prob, axis=1)

    print('\n── Classification Report ────────────────────')
    report_str  = classification_report(
        y_test, y_pred,
        target_names=['Normal','Inner Race','Outer Race','Cage'],
        zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=['Normal','Inner Race','Outer Race','Cage'],
        zero_division=0, output_dict=True
    )
    print(report_str)

    # Save plots
    print('Saving plots...')
    plot_history(history, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    aucs = plot_roc(y_test, y_prob, model_name)

    # Save to MySQL
    save_to_mysql(model_name, model, report_dict,
                  aucs, training_time, epochs_trained,
                  model_path, input_shape)

    return {
        'accuracy' : report_dict['accuracy'],
        'macro_f1' : report_dict['macro avg']['f1-score'],
        'cage_rec' : report_dict['Cage']['recall'],
        'cage_auc' : aucs[3],
        'time'     : training_time,
        'params'   : model.count_params(),
    }

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\nBearingBench — Unified Trainer')
    print('CNN · LSTM · BiLSTM · Transformer')
    print('=' * 55)
    setup_gpu()

    # ── Load data ─────────────────────────────────────────────
    print('\nLoading dataset...')
    X_seq, X_img, y, meta = load_dataset()
    splits = get_splits(X_seq, X_img, y)

    X_seq_tr, y_tr  = splits['seq']['train']
    X_seq_val, y_val = splits['seq']['val']
    X_seq_te, y_te  = splits['seq']['test']

    X_img_tr, _     = splits['img']['train']
    X_img_val, _    = splits['img']['val']
    X_img_te, _     = splits['img']['test']

    print(f'\n  Train : {X_seq_tr.shape}')
    print(f'  Val   : {X_seq_val.shape}')
    print(f'  Test  : {X_seq_te.shape}')
    print(f'\n  Class weights: {CLASS_WEIGHTS}')

    results = {}

    # ── CNN ───────────────────────────────────────────────────
    cnn_path = os.path.join(MODELS_DIR, 'cnn_best.h5')
    if os.path.exists(cnn_path):
        print('\n  CNN already trained — skipping')
        results['CNN'] = {'accuracy':0,'macro_f1':0,
                          'cage_rec':0,'cage_auc':0,
                          'time':0,'params':0}
    else:
        results['CNN'] = train_model(
            build_cnn(), 'CNN', cnn_path,
            X_img_tr, y_tr, X_img_val, y_val, X_img_te, y_te,
            f'(batch, {IMG_H}, {IMG_W}, {N_FEATURES})'
        )

    # ── LSTM ──────────────────────────────────────────────────
    lstm_path = os.path.join(MODELS_DIR, 'lstm_best.h5')
    if os.path.exists(lstm_path):
        print('\n  LSTM already trained — skipping')
        results['LSTM'] = {'accuracy':0,'macro_f1':0,
                           'cage_rec':0,'cage_auc':0,
                           'time':0,'params':0}
    else:
        results['LSTM'] = train_model(
            build_lstm(), 'LSTM', lstm_path,
            X_seq_tr, y_tr, X_seq_val, y_val, X_seq_te, y_te,
            f'(batch, {SEQ_LEN}, {N_FEATURES})'
        )

    # ── BiLSTM ────────────────────────────────────────────────
    bilstm_path = os.path.join(MODELS_DIR, 'bilstm_best.h5')
    if os.path.exists(bilstm_path):
        print('\n  BiLSTM already trained — skipping')
        results['BiLSTM'] = {'accuracy':0,'macro_f1':0,
                             'cage_rec':0,'cage_auc':0,
                             'time':0,'params':0}
    else:
        results['BiLSTM'] = train_model(
            build_bilstm(), 'BiLSTM', bilstm_path,
            X_seq_tr, y_tr, X_seq_val, y_val, X_seq_te, y_te,
            f'(batch, {SEQ_LEN}, {N_FEATURES})'
        )

    # ── Transformer ───────────────────────────────────────────
    trans_path = os.path.join(MODELS_DIR, 'transformer_best.h5')
    if os.path.exists(trans_path):
        print('\n  Transformer already trained — skipping')
        results['Transformer'] = {'accuracy':0,'macro_f1':0,
                                  'cage_rec':0,'cage_auc':0,
                                  'time':0,'params':0}
    else:
        results['Transformer'] = train_model(
            build_transformer(), 'Transformer', trans_path,
            X_seq_tr, y_tr, X_seq_val, y_val, X_seq_te, y_te,
            f'(batch, {SEQ_LEN}, {N_FEATURES})'
        )

    # ── Final comparison ──────────────────────────────────────
    print('\n\n' + '='*60)
    print('BEARINGBENCH — FINAL RESULTS')
    print('='*60)
    print(f'{"Model":<14}{"Accuracy":>10}{"MacroF1":>9}'
          f'{"CageRec":>9}{"CageAUC":>9}'
          f'{"Params":>10}{"Time(s)":>9}')
    print('-'*60)
    for name, r in results.items():
        print(f'{name:<14}{r["accuracy"]:>10.3f}'
              f'{r["macro_f1"]:>9.3f}'
              f'{r["cage_rec"]:>9.3f}'
              f'{r["cage_auc"]:>9.3f}'
              f'{r["params"]:>10,}'
              f'{r["time"]:>9.1f}')

    # Save text report
    report_lines = [
        '='*60,
        'BearingBench — Benchmark Report',
        'XJTU-SY Rolling Element Bearing Fault Detection',
        '4 Classes: Normal / Inner Race / Outer Race / Cage',
        '='*60,
        f'{"Model":<14}{"Accuracy":>10}{"MacroF1":>9}'
        f'{"CageRec":>9}{"CageAUC":>9}{"Params":>10}',
        '-'*60,
    ]
    for name, r in results.items():
        report_lines.append(
            f'{name:<14}{r["accuracy"]:>10.3f}'
            f'{r["macro_f1"]:>9.3f}'
            f'{r["cage_rec"]:>9.3f}'
            f'{r["cage_auc"]:>9.3f}'
            f'{r["params"]:>10,}'
        )

    report_path = os.path.join(RESULTS_DIR, 'benchmark_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'\n  Report saved: {report_path}')
    print('\nBearingBench training complete!\n')
