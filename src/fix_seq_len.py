"""
BearingBench — Fix trainer.py for faster BiLSTM + Transformer training
Run this AFTER LSTM finishes and BEFORE BiLSTM starts

Changes:
    SEQ_LEN    : 1024 → 256   (4x faster, ~same accuracy)
    BATCH_SIZE : 64   → 128   (2x faster)

Then rebuilds cache with new window size.

Run: python src/fix_seq_len.py
"""

import os
import sys
import shutil

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR     = os.path.join(BASE_DIR, 'src')
CACHE_DIR   = os.path.join(BASE_DIR, 'data', 'cache')
TRAINER     = os.path.join(SRC_DIR, 'trainer.py')
DATA_LOADER = os.path.join(SRC_DIR, 'data_loader.py')

print('\nBearingBench — Speed Fix')
print('=' * 50)

# ── Step 1: Backup original files ─────────────────────────────
print('\n1. Backing up original files...')
shutil.copy(TRAINER,     TRAINER     + '.bak')
shutil.copy(DATA_LOADER, DATA_LOADER + '.bak')
print(f'  Backed up: {TRAINER}.bak')
print(f'  Backed up: {DATA_LOADER}.bak')

# ── Step 2: Fix trainer.py ────────────────────────────────────
print('\n2. Updating trainer.py...')
content = open(TRAINER).read()

fixes = [
    ('SEQ_LEN    = 1024', 'SEQ_LEN    = 256 '),
    ('BATCH_SIZE = 64',   'BATCH_SIZE = 128 '),
]

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f'  {old.strip()} → {new.strip()}')
    else:
        print(f'  WARNING: "{old}" not found in trainer.py')

open(TRAINER, 'w').write(content)
print('  trainer.py updated')

# ── Step 3: Fix data_loader.py ────────────────────────────────
print('\n3. Updating data_loader.py...')
content = open(DATA_LOADER).read()

fixes_dl = [
    ('WINDOW_SIZE = 1024', 'WINDOW_SIZE = 256 '),
    ('STRIDE      = 512',  'STRIDE      = 128 '),
]

for old, new in fixes_dl:
    if old in content:
        content = content.replace(old, new)
        print(f'  {old.strip()} → {new.strip()}')
    else:
        print(f'  WARNING: "{old}" not found in data_loader.py')

open(DATA_LOADER, 'w').write(content)
print('  data_loader.py updated')

# ── Step 4: Delete old cache ──────────────────────────────────
print('\n4. Deleting old cache...')
for fname in ['sequences.pkl', 'images.pkl']:
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        print(f'  Deleted: {path}')
    else:
        print(f'  Not found: {path}')

# ── Step 5: Verify ────────────────────────────────────────────
print('\n5. Verifying changes...')
trainer_content    = open(TRAINER).read()
dataloader_content = open(DATA_LOADER).read()

checks = [
    (trainer_content,    'SEQ_LEN    = 256',  'trainer.py SEQ_LEN'),
    (trainer_content,    'BATCH_SIZE = 128',  'trainer.py BATCH_SIZE'),
    (dataloader_content, 'WINDOW_SIZE = 256', 'data_loader.py WINDOW_SIZE'),
    (dataloader_content, 'STRIDE      = 128', 'data_loader.py STRIDE'),
]

all_ok = True
for content, check, label in checks:
    if check in content:
        print(f'  ✓ {label} = {check.split("=")[1].strip()}')
    else:
        print(f'  ✗ {label} NOT updated')
        all_ok = False

# ── Summary ───────────────────────────────────────────────────
print('\n' + '=' * 50)
if all_ok:
    print('All fixes applied successfully!')
    print('\nExpected training times with new settings:')
    print('  LSTM        : ~2-3 hours   (was 22 hours)')
    print('  BiLSTM      : ~4-5 hours   (was 44 hours)')
    print('  Transformer : ~1-2 hours   (was 3-5 hours)')
    print('\nNext steps:')
    print('  1. Wait for LSTM epoch 80 to finish')
    print('  2. Press Ctrl+C to stop trainer')
    print('  3. Run: python src/trainer.py 2>/dev/null')
    print('     (CNN + LSTM will be skipped automatically)')
else:
    print('Some fixes failed — check warnings above')
    print('Restore backups with:')
    print('  cp src/trainer.py.bak src/trainer.py')
    print('  cp src/data_loader.py.bak src/data_loader.py')
