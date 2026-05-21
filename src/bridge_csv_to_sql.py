"""
BearingBench — Bridge CSV Metadata to MySQL
Populates working_conditions, bearings, and files tables.
Does NOT store raw signal data — only metadata.

Run: python src/bridge_csv_to_sql.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, SRC_DIR)

from db import get_conn

# ── Working condition metadata ────────────────────────────────
WC_DATA = [
    {'name': 'WC1', 'rpm': 2100, 'load_kn': 12.0,
     'description': '2100 RPM / 12kN — slow speed, heavy load'},
    {'name': 'WC2', 'rpm': 2250, 'load_kn': 11.0,
     'description': '2250 RPM / 11kN — medium speed, medium load'},
    {'name': 'WC3', 'rpm': 2400, 'load_kn': 10.0,
     'description': '2400 RPM / 10kN — high speed, light load'},
]

# ── Bearing metadata (verified against XJTU-SY paper Table 2) ─
BEARING_DATA = {
    # WC1
    'Bearing1_1': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing1_2': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing1_3': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing1_4': {'fault_label': 3, 'fault_type': 'Cage',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing1_5': {'fault_label': 0, 'fault_type': 'Inner Race + Outer Race',
                   'is_skipped': True,
                   'skip_reason': 'Combined Inner+Outer fault — ambiguous label'},
    # WC2
    'Bearing2_1': {'fault_label': 1, 'fault_type': 'Inner Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing2_2': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing2_3': {'fault_label': 3, 'fault_type': 'Cage',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing2_4': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing2_5': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    # WC3
    'Bearing3_1': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing3_2': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False,
                   'skip_reason': 'Combined fault — labeled as Outer Race (dominant)'},
    'Bearing3_3': {'fault_label': 1, 'fault_type': 'Inner Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing3_4': {'fault_label': 1, 'fault_type': 'Inner Race',
                   'is_skipped': False, 'skip_reason': None},
    'Bearing3_5': {'fault_label': 2, 'fault_type': 'Outer Race',
                   'is_skipped': False, 'skip_reason': None},
}

FAULT_NAMES = {0: 'Normal', 1: 'Inner Race', 2: 'Outer Race', 3: 'Cage'}

WC_BEARINGS = {
    'WC1': ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5'],
    'WC2': ['Bearing2_1','Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5'],
    'WC3': ['Bearing3_1','Bearing3_2','Bearing3_3','Bearing3_4','Bearing3_5'],
}

# ── Insert working conditions ─────────────────────────────────
def insert_working_conditions(conn):
    cursor = conn.cursor()
    for wc in WC_DATA:
        si = int(wc['load_kn'] * wc['rpm'] ** 2)
        cursor.execute("""
            INSERT IGNORE INTO working_conditions
                (name, rpm, load_kn, severity_index, description)
            VALUES (%s, %s, %s, %s, %s)
        """, (wc['name'], wc['rpm'], wc['load_kn'], si, wc['description']))
    conn.commit()

    cursor.execute("SELECT id, name, rpm, load_kn, severity_index "
                   "FROM working_conditions ORDER BY name")
    rows = cursor.fetchall()
    print(f'\n  {"Condition":<8} {"RPM":>6} {"Load":>6} '
          f'{"SI":>12}')
    print(f'  {"-"*36}')
    for r in rows:
        print(f'  {r[1]:<8} {r[2]:>6} {r[3]:>6.0f}kN {r[4]:>12,}')
    cursor.close()

# ── Insert bearings ───────────────────────────────────────────
def insert_bearings(conn):
    cursor = conn.cursor()

    for wc_name, bearing_list in WC_BEARINGS.items():
        cursor.execute("SELECT id FROM working_conditions WHERE name=%s",
                       (wc_name,))
        wc_id = cursor.fetchone()[0]

        for bearing_name in bearing_list:
            info = BEARING_DATA[bearing_name]

            # Count CSV files
            bearing_path = os.path.join(DATA_DIR, wc_name, bearing_name)
            if os.path.exists(bearing_path):
                total_files = len([f for f in os.listdir(bearing_path)
                                   if f.endswith('.csv')])
            else:
                total_files = 0

            cursor.execute("""
                INSERT IGNORE INTO bearings
                    (wc_id, name, total_files, lifetime_min,
                     fault_label, fault_type, is_skipped, skip_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                wc_id, bearing_name, total_files, total_files,
                info['fault_label'], info['fault_type'],
                info['is_skipped'], info['skip_reason']
            ))

    conn.commit()
    cursor.close()

# ── Insert file metadata ──────────────────────────────────────
def insert_files(conn):
    cursor = conn.cursor()
    total_inserted = 0

    for wc_name, bearing_list in WC_BEARINGS.items():
        for bearing_name in bearing_list:
            cursor.execute("SELECT id, total_files, fault_label "
                           "FROM bearings WHERE name=%s", (bearing_name,))
            row = cursor.fetchone()
            if not row:
                continue
            bearing_id, total_files, fault_label = row

            bearing_path = os.path.join(DATA_DIR, wc_name, bearing_name)
            if not os.path.exists(bearing_path):
                continue

            files = sorted(
                [f for f in os.listdir(bearing_path) if f.endswith('.csv')],
                key=lambda x: int(x.replace('.csv', ''))
            )

            cutoff = int(len(files) * 0.8)
            rows   = []

            for i, fname in enumerate(files):
                file_num    = i + 1
                lifetime_pct = round(file_num / len(files) * 100, 2)
                label       = 0 if i < cutoff else fault_label
                state       = FAULT_NAMES[label]
                rows.append((bearing_id, file_num, fname,
                             lifetime_pct, label, state))

            cursor.executemany("""
                INSERT IGNORE INTO files
                    (bearing_id, file_number, filename,
                     lifetime_pct, fault_label, fault_state)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, rows)
            conn.commit()
            total_inserted += len(rows)

            skipped = '(skipped)' if BEARING_DATA[bearing_name]['is_skipped'] else ''
            print(f'  {wc_name}/{bearing_name}: {len(files)} files '
                  f'· Normal={cutoff} · Fault={len(files)-cutoff} {skipped}')

    cursor.close()
    return total_inserted

# ── Print summary ─────────────────────────────────────────────
def print_summary(conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            wc.name, b.name, b.total_files,
            b.fault_type, b.is_skipped,
            SUM(f.fault_label = 0) AS normal,
            SUM(f.fault_label > 0) AS fault
        FROM bearings b
        JOIN working_conditions wc ON wc.id = b.wc_id
        LEFT JOIN files f          ON f.bearing_id = b.id
        GROUP BY b.id
        ORDER BY wc.name, b.name
    """)
    rows = cursor.fetchall()

    print(f'\n{"Cond":<5}{"Bearing":<14}{"Files":>6}'
          f'{"Fault Type":<16}{"Normal":>8}{"Fault":>7}{"Skip":>6}')
    print('-' * 62)
    for r in rows:
        skip = 'YES' if r[4] else ''
        print(f'{r[0]:<5}{r[1]:<14}{r[2]:>6}'
              f'{r[3]:<16}{int(r[5] or 0):>8}'
              f'{int(r[6] or 0):>7}{skip:>6}')

    cursor.execute("SELECT COUNT(*) FROM files")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM bearings WHERE is_skipped=FALSE")
    active = cursor.fetchone()[0]
    print(f'\n  Active bearings : {active}/15')
    print(f'  Total CSV files : {total:,}')
    cursor.close()

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nBearingBench — Bridge CSV to MySQL')
    print('=' * 50)

    conn = get_conn()

    print('\n1. Inserting working conditions...')
    insert_working_conditions(conn)

    print('\n2. Inserting bearings...')
    insert_bearings(conn)

    print('\n3. Inserting file metadata...')
    n = insert_files(conn)

    print('\n4. Summary:')
    print_summary(conn)

    conn.close()
    print('\nBearingBench database populated!\n')
