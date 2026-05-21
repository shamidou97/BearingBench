"""
BearingBench — Shared Database Connection
Auto-detects WSL (socket) vs Windows (TCP)

Usage:
    from db import get_conn, get_engine
"""

import os
import mysql.connector
from sqlalchemy import create_engine

# ── Config ────────────────────────────────────────────────────
USER     = 'bearinguser'
PASSWORD = 'bearing123'
DATABASE = 'bearingbench'
SOCKET   = '/var/run/mysqld/mysqld.sock'
WSL_IP   = '172.25.8.43'   # update if Windows IP changes

def _is_wsl():
    if os.name != 'posix':
        return False
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

def get_conn():
    if _is_wsl():
        return mysql.connector.connect(
            host='localhost',
            unix_socket=SOCKET,
            user=USER, password=PASSWORD, database=DATABASE
        )
    else:
        return mysql.connector.connect(
            host=WSL_IP, port=3306,
            user=USER, password=PASSWORD, database=DATABASE
        )

def get_engine():
    if _is_wsl():
        url = (f'mysql+mysqlconnector://{USER}:{PASSWORD}'
               f'@localhost/{DATABASE}?unix_socket={SOCKET}')
    else:
        url = (f'mysql+mysqlconnector://{USER}:{PASSWORD}'
               f'@{WSL_IP}:3306/{DATABASE}')
    return create_engine(url)

if __name__ == '__main__':
    print('Testing BearingBench database connection...')
    try:
        conn   = get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM bearings')
        count  = cursor.fetchone()[0]
        conn.close()
        print(f'  Connected successfully')
        print(f'  Bearings  : {count}')
        print(f'  Platform  : {"WSL/Linux" if _is_wsl() else "Windows"}')
    except Exception as e:
        print(f'  Connection failed: {e}')
