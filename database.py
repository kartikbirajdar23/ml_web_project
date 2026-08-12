import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect("traffic_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            day TEXT,
            hour INTEGER,
            weather TEXT,
            predicted_vehicles INTEGER,
            traffic_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_prediction(username, day, hour, weather, predicted_vehicles, traffic_level):
    conn = sqlite3.connect("traffic_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (username, day, hour, weather, predicted_vehicles, traffic_level)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, day, hour, weather, predicted_vehicles, traffic_level))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect("traffic_data.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df