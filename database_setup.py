# database_setup.py
import sqlite3
import numpy as np
import io

# --- Functions to convert numpy arrays for SQLite storage ---
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def setup_database(db_file='forensic_database.db'):
    # Register the custom converters
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()

        print("Creating 'identities' table with new forensic fields...")
        # Create table with NIN, DOB, Nationality, and Description.
        # NIN is set to UNIQUE to prevent duplicate entries.
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            nin TEXT UNIQUE NOT NULL,
            dob TEXT,
            nationality TEXT,
            description TEXT,
            embedding array NOT NULL
        )
        ''')
        
        conn.commit()
        print("Database and table have been created successfully.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    setup_database()