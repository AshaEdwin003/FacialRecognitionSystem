# enroll_person.py
import sqlite3
import numpy as np
import argparse
from deepface import DeepFace

# Import the converters from the setup script to handle numpy arrays
from database_setup import adapt_array, convert_array

def enroll_person(image_path, name, db_file='forensic_database.db'):
    """
    Enrolls a new person into the forensic database.
    It detects the face, generates an embedding using ArcFace, and stores it.
    """
    # Register the numpy array converters for this connection
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    try:
        print(f"Attempting to enroll '{name}' from image: {image_path}")
        # Use DeepFace to generate the face embedding.
        # model_name='ArcFace' is specified as per the project goal.
        # We use RetinaFace for detection by specifying it as the detector backend.
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name='ArcFace',
            detector_backend='retinaface'
        )

        # DeepFace.represent returns a list of dictionaries, we take the first one.
        embedding = embedding_objs[0]['embedding']
        embedding_np = np.array(embedding)

        print(f"Successfully generated face embedding for {name}.")

    except Exception as e:
        print(f"Error: Could not process the image {image_path}. No face detected or other issue.")
        print(f"Details: {e}")
        return

    conn = None
    try:
        # Connect to the database to insert the new identity
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()

        # Insert the person's name and their face embedding into the table
        cursor.execute("INSERT INTO identities (name, embedding) VALUES (?, ?)", (name, embedding_np))
        conn.commit()
        print(f"Successfully enrolled '{name}' into the database.")

    except sqlite3.Error as e:
        print(f"Database error during enrollment: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Set up command-line argument parsing to make the script easy to use
    parser = argparse.ArgumentParser(description="Enroll a new person into the face recognition database.")
    parser.add_argument("--image", required=True, help="Path to the image of the person to enroll.")
    parser.add_argument("--name", required=True, help="Name of the person.")
    args = parser.parse_args()

    # Call the enroll function with the provided arguments
    enroll_person(args.image, args.name)