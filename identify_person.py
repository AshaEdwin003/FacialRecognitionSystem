# identify_person.py
import sqlite3
import numpy as np
import argparse
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Import the converters from the setup script
from database_setup import convert_array

def identify_person(image_path, db_file='forensic_database.db', threshold=0.68):
    """
    Identifies a person by comparing their face embedding with those in the database.
    - image_path: Path to the image of the unknown person.
    - threshold: Cosine distance threshold for a match. Lower is better. 0.68 is a good starting point for ArcFace.
    """
    # Register the numpy array converter for this connection
    sqlite3.register_converter("array", convert_array)

    # Step 1: Generate embedding for the unknown person's face
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name='ArcFace',
            detector_backend='retinaface'
        )
        unknown_embedding = np.array(embedding_objs[0]['embedding'])
        print(f"Successfully generated embedding for the unknown person.")
    except Exception as e:
        print(f"Error: Could not process the image {image_path}. No face detected or other issue.")
        print(f"Details: {e}")
        return

    # Step 2: Fetch all known embeddings from the database
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM identities")
        known_identities = cursor.fetchall()
        
        if not known_identities:
            print("Database is empty. Please enroll identities first.")
            return

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return
    finally:
        if conn:
            conn.close()

    # Step 3: Compare the unknown embedding with each known one
    best_match_name = "Unknown Person"
    smallest_distance = float('inf')

    for name, known_embedding in known_identities:
        # Calculate cosine distance
        distance = cosine(unknown_embedding, known_embedding)
        
        print(f"Comparing with {name}... Distance: {distance:.4f}")

        if distance < smallest_distance:
            smallest_distance = distance
            best_match_name = name

    # Step 4: Check if the best match is within the acceptance threshold
    print("-" * 30)
    if smallest_distance <= threshold:
        # Cosine similarity is (1 - cosine_distance)
        similarity = (1 - smallest_distance) * 100
        print(f"✅ Match Found!")
        print(f"   Person: {best_match_name}")
        print(f"   Similarity: {similarity:.2f}% (Distance: {smallest_distance:.4f})")
    else:
        print(f"❌ No match found in the database.")
        print(f"   Closest match was {best_match_name}, but the distance ({smallest_distance:.4f}) was above the threshold ({threshold}).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Identify a person from an image using the forensic database.")
    parser.add_argument("--image", required=True, help="Path to the image of the unknown person.")
    args = parser.parse_args()

    identify_person(args.image)