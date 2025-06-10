import os
import sqlite3
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from deepface import DeepFace
from scipy.spatial.distance import cosine
import io
from datetime import datetime # Import datetime here

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DB_FILE = 'forensic_database.db'
THRESHOLD = 0.68  # Cosine distance threshold for a match

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a_very_secret_key_for_a_forensic_app'

# --- Database Converters (for numpy arrays) ---
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

# --- Context Processor for dynamic year in templates ---
# This function will run before every template is rendered
# and make 'current_year' available in all your Jinja2 templates.
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def identify():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # --- Multi-Face Recognition Logic ---
            all_results = []
            try:
                # DeepFace.represent returns a list of embeddings if multiple faces are found
                embedding_objs = DeepFace.represent(img_path=filepath, model_name='ArcFace', detector_backend='retinaface')

                conn = get_db_connection()
                identities = conn.execute('SELECT * FROM identities').fetchall()
                conn.close()

                if not identities:
                    flash('Database is empty. Please enroll identities first.', 'warning')
                    return redirect(url_for('enroll'))

                # Iterate over each face found in the image
                for emb_obj in embedding_objs:
                    unknown_embedding = np.array(emb_obj['embedding'])

                    best_match_details = None
                    smallest_distance = float('inf')

                    # Compare the current face with all identities in the database
                    for identity in identities:
                        distance = cosine(unknown_embedding, identity['embedding'])
                        if distance < smallest_distance:
                            smallest_distance = distance
                            best_match_details = identity

                    if best_match_details and smallest_distance <= THRESHOLD:
                        similarity = (1 - smallest_distance) * 100
                        all_results.append({
                            'status': 'Match Found',
                            'similarity': f"{similarity:.2f}%",
                            'details': dict(best_match_details) # Convert sqlite3.Row to dict
                        })
                    else:
                        all_results.append({'status': 'Match Not Found'})

            except Exception as e:
                flash(f"Could not process image. Error: {e}", 'danger')
                return redirect(request.url)

            return render_template('result.html', results=all_results, image_path=filepath)

    return render_template('identify.html')


@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        # Retrieve all form data
        name = request.form['name']
        nin = request.form['nin']
        dob = request.form['dob']
        nationality = request.form['nationality']
        description = request.form['description']
        file = request.files['file']

        if not all([name, nin, file]):
            flash('Name, NIN, and Image File are required!', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                embedding_objs = DeepFace.represent(img_path=filepath, model_name='ArcFace', detector_backend='retinaface')
                embedding = np.array(embedding_objs[0]['embedding'])
            except Exception as e:
                flash(f"Could not process image. Is there a clear face? Error: {e}", 'danger')
                return redirect(request.url)

            conn = get_db_connection()
            try:
                conn.execute(
                    'INSERT INTO identities (name, nin, dob, nationality, description, embedding) VALUES (?, ?, ?, ?, ?, ?)',
                    (name, nin, dob, nationality, description, embedding)
                )
                conn.commit()
                flash(f"Successfully enrolled {name} (NIN: {nin}).", 'success')
            except sqlite3.IntegrityError:
                flash(f"Error: NIN '{nin}' already exists in the database.", 'danger')
            finally:
                conn.close()

            return redirect(url_for('enroll'))

    return render_template('enroll.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)