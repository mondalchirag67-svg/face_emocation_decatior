"""
=============================================================================
FaceSense AI - Facial Emotion Detection Web Application
=============================================================================
Framework: Python Flask
Active Model: emotiondetector.h5 (User's Trained CNN Model)
Authentication: Built-in User Registration & Login with SQLite & Werkzeug
Description: Full-stack web server for real-time facial emotion recognition.
             Requires user registration & login before activating the camera
             or detecting facial emotions.
=============================================================================
"""

import os
import io
import time
import base64
import sqlite3
import numpy as np
import cv2
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, session, flash
)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'facesense-ai-secret-key-2026-auth-session'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload limit



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(UPLOAD_FOLDER, 'models')
DB_PATH = os.path.join(BASE_DIR, 'users.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)




def get_db():
    """Returns a SQLite connection with dict-like row access and busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the users table if it does not already exist."""
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()




init_db()

@app.context_processor
def inject_user():
    """
    Injects `current_user` into all Jinja2 templates automatically.
    Returns None if the visitor is not logged in.
    """
    user = None
    user_id = session.get('user_id')
    if user_id:
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                user = dict(row)
        except Exception as e:
            print(f"[DB ERROR] Error fetching current user: {e}")
    return dict(current_user=user)






EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_META = {
    'Angry':    {'emoji': '😠', 'color': '#f87171', 'desc': 'Furrowed brows, tightened lips, intense gaze'},
    'Disgust':  {'emoji': '🤢', 'color': '#34d399', 'desc': 'Wrinkled nose, raised upper lip'},
    'Fear':     {'emoji': '😨', 'color': '#c084fc', 'desc': 'Widened eyes, exposed sclera, raised brows'},
    'Happy':    {'emoji': '😊', 'color': '#fbbf24', 'desc': 'Raised mouth corners, cheek lift, smiling eyes'},
    'Neutral':  {'emoji': '😐', 'color': '#94a3b8', 'desc': 'Relaxed facial musculature, balanced features'},
    'Sad':      {'emoji': '😢', 'color': '#60a5fa', 'desc': 'Downturned mouth corners, inner brows raised'},
    'Surprise': {'emoji': '😲', 'color': '#f472b6', 'desc': 'Dropped jaw, raised curved eyebrows, open eyes'}
}




haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
if not os.path.exists(haar_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    face_cascade = cv2.CascadeClassifier(haar_path)



CUSTOM_MODEL_PATH = "emotiondetector.h5"
CUSTOM_MODEL = None
CUSTOM_MODEL_NAME = "emotiondetector.h5"
CUSTOM_MODEL_INPUT_SHAPE = (48, 48, 1)

def try_load_model(model_filepath):
    """Safely loads a Keras / TensorFlow model without re-compilation."""
    global CUSTOM_MODEL, CUSTOM_MODEL_NAME, CUSTOM_MODEL_INPUT_SHAPE
    try:
        from tensorflow.keras.models import load_model # type: ignore
        print(f"[INFO] Loading emotion model from: {model_filepath}")
        loaded = load_model(model_filepath, compile=False)
        CUSTOM_MODEL = loaded
        CUSTOM_MODEL_NAME = os.path.basename(model_filepath)
        
        try:
            shape = loaded.input_shape
            if shape and len(shape) >= 3:
                h = shape[1] if shape[1] is not None else 48
                w = shape[2] if shape[2] is not None else 48
                c = shape[3] if len(shape) > 3 and shape[3] is not None else 1
                CUSTOM_MODEL_INPUT_SHAPE = (h, w, c)
        except Exception:
            CUSTOM_MODEL_INPUT_SHAPE = (48, 48, 1)

        print(f"[SUCCESS] Model '{CUSTOM_MODEL_NAME}' successfully activated! Input shape: {CUSTOM_MODEL_INPUT_SHAPE}")
        return True, f"Model '{CUSTOM_MODEL_NAME}' loaded successfully!"
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_filepath}': {e}")
        return False, str(e)

# Priority 1: Check root directory
if os.path.exists(os.path.join(BASE_DIR, "emotiondetector.h5")):
    try_load_model(os.path.join(BASE_DIR, "emotiondetector.h5"))
# Priority 2: Check uploads/models/
elif os.path.exists(os.path.join(MODEL_FOLDER, "emotiondetector.h5")):
    try_load_model(os.path.join(MODEL_FOLDER, "emotiondetector.h5"))
# Priority 3: Fallback to any model in uploads/models
else:
    existing = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(('.h5', '.keras'))]
    if existing:
        try_load_model(os.path.join(MODEL_FOLDER, existing[0]))





def predict_emotion_from_face(face_roi_gray, face_roi_color):
    """
    Takes cropped face ROI and runs inference through emotiondetector.h5.
    Preprocesses to (48, 48, 1) and returns 7 emotion probabilities.
    """
    global CUSTOM_MODEL, CUSTOM_MODEL_INPUT_SHAPE, CUSTOM_MODEL_NAME

    if CUSTOM_MODEL is not None:
        try:
            target_h, target_w, target_c = CUSTOM_MODEL_INPUT_SHAPE
            
            if target_c == 1:
                # 48x48x1 Grayscale
                resized = cv2.resize(face_roi_gray, (target_w, target_h))
                input_tensor = resized.astype('float32') / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=-1)
                input_tensor = np.expand_dims(input_tensor, axis=0)
            else:
                # RGB
                resized = cv2.resize(face_roi_color, (target_w, target_h))
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_tensor = resized.astype('float32') / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=0)

            raw_preds = CUSTOM_MODEL.predict(input_tensor, verbose=0)[0]
            
            sum_preds = float(np.sum(raw_preds))
            if 0.95 <= sum_preds <= 1.05 and np.all(raw_preds >= 0):
                probs = raw_preds
            else:
                exp_preds = np.exp(raw_preds - np.max(raw_preds))
                probs = exp_preds / np.sum(exp_preds)

            prob_dict = {emotion: float(round(probs[i] * 100, 2)) for i, emotion in enumerate(EMOTIONS)}
            best_idx = int(np.argmax(probs))
            dominant = EMOTIONS[best_idx]
            confidence = float(round(probs[best_idx] * 100, 2))

            return {
                'dominant': dominant,
                'confidence': confidence,
                'probabilities': prob_dict,
                'engine': f"User Model ({CUSTOM_MODEL_NAME})"
            }
        except Exception as e:
            print(f"[ERROR] Inference error on {CUSTOM_MODEL_NAME}: {e}")

   

    return {
        'dominant': 'Neutral',
        'confidence': 50.0,
        'probabilities': {e: 14.28 for e in EMOTIONS},
        'engine': "Baseline Detector"
    }






@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page & handler."""
    if session.get('user_id'):
        flash('You are already registered and logged in!', 'info')
        return redirect(url_for('detector'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validations
        if not username or not email or not password:
            flash('Please fill in all required fields.', 'danger')
            return render_template('register.html', username=username, email=email)

        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'danger')
            return render_template('register.html', username=username, email=email)

        if '@' not in email or '.' not in email:
            flash('Please enter a valid email address.', 'danger')
            return render_template('register.html', username=username, email=email)

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html', username=username, email=email)

        if password != confirm_password:
            flash('Passwords do not match. Please re-enter.', 'danger')
            return render_template('register.html', username=username, email=email)

        conn = get_db()
        cur = conn.cursor()
        
       
        cur.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cur.fetchone()
        if existing_user:
            conn.close()
            flash('Username or email is already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        
        password_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        new_user_id = cur.lastrowid
        conn.close()

        # Automatically log the user in
        session['user_id'] = new_user_id
        session['username'] = username

        flash(f'🎉 Welcome, {username}! Your account is created. Camera access is now unlocked!', 'success')
        return redirect(url_for('detector'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page & handler."""
    if session.get('user_id'):
        flash('You are already logged in!', 'info')
        return redirect(url_for('detector'))

    if request.method == 'POST':
        login_identifier = request.form.get('username_or_email', '').strip()
        password = request.form.get('password', '')

        if not login_identifier or not password:
            flash('Please enter your username/email and password.', 'danger')
            return render_template('login.html', login_identifier=login_identifier)

        conn = get_db()
        cur = conn.cursor()
        # Allow logging in with either username OR email
        cur.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ? OR email = ?",
            (login_identifier, login_identifier.lower())
        )
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}! Emotion detection is ready.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('detector'))
        else:
            flash('Invalid username/email or password. Please try again or register.', 'danger')
            return render_template('login.html', login_identifier=login_identifier)

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logs the user out and clears the session."""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out. Registration/login is required to use live emotion detection.', 'info')
    return redirect(url_for('home'))




@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    """AJAX endpoint for in-modal login."""
    data = request.get_json(silent=True) or {}
    identifier = data.get('username_or_email', '').strip()
    password = data.get('password', '')

    if not identifier or not password:
        return jsonify({'status': 'error', 'message': 'Username/email and password are required.'}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE username = ? OR email = ?", (identifier, identifier.lower()))
    user = cur.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({
            'status': 'success',
            'message': f'Welcome back, {user["username"]}!',
            'username': user['username']
        })
    return jsonify({'status': 'error', 'message': 'Invalid username/email or password.'}), 401


@app.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    """AJAX endpoint for in-modal registration."""
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')

    if not username or not email or not password:
        return jsonify({'status': 'error', 'message': 'All fields are required.'}), 400

    if len(username) < 3:
        return jsonify({'status': 'error', 'message': 'Username must be at least 3 characters.'}), 400

    if '@' not in email or '.' not in email:
        return jsonify({'status': 'error', 'message': 'Please provide a valid email.'}), 400

    if len(password) < 6:
        return jsonify({'status': 'error', 'message': 'Password must be at least 6 characters.'}), 400

    if password != confirm_password:
        return jsonify({'status': 'error', 'message': 'Passwords do not match.'}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if cur.fetchone():
        conn.close()
        return jsonify({'status': 'error', 'message': 'Username or email already exists. Please log in.'}), 409

    password_hash = generate_password_hash(password)
    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", (username, email, password_hash))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    session['user_id'] = new_id
    session['username'] = username

    return jsonify({
        'status': 'success',
        'message': f'Account created for {username}! You can now start emotion detection.',
        'username': username
    })


@app.route('/api/auth_status', methods=['GET'])
def api_auth_status():
    """Returns whether the current visitor is authenticated."""
    is_auth = 'user_id' in session
    return jsonify({
        'is_authenticated': is_auth,
        'authenticated': is_auth,
        'username': session.get('username') if is_auth else None,
        'user': {'username': session.get('username'), 'id': session.get('user_id')} if is_auth else None
    })


# ---------------------------------------------------------------------------
# 8. WEB PAGE ROUTES
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    """Page 1: Home Page - Overview, Features, Interactive Showcase, and Tech Stack"""
    model_info = {
        'is_loaded': CUSTOM_MODEL is not None,
        'name': CUSTOM_MODEL_NAME,
        'input_shape': CUSTOM_MODEL_INPUT_SHAPE
    }
    return render_template('index.html', page='home', emotions=EMOTIONS, emotion_meta=EMOTION_META, model_info=model_info)


@app.route('/detector')
def detector():
    """
    Page 2: Live Detector & Model Sandbox.
    Visitors can visit and view the page freely without registration,
    but camera & detection require login.
    """
    model_info = {
        'is_loaded': CUSTOM_MODEL is not None,
        'name': CUSTOM_MODEL_NAME,
        'input_shape': CUSTOM_MODEL_INPUT_SHAPE,
        'classes': EMOTIONS
    }
    return render_template('detector.html', page='detector', emotions=EMOTIONS, emotion_meta=EMOTION_META, model_info=model_info)


@app.route('/emotions')
def emotions():
    """Page 3: Emotion Psychology & FACS Micro-Expressions Guide"""
    return render_template('emotions.html', page='emotions', emotions=EMOTIONS, emotion_meta=EMOTION_META)


@app.route('/model-guide')
def model_guide():
    """Page 4: Deep Learning Architecture & Training Blueprint"""
    return render_template('model_guide.html', page='model_guide', emotions=EMOTIONS)





@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Receives frame from webcam or uploaded image, detects faces with Haar Cascade,
    and runs inference through emotiondetector.h5.
    
    [SECURITY CHECK]: Users MUST be registered and logged in to use detection!
    """
    

    if 'user_id' not in session:
        return jsonify({
            'status': 'unauthorized',
            'message': '🔒 Registration or Login required! Please log in to start facial emotion detection.'
        }), 401

    start_time = time.time()
    img_bgr = None

    # Case A: Base64 data URL from live webcam canvas
    data = request.get_json(silent=True)
    if data and 'image' in data:
        header, encoded = data['image'].split(',', 1) if ',' in data['image'] else ('', data['image'])
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    

    elif 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            img_bytes = file.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({'status': 'error', 'message': 'No valid image received'}), 400

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = img_bgr.shape[:2]

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(40, 40)
    )

    results = []

    if len(faces) == 0:
        crop_size = min(h_orig, w_orig)
        y_start = (h_orig - crop_size) // 2
        x_start = (w_orig - crop_size) // 2
        center_gray = gray[y_start:y_start+crop_size, x_start:x_start+crop_size]
        center_color = img_bgr[y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        pred = predict_emotion_from_face(center_gray, center_color)
        results.append({
            'box': [int(x_start), int(y_start), int(crop_size), int(crop_size)],
            'dominant': pred['dominant'],
            'confidence': pred['confidence'],
            'emoji': EMOTION_META[pred['dominant']]['emoji'],
            'color': EMOTION_META[pred['dominant']]['color'],
            'probabilities': pred['probabilities'],
            'engine': pred['engine'],
            'is_fallback': True
        })
    else:
        for (x, y, w, h) in faces:
            pad_w = int(w * 0.08)
            pad_h = int(h * 0.08)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_orig, x + w + pad_w)
            y2 = min(h_orig, y + h + pad_h)

            face_gray = gray[y1:y2, x1:x2]
            face_color = img_bgr[y1:y2, x1:x2]

            pred = predict_emotion_from_face(face_gray, face_color)

            results.append({
                'box': [int(x), int(y), int(w), int(h)],
                'dominant': pred['dominant'],
                'confidence': pred['confidence'],
                'emoji': EMOTION_META[pred['dominant']]['emoji'],
                'color': EMOTION_META[pred['dominant']]['color'],
                'probabilities': pred['probabilities'],
                'engine': pred['engine'],
                'is_fallback': False
            })

    processing_time = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'status': 'success',
        'faces_count': len(results),
        'predictions': results,
        'image_size': {'width': w_orig, 'height': h_orig},
        'processing_time_ms': processing_time,
        'model_name': CUSTOM_MODEL_NAME
    })


@app.route('/api/upload_model', methods=['POST'])
def api_upload_model():
    """Allows uploading new or replacement model weights (Admin/Auth)."""
    if 'user_id' not in session:
        return jsonify({'status': 'unauthorized', 'message': 'Please login to upload models.'}), 401

    if 'model_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    filename = file.filename
    if not (filename.endswith('.h5') or filename.endswith('.keras')):
        return jsonify({'status': 'error', 'message': 'Only .h5 or .keras model files are supported'}), 400

    save_path = os.path.join(MODEL_FOLDER, filename)
    file.save(save_path)

    success, msg = try_load_model(save_path)
    if success:
        return jsonify({
            'status': 'success',
            'message': msg,
            'model_name': filename,
            'input_shape': list(CUSTOM_MODEL_INPUT_SHAPE)
        })
    else:
        return jsonify({'status': 'error', 'message': f'Failed to load model: {msg}'}), 500


@app.route('/api/model_status', methods=['GET'])
def api_model_status():
    """Returns current active model information."""
    return jsonify({
        'is_loaded': CUSTOM_MODEL is not None,
        'model_name': CUSTOM_MODEL_NAME,
        'input_shape': list(CUSTOM_MODEL_INPUT_SHAPE),
        'classes': EMOTIONS
    })





@app.route('/favicon.ico')
def favicon():
    """Serves a clean SVG favicon directly to prevent 404s."""
    from flask import Response
    svg_icon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#6366f1"/>
          <stop offset="100%" stop-color="#0ea5e9"/>
        </linearGradient>
      </defs>
      <rect width="100" height="100" rx="24" fill="url(#grad)"/>
      <circle cx="35" cy="40" r="7" fill="#ffffff"/>
      <circle cx="65" cy="40" r="7" fill="#ffffff"/>
      <path d="M 30 65 Q 50 82 70 65" stroke="#ffffff" stroke-width="7" stroke-linecap="round" fill="none"/>
    </svg>'''
    return Response(svg_icon, mimetype='image/svg+xml')


@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'API endpoint not found'}), 404
    return render_template('error.html', error_code=404, error_title="Page Not Found",
                           error_message="The requested page could not be located. It might have been moved or doesn't exist."), 404


@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500
    return render_template('error.html', error_code=500, error_title="Internal Server Error",
                           error_message="A server error occurred while processing your request. Please try again shortly."), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Payload exceeds 32MB upload limit'}), 413
    return render_template('error.html', error_code=413, error_title="Upload Too Large",
                           error_message="The file you attempted to upload exceeds our 32 MB limit."), 413





if __name__ == '__main__':
    print("=" * 70)
    print("  🚀 FaceSense AI - Facial Emotion Detection Web Server")
    print(f"  👉 Active Model: {CUSTOM_MODEL_NAME}")
    print("  🔐 User Authentication: Active (SQLite & Password Hashing)")
    print("  🌐 Server running at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)
