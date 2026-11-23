# 🎭 FaceSense AI - Deep Learning Facial Emotion Detection System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%20Haar%20Cascade-green.svg)](https://opencv.org/)
[![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205.3%20Glassmorphism-purple.svg)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, full-stack web application for real-time **Facial Emotion Recognition (FER)** powered directly by your custom trained Convolutional Neural Network model **`emotiondetector.h5`**. Features secure user authentication, camera gatekeeping, temporal anti-flicker smoothing, photo upload mode, and CSV telemetry export.

---

## 🌟 Key Highlights & The 4 Main Pages

| Page | Route | Description |
| :--- | :--- | :--- |
| **1. Home Overview** | `/` | Comprehensive overview, interactive FACS Affect showcase, 7 universal emotion cards with realistic photography, 4-stage engineering pipeline, and real-world AI use cases. |
| **2. Live Detector & Sandbox** | `/detector` | Real-time dual-mode detector (Webcam + Photo Upload) featuring prominent **Start Detection** and **Stop Detection** controls, 7 probability meters, anti-flicker smoothing, model swapper, and CSV export. |
| **3. Emotion Science & FACS** | `/emotions` | Scientific handbook based on Dr. Paul Ekman's Facial Action Coding System (FACS), Action Units (AU1, AU4, AU6, AU12), muscle anatomy, and benchmark datasets (FER-2013, AffectNet, CK+). |
| **4. Model Architecture Blueprint** | `/model-guide` | Complete engineering blueprint of the Mini-VGG CNN architecture, layer-by-layer spatial feature mapping, hyperparameter tables, and model integration checklist. |

---

## 🔐 User Authentication & Access Control

FaceSense AI incorporates a secure role-gated access model:

1. **Guest Visitors (No Registration Required)**:
   - Freely browse all 4 pages, view scientific explanations, model blueprints, and live detection tutorials.
   - On the Live Detector (`/detector`), camera activation is **locked**. The screen presents clear login/register options.
   - Calling `/api/predict` without an active session returns an HTTP `401 Unauthorized` response.
2. **Registered & Logged-In Users**:
   - Access is instantly unlocked upon registering or logging in.
   - The user's name is displayed in the navigation bar.
   - The camera displays **"Camera Ready • Signed in as &lt;username&gt;"** with a green **"Start Detection"** button.
   - Full live inference, single-snapshot detection, and portrait analysis are enabled.

---

## 🧠 Active Model Specs (`emotiondetector.h5`)

The system automatically loads your trained model on server boot:

- **Model File**: `emotiondetector.h5` (stored in root workspace)
- **Input Dimensions**: `(None, 48, 48, 1)` (Single-channel Grayscale, normalized `[0.0, 1.0]`)
- **Architecture**: 4-Block Convolutional Neural Network (Conv2D &rarr; BatchNorm &rarr; MaxPool &rarr; Dropout) with a Dense Softmax classification head
- **Target Classes**: 7 Universal Emotions (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`)
- **Face Localization**: OpenCV Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) with 8% physiological margin padding

---

## ⚙️ Anti-Flicker & Stability Innovations

To prevent the erratic flickering common in raw frame-by-frame models:
1. **Exponential Moving Average (EMA) Smoothing**:
   $$\text{Smoothed}_t = \alpha \cdot \text{Raw}_t + (1 - \alpha) \cdot \text{Smoothed}_{t-1} \quad (\alpha = 0.38)$$
   Eliminates instantaneous single-frame jitter while retaining swift response to genuine emotional shifts.
2. **Configurable Detection Rates**:
   Choose between **Relaxed (1.5s)**, **Balanced (1.0s, default)**, or **Brisk (0.6s)** directly from the UI dropdown.
3. **Smart History Deduplication**:
   When an emotion is sustained (e.g., smiling for several seconds), the top table row updates its duration in place (`Held 4s`) rather than flooding the log with repetitive duplicate rows.

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Webcam (built-in or USB)

### 2. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Server
```powershell
python app.py
```

### 4. Open in Browser
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in Chrome, Edge, or Firefox.

---

## 📁 Directory Structure

```text
face_emocation_decatior/
│
├── app.py                     # Flask server, auth handlers, and inference engine
├── emotiondetector.h5         # Active trained CNN emotion model (50.8 MB)
├── users.db                   # SQLite database storing registered accounts
├── requirements.txt           # Python library dependencies
├── README.md                  # Project documentation & reference
│
├── static/
│   ├── css/
│   │   └── style.css          # High-contrast glassmorphism master stylesheet
│   └── js/
│       ├── detector.js        # Webcam streaming, EMA filter, HUD canvas, & CSV export
│       └── main.js            # Tooltip init, password toggles, & image fallbacks
│
├── templates/
│   ├── base.html              # Shell layout with dynamic navbar, auth modals, & footer
│   ├── index.html             # Page 1: Home Overview & Emotional Galleries
│   ├── detector.html          # Page 2: Live Detector & Pluggable Sandbox
│   ├── emotions.html          # Page 3: Emotion Psychology & FACS Science Guide
│   ├── model_guide.html       # Page 4: Deep Learning Architecture & Model Blueprint
│   ├── login.html             # Dedicated high-contrast Sign-In page
│   ├── register.html          # Dedicated Registration page
│   └── error.html             # Branded 404/500/413 error templates
│
└── uploads/
    └── models/                # Storage directory for swapped or custom weights
```

---

## 📡 REST API Reference

| Endpoint | Method | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `/api/auth/register` | `POST` | No | Registers a new user account with hashed password (`username`, `email`, `password`). |
| `/api/auth/login` | `POST` | No | Authenticates user credentials and establishes session cookie. |
| `/api/auth_status` | `GET` | No | Checks current visitor session state (`is_authenticated`, `username`). |
| `/api/predict` | `POST` | **Yes (401)** | Accepts base64 image or multipart file; detects faces and runs `emotiondetector.h5`. |
| `/api/model_status` | `GET` | No | Returns active model name, input shape `[48, 48, 1]`, and class mapping. |
| `/api/upload_model` | `POST` | **Yes** | Uploads and swaps `.h5` or `.keras` weights in-memory dynamically. |

---

## 🛡️ Security & Privacy
- **Client-Side Processing Available**: User webcam streams are processed on local host `127.0.0.1:5000`.
- **Password Protection**: Passwords are never stored in plaintext; hashed via Werkzeug PBKDF2:SHA256 with cryptographically random salts.
- **SQL Injection Prevention**: SQLite queries strictly utilize parameterized placeholders (`?`).
- **File Upload Safeguards**: Multi-layer validation enforces `.h5` / `.keras` extensions and a 32 MB payload ceiling.

---

## 📄 License
This project is open-source under the MIT License.
