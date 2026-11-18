

document.addEventListener('DOMContentLoaded', function () {
  
  const webcamVideo = document.getElementById('webcamVideo');
  const cameraOffPlaceholder = document.getElementById('cameraOffPlaceholder');
  const cameraErrorOverlay = document.getElementById('cameraErrorOverlay');
  const retryCameraBtn = document.getElementById('retryCameraBtn');
  const switchToUploadBtn = document.getElementById('switchToUploadBtn');
  const uploadedImagePreview = document.getElementById('uploadedImagePreview');
  const detectionCanvas = document.getElementById('detectionCanvas');
  
  const startCameraBtn = document.getElementById('startCameraBtn');
  const stopCameraBtn = document.getElementById('stopCameraBtn');
  const quickStartBtn = document.getElementById('quickStartBtn');
  const snapshotBtn = document.getElementById('snapshotBtn');
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  
  
  const detectionSpeedSelect = document.getElementById('detectionSpeedSelect');
  const smoothingToggle = document.getElementById('smoothingToggle');
  
  
  const cameraStatusBadge = document.getElementById('cameraStatusBadge');
  const liveStateIndicator = document.getElementById('liveStateIndicator');
  const liveStateText = document.getElementById('liveStateText');
  const latencyBadge = document.getElementById('latencyBadge');
  const facesCountBadge = document.getElementById('facesCountBadge');
  
  
  const webcamTabBtn = document.getElementById('webcam-tab');
  const uploadTabBtn = document.getElementById('upload-tab');
  const webcamContainer = document.getElementById('webcamContainer');
  const uploadContainer = document.getElementById('uploadContainer');
  const uploadActionButtons = document.getElementById('uploadActionButtons');
  
  // Image Upload Inputs & Buttons
  const imageFileInput = document.getElementById('imageFileInput');
  const dropzoneBox = document.getElementById('dropzoneBox');
  const detectUploadedImageBtn = document.getElementById('detectUploadedImageBtn');
  const clearImageBtn = document.getElementById('clearImageBtn');
  
  // Model Upload Inputs
  const modelFileInput = document.getElementById('modelFileInput');
  const uploadModelBtn = document.getElementById('uploadModelBtn');
  const modelUploadAlert = document.getElementById('modelUploadAlert');
  const currentModelBadge = document.getElementById('currentModelBadge');
  
  // Detection Outcome Displays
  const dominantEmoji = document.getElementById('dominantEmoji');
  const dominantEmotionText = document.getElementById('dominantEmotionText');
  const dominantConfidence = document.getElementById('dominantConfidence');
  const activeEngineBadge = document.getElementById('activeEngineBadge');
  const historyTableBody = document.getElementById('historyTableBody');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  
  // Emotion Meter Bars
  const emotionsList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
  const emotionBars = {};
  const emotionValues = {};
  emotionsList.forEach(emotion => {
    emotionBars[emotion] = document.getElementById(`bar-${emotion}`);
    emotionValues[emotion] = document.getElementById(`val-${emotion}`);
  });

  // =========================================================================
  // STATE MANAGEMENT & CONFIGURATION
  // =========================================================================
  let stream = null;
  let isStreaming = false;
  let detectionInterval = null;
  let isProcessingFrame = false;
  let currentMode = 'webcam'; // 'webcam' or 'upload'
  let detectionHistory = [];
  let currentSelectedFile = null;

  // Slower default detection rate: 1000ms (1 frame every 1.0 second)
  let detectionDelayMs = detectionSpeedSelect ? parseInt(detectionSpeedSelect.value, 10) : 1000;

  // Temporal Smoothing (EMA) state to eliminate flickering / erratic jumping
  let smoothedProbabilities = null;
  const smoothingAlpha = 0.38; // Weight for new frame (0.38) vs historical memory (0.62)
  let lastStableEmotion = null;
  let lastEmotionStartTime = null;
  let currentEmotionCount = 0;

  // Initialize stop button state
  if (stopCameraBtn) {
    stopCameraBtn.classList.add('opacity-50');
    stopCameraBtn.disabled = true;
  }

  // =========================================================================
  // 1. AUTHENTICATION GATE CHECK
  // =========================================================================
  function isUserAuthenticated() {
    const authGate = document.getElementById('detectorAuthGate');
    return authGate && authGate.dataset.authenticated === 'true';
  }

  function requireLogin(message) {
    const loginModalEl = document.getElementById('loginModal');
    if (loginModalEl && window.bootstrap) {
      const alertEl = document.getElementById('modalLoginAlert') || document.getElementById('loginModalAlert');
      if (alertEl && message) {
        alertEl.textContent = message;
        alertEl.classList.remove('d-none');
      }
      const modal = window.bootstrap.Modal.getOrCreateInstance(loginModalEl);
      modal.show();
    } else {
      window.location.href = '/login?next=/detector';
    }
  }

  // =========================================================================
  // 2. WEBCAM STREAMING & DETECTION CONTROLS (START & STOP)
  // =========================================================================

  async function startDetection() {
    if (!isUserAuthenticated()) {
      requireLogin('Please log in or register an account to start the camera and detect emotions.');
      return;
    }

    try {
      updateLiveState('starting', 'Initializing Camera & Model...');

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      webcamVideo.srcObject = stream;
      await webcamVideo.play();

      isStreaming = true;

      // Reset temporal smoothing memory on fresh start
      smoothedProbabilities = null;
      lastStableEmotion = null;
      currentEmotionCount = 0;

      // UI updates
      if (cameraErrorOverlay) cameraErrorOverlay.classList.add('d-none');
      if (cameraOffPlaceholder) cameraOffPlaceholder.classList.add('d-none');
      webcamVideo.classList.remove('d-none');

      startCameraBtn.classList.add('d-none');
      stopCameraBtn.classList.remove('opacity-50');
      stopCameraBtn.disabled = false;

      updateLiveState('active', `Live Detecting (${(detectionDelayMs / 1000).toFixed(1)}s rate)`);

      syncCanvasSize(webcamVideo);

      // Start the paced detection loop
      startDetectionLoop();
    } catch (err) {
      console.error('Camera access error:', err);
      updateLiveState('error', 'Camera Blocked / Not Found');
      if (cameraErrorOverlay) {
        cameraErrorOverlay.classList.remove('d-none');
        if (cameraOffPlaceholder) cameraOffPlaceholder.classList.add('d-none');
        webcamVideo.classList.add('d-none');
      } else {
        alert('Could not access your webcam. Please allow camera permissions in your browser or use the "Upload Photo" tab to test portrait images.');
      }
    }
  }

  function stopDetection() {
    // 1. Clear detection loop timer
    if (detectionInterval) {
      clearInterval(detectionInterval);
      detectionInterval = null;
    }

    // 2. Stop camera tracks
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    isStreaming = false;
    isProcessingFrame = false;
    webcamVideo.srcObject = null;
    webcamVideo.classList.add('d-none');

    // 3. Reset smoothing buffers
    smoothedProbabilities = null;
    lastStableEmotion = null;

    // 4. Show placeholder
    if (cameraErrorOverlay) cameraErrorOverlay.classList.add('d-none');
    if (cameraOffPlaceholder) cameraOffPlaceholder.classList.remove('d-none');

    // 5. Update Buttons
    startCameraBtn.classList.remove('d-none');
    stopCameraBtn.classList.add('opacity-50');
    stopCameraBtn.disabled = true;

    // 6. Update Status
    updateLiveState('stopped', 'Detection Stopped');
    if (latencyBadge) latencyBadge.textContent = '-- ms';
    if (facesCountBadge) facesCountBadge.textContent = '0 detected';

    clearCanvas();
  }

  function startDetectionLoop() {
    if (detectionInterval) clearInterval(detectionInterval);
    // Runs at the selected, comfortable pace (e.g. 1000ms / 1.0 second)
    detectionInterval = setInterval(() => {
      if (isStreaming && !isProcessingFrame) {
        captureAndPredictWebcamFrame();
      }
    }, detectionDelayMs);
  }

  // Handle changing detection speed on the fly without stopping camera
  if (detectionSpeedSelect) {
    detectionSpeedSelect.addEventListener('change', () => {
      detectionDelayMs = parseInt(detectionSpeedSelect.value, 10);
      if (isStreaming) {
        updateLiveState('active', `Live Detecting (${(detectionDelayMs / 1000).toFixed(1)}s rate)`);
        startDetectionLoop();
      }
    });
  }

  function updateLiveState(state, message) {
    if (!liveStateText || !liveStateIndicator || !cameraStatusBadge) return;

    if (state === 'active') {
      liveStateIndicator.className = 'badge rounded-circle bg-success pulse-active';
      liveStateText.textContent = message || 'Live Detecting';
      liveStateText.className = 'small text-success fw-bold';
      cameraStatusBadge.className = 'badge bg-success';
      cameraStatusBadge.textContent = 'Active';
    } else if (state === 'starting') {
      liveStateIndicator.className = 'badge rounded-circle bg-warning';
      liveStateText.textContent = message || 'Connecting...';
      liveStateText.className = 'small text-warning fw-bold';
      cameraStatusBadge.className = 'badge bg-warning text-dark';
      cameraStatusBadge.textContent = 'Starting...';
    } else if (state === 'error') {
      liveStateIndicator.className = 'badge rounded-circle bg-danger';
      liveStateText.textContent = message || 'Error';
      liveStateText.className = 'small text-danger fw-bold';
      cameraStatusBadge.className = 'badge bg-danger';
      cameraStatusBadge.textContent = 'Error';
    } else {
      liveStateIndicator.className = 'badge rounded-circle bg-secondary';
      liveStateText.textContent = message || 'Detection Stopped';
      liveStateText.className = 'small text-white fw-bold';
      cameraStatusBadge.className = 'badge bg-secondary text-white';
      cameraStatusBadge.textContent = 'Stopped';
    }
  }

  function syncCanvasSize(targetElement) {
    if (!targetElement) return;
    const rect = targetElement.getBoundingClientRect();
    detectionCanvas.width = targetElement.videoWidth || targetElement.naturalWidth || rect.width || 640;
    detectionCanvas.height = targetElement.videoHeight || targetElement.naturalHeight || rect.height || 480;
  }

  function clearCanvas() {
    const ctx = detectionCanvas.getContext('2d');
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
  }

  async function captureAndPredictWebcamFrame() {
    if (!isStreaming || webcamVideo.paused || webcamVideo.ended) return;

    isProcessingFrame = true;
    try {
      syncCanvasSize(webcamVideo);

      const offscreenCanvas = document.createElement('canvas');
      offscreenCanvas.width = webcamVideo.videoWidth || 640;
      offscreenCanvas.height = webcamVideo.videoHeight || 480;
      const offCtx = offscreenCanvas.getContext('2d');
      offCtx.drawImage(webcamVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

      const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.82);

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });

      if (response.status === 401) {
        stopDetection();
        requireLogin('Your session has expired or login is required. Please log in to detect emotions.');
        return;
      }

      if (!response.ok) throw new Error('Prediction API error');
      const data = await response.json();
      if (data.status === 'unauthorized') {
        stopDetection();
        requireLogin(data.message || 'Please log in to detect emotions.');
        return;
      }

      handlePredictionResponse(data, webcamVideo, true);
    } catch (err) {
      console.warn('Frame detection skipped/error:', err);
    } finally {
      isProcessingFrame = false;
    }
  }

  // =========================================================================
  // 2. TEMPORAL SMOOTHING & SMART PREDICTION HANDLER (ANTI-FLICKER)
  // =========================================================================

  function handlePredictionResponse(data, sourceElement, isLiveStream = false) {
    if (data.status !== 'success') return;

    if (latencyBadge) latencyBadge.textContent = `${data.processing_time_ms} ms`;
    if (facesCountBadge) facesCountBadge.textContent = `${data.faces_count} detected`;

    if (!data.predictions || data.predictions.length === 0) {
      clearCanvas();
      return;
    }

    const primaryFace = data.predictions[0];
    const rawProbs = primaryFace.probabilities;
    const useSmoothing = smoothingToggle ? smoothingToggle.checked : true;

    // Apply Exponential Moving Average (EMA) Temporal Smoothing
    let finalProbs = { ...rawProbs };

    if (isLiveStream && useSmoothing) {
      if (!smoothedProbabilities) {
        smoothedProbabilities = { ...rawProbs };
      } else {
        // Blend new frame (38%) with historical memory (62%) to eliminate momentary twitches
        for (const emotion of emotionsList) {
          const currentVal = rawProbs[emotion] || 0;
          const prevVal = smoothedProbabilities[emotion] || 0;
          smoothedProbabilities[emotion] = (smoothingAlpha * currentVal) + ((1 - smoothingAlpha) * prevVal);
        }
      }
      finalProbs = { ...smoothedProbabilities };
    } else {
      // For single image upload or when smoothing is off, use raw probabilities
      smoothedProbabilities = null;
    }

    // Determine dominant emotion from smoothed probabilities
    let dominantEmotion = primaryFace.dominant;
    let maxVal = -1;
    for (const emotion of emotionsList) {
      if (finalProbs[emotion] > maxVal) {
        maxVal = finalProbs[emotion];
        dominantEmotion = emotion;
      }
    }

    // Hysteresis stability check: If close tie with previous stable emotion, stay stable
    if (isLiveStream && lastStableEmotion && dominantEmotion !== lastStableEmotion) {
      const prevScore = finalProbs[lastStableEmotion] || 0;
      const newScore = finalProbs[dominantEmotion] || 0;
      // Require at least a 6% clear lead to switch emotions
      if (newScore - prevScore < 6.0) {
        dominantEmotion = lastStableEmotion;
        maxVal = prevScore;
      }
    }

    // Update primary face data with smoothed values
    primaryFace.dominant = dominantEmotion;
    primaryFace.confidence = parseFloat(maxVal.toFixed(1));
    primaryFace.probabilities = finalProbs;

    // Draw stable bounding boxes on canvas
    drawBoundingBoxes([primaryFace], sourceElement);

    // Update Spotlight Card
    if (dominantEmoji) dominantEmoji.textContent = primaryFace.emoji;
    if (dominantEmotionText) {
      dominantEmotionText.textContent = primaryFace.dominant;
      dominantEmotionText.style.color = primaryFace.color;
    }
    if (dominantConfidence) {
      dominantConfidence.textContent = `${primaryFace.confidence}% Confidence`;
      dominantConfidence.className = `badge ${primaryFace.confidence > 70 ? 'bg-success' : 'bg-warning text-dark'} mt-1`;
    }
    if (activeEngineBadge) {
      activeEngineBadge.textContent = primaryFace.engine;
    }

   
    emotionsList.forEach(emotion => {
      const val = finalProbs[emotion] || 0;
      if (emotionBars[emotion]) {
        emotionBars[emotion].style.width = `${Math.min(100, Math.max(0, val))}%`;
      }
      if (emotionValues[emotion]) {
        emotionValues[emotion].textContent = `${val.toFixed(1)}%`;
      }
    });

    // Smart Deduplicated History Logging (No duplicate spamming!)
    updateSmartHistory(primaryFace, data.processing_time_ms);
  }

  // Expose to window for quick testing helpers
  window.handlePredictionResponse = handlePredictionResponse;

 



  function updateSmartHistory(face, latency) {
    if (!historyTableBody) return;
    const now = new Date();
    const timeStr = now.toLocaleTimeString();

    if (lastStableEmotion === face.dominant && detectionHistory.length > 0) {
      // SAME emotion: Increment sustained duration counter
      currentEmotionCount++;
      const topRecord = detectionHistory[0];
      const secondsHeld = Math.max(1, Math.round((now - lastEmotionStartTime) / 1000));
      
      topRecord.confidence = face.confidence;
      topRecord.latency = latency;
      topRecord.heldSeconds = secondsHeld;
      topRecord.count = currentEmotionCount;
    } else {
      // NEW emotion detected: Record fresh row!
      lastStableEmotion = face.dominant;
      lastEmotionStartTime = now;
      currentEmotionCount = 1;

      detectionHistory.unshift({
        time: timeStr,
        emotion: face.dominant,
        emoji: face.emoji,
        confidence: face.confidence,
        latency: latency,
        heldSeconds: 1,
        count: 1
      });

      if (detectionHistory.length > 7) {
        detectionHistory.pop();
      }
    }

    // Render table rows
    renderHistoryTable();
  }

  function renderHistoryTable() {
    if (!historyTableBody) return;

    if (detectionHistory.length === 0) {
      historyTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-2">Click "Start Detection" to stream live results.</td></tr>';
      return;
    }

    historyTableBody.innerHTML = detectionHistory.map((item, idx) => {
      const isTopActive = (idx === 0 && isStreaming);
      const durationBadge = item.heldSeconds > 2 
        ? `<span class="badge border ms-1" style="background: rgba(99, 102, 241, 0.25); color: #e0e7ff; border-color: rgba(99, 102, 241, 0.5); font-size: 0.7rem;">Held ${item.heldSeconds}s</span>` 
        : '';

      return `
        <tr class="${isTopActive ? 'table-active' : ''}">
          <td class="text-light"><small class="fw-semibold">${item.time}</small></td>
          <td>
            <span class="fw-bold text-white">${item.emoji} ${item.emotion}</span>
            ${durationBadge}
          </td>
          <td>
            <span class="badge ${item.confidence > 70 ? 'bg-success text-white' : 'bg-warning text-dark'} fw-bold">
              ${item.confidence}%
            </span>
          </td>
          <td class="text-light"><small class="text-light">${item.latency} ms</small></td>
        </tr>
      `;
    }).join('');
  }

  // =========================================================================
  // 4. CANVAS DRAWING (BOUNDING BOXES & EMOTION BADGES)
  // =========================================================================

  function drawBoundingBoxes(predictions, sourceElement) {
    const ctx = detectionCanvas.getContext('2d');
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);

    if (!predictions || predictions.length === 0) return;

    predictions.forEach(face => {
      const [x, y, w, h] = face.box;
      const color = face.color || '#10b981';
      const label = `${face.emoji} ${face.dominant} (${face.confidence}%)`;

      // Bounding box border
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Corner accent brackets
      const cornerLength = Math.min(w, h) * 0.22;
      ctx.lineWidth = 5;
      
      // Top-Left corner
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLength);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLength, y);
      ctx.stroke();

      // Top-Right corner
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLength, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLength);
      ctx.stroke();

      // Bottom-Left corner
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLength);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerLength, y + h);
      ctx.stroke();

      // Bottom-Right corner
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLength, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerLength);
      ctx.stroke();

      // Emotion banner above bounding box
      ctx.font = 'bold 15px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      const badgeY = Math.max(26, y - 10);

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, badgeY - 22, textWidth + 18, 26, 6);
      ctx.fill();

      // Text label inside banner
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 9, badgeY - 5);
    });
  }

  // =========================================================================
  // 5. IMAGE UPLOAD & ANALYZER
  // =========================================================================

  if (dropzoneBox && imageFileInput) {
    dropzoneBox.addEventListener('click', () => imageFileInput.click());

    dropzoneBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzoneBox.classList.add('drag-over');
    });

    dropzoneBox.addEventListener('dragleave', () => {
      dropzoneBox.classList.remove('drag-over');
    });

    dropzoneBox.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzoneBox.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        imageFileInput.files = e.dataTransfer.files;
        handleSelectedImage(imageFileInput.files[0]);
      }
    });

    imageFileInput.addEventListener('change', () => {
      if (imageFileInput.files && imageFileInput.files.length > 0) {
        handleSelectedImage(imageFileInput.files[0]);
      }
    });
  }

  function handleSelectedImage(file) {
    currentSelectedFile = file;
    const reader = new FileReader();
    reader.onload = function (e) {
      uploadedImagePreview.src = e.target.result;
      uploadedImagePreview.classList.remove('d-none');
      if (uploadActionButtons) uploadActionButtons.classList.remove('d-none');

      uploadedImagePreview.onload = () => {
        syncCanvasSize(uploadedImagePreview);
        if (isUserAuthenticated()) {
          analyzeUploadedImage(file);
        } else {
          requireLogin('Photo loaded! Please log in or register to detect facial emotions.');
        }
      };
    };
    reader.readAsDataURL(file);
  }

  async function analyzeUploadedImage(file) {
    if (!file) return;
    if (!isUserAuthenticated()) {
      requireLogin('Please log in or register an account to detect emotions.');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);

    try {
      if (detectUploadedImageBtn) {
        detectUploadedImageBtn.disabled = true;
        detectUploadedImageBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Detecting...';
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (response.status === 401) {
        requireLogin('Please log in or register to detect emotions.');
        return;
      }

      const data = await response.json();
      if (data.status === 'unauthorized') {
        requireLogin(data.message || 'Please log in to detect emotions.');
        return;
      }

      syncCanvasSize(uploadedImagePreview);
      // isLiveStream=false so image upload gets exact instantaneous prediction
      handlePredictionResponse(data, uploadedImagePreview, false);
    } catch (err) {
      console.error('Error analyzing image:', err);
      alert('Could not analyze the image. Please try again.');
    } finally {
      if (detectUploadedImageBtn) {
        detectUploadedImageBtn.disabled = false;
        detectUploadedImageBtn.innerHTML = '<i class="bi bi-magic me-1"></i> Detect Emotion in Image';
      }
    }
  }

  if (detectUploadedImageBtn) {
    detectUploadedImageBtn.addEventListener('click', () => {
      if (!isUserAuthenticated()) {
        requireLogin('Please log in or register to detect emotions.');
        return;
      }
      if (currentSelectedFile) {
        analyzeUploadedImage(currentSelectedFile);
      }
    });
  }

  if (clearImageBtn) {
    clearImageBtn.addEventListener('click', () => {
      uploadedImagePreview.src = '';
      uploadedImagePreview.classList.add('d-none');
      if (uploadActionButtons) uploadActionButtons.classList.add('d-none');
      if (imageFileInput) imageFileInput.value = '';
      currentSelectedFile = null;
      clearCanvas();
    });
  }

  // =========================================================================
  // 6. SINGLE SNAPSHOT DETECT ONCE
  // =========================================================================

  if (snapshotBtn) {
    snapshotBtn.addEventListener('click', async () => {
      if (!isUserAuthenticated()) {
        requireLogin('Please log in or register to detect emotions.');
        return;
      }
      if (currentMode === 'webcam') {
        if (!isStreaming) {
          // If camera not yet started, start it first
          await startDetection();
          // Give camera 500ms to warm up then take one frame
          setTimeout(() => {
            captureAndPredictWebcamFrame();
          }, 500);
        } else {
          // Take single frame
          captureAndPredictWebcamFrame();
        }
      } else if (currentMode === 'upload' && currentSelectedFile) {
        analyzeUploadedImage(currentSelectedFile);
      }
    });
  }

  // =========================================================================
  // 7. CUSTOM MODEL SWAP HANDLER
  // =========================================================================

  if (uploadModelBtn && modelFileInput) {
    uploadModelBtn.addEventListener('click', async () => {
      if (!modelFileInput.files || modelFileInput.files.length === 0) {
        showModelAlert('Please select a .h5 or .keras model file first.', 'warning');
        return;
      }

      const file = modelFileInput.files[0];
      const formData = new FormData();
      formData.append('model_file', file);

      uploadModelBtn.disabled = true;
      uploadModelBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Loading...';

      try {
        const response = await fetch('/api/upload_model', {
          method: 'POST',
          body: formData
        });

        const res = await response.json();
        if (response.ok && res.status === 'success') {
          showModelAlert(`🎉 Success: ${res.message} Input Shape: [${res.input_shape.join(', ')}]`, 'success');
          if (currentModelBadge) {
            currentModelBadge.className = 'badge bg-success';
            currentModelBadge.textContent = `Active: ${res.model_name}`;
          }
        } else {
          showModelAlert(`❌ Error: ${res.message || 'Failed to load model'}`, 'danger');
        }
      } catch (err) {
        console.error('Error uploading model:', err);
        showModelAlert('Failed to connect to model server.', 'danger');
      } finally {
        uploadModelBtn.disabled = false;
        uploadModelBtn.innerHTML = '<i class="bi bi-upload me-1"></i> Swap Model';
      }
    });
  }

  function showModelAlert(msg, type) {
    if (!modelUploadAlert) return;
    modelUploadAlert.className = `alert alert-${type} alert-dismissible fade show mt-2 mb-0`;
    modelUploadAlert.innerHTML = `
      ${msg}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    modelUploadAlert.classList.remove('d-none');
  }

  // =========================================================================
  // 8. EVENT LISTENERS & TAB SWITCHING
  // =========================================================================

  if (startCameraBtn) startCameraBtn.addEventListener('click', startDetection);
  if (quickStartBtn) quickStartBtn.addEventListener('click', startDetection);
  if (stopCameraBtn) stopCameraBtn.addEventListener('click', stopDetection);

  // Camera permission retry and tab switcher
  if (retryCameraBtn) {
    retryCameraBtn.addEventListener('click', () => {
      if (cameraErrorOverlay) cameraErrorOverlay.classList.add('d-none');
      startDetection();
    });
  }

  if (switchToUploadBtn && uploadTabBtn) {
    switchToUploadBtn.addEventListener('click', () => {
      if (cameraErrorOverlay) cameraErrorOverlay.classList.add('d-none');
      uploadTabBtn.click();
    });
  }

  // Clear & Export CSV History
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      detectionHistory = [];
      lastStableEmotion = null;
      renderHistoryTable();
    });
  }

  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      if (!detectionHistory || detectionHistory.length === 0) {
        alert('No inference records to export yet. Start detection or analyze a photo first.');
        return;
      }

      let csv = "Index,Timestamp,Emotion,Emoji,Confidence(%),Latency(ms),DurationHeld(s),SampleCount\r\n";
      detectionHistory.forEach((row, i) => {
        const cleanEmotion = (row.emotion || '').replace(/,/g, '');
        csv += `${i + 1},"${row.time || ''}","${cleanEmotion}","${row.emoji || ''}",${row.confidence || 0},${row.latency || 0},${row.heldSeconds || 1},${row.count || 1}\r\n`;
      });

      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `FaceSense_Emotion_Log_${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    });
  }

  if (webcamTabBtn && uploadTabBtn) {
    webcamTabBtn.addEventListener('click', () => {
      currentMode = 'webcam';
      webcamContainer.classList.remove('d-none');
      uploadContainer.classList.add('d-none');
      clearCanvas();
      if (isStreaming) {
        syncCanvasSize(webcamVideo);
      }
    });

    uploadTabBtn.addEventListener('click', () => {
      currentMode = 'upload';
      if (isStreaming) {
        stopDetection();
      }
      webcamContainer.classList.add('d-none');
      uploadContainer.classList.remove('d-none');
      clearCanvas();
      if (uploadedImagePreview.src && !uploadedImagePreview.classList.contains('d-none')) {
        syncCanvasSize(uploadedImagePreview);
      }
    });
  }

  window.addEventListener('resize', () => {
    if (currentMode === 'webcam' && isStreaming) {
      syncCanvasSize(webcamVideo);
    } else if (currentMode === 'upload' && !uploadedImagePreview.classList.contains('d-none')) {
      syncCanvasSize(uploadedImagePreview);
    }
  });
});
