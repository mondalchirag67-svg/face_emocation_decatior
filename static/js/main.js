

document.addEventListener('DOMContentLoaded', function () {
  // 1. Initialize Bootstrap Tooltips if available
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.forEach(function (tooltipTriggerEl) {
    if (window.bootstrap) {
      new window.bootstrap.Tooltip(tooltipTriggerEl);
    }
  });

  // 2. Setup Code Snippet Copy Buttons
  const copyButtons = document.querySelectorAll('.btn-copy-code');
  copyButtons.forEach(btn => {
    btn.addEventListener('click', function () {
      const targetId = this.getAttribute('data-target');
      const targetCode = document.getElementById(targetId);
      if (targetCode) {
        navigator.clipboard.writeText(targetCode.innerText).then(() => {
          const originalText = this.innerHTML;
          this.innerHTML = '<i class="bi bi-check2"></i> Copied!';
          this.classList.add('btn-success');
          this.classList.remove('btn-outline-glow');
          setTimeout(() => {
            this.innerHTML = originalText;
            this.classList.remove('btn-success');
            this.classList.add('btn-outline-glow');
          }, 2000);
        });
      }
    });
  });

  // 3. Fallback for broken external images to keep site visually pristine
  document.querySelectorAll('img').forEach(img => {
    img.addEventListener('error', function () {
      // If an external unsplash image is blocked by network, fallback to placeholder
      if (!this.getAttribute('data-fallback-applied')) {
        this.setAttribute('data-fallback-applied', 'true');
        const alt = this.alt || 'Emotion Expression';
        this.src = `https://placehold.co/600x450/1e293b/f8fafc?text=${encodeURIComponent(alt)}`;
      }
    });
  });
});

/**
 * Global helper to toggle password input visibility between 'password' and 'text'.
 */
function togglePasswordVisibility(inputId, btn) {
  const input = document.getElementById(inputId);
  if (!input) return;
  if (input.type === 'password') {
    input.type = 'text';
    if (btn) btn.innerHTML = '<i class="bi bi-eye-slash"></i> Hide';
  } else {
    input.type = 'password';
    if (btn) btn.innerHTML = '<i class="bi bi-eye"></i> Show';
  }
}
window.togglePasswordVisibility = togglePasswordVisibility;

/**
 * Global helper to toggle password input visibility between 'password' and 'text'.
 */
function togglePasswordVisibility(inputId, btn) {
  const input = document.getElementById(inputId);
  if (!input) return;
  if (input.type === 'password') {
    input.type = 'text';
    if (btn) btn.innerHTML = '<i class="bi bi-eye-slash"></i> Hide';
  } else {
    input.type = 'password';
    if (btn) btn.innerHTML = '<i class="bi bi-eye"></i> Show';
  }
}
window.togglePasswordVisibility = togglePasswordVisibility;
