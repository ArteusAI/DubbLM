// Notify the DubbLM website that the extension is installed
const injectPresence = () => {
  // Add a hidden DOM element
  if (!document.getElementById('dubblm-extension-installed')) {
    const el = document.createElement('div');
    el.id = 'dubblm-extension-installed';
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  // Dispatch custom event
  window.dispatchEvent(new CustomEvent('DUBBLM_EXTENSION_INSTALLED'));
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', injectPresence);
} else {
  injectPresence();
}

// Re-notify if requested
window.addEventListener('DUBBLM_PING_EXTENSION', () => {
  injectPresence();
});

// Listen to request from the webpage to sync cookies.
// The server URL and token are intentionally NOT taken from the page event:
// any website could otherwise ask the extension to upload YouTube cookies to
// an attacker-controlled server. The background worker always uses the
// configured DubbLM server from extension storage.
window.addEventListener('DUBBLM_SYNC_COOKIES', () => {
  chrome.runtime.sendMessage({ action: 'sync_cookies' }, (response) => {
    if (chrome.runtime.lastError) {
      window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_ERROR', {
        detail: { error: chrome.runtime.lastError.message }
      }));
      return;
    }
    if (response?.success) {
      window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_SUCCESS'));
    } else {
      window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_ERROR', { detail: { error: response?.error } }));
    }
  });
});
