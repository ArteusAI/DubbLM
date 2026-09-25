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

// Listen to request from the webpage to sync cookies
window.addEventListener('DUBBLM_SYNC_COOKIES', (event) => {
  const serverUrl = event.detail?.serverUrl || window.location.origin;
  const apiToken = event.detail?.apiToken || '';
  chrome.runtime.sendMessage({ action: 'sync_cookies', serverUrl, apiToken }, (response) => {
    if (response?.success) {
      window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_SUCCESS'));
    } else {
      window.dispatchEvent(new CustomEvent('DUBBLM_SYNC_ERROR', { detail: { error: response?.error } }));
    }
  });
});
