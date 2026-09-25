document.addEventListener('DOMContentLoaded', () => {
  // Load saved settings
  chrome.storage.local.get(['serverUrl', 'apiToken'], (result) => {
    if (result.serverUrl) {
      document.getElementById('serverUrl').value = result.serverUrl;
    }
    if (result.apiToken) {
      document.getElementById('apiToken').value = result.apiToken;
    }
  });

  // Save settings
  document.getElementById('saveBtn').addEventListener('click', () => {
    const serverUrl = document.getElementById('serverUrl').value.trim();
    const apiToken = document.getElementById('apiToken').value.trim();
    const statusDiv = document.getElementById('status');

    chrome.storage.local.set({ serverUrl, apiToken }, () => {
      statusDiv.style.color = '#10b981';
      statusDiv.textContent = 'Settings saved successfully!';
      setTimeout(() => {
        statusDiv.textContent = '';
      }, 2000);
    });
  });
});
