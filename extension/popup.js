document.addEventListener('DOMContentLoaded', () => {
  // If no saved values, save the ones from DEFAULT_CONFIG
  chrome.storage.local.get(['serverUrl', 'apiToken'], (result) => {
    let updates = {};
    const defaultServerUrl = (typeof DEFAULT_CONFIG !== 'undefined' && DEFAULT_CONFIG.serverUrl) || 'http://localhost:8000';
    const defaultApiToken = (typeof DEFAULT_CONFIG !== 'undefined' && DEFAULT_CONFIG.apiToken) || '';

    if (!result.serverUrl && defaultServerUrl) {
      updates.serverUrl = defaultServerUrl;
    }
    if (!result.apiToken && defaultApiToken) {
      updates.apiToken = defaultApiToken;
    }

    if (Object.keys(updates).length > 0) {
      chrome.storage.local.set(updates, () => {
        checkConnection(updates.serverUrl || result.serverUrl, updates.apiToken || result.apiToken);
      });
    } else {
      checkConnection(result.serverUrl, result.apiToken);
    }
  });

  const checkConnection = (serverUrl, apiToken) => {
    const statusIndicator = document.getElementById('statusIndicator');
    if (!serverUrl) {
      statusIndicator.textContent = 'Not Configured';
      statusIndicator.style.color = '#fbbf24'; // Yellow
      return;
    }

    const headers = {};
    if (apiToken) {
      headers['Authorization'] = `Bearer ${apiToken}`;
    }

    fetch(`${serverUrl}/health`, { headers })
      .then(response => {
        if (response.ok) {
          statusIndicator.textContent = 'Connected';
          statusIndicator.style.color = '#10b981'; // Green
        } else {
          statusIndicator.textContent = `Error ${response.status}`;
          statusIndicator.style.color = '#ef4444'; // Red
        }
      })
      .catch(err => {
        statusIndicator.textContent = 'Disconnected';
        statusIndicator.style.color = '#ef4444'; // Red
      });
  };

  document.getElementById('syncBtn').addEventListener('click', () => {
    const statusDiv = document.getElementById('status');
    statusDiv.style.color = '#a1a1aa';
    statusDiv.textContent = 'Syncing...';

    chrome.storage.local.get(['serverUrl', 'apiToken'], (result) => {
      const serverUrl = result.serverUrl || 'http://localhost:8000';
      const apiToken = result.apiToken || '';

      chrome.runtime.sendMessage({ action: 'sync_cookies', serverUrl, apiToken }, (response) => {
        if (response && response.success) {
          statusDiv.style.color = '#10b981';
          statusDiv.textContent = '✓ Cookies synced successfully!';
        } else {
          statusDiv.style.color = '#ef4444';
          statusDiv.textContent = '✗ Error: ' + (response ? response.error : 'Unknown error');
        }
      });
    });
  });
});
