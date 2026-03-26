'use strict';

// ── Credential helpers ──────────────────────────────────────────────────────

function getCredentials() {
  return sessionStorage.getItem('admin_auth');
}

function saveCredentials(username, password) {
  // Restrict to ASCII (0–127): btoa accepts Latin-1 (0–255) without throwing,
  // but the backend decodes credentials as UTF-8, so bytes 128–255 cause a 401.
  if (!/^[\x00-\x7F]*$/.test(username + ':' + password)) {
    throw new Error('Username and password must contain only ASCII characters.');
  }
  sessionStorage.setItem('admin_auth', btoa(username + ':' + password));
}

function clearCredentials() {
  sessionStorage.removeItem('admin_auth');
}

function buildAuthHeader(auth) {
  return 'Basic ' + auth;
}

// ── Login modal ─────────────────────────────────────────────────────────────

const loginModal   = document.getElementById('loginModal');
const loginForm    = document.getElementById('loginForm');
const loginError   = document.getElementById('loginError');
const usernameInput = document.getElementById('usernameInput');
const passwordInput = document.getElementById('passwordInput');

let pendingAuthResolve = null;

function showLoginModal() {
  loginError.classList.add('hidden');
  loginForm.reset();
  loginModal.classList.remove('hidden');
  usernameInput.focus();
  return new Promise((resolve) => {
    pendingAuthResolve = resolve;
  });
}

function hideLoginModal() {
  loginModal.classList.add('hidden');
}

loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  if (!username || !password) return;

  // Test credentials with a lightweight request
  // Restrict to ASCII (0–127): btoa accepts Latin-1 without throwing but the
  // backend decodes as UTF-8, so bytes 128–255 would cause a silent 401.
  if (!/^[\x00-\x7F]*$/.test(username + ':' + password)) {
    loginError.textContent = 'Username and password must contain only ASCII characters.';
    loginError.classList.remove('hidden');
    return;
  }
  const encoded = btoa(username + ':' + password);
  const resp = await fetch('/api/admin/documents', {
    headers: { Authorization: 'Basic ' + encoded }
  });

  if (resp.ok) {
    saveCredentials(username, password);
    hideLoginModal();
    if (pendingAuthResolve) {
      pendingAuthResolve(true);
      pendingAuthResolve = null;
    }
  } else if (resp.status === 401) {
    loginError.classList.remove('hidden');
  } else {
    // Server error (e.g. 503 ChromaDB unavailable) — credentials validity unknown
    loginError.textContent = 'Server error. Please try again later.';
    loginError.classList.remove('hidden');
  }
});

// ── Auth-aware fetch ────────────────────────────────────────────────────────

/**
 * authFetch(url, options)
 * Adds Authorization header from sessionStorage.
 * On 401: shows login modal and retries once with new credentials.
 */
async function authFetch(url, options = {}) {
  let creds = getCredentials();

  if (!creds) {
    const ok = await showLoginModal();
    if (!ok) return null;
    creds = getCredentials();
  }

  const headers = Object.assign({}, options.headers || {}, {
    Authorization: buildAuthHeader(creds)
  });

  let resp = await fetch(url, Object.assign({}, options, { headers }));

  if (resp.status === 401) {
    clearCredentials();
    const ok = await showLoginModal();
    if (!ok) return null;
    creds = getCredentials();
    const retryHeaders = Object.assign({}, options.headers || {}, {
      Authorization: buildAuthHeader(creds)
    });
    resp = await fetch(url, Object.assign({}, options, { headers: retryHeaders }));
  }

  return resp;
}

// ── Stats helpers ───────────────────────────────────────────────────────────

const statDocuments  = document.getElementById('statDocuments');
const statChunks     = document.getElementById('statChunks');
const statLastIndexed = document.getElementById('statLastIndexed');

function updateStats(docs) {
  const totalChunks = docs.reduce((sum, d) => sum + (d.chunks || 0), 0);
  statDocuments.textContent = docs.length;
  statChunks.textContent    = totalChunks;

  // Use the most recent uploaded_at as last-indexed date
  const dates = docs.map(d => d.uploaded_at).filter(Boolean).sort();
  if (dates.length) {
    const d = new Date(dates[dates.length - 1]);
    statLastIndexed.textContent = d.toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric'
    });
  } else {
    statLastIndexed.textContent = '—';
  }
}

// ── Documents table ─────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

const docTableBody = document.getElementById('docTableBody');
const emptyRow     = document.getElementById('emptyRow');

function renderDocuments(docs) {
  // Remove all non-empty rows
  const existing = docTableBody.querySelectorAll('tr:not(#emptyRow)');
  existing.forEach(r => r.remove());

  if (!docs.length) {
    emptyRow.classList.remove('hidden');
    return;
  }

  emptyRow.classList.add('hidden');

  docs.forEach(doc => {
    const tr = document.createElement('tr');

    const tdName = document.createElement('td');
    tdName.textContent = doc.filename || doc.file_id;
    tdName.title = doc.filename || '';

    const tdSize = document.createElement('td');
    tdSize.textContent = doc.size != null ? formatBytes(doc.size) : '—';

    const tdChunks = document.createElement('td');
    tdChunks.textContent = doc.chunks != null ? doc.chunks : '—';

    const tdStatus = document.createElement('td');
    const badge = document.createElement('span');
    const isIndexed = (doc.chunks || 0) > 0;
    badge.className = 'status-badge ' + (isIndexed ? 'indexed' : 'not-indexed');
    badge.textContent = isIndexed ? '● Indexed' : '○ Not indexed';
    tdStatus.appendChild(badge);

    const tdActions = document.createElement('td');
    tdActions.style.textAlign = 'right';
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-icon';
    delBtn.title = 'Delete';
    delBtn.textContent = '🗑';
    delBtn.addEventListener('click', () => deleteDocument(doc.file_id, doc.filename));
    tdActions.appendChild(delBtn);

    tr.append(tdName, tdSize, tdChunks, tdStatus, tdActions);
    docTableBody.appendChild(tr);
  });
}

// ── loadDocuments ────────────────────────────────────────────────────────────

async function loadDocuments() {
  const resp = await authFetch('/api/admin/documents');
  if (!resp) return;

  if (!resp.ok) {
    console.error('Failed to load documents:', resp.status);
    return;
  }

  const docs = await resp.json();
  renderDocuments(docs);
  updateStats(docs);
}

// ── deleteDocument ────────────────────────────────────────────────────────────

async function deleteDocument(fileId, filename) {
  const label = filename || fileId;
  if (!confirm(`Delete "${label}" and all its vectors? This cannot be undone.`)) return;

  const resp = await authFetch('/api/admin/documents/' + encodeURIComponent(fileId), {
    method: 'DELETE'
  });

  if (!resp) return;

  if (resp.ok) {
    await loadDocuments();
  } else {
    const body = await resp.json().catch(() => ({}));
    alert('Delete failed: ' + (body.detail || resp.status));
  }
}

// ── uploadFile ────────────────────────────────────────────────────────────────

const uploadProgress   = document.getElementById('uploadProgress');
const progressBarFill  = document.getElementById('progressBarFill');
const uploadStatusText = document.getElementById('uploadStatusText');

async function uploadFile(file) {
  const allowed = ['.pdf', '.docx'];
  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
  if (!allowed.includes(ext)) {
    alert('Only PDF and DOCX files are supported.');
    return;
  }

  uploadProgress.classList.remove('hidden');
  progressBarFill.style.width = '0%';
  uploadStatusText.textContent = 'Uploading…';

  // Animate progress to 60% while waiting (indeterminate)
  let pct = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + 4, 60);
    progressBarFill.style.width = pct + '%';
  }, 150);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await authFetch('/api/admin/upload', { method: 'POST', body: formData });
    clearInterval(ticker);

    if (!resp) {
      uploadProgress.classList.add('hidden');
      return;
    }

    if (resp.ok) {
      progressBarFill.style.width = '100%';
      const data = await resp.json();
      uploadStatusText.textContent = `Done — ${data.chunks_created} chunks created.`;
      setTimeout(() => uploadProgress.classList.add('hidden'), 2000);
      await loadDocuments();
    } else {
      progressBarFill.style.width = '0%';
      const body = await resp.json().catch(() => ({}));
      uploadStatusText.textContent = 'Upload failed: ' + (body.detail || resp.status);
      setTimeout(() => uploadProgress.classList.add('hidden'), 3000);
    }
  } catch (err) {
    clearInterval(ticker);
    uploadStatusText.textContent = 'Upload failed: ' + err.message;
    setTimeout(() => uploadProgress.classList.add('hidden'), 3000);
  }
}

// ── reindexAll ────────────────────────────────────────────────────────────────

const reindexBtn      = document.getElementById('reindexBtn');
const reindexProgress = document.getElementById('reindexProgress');
const reindexStatus   = document.getElementById('reindexStatus');

async function reindexAll() {
  if (!confirm('Rebuild the entire index? All existing vectors will be deleted and re-created from files on disk.')) return;

  reindexBtn.disabled = true;
  reindexProgress.classList.remove('hidden');
  reindexStatus.textContent = 'Rebuilding index…';

  try {
    const resp = await authFetch('/api/admin/reindex', { method: 'POST' });

    if (!resp) {
      reindexBtn.disabled = false;
      reindexProgress.classList.add('hidden');
      return;
    }

    if (resp.ok) {
      const data = await resp.json();
      let msg = `Done — ${data.files_processed} files, ${data.total_chunks} chunks (${data.duration_seconds.toFixed(1)}s)`;
      if (data.failed_files > 0) {
        msg += ` — WARNING: ${data.failed_files} file(s) failed to ingest (check server logs)`;
      }
      reindexStatus.textContent = msg;
      setTimeout(() => reindexProgress.classList.add('hidden'), data.failed_files > 0 ? 6000 : 3000);
      await loadDocuments();
    } else {
      const body = await resp.json().catch(() => ({}));
      reindexStatus.textContent = 'Reindex failed: ' + (body.detail || resp.status);
      setTimeout(() => reindexProgress.classList.add('hidden'), 3000);
    }
  } catch (err) {
    reindexStatus.textContent = 'Reindex failed: ' + err.message;
    setTimeout(() => reindexProgress.classList.add('hidden'), 3000);
  } finally {
    reindexBtn.disabled = false;
  }
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────

const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    uploadFile(fileInput.files[0]);
    fileInput.value = '';
  }
});

// ── Refresh and reindex buttons ───────────────────────────────────────────────

document.getElementById('refreshBtn').addEventListener('click', loadDocuments);
reindexBtn.addEventListener('click', reindexAll);

// ── Init ──────────────────────────────────────────────────────────────────────

(async function init() {
  // If credentials already in session, load immediately; otherwise modal will appear
  await loadDocuments();
})();
