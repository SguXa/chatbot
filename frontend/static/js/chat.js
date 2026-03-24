(function () {
  'use strict';

  const messagesEl = document.getElementById('messages');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('questionInput');
  const sendBtn = document.getElementById('sendBtn');
  const typingIndicator = document.getElementById('typingIndicator');
  const statusDot = document.getElementById('statusDot');
  const statusLabel = document.getElementById('statusLabel');

  // ── Health check ────────────────────────────────────────────────────────────

  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('non-2xx');
      const data = await res.json();

      // Update app name from health response if present
      if (data.app_name) {
        const appNameEl = document.querySelector('.app-name');
        if (appNameEl) appNameEl.textContent = data.app_name;
        document.title = data.app_name;
      }

      const ok = data.status === 'ok';
      statusDot.className = 'status-dot ' + (ok ? 'ok' : 'err');
      statusLabel.textContent = ok ? 'Online' : 'Degraded';
    } catch (_) {
      statusDot.className = 'status-dot err';
      statusLabel.textContent = 'Offline';
    }
  }

  checkHealth();

  // ── Helpers ─────────────────────────────────────────────────────────────────

  function setInputDisabled(disabled) {
    input.disabled = disabled;
    sendBtn.disabled = disabled;
  }

  function showTyping() {
    typingIndicator.classList.remove('hidden');
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function hideTyping() {
    typingIndicator.classList.add('hidden');
  }

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message user';
    msg.innerHTML = `
      <div class="message-avatar">You</div>
      <div class="message-body">
        <div class="bubble">${escapeHtml(text)}</div>
      </div>
    `;
    messagesEl.appendChild(msg);
    scrollBottom();
  }

  function createBotMessage() {
    const msg = document.createElement('div');
    msg.className = 'message bot';
    msg.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-body">
        <div class="bubble"></div>
        <div class="sources"></div>
      </div>
    `;
    messagesEl.appendChild(msg);
    scrollBottom();
    return {
      bubble: msg.querySelector('.bubble'),
      sources: msg.querySelector('.sources'),
    };
  }

  function appendErrorMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message bot error';
    msg.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-body">
        <div class="bubble">${escapeHtml(text)}</div>
      </div>
    `;
    messagesEl.appendChild(msg);
    scrollBottom();
  }

  function renderSources(sourcesEl, sources) {
    if (!sources || sources.length === 0) return;
    sources.forEach(function (src) {
      const pill = document.createElement('span');
      pill.className = 'source-pill';
      const label = src.page
        ? `${src.filename} · p.${src.page}`
        : src.filename;
      pill.textContent = label;
      sourcesEl.appendChild(pill);
    });
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ── SSE streaming via fetch + ReadableStream ────────────────────────────────

  async function sendMessage(question) {
    setInputDisabled(true);
    showTyping();

    let botMessage = null;

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question }),
      });

      if (!res.ok) {
        const errText = await res.text();
        hideTyping();
        appendErrorMessage('Error ' + res.status + ': ' + errText);
        return;
      }

      hideTyping();
      botMessage = createBotMessage();

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE lines from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;

          let event;
          try {
            event = JSON.parse(payload);
          } catch (_) {
            continue;
          }

          if (event.error) {
            botMessage.bubble.textContent = 'Error: ' + event.error;
            botMessage.bubble.classList.add('error-text');
            scrollBottom();
          } else if (event.token) {
            botMessage.bubble.textContent += event.token;
            scrollBottom();
          }

          if (event.done && !event.error) {
            renderSources(botMessage.sources, event.sources);
            scrollBottom();
          }
        }
      }

      // Flush remaining buffer
      if (buffer.startsWith('data:')) {
        const payload = buffer.slice(5).trim();
        if (payload) {
          try {
            const event = JSON.parse(payload);
            if (event.error) { botMessage.bubble.textContent = 'Error: ' + event.error; }
            else if (event.token) botMessage.bubble.textContent += event.token;
            if (event.done && !event.error) renderSources(botMessage.sources, event.sources);
            scrollBottom();
          } catch (_) { /* ignore */ }
        }
      }

    } catch (err) {
      hideTyping();
      if (!botMessage) {
        appendErrorMessage('Could not reach the server. Please try again.');
      } else {
        botMessage.bubble.textContent += '\n[Connection lost]';
      }
    } finally {
      setInputDisabled(false);
      input.focus();
    }
  }

  // ── Form submit ─────────────────────────────────────────────────────────────

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = '';
    appendUserMessage(question);
    sendMessage(question);
  });

  // Focus input on load
  input.focus();
})();
