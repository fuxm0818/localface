// 摄像头权限状态与引导：桌面版走 Tauri 命令，浏览器版给浏览器提示
(() => {
  const isDesktop = () => !!window.__TAURI_INTERNALS__;
  const labels = {
    authorized: '已授权',
    denied: '已拒绝',
    restricted: '被系统限制',
    not_determined: '尚未询问',
    unknown: '未知',
  };

  async function status() {
    if (!isDesktop()) return null;
    try {
      return await window.__TAURI_INTERNALS__.invoke('camera_permission_status');
    } catch {
      return 'unknown';
    }
  }

  async function request() {
    if (!isDesktop()) return null;
    try {
      return await window.__TAURI_INTERNALS__.invoke('request_camera_permission');
    } catch {
      return null;
    }
  }

  function openSettings() {
    if (!isDesktop()) return false;
    window.__TAURI_INTERNALS__.invoke('open_camera_settings').catch(() => {});
    return true;
  }

  let dialog = null;
  let retry = null;

  function ensureDialog() {
    if (dialog) return dialog;
    dialog = document.createElement('dialog');
    dialog.innerHTML = `
      <h2 id="camPermTitle">摄像头没有权限</h2>
      <p id="camPermText"></p>
      <ol id="camPermSteps">
        <li>点下面的「打开系统设置」</li>
        <li>在「隐私与安全性 → 摄像头」里打开 <b>LocalFace</b></li>
        <li>回到这里点「重试」</li>
      </ol>
      <p class="note" id="camPermNote"></p>
      <div class="actions">
        <button type="button" class="ghost" data-act="close">关闭</button>
        <button type="button" class="ghost" data-act="settings">打开系统设置</button>
        <button type="button" class="primary" data-act="retry">重试</button>
      </div>`;
    dialog.addEventListener('click', (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'close') dialog.close();
      else if (act === 'settings') openSettings();
      else if (act === 'retry') {
        dialog.close();
        retry?.();
      }
    });
    document.body.appendChild(dialog);
    return dialog;
  }

  function explain(state, onRetry) {
    retry = onRetry || null;
    const d = ensureDialog();
    const browser = !isDesktop();
    const title = d.querySelector('#camPermTitle');
    const text = d.querySelector('#camPermText');
    const steps = d.querySelector('#camPermSteps');
    const note = d.querySelector('#camPermNote');

    if (browser) {
      title.textContent = '浏览器没有摄像头权限';
      text.textContent = '请点地址栏左侧的网站设置图标，把「摄像头」改为「允许」，然后点「重试」。';
      steps.hidden = true;
      note.hidden = true;
      d.querySelector('[data-act="settings"]').hidden = true;
    } else {
      const denied = state === 'denied' || state === 'restricted';
      title.textContent = state === 'not_determined' ? '等待系统授权' : '摄像头没有权限';
      text.textContent = state === 'not_determined'
        ? '系统授权框没有出现。可以点「重试」再触发一次，或去系统设置里手动打开。'
        : 'macOS 没有允许 LocalFace 使用摄像头，所以画面打不开。';
      steps.hidden = false;
      note.hidden = state === 'not_determined';
      note.textContent = denied ? '如果「摄像头」列表里没有 LocalFace：先点「重试」，系统会弹出授权框。' : '';
      d.querySelector('[data-act="settings"]').hidden = false;
    }

    if (!d.open) d.showModal();
  }

  window.CameraPermission = { status, request, explain, label: (state) => labels[state] || state, isDesktop };
})();
