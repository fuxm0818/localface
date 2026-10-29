// 本地识别服务：静态页面 + 多人物人脸库 + 管理 API。所有数据只走 127.0.0.1，照片不出本机。
import { existsSync, statSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, relative, resolve, sep } from 'node:path';
import jpeg from 'jpeg-js';

import { openDb } from './db.mjs';
import { FacePipe, cropFace } from './facepipe.mjs';
import { Store, StoreError } from './store.mjs';

const ROOT = import.meta.dirname;
const PORT = Number(process.env.FACE_PORT || process.env.PORT) || 5273;
const DATA_DIR = process.env.FACE_DATA_DIR || process.env.DATA_DIR
  ? resolve(process.env.FACE_DATA_DIR || process.env.DATA_DIR)
  : join(ROOT, 'data');
const STATIC_DIR = process.env.FACE_STATIC_DIR ? resolve(process.env.FACE_STATIC_DIR) : ROOT;
const MODELS_DIR = process.env.FACE_MODELS_DIR ? resolve(process.env.FACE_MODELS_DIR) : join(ROOT, 'models');
const SHUTDOWN_TOKEN = process.env.FACE_SHUTDOWN_TOKEN || '';
const DET_SIZE = 640; // 与 models/scrfd_500m.onnx 的固定输入一致
const MAX_FRAME = 12 * 1024 * 1024;
const MODEL = 'insightface buffalo_l w600k_r50';
const DB_LABEL = DATA_DIR.startsWith(ROOT + sep) ? relative(ROOT, join(DATA_DIR, 'app.db')) : join(DATA_DIR, 'app.db');
// 桌面壳的 webview 源（macOS 是 tauri://localhost，Windows 是 http://tauri.localhost）
const SHELL_ORIGINS = new Set(['tauri://localhost', 'http://tauri.localhost', 'https://tauri.localhost']);
const CAMERA_LEASE_MS = 8000;

// 摄像头同一时间只能被一个页面占用：后台录入会临时接管，护栏页让出
const cameraLease = { owner: '', at: 0 };
let guardReport = { at: 0 };

function guardStatus() {
  const now = Date.now();
  return {
    ...guardReport,
    at: guardReport.at ? new Date(guardReport.at).toISOString() : '',
    ageMs: guardReport.at ? now - guardReport.at : null,
    live: now - guardReport.at < 6000,
    cameraOwner: cameraLease.owner,
    cameraOwnerFresh: now - cameraLease.at < CAMERA_LEASE_MS,
  };
}

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
};

// 首次启动模型缺失时自动下载（模型许可限制不入仓库，见 README 第 15/17 节）
const MODEL_SIZES = { 'scrfd_500m.onnx': 2524817, 'arcface_r50.onnx': 174383860 };
const modelsOk = Object.entries(MODEL_SIZES).every(
  ([f, size]) => existsSync(join(MODELS_DIR, f)) && statSync(join(MODELS_DIR, f)).size === size
);
if (!modelsOk) {
  console.log('首次启动：模型文件缺失，自动下载（约 170 MB，视网速需要几分钟）…');
  await import('./scripts/download-models.mjs');
}

const pipe = await FacePipe.create(MODELS_DIR, DET_SIZE);
const db = openDb(DATA_DIR);
const store = new Store(db, { dataDir: DATA_DIR, pipe, model: MODEL, dbLabel: DB_LABEL });
console.log(`模型就绪：${MODEL}`);
console.log(`人物库就绪：${store.persons.size} 人 / ${store.faceById.size} 张人脸 → ${DB_LABEL}`);

// 串行化推理，避免多帧并发互相抢核
let chain = Promise.resolve();
const serialize = fn => { const p = chain.then(fn, fn); chain = p.then(() => {}, () => {}); return p; };

function readBody(req, limit) {
  return new Promise((resolveBody, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', c => {
      size += c.length;
      if (size > limit) {
        const e = new Error('数据太大');
        e.status = 413;
        reject(e);
        req.removeAllListeners('data');
        req.resume();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolveBody(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function readJson(req) {
  const body = await readBody(req, 64 * 1024);
  if (!body.length) return {};
  try {
    return JSON.parse(body.toString('utf8'));
  } catch {
    throw new StoreError('请求数据不是合法 JSON');
  }
}

function frameDims(params) {
  const w = Number(params.get('w'));
  const h = Number(params.get('h'));
  if (!Number.isInteger(w) || !Number.isInteger(h) || w < 16 || h < 16 || w > 4096 || h > 4096) {
    throw new StoreError('画面尺寸无效');
  }
  return { w, h };
}

function toRgba(body, w, h) {
  if (body.length !== w * h * 4) throw new StoreError(`帧尺寸不符：期望 ${w * h * 4} 字节，实到 ${body.length}`);
  return new Uint8Array(body.buffer, body.byteOffset, body.length);
}

const idOf = value => {
  const n = Number(value);
  if (!Number.isInteger(n) || n <= 0) throw new StoreError('ID 无效');
  return n;
};

const byFaceW = (a, b) => b.faceW - a.faceW;

async function scanFrame(body, params) {
  const { w, h } = frameDims(params);
  const rgba = toRgba(body, w, h);
  store.ensureFresh();
  const threshold = store.threshold();
  const identity = params.get('identity') !== '0';
  const t0 = performance.now();
  const faces = await pipe.detect(rgba, w, h, {
    threshold: Number(params.get('detThreshold')) || 0.5,
    maxFaces: 8,
  });
  const tDet = performance.now();
  if (!identity) {
    for (const f of faces) {
      f.personId = null;
      f.personName = null;
      f.sim = null;
      f.known = false;
    }
    return { faces, threshold, identity, ms: { det: Math.round(tDet - t0), rec: 0 } };
  }
  for (const f of faces) {
    f.emb = await pipe.embed(rgba, w, h, f.kps);
    const r = f.emb ? store.recognize(f.emb) : null;
    f.personId = r?.personId ?? null;
    f.personName = r?.name ?? null;
    f.sim = r ? +r.sim.toFixed(4) : null;
    f.known = !!(r && r.sim >= threshold);
  }
  return { faces, threshold, identity, ms: { det: Math.round(tDet - t0), rec: Math.round(performance.now() - tDet) } };
}

async function enrollFrame(personId, body, params) {
  const { w, h } = frameDims(params);
  const rgba = toRgba(body, w, h);
  const source = ['upload', 'camera', 'video', 'cli'].includes(params.get('source')) ? params.get('source') : 'upload';
  const file = params.get('file') || '';
  return serialize(async () => {
    const found = await pipe.detect(rgba, w, h, { threshold: 0.5, maxFaces: 8 });
    const best = found.slice().sort(byFaceW)[0];
    if (!best) return { error: '画面里没有检测到人脸，请正对镜头、光线充足时再试' };
    if (found.filter(f => f.faceW > best.faceW * 0.6).length > 1) {
      return { error: '画面里有多张差不多大的人脸，请只让一个人出现在照片 / 镜头里' };
    }
    const emb = await pipe.embed(rgba, w, h, best.kps);
    const result = await store.addFace(personId, { rgba, w, h, face: best, emb, source, file });
    return { ...result, detected: Math.round(best.faceW) };
  });
}

async function fillLegacyThumbs(personId) {
  const { faces } = store.personFaces(personId);
  const pending = faces.filter(f => !f.thumb && f.file);
  if (!pending.length) return;
  console.log(`正在后台生成 ${pending.length} 张旧样本的缩略图（不影响识别）…`);
  let done = 0;
  for (const face of pending) {
    try {
      const file = join(ROOT, 'photos', face.file);
      const img = jpeg.decode(await readFile(file), { useArrayBuffer: false });
      const found = await serialize(() => pipe.detect(img.data, img.width, img.height, { threshold: 0.5, maxFaces: 4 }));
      const best = found.slice().sort(byFaceW)[0];
      if (!best) continue;
      const crop = cropFace(img.data, img.width, img.height, best.kps, 112);
      if (!crop) continue;
      const thumb = `${randomUUID()}.jpg`;
      await writeFile(join(DATA_DIR, 'thumbs', thumb), jpeg.encode({ data: crop.data, width: crop.size, height: crop.size }, 82).data);
      store.setFaceThumb(face.id, thumb);
      done++;
    } catch {
      // 原照片缺失或损坏时保留无缩略图样本，不影响识别
    }
  }
  console.log(`缩略图生成完成：${done}/${pending.length}`);
}

async function migrateLegacyTemplate() {
  if (process.env.SKIP_MIGRATE === '1' || store.persons.size > 0) return;
  let tpl;
  try {
    tpl = JSON.parse(await readFile(join(ROOT, 'son-template.json'), 'utf8'));
  } catch {
    return;
  }
  const rows = (tpl.emb || []).map((e, i) => ({ emb: Float32Array.from(e), file: tpl.files?.[i] || '' }));
  if (!rows.length) return;
  const { personId, count } = store.importEmbeddings('儿子', rows, '从第一版 son-template.json 自动迁移');
  console.log(`已迁移 v1 模板：${count} 张人脸 → 人物「儿子」`);
  fillLegacyThumbs(personId).catch(e => console.log('缩略图生成失败：' + e.message));
}

await migrateLegacyTemplate();

const sendJson = (res, code, obj) => {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(obj));
};

async function handle(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const { pathname } = url;
  const method = req.method;
  const params = url.searchParams;

  if (process.env.FACE_LOG_REQUESTS === '1') console.log(`[req] ${method} ${pathname}`);

  // 桌面壳里页面来自 tauri:// 源，访问 127.0.0.1 属于跨源请求
  const origin = req.headers.origin;
  if (origin && (SHELL_ORIGINS.has(origin) || origin === `http://127.0.0.1:${PORT}` || origin === `http://localhost:${PORT}`)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  }
  if (method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-Face-Shutdown-Token',
      'Access-Control-Max-Age': '86400',
    });
    return res.end();
  }

  if (method === 'POST' && pathname === '/api/shutdown') {
    if (!SHUTDOWN_TOKEN || req.headers['x-face-shutdown-token'] !== SHUTDOWN_TOKEN) {
      return sendJson(res, 403, { error: 'forbidden' });
    }
    sendJson(res, 200, { ok: true });
    setTimeout(() => shutdown('shell'), 50);
    return;
  }

  if (method === 'GET' && pathname === '/api/state') {
    return sendJson(res, 200, store.state());
  }

  if (method === 'GET' && pathname === '/template') {
    const s = store.state();
    return sendJson(res, 200, { count: s.faceCount, personCount: s.personCount, model: s.model, threshold: s.threshold, agg: 'top5' });
  }

  if (method === 'POST' && pathname === '/scan') {
    const body = await readBody(req, MAX_FRAME);
    const r = await serialize(() => scanFrame(body, params));
    const faces = r.faces.map(({ box, faceW, score, personId, personName, sim, known }) => ({
      box, faceW: Math.round(faceW), score: +score.toFixed(3), personId, personName, sim, known,
    }));
    return sendJson(res, 200, { threshold: r.threshold, identity: r.identity, ms: r.ms, faces });
  }

  if (method === 'GET' && pathname === '/api/guard') {
    return sendJson(res, 200, { ...store.guardConfig(), status: guardStatus() });
  }

  if ((method === 'PUT' || method === 'PATCH') && pathname === '/api/guard') {
    const body = await readJson(req);
    return sendJson(res, 200, store.setGuardEnabled(body.enabled));
  }

  let m = pathname.match(/^\/api\/guard\/persons\/(\d+)$/);
  if (m && (method === 'PUT' || method === 'PATCH')) {
    const personId = idOf(m[1]);
    const body = await readJson(req);
    return sendJson(res, 200, store.setGuardProfile(personId, body));
  }

  m = pathname.match(/^\/api\/guard\/persons\/(\d+)\/calibrate$/);
  if (m && method === 'POST') {
    const personId = idOf(m[1]);
    const body = await readJson(req);
    const config = store.setGuardProfile(personId, {
      refRatio: body.ratio,
      refCm: body.cm,
      warnCm: body.warnCm,
      enabled: body.enabled === undefined ? undefined : !!body.enabled,
    });
    return sendJson(res, 200, { ...config, person: config.profiles.find(p => p.personId === personId) });
  }

  if (method === 'GET' && pathname === '/api/guard/status') {
    return sendJson(res, 200, guardStatus());
  }

  if (method === 'POST' && pathname === '/api/guard/report') {
    const body = await readJson(req);
    guardReport = {
      at: Date.now(),
      mode: String(body.mode || 'idle').slice(0, 20),
      ratio: Number(body.ratio) || 0,
      seen: !!body.seen,
      close: !!body.close,
      personId: Number(body.personId) || null,
      personName: String(body.personName || '').slice(0, 30),
      fps: Number(body.fps) || 0,
      camera: String(body.camera || '').slice(0, 20),
      error: String(body.error || '').slice(0, 200),
    };
    return sendJson(res, 200, { ok: true });
  }

  if (method === 'POST' && pathname === '/api/camera/heartbeat') {
    const body = await readJson(req);
    const owner = String(body.owner || 'ui').slice(0, 20);
    const fresh = Date.now() - cameraLease.at < CAMERA_LEASE_MS;
    // 前台页面（识别 / 后台）优先；护栏只在没人占用时拿摄像头
    if (owner !== 'guard') cameraLease.owner = owner;
    else if (!(cameraLease.owner && cameraLease.owner !== 'guard' && fresh)) cameraLease.owner = 'guard';
    cameraLease.at = Date.now();
    return sendJson(res, 200, { owner: cameraLease.owner });
  }

  if (method === 'POST' && pathname === '/api/camera/release') {
    const body = await readJson(req);
    if (!body.owner || body.owner === cameraLease.owner) {
      cameraLease.owner = '';
      cameraLease.at = 0;
    }
    return sendJson(res, 200, { owner: cameraLease.owner });
  }

  if (method === 'POST' && pathname === '/api/persons') {
    const body = await readJson(req);
    return sendJson(res, 201, { person: store.createPerson(body.name, body.note) });
  }

  m = pathname.match(/^\/api\/persons\/(\d+)\/faces$/);
  if (m) {
    const personId = idOf(m[1]);
    if (method === 'GET') return sendJson(res, 200, store.personFaces(personId));
    if (method === 'POST') {
      const body = await readBody(req, MAX_FRAME);
      const r = await enrollFrame(personId, body, params);
      if (r.error) return sendJson(res, 400, { error: r.error });
      return sendJson(res, 200, r);
    }
  }

  m = pathname.match(/^\/api\/persons\/(\d+)$/);
  if (m) {
    const personId = idOf(m[1]);
    if (method === 'PATCH' || method === 'PUT') {
      const body = await readJson(req);
      return sendJson(res, 200, { person: store.updatePerson(personId, { name: body.name, note: body.note }) });
    }
    if (method === 'DELETE') {
      return sendJson(res, 200, await store.deletePerson(personId));
    }
    if (method === 'GET') return sendJson(res, 200, { person: store.getPerson(personId) });
  }

  m = pathname.match(/^\/api\/faces\/(\d+)$/);
  if (m && method === 'DELETE') {
    return sendJson(res, 200, await store.deleteFace(idOf(m[1])));
  }

  if ((method === 'PUT' || method === 'PATCH') && pathname === '/api/settings') {
    const body = await readJson(req);
    return sendJson(res, 200, { threshold: store.setThreshold(body.threshold) });
  }

  if (method === 'GET' && pathname.startsWith('/thumbs/')) {
    const name = decodeURIComponent(pathname.slice('/thumbs/'.length));
    if (!/^[a-f0-9-]{36}\.jpg$/.test(name)) return sendJson(res, 404, { error: '缩略图不存在' });
    try {
      const buf = await readFile(join(DATA_DIR, 'thumbs', name));
      res.writeHead(200, { 'Content-Type': 'image/jpeg', 'Cache-Control': 'public, max-age=31536000, immutable' });
      return res.end(buf);
    } catch {
      return sendJson(res, 404, { error: '缩略图不存在' });
    }
  }

  if (method === 'GET' || method === 'HEAD') {
    if (pathname === '/favicon.ico') { res.writeHead(204); return res.end(); }
    const file = pathname === '/' ? 'index.html'
      : pathname === '/admin' ? 'admin.html'
      : pathname === '/guard' ? 'guard.html'
      : ['/index.html', '/admin.html', '/style.css', '/guard.html'].includes(pathname) ? pathname.slice(1)
      : null;
    if (file) {
      try {
        const buf = await readFile(join(STATIC_DIR, file));
        res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
        return res.end(method === 'HEAD' ? undefined : buf);
      } catch {
        return sendJson(res, 404, { error: '页面不存在' });
      }
    }
  }

  return sendJson(res, 404, { error: 'Not Found' });
}

const server = createServer((req, res) => {
  handle(req, res).catch(e => {
    const status = e.status || (e.message === '数据太大' ? 413 : 500);
    if (status >= 500) console.error('[请求出错]', e);
    if (!res.headersSent) sendJson(res, status, { error: e.message });
    else res.end();
  });
});

server.on('error', e => {
  if (e.code === 'EADDRINUSE') console.error(`端口 ${PORT} 已被占用：可以换端口启动，如 PORT=5274 npm start`);
  else console.error('服务启动失败：', e);
  process.exit(1);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`识别服务已启动： http://localhost:${PORT}`);
  console.log(`管理后台： http://localhost:${PORT}/admin`);
  console.log('摄像头画面只在本机 127.0.0.1 处理，不会上传。');
});

let closing = false;
function shutdown(reason = 'signal') {
  if (closing) return;
  closing = true;
  console.log(`\n正在关闭服务（${reason}）…`);
  server.close(() => { db.close(); process.exit(0); });
  // 识别请求可能还在跑，给 1.5 秒收尾
  setTimeout(() => process.exit(0), 1500).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
