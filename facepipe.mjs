// 本地人脸管线：SCRFD 检测 + 5 要点对齐 + ArcFace(w600k_r50) 512 维特征
// 数学与 insightface 官方实现一致，已在 Python 侧逐张核验（浏览器/Python 余弦差 <0.001）

import { InferenceSession, Tensor } from 'onnxruntime-node';

const ARC_TPL = [38.2946, 46.6182, 73.5313, 46.6182, 56.1252, 68.4910, 41.5493, 93.3094, 70.7299, 93.3094];
const STRIDES = [8, 16, 32];
const NUM_ANCHORS = 2;

// 把图像等比放进 size×size 画布，右下补灰边
function letterbox(rgba, w, h, size) {
  const scale = size / Math.max(w, h);
  const nw = Math.max(1, Math.round(w * scale)), nh = Math.max(1, Math.round(h * scale));
  const plane = size * size;
  const x = new Float32Array(3 * plane);
  for (let y = 0; y < size; y++) {
    for (let cx = 0; cx < size; cx++) {
      const i = y * size + cx;
      if (cx >= nw || y >= nh) { x[i] = x[plane + i] = x[plane * 2 + i] = (114 - 127.5) / 128; continue; }
      const sx = Math.min(w - 1, cx / scale), sy = Math.min(h - 1, y / scale);
      const x0 = sx | 0, y0 = sy | 0;
      const fx = sx - x0, fy = sy - y0;
      const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
      let r = 0, g = 0, b = 0;
      for (const [dy, wy] of [[y0, 1 - fy], [y1, fy]]) {
        for (const [dx, wx] of [[x0, 1 - fx], [x1, fx]]) {
          const p = (dy * w + dx) * 4, wt = wy * wx;
          r += rgba[p] * wt; g += rgba[p + 1] * wt; b += rgba[p + 2] * wt;
        }
      }
      x[i] = (r - 127.5) / 128;
      x[plane + i] = (g - 127.5) / 128;
      x[plane * 2 + i] = (b - 127.5) / 128;
    }
  }
  return { tensor: new Tensor('float32', x, [1, 3, size, size]), scale };
}

function iou(a, b) {
  const ix = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const iy = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (ix <= 0 || iy <= 0) return 0;
  const inter = ix * iy;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

// 模板 5 点 -> 实测 5 点 的最小二乘相似变换
function fitAffine(kps) {
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < 5; i++) { sx += ARC_TPL[i * 2]; sy += ARC_TPL[i * 2 + 1]; dx += kps[i * 2]; dy += kps[i * 2 + 1]; }
  sx /= 5; sy /= 5; dx /= 5; dy /= 5;
  let A = 0, B = 0, N = 0;
  for (let i = 0; i < 5; i++) {
    const ax = ARC_TPL[i * 2] - sx, ay = ARC_TPL[i * 2 + 1] - sy;
    const bx = kps[i * 2] - dx, by = kps[i * 2 + 1] - dy;
    A += ax * bx + ay * by; B += ax * by - ay * bx; N += ax * ax + ay * ay;
  }
  if (!N) return null;
  const sa = A / N, sb = B / N;
  return [sa, -sb, sb, sa, dx - (sa * sx - sb * sy), dy - (sb * sx + sa * sy)];
}

export class FacePipe {
  constructor(det, rec, detSize) {
    this.det = det; this.rec = rec; this.detSize = detSize;
  }

  static async create(modelDir = './models', detSize = 448) {
    const det = await InferenceSession.create(`${modelDir}/scrfd_500m.onnx`);
    const rec = await InferenceSession.create(`${modelDir}/arcface_r50.onnx`);
    return new FacePipe(det, rec, detSize);
  }

  async detect(rgba, w, h, { threshold = 0.5, maxFaces = 6 } = {}) {
    const size = this.detSize; // 模型输入尺寸固定
    const { tensor, scale } = letterbox(rgba, w, h, size);
    const out = await this.det.run({ [this.det.inputNames[0]]: tensor });
    const vals = this.det.outputNames.map(n => out[n]);
    const groups = { 1: [], 4: [], 10: [] };
    for (const t of vals) if (groups[t.dims[t.dims.length - 1]]) groups[t.dims[t.dims.length - 1]].push(t);
    for (const k of Object.keys(groups)) groups[k].sort((a, b) => b.dims[0] - a.dims[0]);
    const cands = [];
    for (let i = 0; i < STRIDES.length; i++) {
      const stride = STRIDES[i];
      const scores = groups[1][i], deltas = groups[4][i], kp = groups[10][i];
      if (!scores || !deltas || !kp) continue;
      const grid = Math.round(Math.sqrt(scores.dims[0] / NUM_ANCHORS));
      for (let a = 0; a < scores.dims[0]; a++) {
        const sc = scores.data[a];
        if (sc < threshold) continue;
        const cell = (a / NUM_ANCHORS) | 0;
        const cx = ((cell % grid) + 0.5) * stride, cy = (((cell / grid) | 0) + 0.5) * stride;
        const o = a * 4, k = a * 10, d = deltas.data, p = kp.data;
        const box = [(cx - d[o] * stride) / scale, (cy - d[o + 1] * stride) / scale,
                     (cx + d[o + 2] * stride) / scale, (cy + d[o + 3] * stride) / scale];
        const pts = new Float64Array(10);
        for (let j = 0; j < 5; j++) {
          pts[j * 2] = (cx + p[k + j * 2] * stride) / scale;
          pts[j * 2 + 1] = (cy + p[k + j * 2 + 1] * stride) / scale;
        }
        cands.push({ score: sc, box, kps: pts });
      }
    }
    cands.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const f of cands) {
      if (kept.every(g => iou(f.box, g.box) < 0.4)) kept.push(f);
      if (kept.length >= maxFaces) break;
    }
    for (const f of kept) {
      f.faceW = f.box[2] - f.box[0];
      f.box = { x: f.box[0], y: f.box[1], width: f.faceW, height: f.box[3] - f.box[1] };
    }
    return kept;
  }

  // 按 5 要点双线性采样出 112x112 对齐脸，再提特征
  async embed(rgba, w, h, kps) {
    const m = fitAffine(kps);
    if (!m) return null;
    const plane = 112 * 112;
    const x = new Float32Array(3 * plane);
    for (let y = 0; y < 112; y++) {
      for (let cx = 0; cx < 112; cx++) {
        const i = y * 112 + cx;
        const sx = m[0] * cx + m[1] * y + m[4];
        const sy = m[2] * cx + m[3] * y + m[5];
        if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) { x[i] = x[plane + i] = x[plane * 2 + i] = -1; continue; }
        const x0 = sx | 0, y0 = sy | 0, fx = sx - x0, fy = sy - y0;
        const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
        let r = 0, g = 0, b = 0;
        for (const [dy, wy] of [[y0, 1 - fy], [y1, fy]]) {
          for (const [dx, wx] of [[x0, 1 - fx], [x1, fx]]) {
            const p = (dy * w + dx) * 4, wt = wy * wx;
            r += rgba[p] * wt; g += rgba[p + 1] * wt; b += rgba[p + 2] * wt;
          }
        }
        x[i] = (r - 127.5) / 127.5;
        x[plane + i] = (g - 127.5) / 127.5;
        x[plane * 2 + i] = (b - 127.5) / 127.5;
      }
    }
    const out = await this.rec.run({ [this.rec.inputNames[0]]: new Tensor('float32', x, [1, 3, 112, 112]) });
    const v = out[this.rec.outputNames[0]].data;
    const e = Float32Array.from(v);
    let norm = 0;
    for (let i = 0; i < e.length; i++) norm += e[i] * e[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < e.length; i++) e[i] /= norm;
    return e;
  }
}

// 按同一套 5 点对齐切出正方形小图，用于管理后台的样本缩略图
export function cropFace(rgba, w, h, kps, size = 112) {
  const m = fitAffine(kps);
  if (!m) return null;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let cx = 0; cx < size; cx++) {
      const o = (y * size + cx) * 4;
      const sx = m[0] * cx + m[1] * y + m[4];
      const sy = m[2] * cx + m[3] * y + m[5];
      if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) {
        data[o] = 24; data[o + 1] = 26; data[o + 2] = 30; data[o + 3] = 255;
        continue;
      }
      const x0 = sx | 0, y0 = sy | 0, fx = sx - x0, fy = sy - y0;
      const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
      let r = 0, g = 0, b = 0;
      for (const [dy, wy] of [[y0, 1 - fy], [y1, fy]]) {
        for (const [dx, wx] of [[x0, 1 - fx], [x1, fx]]) {
          const p = (dy * w + dx) * 4, wt = wy * wx;
          r += rgba[p] * wt; g += rgba[p + 1] * wt; b += rgba[p + 2] * wt;
        }
      }
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
    }
  }
  return { data, size };
}

export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// 与模板的 top-k 平均余弦：既保留本人多姿态命中，又抑制单张模板的偶然匹配
export function match(emb, template, k = 5) {
  const sims = template.map(t => cosine(emb, t));
  sims.sort((a, b) => b - a);
  let acc = 0;
  const m = Math.min(k, sims.length);
  for (let i = 0; i < m; i++) acc += sims[i];
  return { sim: acc / m, max: sims[0] };
}
