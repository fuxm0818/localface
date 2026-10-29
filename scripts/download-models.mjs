// 首次安装/启动时自动下载人脸模型（模型体积大且许可限制，不入仓库）
// 来源：deepghs/insightface 镜像（buffalo_l 的识别模型 + buffalo_s 的检测模型）
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from 'node:fs';
import { get } from 'node:https';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MODELS_DIR = process.env.FACE_MODELS_DIR
  ? path.resolve(process.env.FACE_MODELS_DIR)
  : path.join(ROOT, 'models');

// name: 仓库内文件名；size: 期望字节数（校验下载完整性）
const MODELS = [
  {
    file: 'scrfd_500m.onnx',
    size: 2524817,
    sources: [
      'https://hf-mirror.com/deepghs/insightface/resolve/main/buffalo_s/det_500m.onnx',
      'https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/det_500m.onnx',
    ],
  },
  {
    file: 'arcface_r50.onnx',
    size: 174383860,
    sources: [
      'https://hf-mirror.com/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx',
      'https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx',
    ],
  },
];

const TIMEOUT_MS = 60000;

function fetchRedirect(url, redirects = 5) {
  return new Promise((resolve, reject) => {
    const req = get(url, { headers: { 'User-Agent': 'localface-setup' }, timeout: TIMEOUT_MS }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && redirects > 0) {
        res.resume();
        const next = new URL(res.headers.location, url).toString();
        resolve(fetchRedirect(next, redirects - 1));
        return;
      }
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`HTTP ${res.statusCode}`));
        return;
      }
      resolve(res);
    });
    req.on('timeout', () => req.destroy(new Error('连接超时')));
    req.on('error', reject);
  });
}

async function download(url, dest) {
  const res = await fetchRedirect(url);
  const tmp = dest + '.part';
  await new Promise((resolve, reject) => {
    const out = createWriteStream(tmp);
    res.pipe(out);
    res.on('error', reject);
    out.on('error', reject);
    out.on('finish', resolve);
  });
  const size = statSync(tmp).size;
  if (size !== MODELS.find((m) => m.file === path.basename(dest)).size) {
    throw new Error(`文件大小不符（收到 ${size} 字节）`);
  }
  renameSync(tmp, dest);
}

let missing = 0;
mkdirSync(MODELS_DIR, { recursive: true }); // 全新 clone 连 models/ 目录都没有（内容全被 gitignore）
for (const m of MODELS) {
  const dest = path.join(MODELS_DIR, m.file);
  if (existsSync(dest) && statSync(dest).size === m.size) continue;
  if (existsSync(dest)) {
    console.log(`models/${m.file} 大小异常，重新下载…`);
  } else {
    console.log(`models/${m.file} 缺失，开始下载（${(m.size / 1024 / 1024).toFixed(1)} MB）…`);
  }
  missing++;
  let ok = false;
  let lastErr;
  for (const url of m.sources) {
    try {
      console.log(`  尝试 ${new URL(url).host} …`);
      await download(url, dest);
      console.log(`  已下载 models/${m.file}`);
      ok = true;
      break;
    } catch (e) {
      lastErr = e;
      console.log(`  失败：${e.message}`);
    }
  }
  if (!ok) {
    console.error(`\n模型 ${m.file} 下载失败：${lastErr?.message}`);
    console.error('请检查网络后重新运行：node scripts/download-models.mjs');
    process.exit(1);
  }
}

if (missing === 0) {
  console.log('模型文件已就绪。');
} else {
  console.log('全部模型下载完成。');
}
