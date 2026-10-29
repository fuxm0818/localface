// 打包后硬校验：安装包里不能出现任何照片、视频、数据库或 v1 模板
import { readdir, stat } from 'node:fs/promises';
import { extname, join } from 'node:path';

const ROOT = join(import.meta.dirname, '..');
const APP = process.argv[2] || join(ROOT, 'src-tauri', 'target', 'release', 'bundle', 'macos', 'LocalFace.app');
const MEDIA = new Set([
  '.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.heic', '.heif', '.tif', '.tiff',
  '.mp4', '.mov', '.m4v', '.avi', '.mkv',
  '.db', '.db-wal', '.db-shm', '.sqlite', '.sqlite3',
]);
const NAMES = new Set(['son-template.json', 'photos', 'thumbs', 'app.db']);

const bad = [];
let total = 0;

async function walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    total++;
    if (NAMES.has(entry.name) || MEDIA.has(extname(entry.name).toLowerCase())) bad.push(path);
    if (entry.isDirectory()) await walk(path);
  }
}

try {
  await stat(APP);
} catch {
  console.error(`找不到安装包：${APP}`);
  process.exit(1);
}

await walk(APP);

if (bad.length) {
  console.error(`安装包里发现 ${bad.length} 个个人数据文件，打包中止：`);
  for (const path of bad.slice(0, 20)) console.error('  ' + path);
  process.exit(1);
}

console.log(`安装包校验通过：${total} 个文件，没有照片 / 视频 / 数据库 / v1 模板`);
