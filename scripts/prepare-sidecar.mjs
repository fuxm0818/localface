// 组装生产包所需资源：Node 运行时（sidecar）+ 精简后的依赖 + 模型 + 静态页
import { execFileSync } from 'node:child_process';
import { cp, mkdir, rm, readdir } from 'node:fs/promises';
import { join } from 'node:path';

const ROOT = join(import.meta.dirname, '..');
const TAURI = join(ROOT, 'src-tauri');
const BIN = join(TAURI, 'binaries');
const SERVER = join(TAURI, 'sidecar', 'server');
const STATIC = join(TAURI, 'resources', 'static');
const MODELS = join(TAURI, 'resources', 'models');

const ARCH = { arm64: 'aarch64', x64: 'x86_64' }[process.arch];
const OS = { darwin: 'apple-darwin', win32: 'pc-windows-msvc', linux: 'unknown-linux-gnu' }[process.platform];
if (!ARCH || !OS) throw new Error(`不支持的平台：${process.platform}/${process.arch}`);
const TRIPLE = `${ARCH}-${OS}`;

const SERVER_FILES = ['server.mjs', 'db.mjs', 'store.mjs', 'facepipe.mjs'];
const PACKAGES = ['jpeg-js', 'onnxruntime-common', 'onnxruntime-node'];

async function copy(src, dest) {
  if (process.platform === 'darwin') {
    try {
      execFileSync('cp', ['-cR', src, dest], { stdio: 'inherit' });
      return;
    } catch {}
  }
  await cp(src, dest, { recursive: true });
}

await rm(BIN, { recursive: true, force: true });
await rm(SERVER, { recursive: true, force: true });
await rm(join(TAURI, 'resources'), { recursive: true, force: true });
await mkdir(BIN, { recursive: true });
await mkdir(SERVER, { recursive: true });
await mkdir(STATIC, { recursive: true });
await mkdir(MODELS, { recursive: true });

const exe = process.platform === 'win32' ? '.exe' : '';
const sidecar = join(BIN, `localface-server-${TRIPLE}${exe}`);
await copy(process.execPath, sidecar);
console.log(`Node 运行时 → ${sidecar}`);

for (const file of SERVER_FILES) await cp(join(ROOT, file), join(SERVER, file));
await cp(join(ROOT, 'package.json'), join(SERVER, 'package.json'));

const modules = join(SERVER, 'node_modules');
await mkdir(modules, { recursive: true });
for (const pkg of PACKAGES) await copy(join(ROOT, 'node_modules', pkg), join(modules, pkg));

// 只保留当前平台的 onnxruntime 二进制，其余平台目录删掉可省 200MB
const napi = join(modules, 'onnxruntime-node', 'bin', 'napi-v6');
const keep = { darwin: 'darwin', win32: 'win32', linux: 'linux' }[process.platform];
for (const dir of await readdir(napi)) {
  if (dir !== keep) await rm(join(napi, dir), { recursive: true, force: true });
}
const platformDir = join(napi, keep);
for (const dir of await readdir(platformDir)) {
  if (dir !== process.arch) await rm(join(platformDir, dir), { recursive: true, force: true });
}

for (const file of ['index.html', 'admin.html', 'guard.html', 'style.css', 'permission-ui.js']) await cp(join(ROOT, file), join(STATIC, file));
for (const file of ['arcface_r50.onnx', 'scrfd_500m.onnx']) await copy(join(ROOT, 'models', file), join(MODELS, file));

console.log(`服务代码 → ${SERVER}`);
console.log(`模型与页面 → ${join(TAURI, 'resources')}`);
console.log(`目标平台：${TRIPLE}`);
