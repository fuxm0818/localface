// 同步桌面壳需要的前端资源：壳用 splash.html 起窗口，服务进程从 desktop-dist 提供页面
import { cp, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';

const ROOT = join(import.meta.dirname, '..');
const OUT = join(ROOT, 'desktop-dist');
const FILES = ['index.html', 'admin.html', 'style.css'];

await rm(OUT, { recursive: true, force: true });
await mkdir(OUT, { recursive: true });
for (const file of FILES) await cp(join(ROOT, file), join(OUT, file));
await cp(join(ROOT, 'desktop', 'splash.html'), join(OUT, 'splash.html'));
console.log(`桌面资源已同步：${OUT}`);
