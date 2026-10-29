// 批量导入照片目录到某个人物：npm run import -- photos 儿子
import { readdir, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import jpeg from 'jpeg-js';

import { openDb } from './db.mjs';
import { FacePipe } from './facepipe.mjs';
import { Store } from './store.mjs';

const ROOT = import.meta.dirname;
const DATA_DIR = process.env.DATA_DIR ? resolve(process.env.DATA_DIR) : join(ROOT, 'data');
const MIN_FACE_W = 100;
const MIN_DET = 0.6;

const args = process.argv.slice(2).filter(a => !a.startsWith('--'));
if (!args[0]) {
  console.log('用法：npm run import -- <照片目录> [人物名]');
  console.log('例如：npm run import -- photos 儿子');
  process.exit(1);
}
const folder = resolve(args[0]);
const personName = args[1] || '儿子';

const pipe = await FacePipe.create(join(ROOT, 'models'), 640);
console.log('模型已加载，正在读取照片目录…');

const db = openDb(DATA_DIR);
const store = new Store(db, { dataDir: DATA_DIR, pipe, dbLabel: join(DATA_DIR, 'app.db') });
const existing = store.state().persons.find(p => p.name === personName);
const person = existing || store.createPerson(personName, '批量导入');
if (!existing) console.log(`已创建人物「${person.name}」`);

let files;
try {
  files = (await readdir(folder)).filter(f => /\.jpe?g$/i.test(f)).sort();
} catch (e) {
  console.error(`打不开照片目录：${folder}\n${e.message}`);
  process.exit(1);
}
if (!files.length) {
  console.log('目录里没有 jpg / jpeg 照片');
  process.exit(0);
}

console.log(`开始导入 ${files.length} 张照片到「${person.name}」，数据写入 ${join(DATA_DIR, 'app.db')}`);
let added = 0, skipped = 0, failed = 0;

for (let i = 0; i < files.length; i++) {
  const name = files[i];
  const tag = `[${i + 1}/${files.length}]`;
  try {
    const img = jpeg.decode(await readFile(join(folder, name)), { useArrayBuffer: false });
    const faces = await pipe.detect(img.data, img.width, img.height, { threshold: 0.5, maxFaces: 8 });
    if (faces.length !== 1) {
      skipped++;
      console.log(`${tag} • ${name}：检测到 ${faces.length} 张脸，跳过（要求单人照）`);
      continue;
    }
    const face = faces[0];
    if (face.faceW < MIN_FACE_W || face.score < MIN_DET) {
      skipped++;
      console.log(`${tag} • ${name}：脸太小或不够清晰（${Math.round(face.faceW)}px / ${face.score.toFixed(2)}），跳过`);
      continue;
    }
    const emb = await pipe.embed(img.data, img.width, img.height, face.kps);
    const r = await store.addFace(person.id, { rgba: img.data, w: img.width, h: img.height, face, emb, source: 'cli', file: name });
    if (r.added) {
      added++;
      console.log(`${tag} ✓ ${name} · 脸 ${r.faceW}px · 现有 ${r.total} 张`);
    } else {
      skipped++;
      console.log(`${tag} • ${name}：${r.reason}`);
    }
  } catch (e) {
    failed++;
    console.log(`${tag} ✗ ${name}：${e.message}`);
  }
}

console.log(`\n导入完成：新增 ${added}，跳过 ${skipped}，失败 ${failed}；「${person.name}」现在共有 ${store.personFaces(person.id).person.faceCount} 张样本。`);
if (added) console.log('正在运行的识别服务会自动加载新样本，不用重启。');
db.close();
