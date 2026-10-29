// 人物库：SQLite 持久化 + 内存索引。每张脸一条 512 维特征，识别时按人取 top-5 平均。
import { randomUUID } from 'node:crypto';
import { unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import jpeg from 'jpeg-js';

import { DEFAULT_THRESHOLD, bumpRevision, getSetting, nowIso, readRevision, setSetting } from './db.mjs';
import { cropFace, match } from './facepipe.mjs';

export const TOP_K = 5;
export const MIN_FACE_W = 90;
export const DEDUPE_SIM = 0.97;
const MIN_DET_SCORE = 0.6;
const GUARD_WARN_MIN = 25;
const GUARD_WARN_MAX = 150;
const GUARD_REF_CM_MIN = 20;
const GUARD_REF_CM_MAX = 200;

function inRange(value, min, max, label) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < min || n > max) throw new StoreError(`${label}需要在 ${min}～${max} 之间`);
  return n;
}

export class StoreError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

function toEmb(blob) {
  const u8 = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
  return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
}

function faceView(row) {
  return {
    id: row.id,
    personId: row.person_id,
    source: row.source,
    file: row.file,
    thumb: row.thumb,
    thumbUrl: row.thumb ? `/thumbs/${row.thumb}` : '',
    quality: Number(row.quality),
    faceW: Math.round(row.face_w),
    detScore: +Number(row.det_score).toFixed(3),
    createdAt: row.created_at,
  };
}

function cleanName(name) {
  const s = String(name ?? '').trim().replace(/\s+/g, ' ');
  if (!s) throw new StoreError('名字不能为空');
  if (s.length > 30) throw new StoreError('名字最多 30 个字');
  return s;
}

export class Store {
  constructor(db, { dataDir, pipe, model = 'insightface buffalo_l w600k_r50', dbLabel = '' }) {
    this.db = db;
    this.pipe = pipe;
    this.model = model;
    this.dataDir = dataDir;
    this.thumbsDir = join(dataDir, 'thumbs');
    this.dbLabel = dbLabel || join(dataDir, 'app.db');
    this.persons = new Map();
    this.faceById = new Map();
    this.revision = -1;
    this.reload();
  }

  reload() {
    const persons = this.db.prepare('SELECT * FROM persons ORDER BY id').all();
    const faces = this.db.prepare('SELECT * FROM faces ORDER BY id').all();
    this.persons = new Map();
    this.faceById = new Map();
    for (const row of persons) {
      this.persons.set(row.id, {
        id: row.id, name: row.name, note: row.note,
        createdAt: row.created_at, updatedAt: row.updated_at,
        faces: [], embs: [],
      });
    }
    for (const row of faces) {
      const p = this.persons.get(row.person_id);
      if (!p) continue;
      const meta = faceView(row);
      p.faces.push(meta);
      p.embs.push(toEmb(row.emb));
      this.faceById.set(row.id, meta);
    }
    this.revision = readRevision(this.db);
  }

  // CLI 批量导入会直接写库；发现 revision 变了就重新加载，运行中的服务无需重启
  ensureFresh() {
    if (readRevision(this.db) !== this.revision) this.reload();
  }

  threshold() {
    return Number(getSetting(this.db, 'threshold', String(DEFAULT_THRESHOLD)));
  }

  setThreshold(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n < 0.2 || n > 0.8) throw new StoreError('阈值需要在 0.20～0.80 之间');
    setSetting(this.db, 'threshold', n.toFixed(2));
    return this.threshold();
  }

  // 距离提醒：全局开关 + 每个人各自的警告距离。阈值不直接存，按「校准时的脸宽比例 × 校准距离 ÷ 警告距离」换算。
  guardProfileView(person, row) {
    const warnCm = row ? Number(row.warn_cm) : 60;
    const refRatio = row ? Number(row.ref_ratio) : 0;
    const refCm = row ? Number(row.ref_cm) : 0;
    const threshold = refRatio > 0 && refCm > 0 && warnCm > 0 ? (refRatio * refCm) / warnCm : 0;
    return {
      personId: person.id,
      name: person.name,
      enabled: row ? !!row.enabled : false,
      warnCm,
      refRatio: +refRatio.toFixed(5),
      refCm,
      threshold: +threshold.toFixed(5),
      calibrated: threshold > 0,
      updatedAt: row?.updated_at || '',
    };
  }

  guardConfig() {
    this.ensureFresh();
    const rows = this.db.prepare('SELECT * FROM guard_profiles').all();
    const byPerson = new Map(rows.map(r => [r.person_id, r]));
    const profiles = [...this.persons.values()].map(p => this.guardProfileView(p, byPerson.get(p.id)));
    return {
      enabled: getSetting(this.db, 'guard_enabled', '1') !== '0',
      activeCount: profiles.filter(p => p.enabled && p.calibrated).length,
      profiles,
    };
  }

  setGuardEnabled(value) {
    setSetting(this.db, 'guard_enabled', value ? '1' : '0');
    return this.guardConfig();
  }

  setGuardProfile(personId, { enabled, warnCm, refRatio, refCm } = {}) {
    const p = this.persons.get(Number(personId));
    if (!p) throw new StoreError('人物不存在', 404);
    const row = this.db.prepare('SELECT * FROM guard_profiles WHERE person_id = ?').get(p.id);

    const nextEnabled = enabled === undefined ? !!row?.enabled : !!enabled;
    const nextWarn = warnCm === undefined
      ? Number(row?.warn_cm ?? 60)
      : inRange(warnCm, GUARD_WARN_MIN, GUARD_WARN_MAX, '警告距离');
    let nextRatio = row ? Number(row.ref_ratio) : 0;
    let nextRefCm = row ? Number(row.ref_cm) : 0;
    if (refRatio !== undefined || refCm !== undefined) {
      nextRatio = inRange(refRatio, 0.01, 0.95, '校准比例');
      nextRefCm = inRange(refCm, GUARD_REF_CM_MIN, GUARD_REF_CM_MAX, '校准距离');
    }

    this.db.prepare(`INSERT INTO guard_profiles(person_id, enabled, warn_cm, ref_ratio, ref_cm, updated_at)
      VALUES(?, ?, ?, ?, ?, ?)
      ON CONFLICT(person_id) DO UPDATE SET
        enabled = excluded.enabled, warn_cm = excluded.warn_cm,
        ref_ratio = excluded.ref_ratio, ref_cm = excluded.ref_cm, updated_at = excluded.updated_at`)
      .run(p.id, nextEnabled ? 1 : 0, nextWarn, nextRatio, nextRefCm, nowIso());
    return this.guardConfig();
  }

  personView(p) {
    const withThumb = p.faces.filter(f => f.thumb).sort((a, b) => b.faceW - a.faceW || b.id - a.id);
    return {
      id: p.id,
      name: p.name,
      note: p.note,
      faceCount: p.faces.length,
      avatar: withThumb[0]?.thumbUrl || '',
      createdAt: p.createdAt,
      updatedAt: p.updatedAt,
    };
  }

  state() {
    this.ensureFresh();
    const persons = [...this.persons.values()].map(p => this.personView(p));
    return {
      persons,
      personCount: persons.length,
      faceCount: persons.reduce((n, p) => n + p.faceCount, 0),
      threshold: this.threshold(),
      model: this.model,
      topK: TOP_K,
      minFaceW: MIN_FACE_W,
      storage: this.dbLabel,
    };
  }

  getPerson(id) {
    const p = this.persons.get(Number(id));
    if (!p) throw new StoreError('人物不存在', 404);
    return this.personView(p);
  }

  personFaces(id) {
    this.ensureFresh();
    const p = this.persons.get(Number(id));
    if (!p) throw new StoreError('人物不存在', 404);
    return {
      person: this.personView(p),
      faces: p.faces.slice().sort((a, b) => b.id - a.id),
    };
  }

  createPerson(name, note = '') {
    const clean = cleanName(name);
    const t = nowIso();
    let id;
    try {
      id = Number(this.db.prepare('INSERT INTO persons(name, note, created_at, updated_at) VALUES(?, ?, ?, ?)')
        .run(clean, String(note ?? '').trim().slice(0, 200), t, t).lastInsertRowid);
    } catch (e) {
      if (String(e.message).includes('UNIQUE')) throw new StoreError(`已经有一个叫「${clean}」的人物了`);
      throw e;
    }
    bumpRevision(this.db);
    this.reload();
    return this.getPerson(id);
  }

  updatePerson(id, { name, note } = {}) {
    const p = this.persons.get(Number(id));
    if (!p) throw new StoreError('人物不存在', 404);
    const nextName = name === undefined ? p.name : cleanName(name);
    const nextNote = note === undefined ? p.note : String(note ?? '').trim().slice(0, 200);
    try {
      this.db.prepare('UPDATE persons SET name = ?, note = ?, updated_at = ? WHERE id = ?')
        .run(nextName, nextNote, nowIso(), p.id);
    } catch (e) {
      if (String(e.message).includes('UNIQUE')) throw new StoreError(`已经有一个叫「${nextName}」的人物了`);
      throw e;
    }
    bumpRevision(this.db);
    this.reload();
    return this.getPerson(p.id);
  }

  async deletePerson(id) {
    const p = this.persons.get(Number(id));
    if (!p) throw new StoreError('人物不存在', 404);
    const thumbs = p.faces.map(f => f.thumb).filter(Boolean);
    this.db.prepare('DELETE FROM persons WHERE id = ?').run(p.id);
    bumpRevision(this.db);
    this.reload();
    await Promise.allSettled(thumbs.map(t => unlink(join(this.thumbsDir, t))));
    return { ok: true, personId: p.id };
  }

  async deleteFace(id) {
    const f = this.faceById.get(Number(id));
    if (!f) throw new StoreError('人脸样本不存在', 404);
    this.db.prepare('DELETE FROM faces WHERE id = ?').run(f.id);
    this.db.prepare('UPDATE persons SET updated_at = ? WHERE id = ?').run(nowIso(), f.personId);
    bumpRevision(this.db);
    this.reload();
    if (f.thumb) await unlink(join(this.thumbsDir, f.thumb)).catch(() => {});
    return { ok: true, faceId: f.id, personId: f.personId };
  }

  recognize(emb) {
    let best = null;
    for (const p of this.persons.values()) {
      if (!p.embs.length) continue;
      const { sim } = match(emb, p.embs, TOP_K);
      if (!best || sim > best.sim) best = { personId: p.id, name: p.name, sim };
    }
    return best;
  }

  // face: pipe.detect 的结果（含 kps / faceW / score），emb 可复用扫描阶段已算好的特征
  async addFace(personId, { rgba, w, h, face, emb, source = 'upload', file = '' }) {
    const p = this.persons.get(Number(personId));
    if (!p) throw new StoreError('人物不存在', 404);
    const faceW = Math.round(face.faceW);
    if (faceW < MIN_FACE_W) {
      return { added: false, faceW, reason: `脸太小（约 ${faceW}px，至少 ${MIN_FACE_W}px），请靠近一点或换一张更清晰的照片` };
    }
    if (face.score < MIN_DET_SCORE) return { added: false, faceW, reason: '人脸不够清晰，已跳过' };
    if (!emb) return { added: false, faceW, reason: '提取人脸特征失败' };

    const near = p.embs.length ? match(emb, p.embs, 1).max : 0;
    if (near >= DEDUPE_SIM) {
      return { added: false, duplicate: true, faceW, sim: +near.toFixed(4), reason: `与已有样本几乎相同（${Math.round(near * 100)}%），已跳过` };
    }

    let thumb = '';
    if (face.kps) {
      try {
        const crop = cropFace(rgba, w, h, face.kps, 112);
        if (crop) {
          thumb = `${randomUUID()}.jpg`;
          const enc = jpeg.encode({ data: crop.data, width: crop.size, height: crop.size }, 82);
          await writeFile(join(this.thumbsDir, thumb), enc.data);
        }
      } catch {
        thumb = '';
      }
    }

    const t = nowIso();
    const info = this.db.prepare(`INSERT INTO faces
      (person_id, source, file, thumb, quality, face_w, det_score, emb, created_at)
      VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      p.id, source, String(file).slice(0, 300), thumb,
      +Number(face.score).toFixed(3), Math.round(face.faceW * 10) / 10, +Number(face.score).toFixed(3),
      Buffer.from(new Uint8Array(emb.buffer, emb.byteOffset, emb.byteLength)), t,
    );
    this.db.prepare('UPDATE persons SET updated_at = ? WHERE id = ?').run(t, p.id);
    bumpRevision(this.db);
    this.reload();

    return {
      added: true,
      id: Number(info.lastInsertRowid),
      faceW,
      sim: near ? +near.toFixed(4) : null,
      total: this.persons.get(p.id)?.faces.length ?? 0,
      thumb: thumb ? `/thumbs/${thumb}` : '',
    };
  }

  // 只用于从 v1 son-template.json 迁移；不重新提特征，也不需要缩略图
  importEmbeddings(name, rows, note = '') {
    const clean = cleanName(name);
    let person = this.db.prepare('SELECT * FROM persons WHERE name = ?').get(clean);
    const t = nowIso();
    this.db.exec('BEGIN');
    try {
      if (!person) {
        const id = Number(this.db.prepare('INSERT INTO persons(name, note, created_at, updated_at) VALUES(?, ?, ?, ?)')
          .run(clean, note, t, t).lastInsertRowid);
        person = this.db.prepare('SELECT * FROM persons WHERE id = ?').get(id);
      }
      const st = this.db.prepare(`INSERT INTO faces
        (person_id, source, file, thumb, quality, face_w, det_score, emb, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)`);
      for (const r of rows) {
        st.run(person.id, 'legacy', String(r.file || '').slice(0, 300), '', 0, 0, 0,
          Buffer.from(new Uint8Array(r.emb.buffer, r.emb.byteOffset, r.emb.byteLength)), t);
      }
      this.db.prepare('UPDATE persons SET updated_at = ? WHERE id = ?').run(t, person.id);
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
    bumpRevision(this.db);
    this.reload();
    return { personId: Number(person.id), count: rows.length };
  }

  setFaceThumb(faceId, thumb) {
    this.db.prepare('UPDATE faces SET thumb = ? WHERE id = ?').run(thumb, Number(faceId));
    this.reload();
  }
}
