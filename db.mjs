// SQLite 数据层：人物、人脸样本、设置。使用 Node 内置 node:sqlite，无额外依赖。
import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

export const DEFAULT_THRESHOLD = 0.45;

export function openDb(dataDir) {
  mkdirSync(join(dataDir, 'thumbs'), { recursive: true });
  const db = new DatabaseSync(join(dataDir, 'app.db'));
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;

    CREATE TABLE IF NOT EXISTS persons (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      name       TEXT NOT NULL UNIQUE,
      note       TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS faces (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
      source     TEXT NOT NULL DEFAULT 'upload',
      file       TEXT NOT NULL DEFAULT '',
      thumb      TEXT NOT NULL DEFAULT '',
      quality    REAL NOT NULL DEFAULT 0,
      face_w     REAL NOT NULL DEFAULT 0,
      det_score  REAL NOT NULL DEFAULT 0,
      emb        BLOB NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

    CREATE TABLE IF NOT EXISTS settings (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS guard_profiles (
      person_id  INTEGER PRIMARY KEY REFERENCES persons(id) ON DELETE CASCADE,
      enabled    INTEGER NOT NULL DEFAULT 0,
      warn_cm    REAL NOT NULL DEFAULT 60,
      ref_ratio  REAL NOT NULL DEFAULT 0,
      ref_cm     REAL NOT NULL DEFAULT 60,
      updated_at TEXT NOT NULL
    );
  `);
  db.prepare('INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)').run('threshold', String(DEFAULT_THRESHOLD));
  db.prepare('INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)').run('revision', '1');
  db.prepare('INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)').run('guard_enabled', '1');
  return db;
}

export const nowIso = () => new Date().toISOString();

export function getSetting(db, key, fallback = '') {
  const row = db.prepare('SELECT value FROM settings WHERE key = ?').get(key);
  return row ? row.value : fallback;
}

export function setSetting(db, key, value) {
  db.prepare('INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value')
    .run(key, String(value));
}

export function bumpRevision(db) {
  db.exec("UPDATE settings SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'");
}

export function readRevision(db) {
  return Number(getSetting(db, 'revision', '0'));
}
