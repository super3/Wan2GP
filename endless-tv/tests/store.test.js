const fs = require('fs');
const path = require('path');
const { Store, HttpError } = require('../server/store');
const { tmpDir, writeTmp, PNG } = require('./helpers');

async function fresh(options) {
  return new Store(tmpDir(), options).init();
}

function upload(store, name = 'x', data = 'data') {
  return writeTmp(store.tmpDir, `.upload-${name}-${Math.random()}`, data);
}

test('ids are lowercase letters, digits and dashes', () => {
  expect(Store.validId('laugh-track')).toBe(true);
  expect(Store.validId('a'.repeat(40))).toBe(true);
  for (const bad of ['', '-x', 'Upper', 'a/b', '..', 'a'.repeat(41), 7]) expect(Store.validId(bad)).toBe(false);
});

test('creates a channel, updates only the fields given, and keeps it across a restart', async () => {
  const store = await fresh();
  await store.upsertChannel('drift', { number: '106', name: '  Drift  ', genre: 'Slow TV', schedule: [{ title: 'Fire', minutes: 300, badges: ['TV-G', 5, ''], extra: 'dropped' }] });
  await store.upsertChannel('drift', { blurb: 'Rain on the window.' });
  const channel = store.get('drift');
  expect(channel).toMatchObject({ number: 106, name: 'Drift', genre: 'Slow TV', blurb: 'Rain on the window.' });
  expect(channel.schedule).toEqual([{ title: 'Fire', episode: '', blurb: '', badges: ['TV-G'], isNew: false, minutes: 300 }]);
  const again = await new Store(store.dir).init();
  expect(again.get('drift')).toEqual(channel);
});

test('a channel is pre-generated unless marked live', async () => {
  const store = await fresh();
  expect((await store.upsertChannel('drift', {})).live).toBe(false);
  expect((await store.upsertChannel('news', { live: true })).live).toBe(true);
  expect((await store.upsertChannel('news', { name: 'News' })).live).toBe(true);  // other updates keep it
  expect((await store.upsertChannel('news', { live: 'yes' })).live).toBe(false);  // only true means live
});

test('rejects a bad number or schedule and leaves the channel unchanged', async () => {
  const store = await fresh();
  await store.upsertChannel('news', { number: 101 });
  await expect(store.upsertChannel('news', { number: -1 })).rejects.toMatchObject({ status: 400 });
  await expect(store.upsertChannel('news', { schedule: 'soon' })).rejects.toMatchObject({ status: 400 });
  await expect(store.upsertChannel('news', { schedule: new Array(49).fill({ minutes: 1 }) })).rejects.toMatchObject({ status: 400 });
  await expect(store.upsertChannel('news', { schedule: [{ title: 'Forever', minutes: 2000 }] })).rejects.toMatchObject({ status: 400 });
  await expect(store.upsertChannel('news', { schedule: [null] })).rejects.toBeInstanceOf(HttpError);
  expect(store.get('news').number).toBe(101);
  const untitled = await store.upsertChannel('news', { schedule: [{ minutes: 30 }] });
  expect(untitled.schedule[0].title).toBe('Untitled');
});

test('lists channels by number', async () => {
  const store = await fresh();
  await store.upsertChannel('b', { number: 2 });
  await store.upsertChannel('a', { number: 2 });
  await store.upsertChannel('c', { number: 1 });
  expect(store.list().map((c) => c.id)).toEqual(['c', 'a', 'b']);
});

test('a logo or poster replaces the previous file', async () => {
  const store = await fresh();
  await store.upsertChannel('doodle', {});
  await store.setImage('doodle', 'logo', upload(store, 'logo', PNG), 'png');
  const first = store.get('doodle').logo;
  await store.setImage('doodle', 'logo', upload(store, 'logo2', PNG), 'png');
  const second = store.get('doodle').logo;
  expect(second).not.toBe(first);
  expect(fs.existsSync(path.join(store.mediaDir, 'doodle', first))).toBe(false);
  expect(fs.existsSync(path.join(store.mediaDir, 'doodle', second))).toBe(true);
  expect(() => store.setImage('doodle', 'banner', 'x', 'png')).toThrow(HttpError);
  await expect(store.setImage('nobody', 'logo', upload(store), 'png')).rejects.toMatchObject({ status: 404 });
});

test('clips are kept in sequence order, appended by default', async () => {
  const store = await fresh();
  await store.upsertChannel('lt', {});
  const one = await store.addClip('lt', upload(store), { duration: 5, bytes: 4, title: 'one', seed: 1 });
  const two = await store.addClip('lt', upload(store), { duration: 5, bytes: 4, title: 'two', generateS: 4.2, run: 'run-1' });
  const first = await store.addClip('lt', upload(store), { duration: 5, bytes: 4, title: 'zero', sequence: 0.5, run: 'Bad Run!' });
  expect(store.get('lt').clips.map((c) => c.title)).toEqual(['zero', 'one', 'two']);
  expect(one.sequence).toBe(1);
  expect(two).toMatchObject({ sequence: 2, generateS: 4.2, run: 'run-1', seed: null });
  expect(first.run).toBeNull();
  expect(fs.existsSync(path.join(store.mediaDir, 'lt', one.file))).toBe(true);
  expect(await store.usage()).toEqual({ channels: 1, clips: 3, clipBytes: 12 });
});

test('clips with the same sequence keep their upload order; a clip without a size counts as 0 bytes', async () => {
  const store = await fresh();
  await store.upsertChannel('lt', {});
  await store.addClip('lt', upload(store), { duration: 5, title: 'first', sequence: 1 });
  await new Promise((resolve) => setTimeout(resolve, 5));
  await store.addClip('lt', upload(store), { duration: 5, title: 'second', sequence: 1 });
  expect(store.get('lt').clips.map((c) => c.title)).toEqual(['first', 'second']);
  expect((await store.usage()).clipBytes).toBe(0);
});

test('keepClips drops the oldest clips and their files', async () => {
  const store = await fresh({ keepClips: 2 });
  await store.upsertChannel('lt', {});
  const old = await store.addClip('lt', upload(store), { duration: 5 });
  await new Promise((resolve) => setTimeout(resolve, 5));
  await store.addClip('lt', upload(store), { duration: 5 });
  await new Promise((resolve) => setTimeout(resolve, 5));
  await store.addClip('lt', upload(store), { duration: 5, sequence: 0 });
  expect(store.get('lt').clips).toHaveLength(2);
  expect(store.get('lt').clips.find((c) => c.id === old.id)).toBeUndefined();
  expect(fs.existsSync(path.join(store.mediaDir, 'lt', old.file))).toBe(false);
});

test('removes a clip or a whole channel with its files', async () => {
  const store = await fresh();
  await store.upsertChannel('hw', {});
  const clip = await store.addClip('hw', upload(store), { duration: 5 });
  await expect(store.removeClip('hw', 'nope')).rejects.toMatchObject({ status: 404 });
  await store.removeClip('hw', clip.id);
  expect(store.get('hw').clips).toEqual([]);
  expect(fs.existsSync(path.join(store.mediaDir, 'hw', clip.file))).toBe(false);
  await store.addClip('hw', upload(store), { duration: 5 });
  await store.removeChannel('hw');
  expect(store.get('hw')).toBeUndefined();
  expect(fs.existsSync(path.join(store.mediaDir, 'hw'))).toBe(false);
  await expect(store.removeChannel('hw')).rejects.toMatchObject({ status: 404 });
  await expect(store.addClip('hw', upload(store), { duration: 5 })).rejects.toMatchObject({ status: 404 });
});

test('a failed write does not block the next one', async () => {
  const store = await fresh();
  await expect(store.write(async () => { throw new Error('boom'); })).rejects.toThrow('boom');
  await store.upsertChannel('ok', {});
  expect(store.get('ok')).toBeDefined();
});

test('runs are saved as sent, newest first', async () => {
  const store = await fresh();
  const first = await store.addRun({ gpu: 'RTX PRO 6000', realtimeX: 1.07 });
  await new Promise((resolve) => setTimeout(resolve, 5));
  const second = await store.addRun({ id: 'preload-v2012', cycle_s: 4.83 });
  expect(first.id).toMatch(/^r/);
  expect(second.id).toBe('preload-v2012');
  fs.writeFileSync(path.join(store.runsDir, 'notes.txt'), 'ignored');
  const runs = await store.listRuns();
  expect(runs.map((r) => r.id)).toEqual(['preload-v2012', first.id]);
  expect(await store.listRuns(1)).toHaveLength(1);
});

test('init clears uploads a restart cut short, and fails on an unreadable channels file', async () => {
  const store = await fresh();
  const leftover = upload(store, 'leftover');
  await new Store(store.dir).init();
  expect(fs.existsSync(leftover)).toBe(false);
  const broken = tmpDir();
  fs.mkdirSync(path.join(broken, 'channels.json'));
  await expect(new Store(broken).init()).rejects.toMatchObject({ code: 'EISDIR' });
});
