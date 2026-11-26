// tools/publish.py against a running server, as a GPU box would use it.
const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const { createApp } = require('../server/app');
const { Store } = require('../server/store');
const { mp4, PNG, SVG, tmpDir, writeTmp } = require('./helpers');

const run = promisify(execFile);
const SCRIPT = path.join(__dirname, '..', 'tools', 'publish.py');
const TOKEN = 'box-token';
let server;
let store;
let url;

beforeAll(async () => {
  store = await new Store(tmpDir()).init();
  server = createApp({ store, token: TOKEN }).listen(0);
  await new Promise((resolve) => server.on('listening', resolve));
  url = `http://127.0.0.1:${server.address().port}`;
});

afterAll(() => new Promise((resolve) => server.close(resolve)));

function publish(args, env = {}) {
  return run('python3', [SCRIPT, ...args], { env: { PATH: process.env.PATH, ENDLESS_TV_URL: url, ENDLESS_TV_TOKEN: TOKEN, ...env } });
}

test('seeds channels with their logos, posters and clips', async () => {
  const media = tmpDir();
  writeTmp(media, 'logo.svg', SVG);
  writeTmp(media, 'poster.png', PNG);
  writeTmp(media, 'clip.mp4', mp4({ duration: 6 }));
  const seed = writeTmp(media, 'seed.json', JSON.stringify({ channels: [
    { id: 'drift', number: 106, name: 'Drift', genre: 'Slow TV', live: true, logo: 'logo.svg', poster: 'poster.png', clips: ['clip.mp4', 'missing.mp4'],
      schedule: [{ title: 'Cabin Fire', minutes: 300 }] },
  ] }));
  const { stdout, stderr } = await publish(['seed', seed]);
  expect(stdout).toContain('drift: channel 106, 1 clip(s)');
  expect(stderr).toContain('missing.mp4');
  const drift = store.get('drift');
  expect(drift).toMatchObject({ name: 'Drift', live: true, schedule: [expect.objectContaining({ title: 'Cabin Fire' })] });
  expect(drift.logo).toMatch(/\.svg$/);
  expect(drift.poster).toMatch(/\.png$/);
  expect(drift.clips[0].duration).toBeCloseTo(6, 3);
  expect((await publish(['list'], { ENDLESS_TV_TOKEN: '' })).stdout).toMatch(/106\s+drift\s+1 clips\s+Drift/);
});

test('publishes a channel, a clip with its prompt, a run, then deletes them', async () => {
  const dir = tmpDir();
  const schedule = writeTmp(dir, 'schedule.json', JSON.stringify([{ title: 'Open Mic', minutes: 30 }]));
  await publish(['channel', 'laugh-track', '--number', '103', '--name', 'Laugh Track', '--schedule', schedule, '--logo', writeTmp(dir, 'l.png', PNG)]);
  expect(store.get('laugh-track').live).toBe(false);
  await publish(['channel', 'laugh-track', '--live']);
  expect(store.get('laugh-track').live).toBe(true);
  await publish(['channel', 'laugh-track', '--no-live']);
  expect(store.get('laugh-track').live).toBe(false);
  const promptFile = writeTmp(dir, 'prompt.txt', 'A comedian — “tiny apartment”, 100% real');
  const clip = await publish(['clip', 'laugh-track', writeTmp(dir, 'c.mp4', mp4({ duration: 5.17 })), '--title', 'Bit 1', '--prompt-file', promptFile, '--seed', '9']);
  expect(clip.stdout).toMatch(/clip c\S+: 5\.17 s/);
  const saved = store.get('laugh-track').clips[0];
  expect(saved).toMatchObject({ title: 'Bit 1', seed: 9, prompt: 'A comedian — “tiny apartment”, 100% real' });
  expect((await publish(['run', writeTmp(dir, 'r.json', JSON.stringify({ realtimeX: 1.07 })), '--id', 'rt-check'])).stdout).toContain('run rt-check');
  expect((await store.listRuns())[0]).toMatchObject({ id: 'rt-check', realtimeX: 1.07 });
  await publish(['delete-clip', 'laugh-track', saved.id]);
  expect(store.get('laugh-track').clips).toEqual([]);
  await publish(['delete-channel', 'laugh-track']);
  expect(store.get('laugh-track')).toBeUndefined();
});

test('publishes an H3 Live session folder: its clips in order and its numbers as a run', async () => {
  const dir = tmpDir('live_20261124-001054-');
  await publish(['channel', 'doodle', '--name', 'Doodle']);
  writeTmp(dir, 'clip_00000.mp4', mp4({ duration: 5.17 }));
  writeTmp(dir, 'clip_00001.mp4', mp4({ duration: 5.17 }));
  fs.writeFileSync(path.join(dir, 'session.json'), JSON.stringify({ clips: 3, cycle_s: 4.83, realtime_x: 1.07, records: [
    { clip: 0, prompt: 'Pickles wants the cookie', seed: 42, generate_s: 4.9, mp4: 'clip_00000.mp4' },
    { clip: 1, prompt: 'Pickles brings a ladder', seed: 43, generate_s: 4.8, mp4: 'clip_00001.mp4' },
    { clip: 2, error: 'out of memory' },
    { clip: 3, mp4: 'clip_00003.mp4' },
  ] }));
  const { stdout } = await publish(['session', 'doodle', dir]);
  expect(stdout).toMatch(/run s-live-20261124-001054-\S*: 2 clip\(s\) on doodle/);
  const clips = store.get('doodle').clips;
  expect(clips.map((c) => [c.title, c.seed, c.generateS])).toEqual([['Clip 1', 42, 4.9], ['Clip 2', 43, 4.8]]);
  expect(clips[0].run).toMatch(/^s-live-20261124-001054/);
  const runs = await store.listRuns();
  expect(runs.find((r) => r.id === clips[0].run)).toMatchObject({ kind: 'h3-live-session', channel: 'doodle', realtime_x: 1.07 });
});

test('fails clearly without a server, a token, or a reachable host', async () => {
  await expect(publish(['list'], { ENDLESS_TV_URL: '' })).rejects.toMatchObject({ code: 1, stderr: expect.stringContaining('ENDLESS_TV_URL') });
  await expect(publish(['channel', 'x'], { ENDLESS_TV_TOKEN: '' })).rejects.toMatchObject({ code: 1, stderr: expect.stringContaining('ENDLESS_TV_TOKEN') });
  await expect(publish(['channel', 'x'], { ENDLESS_TV_TOKEN: 'wrong' })).rejects.toMatchObject({ stderr: expect.stringContaining('HTTP 401') });
  await expect(publish(['list'], { ENDLESS_TV_URL: 'http://127.0.0.1:9' })).rejects.toMatchObject({ code: 1 });
});
