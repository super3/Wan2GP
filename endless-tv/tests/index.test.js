const path = require('path');
const { config, start } = require('../server/index');
const { tmpDir } = require('./helpers');

test('reads its settings from the environment, the Railway volume first', () => {
  expect(config({})).toMatchObject({ port: 3000, persistent: false, token: '', maxUploadBytes: 300e6, keepClips: 0 });
  expect(config({}).dataDir).toBe(path.resolve('data'));
  expect(config({ RAILWAY_VOLUME_MOUNT_PATH: '/data', PORT: '8080', INGEST_TOKEN: 't', MAX_UPLOAD_MB: '50', KEEP_CLIPS: '200' }))
    .toEqual({ port: 8080, dataDir: '/data', persistent: true, token: 't', maxUploadBytes: 50e6, keepClips: 200 });
  expect(config({ RAILWAY_VOLUME_MOUNT_PATH: '/data', DATA_DIR: '/other' }).dataDir).toBe('/other');
  expect(config().port).toBe(Number(process.env.PORT) || 3000);  // the process environment by default
});

async function startAndCheck(env) {
  const log = jest.spyOn(console, 'log').mockImplementation(() => {});
  const server = await start(env);
  if (!server.listening) await new Promise((resolve) => server.on('listening', resolve));
  const res = await fetch(`http://127.0.0.1:${server.address().port}/health`);
  const health = await res.json();
  await new Promise((resolve) => server.close(resolve));
  const message = log.mock.calls[0][0];
  log.mockRestore();
  return { health, message };
}

test('starts and answers its health check, saying where its data is', async () => {
  const off = await startAndCheck({ PORT: '0', DATA_DIR: tmpDir() });
  expect(off.health).toMatchObject({ status: 'ok', persistent: true, uploads: false });
  expect(off.message).toMatch(/uploads off/);
  const on = await startAndCheck({ PORT: '0', DATA_DIR: tmpDir(), INGEST_TOKEN: 'x' });
  expect(on.message).toMatch(/uploads on/);
  expect(on.message).not.toMatch(/not persistent/);
});

test('says when its data is not persistent', async () => {
  const cwd = process.cwd();
  process.chdir(tmpDir());  // with no volume and no DATA_DIR it writes to ./data
  try {
    const { message } = await startAndCheck({ PORT: '0' });
    expect(message).toMatch(/not persistent/);
  } finally {
    process.chdir(cwd);
  }
});
