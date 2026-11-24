const request = require('supertest');
const { createApp } = require('../server/app');
const { Store } = require('../server/store');
const { mp4, box, ftyp, PNG, SVG, tmpDir } = require('./helpers');

const TOKEN = 'test-token';
const auth = { Authorization: `Bearer ${TOKEN}` };

async function setup(options = {}) {
  const store = await new Store(tmpDir()).init();
  const app = createApp({ store, token: TOKEN, persistent: true, ...options });
  return { store, app };
}

async function withChannel() {
  const ctx = await setup();
  await request(ctx.app).put('/api/channels/laugh-track').set(auth).send({ number: 103, name: 'Laugh Track', genre: 'Comedy' }).expect(200);
  return ctx;
}

describe('read API', () => {
  test('health says what is stored and whether uploads are on', async () => {
    const { app } = await setup();
    const res = await request(app).get('/health').expect(200);
    expect(res.body).toMatchObject({ status: 'ok', persistent: true, uploads: true, channels: 0, clips: 0 });
  });

  test('lists channels with the server clock, and returns one channel', async () => {
    const { app } = await withChannel();
    const list = await request(app).get('/api/channels').expect(200);
    expect(list.body.now).toBeGreaterThan(0);
    expect(list.body.channels).toEqual([expect.objectContaining({ id: 'laugh-track', number: 103, logo: null, clips: [] })]);
    expect(list.headers['cache-control']).toBe('no-cache');
    const one = await request(app).get('/api/channels/laugh-track').expect(200);
    expect(one.body.channel.name).toBe('Laugh Track');
    await request(app).get('/api/channels/nobody').expect(404);
    await request(app).get('/api/channels/Bad_Id').expect(400);
  });

  test('every response carries the security headers', async () => {
    const { app } = await setup();
    const res = await request(app).get('/api/channels');
    expect(res.headers['x-content-type-options']).toBe('nosniff');
    expect(res.headers['content-security-policy']).toContain("default-src 'self'");
    expect(res.headers['x-powered-by']).toBeUndefined();
  });
});

describe('upload token', () => {
  test('writes need the token', async () => {
    const { app } = await setup();
    await request(app).put('/api/channels/x').send({}).expect(401);
    await request(app).put('/api/channels/x').set('Authorization', 'Bearer wrong').send({}).expect(401);
    await request(app).put('/api/channels/x').set('Authorization', TOKEN).send({}).expect(401);
    await request(app).put('/api/channels/x').set(auth).send({}).expect(200);
  });

  test('without a token configured, uploads are off', async () => {
    const { app } = await setup({ token: '' });
    const res = await request(app).put('/api/channels/x').set(auth).send({}).expect(503);
    expect(res.body.error).toMatch(/INGEST_TOKEN/);
    expect((await request(app).get('/health')).body.uploads).toBe(false);
  });
});

describe('channels', () => {
  test('a channel can be created without a body, and the app has sane defaults', async () => {
    const store = await new Store(tmpDir()).init();
    const bare = createApp({ store });
    expect((await request(bare).get('/health')).body).toMatchObject({ persistent: false, uploads: false });
    const { app } = await setup();
    const res = await request(app).put('/api/channels/blank').set(auth).expect(200);
    expect(res.body.channel).toMatchObject({ id: 'blank', name: 'blank', number: 0 });
  });

  test('a bad body is refused', async () => {
    const { app } = await setup();
    await request(app).put('/api/channels/x').set(auth).send({ number: 'lots' }).expect(400);
    await request(app).put('/api/channels/x').set(auth).set('Content-Type', 'application/json').send('{not json').expect(400);
  });

  test('a channel can be deleted', async () => {
    const { app } = await withChannel();
    await request(app).delete('/api/channels/laugh-track').set(auth).expect(204);
    await request(app).get('/api/channels/laugh-track').expect(404);
    await request(app).delete('/api/channels/laugh-track').set(auth).expect(404);
  });
});

describe('images', () => {
  test('a PNG logo and an SVG poster are stored and served sandboxed, cacheable for good', async () => {
    const { app } = await withChannel();
    const logo = await request(app).put('/api/channels/laugh-track/logo').set(auth).set('Content-Type', 'image/png').send(PNG).expect(200);
    expect(logo.body.channel.logo).toMatch(/^\/media\/laugh-track\/logo-.+\.png$/);
    const poster = await request(app).put('/api/channels/laugh-track/poster').set(auth).set('Content-Type', 'image/svg+xml').send(SVG).expect(200);
    const media = await request(app).get(poster.body.channel.poster).expect(200);
    expect(media.headers['content-security-policy']).toContain('sandbox');
    expect(media.headers['cache-control']).toContain('immutable');
  });

  test('anything else is refused', async () => {
    const { app, store } = await withChannel();
    await request(app).put('/api/channels/laugh-track/logo').set(auth).set('Content-Type', 'image/svg+xml').send('<svg><script>x()</script></svg>').expect(415);
    await request(app).put('/api/channels/laugh-track/banner').set(auth).set('Content-Type', 'image/png').send(PNG).expect(404);
    await request(app).put('/api/channels/nobody/logo').set(auth).set('Content-Type', 'image/png').send(PNG).expect(404);
    const big = Buffer.concat([PNG, Buffer.alloc(11e6)]);
    await request(app).put('/api/channels/laugh-track/logo').set(auth).set('Content-Type', 'image/png').send(big).expect(413);
    expect(store.get('laugh-track').logo).toBeNull();
  });
});

describe('clips', () => {
  test('an MP4 is stored with its duration read from the file and its details from the query', async () => {
    const { app, store } = await withChannel();
    const prompt = 'A comedian on a brick-wall stage — “tiny apartment”';
    const res = await request(app).post('/api/channels/laugh-track/clips?title=Open%20mic&seed=42&generate_s=4.8&run=run-7&sequence=3')
      .set(auth).set('Content-Type', 'video/mp4').set('X-Clip-Prompt', encodeURIComponent(prompt)).send(mp4({ duration: 5.17 })).expect(201);
    expect(res.body.clip).toMatchObject({ title: 'Open mic', seed: 42, generateS: 4.8, run: 'run-7', prompt });
    expect(res.body.clip.duration).toBeCloseTo(5.17, 3);
    expect(res.body.clip.url).toMatch(/^\/media\/laugh-track\/c.+\.mp4$/);
    expect(store.get('laugh-track').clips[0].sequence).toBe(3);
    const ranged = await request(app).get(res.body.clip.url).set('Range', 'bytes=0-9').expect(206);
    expect(ranged.body.length).toBe(10);
    const plain = await request(app).post('/api/channels/laugh-track/clips?prompt=short&seed=&sequence=x').set(auth).set('Content-Type', 'video/mp4').send(mp4()).expect(201);
    expect(plain.body.clip).toMatchObject({ prompt: 'short', seed: null });
  });

  test('anything but a readable MP4 is refused, and nothing is left behind', async () => {
    const { app, store } = await withChannel();
    await request(app).post('/api/channels/laugh-track/clips').set(auth).set('Content-Type', 'video/mp4').send(Buffer.from('not a video')).expect(415);
    await request(app).post('/api/channels/laugh-track/clips').set(auth).set('Content-Type', 'video/mp4').send(Buffer.concat([ftyp(), box('mdat')])).expect(422);
    await request(app).post('/api/channels/laugh-track/clips').set(auth).set('Content-Type', 'video/mp4').set('X-Clip-Prompt', '%E0%A4%A').send(mp4()).expect(400);
    await request(app).post('/api/channels/nobody/clips').set(auth).set('Content-Type', 'video/mp4').send(mp4()).expect(404);
    const { app: small } = { app: createApp({ store, token: TOKEN, maxUploadBytes: 100 }) };
    await request(small).post('/api/channels/laugh-track/clips').set(auth).set('Content-Type', 'video/mp4').send(mp4()).expect(413);
    expect(store.get('laugh-track').clips).toEqual([]);
    expect(require('fs').readdirSync(store.tmpDir)).toEqual([]);
  });

  test('a clip can be deleted', async () => {
    const { app } = await withChannel();
    const res = await request(app).post('/api/channels/laugh-track/clips').set(auth).set('Content-Type', 'video/mp4').send(mp4()).expect(201);
    await request(app).delete(`/api/channels/laugh-track/clips/${res.body.clip.id}`).set(auth).expect(204);
    await request(app).get(res.body.clip.url).expect(404);
    await request(app).delete(`/api/channels/laugh-track/clips/${res.body.clip.id}`).set(auth).expect(404);
  });
});

describe('runs', () => {
  test('a GPU run is saved and listed publicly', async () => {
    const { app } = await setup();
    const res = await request(app).post('/api/runs').set(auth).send({ id: 'preload-v2012', gpu: 'RTX PRO 6000', cycle_s: 4.83 }).expect(201);
    expect(res.body.run).toMatchObject({ id: 'preload-v2012', cycle_s: 4.83 });
    const list = await request(app).get('/api/runs').expect(200);
    expect(list.body.runs.map((r) => r.id)).toEqual(['preload-v2012']);
    await request(app).post('/api/runs').set(auth).send([1, 2]).expect(400);
    await request(app).post('/api/runs').set(auth).set('Content-Type', 'text/plain').send('x').expect(400);
  });
});

describe('pages', () => {
  test('the guide, the watch page and the scripts are served', async () => {
    const { app } = await setup();
    expect((await request(app).get('/').expect(200)).text).toContain('id="guide"');
    expect((await request(app).get('/watch/drift').expect(200)).text).toContain('id="watch"');
    await request(app).get('/watch/Bad_Id').expect(400);
    await request(app).get('/js/live.js').expect(200).expect('Content-Type', /javascript/);
    await request(app).get('/nothing-here').expect(404);
  });

  test('an unexpected error is a 500 without details', async () => {
    const { app, store } = await setup();
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    store.listRuns = async () => {
      throw new Error('disk on fire');
    };
    const res = await request(app).get('/api/runs').expect(500);
    expect(res.body).toEqual({ error: 'server error' });
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});
