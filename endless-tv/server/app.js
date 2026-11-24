// The Express app: the public guide and watch pages, the read API they use, and the upload API the GPU boxes use.
const crypto = require('crypto');
const fsp = require('fs/promises');
const path = require('path');
const express = require('express');
const { Store, HttpError } = require('./store');
const { receive, imageExtension } = require('./upload');
const { looksLikeMp4, mp4Duration } = require('./mp4');

const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const CSP = [
  "default-src 'self'",
  "style-src 'self' https://fonts.googleapis.com",
  'font-src https://fonts.gstatic.com',
  "img-src 'self' data:",
  "media-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "frame-ancestors 'none'",
].join('; ');

const mediaUrl = (channel, file) => (file ? `/media/${channel.id}/${file}` : null);

// A channel as the public pages see it.
function publicChannel(channel) {
  return {
    id: channel.id,
    number: channel.number,
    name: channel.name,
    genre: channel.genre,
    blurb: channel.blurb,
    schedule: channel.schedule,
    logo: mediaUrl(channel, channel.logo),
    poster: mediaUrl(channel, channel.poster),
    clips: channel.clips.map((clip) => ({
      id: clip.id, url: mediaUrl(channel, clip.file), duration: clip.duration, title: clip.title, prompt: clip.prompt,
      seed: clip.seed, generateS: clip.generateS, run: clip.run, createdAt: clip.createdAt,
    })),
    updatedAt: channel.updatedAt,
  };
}

function finiteOrUndefined(value) {
  const number = Number(value);
  return value === undefined || value === '' || !Number.isFinite(number) ? undefined : number;
}

// Only requests carrying the upload token may write. Without a token configured, nothing may.
function requireToken(token) {
  const expected = token ? crypto.createHash('sha256').update(token).digest() : null;
  return (req, res, next) => {
    if (!expected) return next(new HttpError(503, 'uploads are off: set INGEST_TOKEN on the server'));
    const match = /^Bearer (.+)$/.exec(req.headers.authorization || '');
    const given = crypto.createHash('sha256').update(match ? match[1] : '').digest();
    if (!match || !crypto.timingSafeEqual(given, expected)) return next(new HttpError(401, 'wrong or missing token'));
    return next();
  };
}

function createApp({ store, token = '', maxUploadBytes = 300e6, persistent = false }) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', true);  // behind Railway's proxy
  const auth = requireToken(token);
  const json = express.json({ limit: '1mb' });

  app.use((req, res, next) => {
    res.set({ 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'strict-origin-when-cross-origin', 'Content-Security-Policy': CSP });
    next();
  });

  const channelParam = (req) => {
    if (!Store.validId(req.params.id)) throw new HttpError(400, 'channel ids are lowercase letters, digits and dashes');
    return req.params.id;
  };

  app.get('/health', async (req, res) => {
    res.json({ status: 'ok', persistent, uploads: Boolean(token), ...(await store.usage()), timestamp: new Date().toISOString() });
  });

  // --- read API (public) -------------------------------------------------------------------------------------------
  app.get('/api/channels', (req, res) => {
    res.set('Cache-Control', 'no-cache');
    res.json({ now: Date.now(), channels: store.list().map(publicChannel) });
  });

  app.get('/api/channels/:id', (req, res) => {
    res.set('Cache-Control', 'no-cache');
    res.json({ now: Date.now(), channel: publicChannel(store.require(channelParam(req))) });
  });

  app.get('/api/runs', async (req, res) => {
    res.set('Cache-Control', 'no-cache');
    res.json({ runs: await store.listRuns() });
  });

  // --- upload API (token) ------------------------------------------------------------------------------------------
  app.put('/api/channels/:id', auth, json, async (req, res) => {
    const channel = await store.upsertChannel(channelParam(req), req.body || {});
    res.json({ channel: publicChannel(channel) });
  });

  app.delete('/api/channels/:id', auth, async (req, res) => {
    await store.removeChannel(channelParam(req));
    res.status(204).end();
  });

  app.put('/api/channels/:id/:kind', auth, async (req, res) => {
    const id = channelParam(req);
    if (!['logo', 'poster'].includes(req.params.kind)) throw new HttpError(404, 'not found');
    store.require(id);
    const upload = await receive(req, store.tmpDir, Math.min(maxUploadBytes, 10e6));
    try {
      const extension = await imageExtension(upload.path, upload.head);
      if (!extension) throw new HttpError(415, 'send a PNG, JPEG, WebP or script-free SVG');
      const channel = await store.setImage(id, req.params.kind, upload.path, extension);
      res.json({ channel: publicChannel(channel) });
    } finally {
      await fsp.rm(upload.path, { force: true });
    }
  });

  // A clip: the MP4 as the body, its details in the query string (and a long prompt in X-Clip-Prompt, URI-encoded).
  app.post('/api/channels/:id/clips', auth, async (req, res) => {
    const id = channelParam(req);
    store.require(id);
    const upload = await receive(req, store.tmpDir, maxUploadBytes);
    try {
      if (!looksLikeMp4(upload.head)) throw new HttpError(415, 'send an MP4');
      const duration = await mp4Duration(upload.path);
      if (!(duration > 0)) throw new HttpError(422, 'could not read the MP4 duration');
      let prompt = req.query.prompt;
      if (req.headers['x-clip-prompt']) {
        try {
          prompt = decodeURIComponent(req.headers['x-clip-prompt']);
        } catch {
          throw new HttpError(400, 'X-Clip-Prompt must be URI-encoded');
        }
      }
      const clip = await store.addClip(id, upload.path, {
        duration, bytes: upload.bytes, title: req.query.title, prompt, run: req.query.run,
        seed: finiteOrUndefined(req.query.seed), sequence: finiteOrUndefined(req.query.sequence), generateS: finiteOrUndefined(req.query.generate_s),
      });
      res.status(201).json({ clip: publicChannel({ ...store.get(id), clips: [clip] }).clips[0] });
    } finally {
      await fsp.rm(upload.path, { force: true });
    }
  });

  app.delete('/api/channels/:id/clips/:clipId', auth, async (req, res) => {
    await store.removeClip(channelParam(req), req.params.clipId);
    res.status(204).end();
  });

  app.post('/api/runs', auth, json, async (req, res) => {
    if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body)) throw new HttpError(400, 'send the run as a JSON object');
    res.status(201).json({ run: await store.addRun(req.body) });
  });

  // --- media and pages ---------------------------------------------------------------------------------------------
  // Every media file name is new, so browsers and Railway's edge cache may keep them for good. Uploaded files are
  // sandboxed: an SVG opened on its own can't run anything.
  app.use('/media', express.static(store.mediaDir, {
    immutable: true,
    maxAge: '365d',
    index: false,
    dotfiles: 'ignore',
    setHeaders: (res) => res.set('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; sandbox"),
  }));

  app.get('/watch/:id', (req, res) => {
    channelParam(req);
    res.sendFile(path.join(PUBLIC_DIR, 'watch.html'));
  });

  app.use(express.static(PUBLIC_DIR, { extensions: ['html'], maxAge: '5m' }));

  app.use((req, res) => res.status(404).json({ error: 'not found' }));

  // eslint-disable-next-line no-unused-vars
  app.use((error, req, res, next) => {
    const status = error.status || error.statusCode || 500;
    if (status >= 500 && status !== 503) console.error(error);
    res.status(status).json({ error: status >= 500 && status !== 503 ? 'server error' : error.message });
  });

  return app;
}

module.exports = { createApp, publicChannel };
