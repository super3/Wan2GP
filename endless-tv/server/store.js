// Everything the GPU boxes upload, kept on disk (a Railway volume in production):
//   channels.json            the channels: name, number, genre, schedule, logo/poster file, clips
//   media/<channel>/<file>   their clips (MP4), logos and posters; every file name is new, so it can be cached forever
//   runs/<id>.json           results of GPU runs (speed, GPU, settings), as the boxes sent them
// Writes are serialized and channels.json is replaced atomically, so a crash never leaves it half written.
const crypto = require('crypto');
const fs = require('fs/promises');
const path = require('path');

const ID = /^[a-z0-9][a-z0-9-]{0,39}$/;
const IMAGE_KINDS = ['logo', 'poster'];
const TEXT_LIMITS = { name: 60, genre: 30, blurb: 600, title: 120, episode: 120, badge: 12, prompt: 8000 };

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function text(value, limit) {
  return typeof value === 'string' ? value.trim().slice(0, limit) : '';
}

function newId(prefix) {
  return `${prefix}${Date.now().toString(36)}${crypto.randomBytes(4).toString('hex')}`;
}

// A schedule entry as the guide uses it; anything else in the request is dropped.
function cleanProgram(program) {
  const minutes = Math.round(Number(program && program.minutes));
  if (!(minutes >= 1 && minutes <= 24 * 60)) throw new HttpError(400, 'each program needs minutes between 1 and 1440');
  return {
    title: text(program.title, TEXT_LIMITS.title) || 'Untitled',
    episode: text(program.episode, TEXT_LIMITS.episode),
    blurb: text(program.blurb, TEXT_LIMITS.blurb),
    badges: (Array.isArray(program.badges) ? program.badges : []).slice(0, 4).map((badge) => text(badge, TEXT_LIMITS.badge)).filter(Boolean),
    isNew: program.isNew === true,
    minutes,
  };
}

class Store {
  constructor(dir, { keepClips = 0 } = {}) {
    this.dir = dir;
    this.mediaDir = path.join(dir, 'media');
    this.runsDir = path.join(dir, 'runs');
    this.tmpDir = path.join(dir, 'tmp');  // uploads land here first: on the same disk, so placing them is a rename
    this.file = path.join(dir, 'channels.json');
    this.keepClips = keepClips;  // clips kept per channel (oldest dropped first); 0 keeps them all
    this.channels = new Map();
    this.queue = Promise.resolve();
    this.runsCache = new Map();  // limit -> the listing, until the next run arrives
  }

  static validId(id) {
    return typeof id === 'string' && ID.test(id);
  }

  async init() {
    await fs.mkdir(this.mediaDir, { recursive: true });
    await fs.mkdir(this.runsDir, { recursive: true });
    await fs.rm(this.tmpDir, { recursive: true, force: true });  // uploads a restart cut short
    await fs.mkdir(this.tmpDir, { recursive: true });
    let saved = [];
    try {
      saved = JSON.parse(await fs.readFile(this.file, 'utf8'));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    this.channels = new Map(saved.map((channel) => [channel.id, channel]));
    return this;
  }

  // Runs `change` after every earlier write, then saves channels.json.
  write(change) {
    const run = this.queue.then(async () => {
      const result = await change();
      const tmp = `${this.file}.${process.pid}.tmp`;
      await fs.writeFile(tmp, JSON.stringify([...this.channels.values()], null, 1));
      await fs.rename(tmp, this.file);
      return result;
    });
    this.queue = run.catch(() => {});
    return run;
  }

  list() {
    return [...this.channels.values()].sort((a, b) => a.number - b.number || a.id.localeCompare(b.id));
  }

  get(id) {
    return this.channels.get(id);
  }

  require(id) {
    const channel = this.channels.get(id);
    if (!channel) throw new HttpError(404, `no channel ${id}`);
    return channel;
  }

  // Creates the channel or updates the fields given.
  upsertChannel(id, fields) {
    return this.write(async () => {
      const now = new Date().toISOString();
      const channel = this.channels.get(id) || { id, number: 0, name: id, genre: '', blurb: '', live: false, schedule: [], logo: null, poster: null, clips: [], createdAt: now };
      if (fields.number !== undefined) {
        const number = Math.round(Number(fields.number));
        if (!(number >= 0 && number <= 9999)) throw new HttpError(400, 'number must be between 0 and 9999');
        channel.number = number;
      }
      if (fields.live !== undefined) channel.live = fields.live === true;  // runs live (a GPU making it now), not pre-generated
      if (fields.uploadKey !== undefined) {  // a key that may upload clips to this channel only; kept as its hash; null removes it
        if (fields.uploadKey !== null && !(typeof fields.uploadKey === 'string' && fields.uploadKey.length >= 32 && fields.uploadKey.length <= 200)) {
          throw new HttpError(400, 'uploadKey must be 32 to 200 characters, or null');
        }
        channel.uploadKeyHash = fields.uploadKey === null ? null : crypto.createHash('sha256').update(fields.uploadKey).digest('hex');
      }
      if (fields.loopStart !== undefined) {  // where the channel's loop starts (ms since 1970); null: the shared EPOCH
        const loopStart = fields.loopStart === null ? null : Number(fields.loopStart);
        if (loopStart !== null && !(Number.isInteger(loopStart) && loopStart > 0)) throw new HttpError(400, 'loopStart must be a time in ms, or null');
        channel.loopStart = loopStart;
      }
      for (const key of ['name', 'genre', 'blurb']) {
        if (fields[key] !== undefined) channel[key] = text(fields[key], TEXT_LIMITS[key]);
      }
      if (fields.schedule !== undefined) {
        if (!Array.isArray(fields.schedule) || fields.schedule.length > 48) throw new HttpError(400, 'schedule must be a list of at most 48 programs');
        channel.schedule = fields.schedule.map(cleanProgram);
      }
      channel.updatedAt = now;
      this.channels.set(id, channel);
      return channel;
    });
  }

  removeChannel(id) {
    return this.write(async () => {
      this.require(id);
      this.channels.delete(id);
      await fs.rm(path.join(this.mediaDir, id), { recursive: true, force: true });
    });
  }

  // Moves an uploaded file into the channel's media folder under a new name.
  async place(id, tmpPath, name) {
    const dir = path.join(this.mediaDir, id);
    await fs.mkdir(dir, { recursive: true });
    await fs.rename(tmpPath, path.join(dir, name));
    return name;
  }

  async removeFile(id, name) {
    if (name) await fs.rm(path.join(this.mediaDir, id, name), { force: true });
  }

  setImage(id, kind, tmpPath, extension) {
    if (!IMAGE_KINDS.includes(kind)) throw new HttpError(404, `no image kind ${kind}`);
    return this.write(async () => {
      const channel = this.require(id);
      const old = channel[kind];
      channel[kind] = await this.place(id, tmpPath, `${newId(kind + '-')}.${extension}`);
      channel.updatedAt = new Date().toISOString();
      await this.removeFile(id, old);
      return channel;
    });
  }

  addClip(id, tmpPath, meta) {
    return this.write(async () => {
      const channel = this.require(id);
      const clipId = newId('c');
      const clip = {
        id: clipId,
        file: await this.place(id, tmpPath, `${clipId}.mp4`),
        duration: meta.duration,
        bytes: meta.bytes,
        title: text(meta.title, TEXT_LIMITS.title),
        prompt: text(meta.prompt, TEXT_LIMITS.prompt),
        seed: Number.isFinite(meta.seed) ? meta.seed : null,
        generateS: Number.isFinite(meta.generateS) ? meta.generateS : null,
        run: Store.validId(meta.run) ? meta.run : null,
        sequence: Number.isFinite(meta.sequence) ? meta.sequence : (channel.clips.reduce((max, c) => Math.max(max, c.sequence), 0) + 1),
        createdAt: new Date().toISOString(),
      };
      channel.clips.push(clip);
      channel.clips.sort((a, b) => a.sequence - b.sequence || a.createdAt.localeCompare(b.createdAt));
      if (this.keepClips > 0 && channel.clips.length > this.keepClips) {
        const dropped = [...channel.clips].sort((a, b) => a.createdAt.localeCompare(b.createdAt)).slice(0, channel.clips.length - this.keepClips);
        channel.clips = channel.clips.filter((c) => !dropped.includes(c));
        for (const old of dropped) await this.removeFile(id, old.file);
      }
      channel.updatedAt = clip.createdAt;
      return clip;
    });
  }

  removeClip(id, clipId) {
    return this.write(async () => {
      const channel = this.require(id);
      const clip = channel.clips.find((c) => c.id === clipId);
      if (!clip) throw new HttpError(404, `no clip ${clipId} on ${id}`);
      channel.clips = channel.clips.filter((c) => c !== clip);
      await this.removeFile(id, clip.file);
    });
  }

  async addRun(run) {
    const id = Store.validId(run.id) ? run.id : newId('r');
    const record = { ...run, id, receivedAt: new Date().toISOString() };
    await fs.writeFile(path.join(this.runsDir, `${id}.json`), JSON.stringify(record, null, 1));
    this.runsCache.clear();
    return record;
  }

  // The newest runs first. A run can be large (key frames for the GPU boxes), and reading every run on every poll once ran
  // the app out of memory, so polls between two new runs share one listing (kept until the next run arrives), and a
  // listing reads only the newest files (by modification time, with a few spare for files written in the same instant).
  listRuns(limit = 50) {
    if (!this.runsCache.has(limit)) {
      const listing = this.readRuns(limit);
      listing.catch(() => this.runsCache.delete(limit));  // a failed read is not kept
      this.runsCache.set(limit, listing);
    }
    return this.runsCache.get(limit);
  }

  async readRuns(limit) {
    const names = (await fs.readdir(this.runsDir)).filter((name) => name.endsWith('.json'));
    const files = await Promise.all(names.map(async (name) => {
      const file = path.join(this.runsDir, name);
      return { file, mtime: (await fs.stat(file)).mtimeMs };
    }));
    const newest = files.sort((a, b) => b.mtime - a.mtime).slice(0, limit + 10);
    const runs = await Promise.all(newest.map(async ({ file }) => JSON.parse(await fs.readFile(file, 'utf8'))));
    return runs.sort((a, b) => b.receivedAt.localeCompare(a.receivedAt)).slice(0, limit);
  }

  // Bytes of media and runs on disk.
  async usage() {
    let bytes = 0;
    for (const channel of this.channels.values()) {
      for (const clip of channel.clips) bytes += clip.bytes || 0;
    }
    return { channels: this.channels.size, clips: [...this.channels.values()].reduce((n, c) => n + c.clips.length, 0), clipBytes: bytes };
  }
}

module.exports = { Store, HttpError };
