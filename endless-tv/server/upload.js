// Streams a request body to a file on the volume, with a size limit, and recognizes what was uploaded from its first
// bytes rather than trusting the Content-Type the sender set.
const crypto = require('crypto');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { HttpError } = require('./store');

// The body of `req` written to a new temporary file in `dir`; resolves to { path, bytes, head }.
function receive(req, dir, maxBytes) {
  const declared = Number(req.headers['content-length']);
  if (declared > maxBytes) return Promise.reject(new HttpError(413, `larger than ${Math.round(maxBytes / 1e6)} MB`));
  const tmpPath = path.join(dir, `.upload-${crypto.randomBytes(8).toString('hex')}`);
  return new Promise((resolve, reject) => {
    const out = fs.createWriteStream(tmpPath);
    let bytes = 0;
    let head = Buffer.alloc(0);
    let settled = false;
    const fail = (error) => {
      if (settled) return;  // the first outcome stands
      settled = true;
      req.unpipe(out);
      out.destroy();
      fsp.rm(tmpPath, { force: true }).finally(() => reject(error));
    };
    req.on('data', (chunk) => {
      bytes += chunk.length;
      if (head.length < 64) head = Buffer.concat([head, chunk.subarray(0, 64 - head.length)]);
      if (bytes > maxBytes) {
        fail(new HttpError(413, `larger than ${Math.round(maxBytes / 1e6)} MB`));
        req.resume();
      }
    });
    req.on('aborted', () => fail(new HttpError(400, 'upload aborted')));
    req.on('error', fail);
    out.on('error', fail);
    out.on('finish', () => {  // a failed upload destroyed `out`, which then never finishes
      if (!bytes) return fail(new HttpError(400, 'empty upload'));
      settled = true;
      resolve({ path: tmpPath, bytes, head });
    });
    req.pipe(out);
  });
}

// The file extension for an uploaded image, or null when it is not one we serve. SVG is allowed for logos only when it
// carries no script; every media response is also sandboxed (see app.js).
async function imageExtension(file, head) {
  if (head.length >= 8 && head.readUInt32BE(0) === 0x89504e47) return 'png';
  if (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return 'jpg';
  if (head.length >= 12 && head.toString('latin1', 0, 4) === 'RIFF' && head.toString('latin1', 8, 12) === 'WEBP') return 'webp';
  const start = head.toString('utf8').trimStart().toLowerCase();
  if (start.startsWith('<svg') || start.startsWith('<?xml')) {
    const svg = (await fsp.readFile(file, 'utf8')).toLowerCase();
    if (!svg.includes('<svg') || /<script|<foreignobject|\son[a-z]+\s*=|javascript:/.test(svg)) return null;
    return 'svg';
  }
  return null;
}

module.exports = { receive, imageExtension };
