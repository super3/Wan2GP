// Reads what the server needs to know about an uploaded MP4 without ffmpeg: that it is one (an `ftyp` box first) and
// its duration (the `mvhd` box inside `moov`, which ffmpeg writes at the end of the file unless told otherwise).
const fs = require('fs/promises');

const HEADER = 16;  // a box header: 32-bit size and type, then a 64-bit size when the 32-bit one is 1

async function readAt(handle, position, length) {
  const buffer = Buffer.alloc(length);
  const { bytesRead } = await handle.read(buffer, 0, length, position);
  return buffer.subarray(0, bytesRead);
}

// The boxes between `start` and `end`: [{ type, start (of the payload), end }].
async function boxes(handle, start, end) {
  const found = [];
  let position = start;
  while (position + 8 <= end) {  // `end` never passes the end of the file, so the 8 bytes are there
    const header = await readAt(handle, position, HEADER);
    let size = header.readUInt32BE(0);
    const type = header.toString('latin1', 4, 8);
    let headerSize = 8;
    if (size === 1) {
      if (header.length < 16) break;
      size = Number(header.readBigUInt64BE(8));
      headerSize = 16;
    } else if (size === 0) {
      size = end - position;  // the box runs to the end of its parent
    }
    if (size < headerSize) break;
    found.push({ type, start: position + headerSize, end: Math.min(position + size, end) });
    position += size;
  }
  return found;
}

// Whether the file starts like an MP4 (ISO base media) file.
function looksLikeMp4(head) {
  return head.length >= 8 && head.toString('latin1', 4, 8) === 'ftyp';
}

// The duration in seconds, or null when the file has no readable `moov`/`mvhd`.
async function mp4Duration(file) {
  const handle = await fs.open(file, 'r');
  try {
    const { size } = await handle.stat();
    const moov = (await boxes(handle, 0, size)).find((box) => box.type === 'moov');
    if (!moov) return null;
    const mvhd = (await boxes(handle, moov.start, moov.end)).find((box) => box.type === 'mvhd');
    if (!mvhd) return null;
    const body = await readAt(handle, mvhd.start, 32);
    const version = body[0];
    let timescale;
    let duration;
    if (version === 1 && body.length >= 32) {
      timescale = body.readUInt32BE(20);
      duration = Number(body.readBigUInt64BE(24));
    } else if (version === 0 && body.length >= 20) {
      timescale = body.readUInt32BE(12);
      duration = body.readUInt32BE(16);
    } else {
      return null;
    }
    return timescale ? duration / timescale : null;
  } finally {
    await handle.close();
  }
}

module.exports = { looksLikeMp4, mp4Duration };
