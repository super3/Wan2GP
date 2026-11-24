// Small files for the tests: MP4s built box by box (no ffmpeg needed) and images by their first bytes.
const fs = require('fs');
const os = require('os');
const path = require('path');

function box(type, payload = Buffer.alloc(0)) {
  const header = Buffer.alloc(8);
  header.writeUInt32BE(8 + payload.length, 0);
  header.write(type, 4, 'latin1');
  return Buffer.concat([header, payload]);
}

// A box written with a 64-bit size (32-bit size field set to 1).
function largeBox(type, payload) {
  const header = Buffer.alloc(16);
  header.writeUInt32BE(1, 0);
  header.write(type, 4, 'latin1');
  header.writeBigUInt64BE(BigInt(16 + payload.length), 8);
  return Buffer.concat([header, payload]);
}

function mvhd({ duration = 5, timescale = 1000, version = 0 } = {}) {
  if (version === 1) {
    const payload = Buffer.alloc(112);
    payload[0] = 1;
    payload.writeUInt32BE(timescale, 20);
    payload.writeBigUInt64BE(BigInt(Math.round(duration * timescale)), 24);
    return box('mvhd', payload);
  }
  const payload = Buffer.alloc(100);
  payload[0] = version;
  payload.writeUInt32BE(timescale, 12);
  payload.writeUInt32BE(Math.round(duration * timescale), 16);
  return box('mvhd', payload);
}

const ftyp = () => box('ftyp', Buffer.from('isom\0\0\x02\0isomiso2avc1mp41', 'latin1'));

// An MP4 of `duration` seconds: ftyp, mdat, then moov (as ffmpeg writes it), or moov first.
function mp4({ duration = 5, timescale = 1000, version = 0, moovFirst = false, mdatBytes = 256 } = {}) {
  const moov = box('moov', Buffer.concat([mvhd({ duration, timescale, version }), box('trak')]));
  const mdat = box('mdat', Buffer.alloc(mdatBytes, 7));
  return Buffer.concat(moovFirst ? [ftyp(), moov, mdat] : [ftyp(), mdat, moov]);
}

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(40)]);
const JPEG = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(40)]);
const WEBP = Buffer.concat([Buffer.from('RIFF\0\0\0\0WEBPVP8 ', 'latin1'), Buffer.alloc(40)]);
const SVG = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>');

function tmpDir(prefix = 'etv-') {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function writeTmp(dir, name, data) {
  const file = path.join(dir, name);
  fs.writeFileSync(file, data);
  return file;
}

module.exports = { box, largeBox, mvhd, ftyp, mp4, PNG, JPEG, WEBP, SVG, tmpDir, writeTmp };
