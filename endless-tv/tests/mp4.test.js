const { looksLikeMp4, mp4Duration } = require('../server/mp4');
const { box, largeBox, mvhd, ftyp, mp4, tmpDir, writeTmp } = require('./helpers');

const dir = tmpDir();
const write = (name, data) => writeTmp(dir, name, data);

test('recognizes an MP4 by its ftyp box', () => {
  expect(looksLikeMp4(mp4())).toBe(true);
  expect(looksLikeMp4(Buffer.from('not a video at all'))).toBe(false);
  expect(looksLikeMp4(Buffer.from('abc'))).toBe(false);
});

test('reads the duration with moov at the end (as ffmpeg writes it) or at the start', async () => {
  expect(await mp4Duration(write('end.mp4', mp4({ duration: 5.17 })))).toBeCloseTo(5.17, 3);
  expect(await mp4Duration(write('start.mp4', mp4({ duration: 31.05, moovFirst: true })))).toBeCloseTo(31.05, 3);
});

test('reads a version 1 (64-bit) mvhd and a 64-bit box size', async () => {
  expect(await mp4Duration(write('v1.mp4', mp4({ duration: 7.5, timescale: 90000, version: 1 })))).toBeCloseTo(7.5, 3);
  const file = Buffer.concat([ftyp(), largeBox('mdat', Buffer.alloc(100)), box('moov', mvhd({ duration: 2 }))]);
  expect(await mp4Duration(write('large.mp4', file))).toBeCloseTo(2, 3);
});

test('a box of size 0 runs to the end of the file', async () => {
  const moov = box('moov', mvhd({ duration: 3 }));
  moov.writeUInt32BE(0, 0);
  expect(await mp4Duration(write('zero.mp4', Buffer.concat([ftyp(), moov])))).toBeCloseTo(3, 3);
});

test('returns null when there is no duration to read', async () => {
  expect(await mp4Duration(write('nomoov.mp4', Buffer.concat([ftyp(), box('mdat', Buffer.alloc(8))])))).toBeNull();
  expect(await mp4Duration(write('nomvhd.mp4', Buffer.concat([ftyp(), box('moov', box('trak'))])))).toBeNull();
  expect(await mp4Duration(write('version2.mp4', Buffer.concat([ftyp(), box('moov', mvhd({ version: 2 }))])))).toBeNull();
  expect(await mp4Duration(write('notimescale.mp4', Buffer.concat([ftyp(), box('moov', mvhd({ timescale: 0 }))])))).toBeNull();
  const shortMvhd = box('moov', box('mvhd', Buffer.alloc(6)));
  expect(await mp4Duration(write('short.mp4', Buffer.concat([ftyp(), shortMvhd])))).toBeNull();
});

test('stops on broken box headers instead of looping', async () => {
  const tooSmall = Buffer.alloc(8);
  tooSmall.writeUInt32BE(4, 0);
  tooSmall.write('junk', 4, 'latin1');
  expect(await mp4Duration(write('small.mp4', Buffer.concat([ftyp(), tooSmall])))).toBeNull();
  const cutLarge = Buffer.alloc(12);
  cutLarge.writeUInt32BE(1, 0);
  cutLarge.write('mdat', 4, 'latin1');
  expect(await mp4Duration(write('cut.mp4', Buffer.concat([ftyp(), cutLarge])))).toBeNull();
  expect(await mp4Duration(write('tiny.mp4', Buffer.from('abcdefghij')))).toBeNull();
});
