const fs = require('fs');
const { PassThrough } = require('stream');
const { receive, imageExtension } = require('../server/upload');
const { tmpDir, writeTmp, PNG, JPEG, WEBP, SVG } = require('./helpers');

// A request-like stream with the headers receive() reads.
function fakeRequest(chunks, headers = {}) {
  const req = new PassThrough();
  req.headers = headers;
  setImmediate(() => {
    for (const chunk of chunks) req.write(chunk);
    req.end();
  });
  return req;
}

test('writes the body to a new file and keeps its first bytes', async () => {
  const dir = tmpDir();
  const result = await receive(fakeRequest([Buffer.from('hello '), Buffer.from('world')]), dir, 100);
  expect(result.bytes).toBe(11);
  expect(fs.readFileSync(result.path, 'utf8')).toBe('hello world');
  expect(result.head.toString()).toBe('hello world');
});

test('keeps only the first 64 bytes for recognizing the file', async () => {
  const result = await receive(fakeRequest([Buffer.alloc(100, 1), Buffer.alloc(100, 2)]), tmpDir(), 1000);
  expect(result.bytes).toBe(200);
  expect(result.head).toEqual(Buffer.alloc(64, 1));
});

test('refuses a body declared or found larger than the limit, and leaves no file', async () => {
  const dir = tmpDir();
  await expect(receive(fakeRequest([], { 'content-length': '1000' }), dir, 100)).rejects.toMatchObject({ status: 413 });
  await expect(receive(fakeRequest([Buffer.alloc(60), Buffer.alloc(60)]), dir, 100)).rejects.toMatchObject({ status: 413 });
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(fs.readdirSync(dir)).toEqual([]);
});

test('refuses an empty body', async () => {
  await expect(receive(fakeRequest([]), tmpDir(), 100)).rejects.toMatchObject({ status: 400, message: 'empty upload' });
});

test('an aborted upload is an error and leaves no file', async () => {
  const dir = tmpDir();
  const req = new PassThrough();
  req.headers = {};
  const pending = receive(req, dir, 100);
  req.write('partial');
  req.emit('aborted');
  await expect(pending).rejects.toMatchObject({ status: 400, message: 'upload aborted' });
  req.emit('error', new Error('late'));  // a second failure changes nothing
  expect(fs.readdirSync(dir)).toEqual([]);
});

test('a disk error is passed on', async () => {
  await expect(receive(fakeRequest([Buffer.from('x')]), '/nonexistent/dir', 100)).rejects.toMatchObject({ code: 'ENOENT' });
});

test('recognizes images by their first bytes, and only script-free SVG', async () => {
  const dir = tmpDir();
  const check = (name, data) => imageExtension(writeTmp(dir, name, data), Buffer.from(data).subarray(0, 64));
  expect(await check('a.png', PNG)).toBe('png');
  expect(await check('a.jpg', JPEG)).toBe('jpg');
  expect(await check('a.webp', WEBP)).toBe('webp');
  expect(await check('a.svg', SVG)).toBe('svg');
  expect(await check('b.svg', Buffer.from(`<?xml version="1.0"?>\n${SVG}`))).toBe('svg');
  expect(await check('c.svg', Buffer.from('<svg><script>alert(1)</script></svg>'))).toBeNull();
  expect(await check('d.svg', Buffer.from('<svg><rect onclick="x()"/></svg>'))).toBeNull();
  expect(await check('e.svg', Buffer.from('<svg><a href="javascript:x()"/></svg>'))).toBeNull();
  expect(await check('f.svg', Buffer.from('<svg><foreignObject/></svg>'))).toBeNull();
  expect(await check('g.xml', Buffer.from('<?xml version="1.0"?><html/>'))).toBeNull();
  expect(await check('h.gif', Buffer.from('GIF89a......'))).toBeNull();
});
