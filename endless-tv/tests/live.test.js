const fs = require('fs');
const path = require('path');
const vm = require('vm');
const Live = require('../public/js/live');

test('in a browser (no module), it defines the Live global', () => {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '..', 'public', 'js', 'live.js'), 'utf8'), sandbox);
  expect(Object.keys(sandbox.Live)).toEqual(Object.keys(Live));
});

const MIN = 60000;
const clips = [{ id: 'a', duration: 10 }, { id: 'b', duration: 5 }, { id: 'c', duration: 15 }];

describe('livePosition', () => {
  test('finds the clip on now and the offset into it, the same for everyone', () => {
    expect(Live.livePosition(clips, Live.EPOCH)).toMatchObject({ index: 0, offset: 0 });
    expect(Live.livePosition(clips, Live.EPOCH + 12000)).toMatchObject({ index: 1, offset: 2 });
    expect(Live.livePosition(clips, Live.EPOCH + 29000)).toMatchObject({ index: 2, offset: 14 });
  });

  test('loops forever, before and after the epoch', () => {
    expect(Live.livePosition(clips, Live.EPOCH + 30000 * 1000 + 11000)).toMatchObject({ index: 1, offset: 1 });
    expect(Live.livePosition(clips, Live.EPOCH - 1000)).toMatchObject({ index: 2, offset: 14 });
  });

  test('skips clips without a duration and returns null with nothing to play', () => {
    expect(Live.livePosition([{ id: 'x', duration: 0 }, { id: 'y', duration: 4 }], Live.EPOCH + 1000)).toMatchObject({ clip: { id: 'y' } });
    expect(Live.livePosition([], Live.EPOCH)).toBeNull();
    expect(Live.livePosition(undefined, Live.EPOCH)).toBeNull();
    expect(Live.livePosition([{ id: 'x', duration: -1 }], Live.EPOCH)).toBeNull();
  });

  test('a floating-point remainder at the very end wraps to the first clip', () => {
    const tiny = [{ id: 'p', duration: 0.1 }, { id: 'q', duration: 0.2 }];
    for (let ms = 0; ms < 1000; ms += 1) {
      const position = Live.livePosition(tiny, Live.EPOCH + ms);
      expect(position.offset).toBeLessThan(position.clip.duration + 1e-9);
    }
  });

  test('loopSeconds adds the playable durations', () => {
    expect(Live.loopSeconds(clips)).toBe(30);
    expect(Live.loopSeconds(undefined)).toBe(0);
    expect(Live.loopSeconds([{ duration: -2 }, { duration: 3 }])).toBe(3);
  });
});

describe('schedule', () => {
  const channel = { name: 'News', schedule: [{ title: 'A', minutes: 30 }, { title: 'B', minutes: 60 }, { title: 'Gone', minutes: 0 }] };

  test('a channel without a schedule gets one program named after it', () => {
    expect(Live.scheduleOf({ name: 'Drift' })).toEqual([expect.objectContaining({ title: 'Drift', minutes: 60 })]);
    expect(Live.scheduleOf(channel).map((p) => p.title)).toEqual(['A', 'B']);
  });

  test('airings repeat the schedule and cover the window', () => {
    const list = Live.airings(channel, Live.EPOCH, Live.EPOCH + 120 * MIN);
    expect(list.map((a) => [a.program.title, (a.start - Live.EPOCH) / MIN, (a.end - Live.EPOCH) / MIN])).toEqual([
      ['A', 0, 30], ['B', 30, 90], ['A', 90, 120],
    ]);
  });

  test('an airing started before the window is included', () => {
    const list = Live.airings(channel, Live.EPOCH + 45 * MIN, Live.EPOCH + 50 * MIN);
    expect(list).toHaveLength(1);
    expect(list[0]).toMatchObject({ program: { title: 'B' }, start: Live.EPOCH + 30 * MIN });
  });

  test('onNow is the airing on at that moment, years later too', () => {
    expect(Live.onNow(channel, Live.EPOCH + 10 * MIN).program.title).toBe('A');
    const later = Live.EPOCH + 1000 * 90 * MIN + 31 * MIN;
    expect(Live.onNow(channel, later).program.title).toBe('B');
  });

  test('halfHourStart rounds down to the local half hour', () => {
    const t = new Date(2026, 10, 24, 1, 47, 12).getTime();
    expect(new Date(Live.halfHourStart(t)).getMinutes()).toBe(30);
    expect(new Date(Live.halfHourStart(new Date(2026, 10, 24, 1, 12).getTime())).getMinutes()).toBe(0);
  });
});
