// What is on every channel at a given moment, the same for every viewer: a channel's clips play in a loop, and its
// schedule (the programs the guide shows) repeats, both counted from one fixed start (EPOCH). Runs in the browser and,
// for the tests, in Node.
(function (root, factory) {
  const live = factory();
  /* istanbul ignore else -- the browser path, tested by running this file in a sandbox (live.test.js) */
  if (typeof module === 'object' && module.exports) module.exports = live;
  else root.Live = live;
})(this, function () {  // a script's top-level `this` is the global object, in old browsers too
  const EPOCH = Date.UTC(2026, 0, 1);  // midnight UTC, Jan 1 2026: where every loop starts
  const DEFAULT_PROGRAM_MINUTES = 60;

  // Total length of a channel's clips, in seconds.
  function loopSeconds(clips) {
    return (clips || []).reduce((sum, clip) => sum + (clip.duration > 0 ? clip.duration : 0), 0);
  }

  // The clip playing at `nowMs` and how far into it, or null when the channel has nothing to play.
  function livePosition(clips, nowMs) {
    const playable = (clips || []).filter((clip) => clip.duration > 0);
    const total = loopSeconds(playable);
    if (!total) return null;
    let t = (((nowMs - EPOCH) / 1000) % total + total) % total;
    let index = 0;
    // the last clip also takes a floating-point remainder at the very end of the loop
    while (index < playable.length - 1 && t >= playable[index].duration) t -= playable[index++].duration;
    return { index, clip: playable[index], offset: Math.min(t, playable[index].duration), clips: playable };
  }

  // A channel's schedule, or one program named after the channel when it has none.
  function scheduleOf(channel) {
    const programs = (channel.schedule || []).filter((program) => program.minutes > 0);
    if (programs.length) return programs;
    return [{ title: channel.name, episode: 'Live', minutes: DEFAULT_PROGRAM_MINUTES, badges: [] }];
  }

  // Every airing of the channel's looping schedule that overlaps [fromMs, toMs), in order.
  function airings(channel, fromMs, toMs) {
    const programs = scheduleOf(channel);
    const loopMs = programs.reduce((sum, program) => sum + program.minutes * 60000, 0);
    const out = [];
    let start = EPOCH + Math.floor((fromMs - EPOCH) / loopMs) * loopMs;
    for (let guard = 0; start < toMs && guard < 10000; guard++) {
      for (const program of programs) {
        const end = start + program.minutes * 60000;
        if (end > fromMs && start < toMs) out.push({ program, start, end });
        start = end;
      }
    }
    return out;
  }

  // The airing on now.
  function onNow(channel, nowMs) {
    return airings(channel, nowMs, nowMs + 1)[0];
  }

  // The start of the local half hour `nowMs` falls in (the guide's window starts there).
  function halfHourStart(nowMs) {
    const date = new Date(nowMs);
    date.setMinutes(date.getMinutes() < 30 ? 0 : 30, 0, 0);
    return date.getTime();
  }

  // A clip's short name for "Now playing": its title up to a colon ("Larry: stand-up set" is Larry), or all of it.
  function clipName(clip) {
    return ((clip && clip.title) || '').split(':')[0].trim() || 'Untitled';
  }

  return { EPOCH, loopSeconds, livePosition, scheduleOf, airings, onNow, halfHourStart, clipName };
});
