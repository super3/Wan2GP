// The watch page: one channel full screen, live, with channel up/down. The channel id is the last part of the URL.
/* global Live, LivePlayer */
(function () {
  const $ = (id) => document.getElementById(id);
  const SOUND_ON = 'M4 9h4l5-4v14l-5-4H4zM16.5 8.5a5 5 0 010 7M19 6a8.5 8.5 0 010 12';
  const SOUND_OFF = 'M4 9h4l5-4v14l-5-4H4zM17 9l5 6M22 9l-5 6';
  const state = { channels: [], index: 0, skew: 0, muted: true };
  const now = () => Date.now() + state.skew;
  let player;

  function channel() {
    return state.channels[state.index];
  }

  function renderInfo(position) {
    const current = channel();
    if (!current) return;
    const airing = Live.onNow(current, now());
    $('title').textContent = airing.program.title;
    $('episode').textContent = [current.name, airing.program.episode].filter(Boolean).join(' · ');
    if (current.logo) {
      $('logo').src = current.logo;
      $('logo').alt = current.name;
    }
    $('logo').hidden = !current.logo;
    $('chNum').textContent = String(current.number);
    $('livePill').hidden = !position;
    $('clipInfo').textContent = position
      ? `Clip ${position.index + 1} of ${position.clips.length}${position.clip.title ? `: ${position.clip.title}` : ''}`
      : 'No clips yet';
    document.title = `${current.name} · endless TV`;
  }

  function tune(index) {
    if (!state.channels.length) return;
    state.index = (index + state.channels.length) % state.channels.length;
    history.replaceState(null, '', `/watch/${channel().id}`);
    player.setChannel(channel());
    renderInfo(player.position);
  }

  function setMuted(muted) {
    state.muted = muted;
    player.setMuted(muted);
    $('muteIcon').setAttribute('d', muted ? SOUND_OFF : SOUND_ON);
    $('mute').setAttribute('aria-label', muted ? 'Turn on sound' : 'Turn off sound');
    $('unmute').hidden = !muted;
  }

  async function load() {
    const response = await fetch('/api/channels', { cache: 'no-store' });
    const data = await response.json();
    state.skew = data.now - Date.now();
    state.channels = data.channels;
    player.skew = state.skew;
  }

  async function start() {
    player = new LivePlayer($('player'), { muted: true, onClip: renderInfo });
    await load();
    const wanted = decodeURIComponent(location.pathname.split('/').pop());
    const index = state.channels.findIndex((c) => c.id === wanted);
    tune(index < 0 ? 0 : index);
    setInterval(async () => {  // new clips
      const id = channel() && channel().id;
      try {
        await load();
      } catch {
        return;
      }
      const index = state.channels.findIndex((c) => c.id === id);
      state.index = index < 0 ? 0 : index;
      if (channel()) player.setChannel(channel());
    }, 60000);
    setInterval(() => renderInfo(player.position), 15000);
  }

  $('unmute').addEventListener('click', () => setMuted(false));
  $('mute').addEventListener('click', () => setMuted(!state.muted));
  $('chUp').addEventListener('click', () => tune(state.index + 1));
  $('chDown').addEventListener('click', () => tune(state.index - 1));
  $('fullscreen').addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else $('watch').requestFullscreen().catch(() => {});
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp') tune(state.index + 1);
    else if (event.key === 'ArrowDown') tune(state.index - 1);
    else if (event.key === 'm') setMuted(!state.muted);
    else if (event.key === 'f') $('fullscreen').click();
  });

  // the controls fade when the mouse rests
  let idle;
  const wake = () => {
    $('watch').classList.remove('idle');
    clearTimeout(idle);
    idle = setTimeout(() => $('watch').classList.add('idle'), 3000);
  };
  ['mousemove', 'touchstart', 'keydown'].forEach((name) => document.addEventListener(name, wake));
  wake();

  start();
})();
