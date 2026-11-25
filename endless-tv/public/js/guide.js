// The guide: the selected program on top (playing live when it is on now), and a two-hour grid of every channel.
/* global Live, LivePlayer */
(function () {
  const WINDOW_MIN = 120;
  const $ = (id) => document.getElementById(id);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const fmt = (ms) => new Date(ms).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  // A node's children replaced (Element.replaceChildren needs Chromium 86, newer than most TV browsers).
  const fill = (node, children) => {
    while (node.firstChild) node.removeChild(node.firstChild);
    for (const child of children) node.appendChild(child);
  };

  const state = { channels: [], skew: 0, filter: 'All', selected: null, player: null, muted: true };
  const now = () => Date.now() + state.skew;

  async function load() {
    const response = await fetch('/api/channels', { cache: 'no-store' });
    const data = await response.json();
    state.skew = data.now - Date.now();
    state.channels = data.channels;
    if (state.player) state.player.skew = state.skew;
    const discord = data.site && data.site.discord;  // set by DISCORD_URL on the server
    if (discord) $('discord').href = discord;
    $('discord').hidden = !discord;
  }

  function genres() {
    const seen = [];
    for (const channel of state.channels) if (channel.genre && !seen.includes(channel.genre)) seen.push(channel.genre);
    return ['All', ...seen];
  }

  // The selected airing: what the viewer picked, or what is on now on the first channel.
  function selection() {
    const t = now();
    if (state.selected) {
      const channel = state.channels.find((c) => c.id === state.selected.channelId);
      const airing = channel && Live.airings(channel, state.selected.start, state.selected.start + 1)[0];
      if (airing && airing.end > t) return { channel, airing };
    }
    const channel = state.channels[0];
    return channel ? { channel, airing: Live.onNow(channel, t) } : null;
  }

  function renderFilters() {
    const box = $('filters');
    fill(box, genres().map((label) => {
      const chip = el('button', 'chip', label);
      chip.type = 'button';
      chip.setAttribute('aria-pressed', String(state.filter === label));
      chip.addEventListener('click', () => {
        state.filter = label;
        renderFilters();
        renderGrid();
      });
      return chip;
    }));
  }

  function renderGrid() {
    const t = now();
    const base = Live.halfHourStart(t);
    const end = base + WINDOW_MIN * 60000;
    const sel = selection();
    const timeline = $('timeline');
    timeline.querySelectorAll('.tick').forEach((tick) => tick.remove());
    for (const minutes of [0, 30, 60, 90]) {
      const tick = el('span', 'tick', fmt(base + minutes * 60000));
      tick.style.left = `${(100 * minutes) / WINDOW_MIN}%`;
      timeline.prepend(tick);
    }
    const rows = state.channels.filter((c) => state.filter === 'All' || c.genre === state.filter).map((channel) => {
      const row = el('div', 'row');
      const name = el('a', 'chan');
      name.href = `/watch/${channel.id}`;
      name.setAttribute('aria-label', `Watch ${channel.name}`);
      const logo = el('span', 'chan-logo');
      if (channel.logo) {
        const img = el('img');
        img.src = channel.logo;
        img.alt = channel.name;
        logo.append(img);
      } else {
        logo.append(el('span', '', channel.name));
      }
      name.append(logo, el('span', 'num', String(channel.number)));
      const lane = el('div', 'lane');
      for (const airing of Live.airings(channel, base, end)) {
        const from = Math.max(airing.start, base);
        const to = Math.min(airing.end, end);
        const isNow = airing.start <= t && t < airing.end;
        const isSelected = sel && sel.channel.id === channel.id && sel.airing.start === airing.start;
        const cell = el('button', `cell${isNow ? ' now' : ''}${isSelected ? ' selected' : ''}${airing.end <= t ? ' past' : ''}`);
        cell.type = 'button';
        cell.setAttribute('aria-pressed', String(Boolean(isSelected)));
        cell.style.left = `${(100 * (from - base)) / (end - base)}%`;
        cell.style.width = `calc(${(100 * (to - from)) / (end - base)}% - 4px)`;
        const title = el('span', 'cell-title');
        if (airing.start < base) title.append(svgArrow());
        title.append(document.createTextNode(airing.program.title));
        const sub = el('span', 'cell-sub');
        if (airing.program.isNew) sub.append(el('span', 'tag-new', 'NEW'));
        sub.append(document.createTextNode(`${fmt(airing.start)} – ${fmt(airing.end)}`));
        cell.append(title, sub);
        cell.addEventListener('click', () => {
          state.selected = { channelId: channel.id, start: airing.start };
          renderHero();
          renderGrid();
        });
        lane.append(cell);
      }
      row.append(name, lane);
      return row;
    });
    fill($('rows'), rows);
    renderClock();
  }

  function svgArrow() {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('width', '10');
    svg.setAttribute('height', '10');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', '#ADADB8');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', 'M16 4L6 12l10 8z');
    svg.append(path);
    return svg;
  }

  function renderClock() {
    const t = now();
    const base = Live.halfHourStart(t);
    const frac = (t - base) / (WINDOW_MIN * 60000);
    $('nowLabel').textContent = fmt(t);
    $('nowLabel').style.left = `${100 * frac}%`;
    $('nowLine').style.left = `calc(200px + (100% - 200px) * ${frac})`;
  }

  // The hero plays muted (browsers only autoplay muted video); the button on it turns the sound on and off.
  const SOUND_ON = 'M4 9h4l5-4v14l-5-4H4zM16.5 8.5a5 5 0 010 7M19 6a8.5 8.5 0 010 12';
  const SOUND_OFF = 'M4 9h4l5-4v14l-5-4H4zM17 9l5 6M22 9l-5 6';
  function setMuted(muted) {
    state.muted = muted;
    state.player.setMuted(muted);
    $('muteIcon').setAttribute('d', muted ? SOUND_OFF : SOUND_ON);
    $('mute').setAttribute('aria-label', muted ? 'Turn on sound' : 'Turn off sound');
  }

  // Full screen for the video (the button beside mute). Older browsers have only the webkit-prefixed calls, without promises.
  const FS_ENTER = 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5';
  const FS_EXIT = 'M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5';
  const fullscreenElement = () => document.fullscreenElement || document.webkitFullscreenElement;
  const canFullscreen = () => Boolean($('screenBox').requestFullscreen || $('screenBox').webkitRequestFullscreen);
  function toggleFullscreen() {
    if (fullscreenElement()) {
      (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      return;
    }
    const box = $('screenBox');
    const result = (box.requestFullscreen || box.webkitRequestFullscreen).call(box);
    if (result && result.catch) result.catch(() => {});
  }
  function renderFullscreen() {
    const on = Boolean(fullscreenElement());
    $('fsIcon').setAttribute('d', on ? FS_EXIT : FS_ENTER);
    $('fs').setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
  }

  function renderHero() {
    const sel = selection();
    if (!sel) return;
    const { channel, airing } = sel;
    const t = now();
    const live = airing.start <= t && t < airing.end;
    const program = airing.program;
    if (channel.logo) $('selLogo').src = channel.logo;
    $('selLogo').alt = channel.name;
    $('selLogo').hidden = !channel.logo;
    $('selChannel').textContent = `Channel ${channel.number}`;
    $('selTitle').textContent = program.title;
    $('selEpisode').textContent = program.episode || '';
    const meta = $('selMeta');
    fill(meta, [el('span', '', `${fmt(airing.start)} – ${fmt(airing.end)}`)].concat((program.badges || []).map((b) => el('span', 'rating', b))));
    $('selBlurb').textContent = program.blurb || channel.blurb || '';
    const watch = `/watch/${channel.id}`;
    $('screen').href = watch;
    $('watchBtn').href = watch;
    $('watchLabel').textContent = live ? 'Watch now' : `Watch ${channel.name} now`;
    $('screen').setAttribute('aria-label', `Watch ${channel.name}`);
    const clips = channel.clips.length;

    const cover = $('cover');
    const note = $('screenNote');
    $('mute').hidden = !(live && clips);
    $('fs').hidden = !(live && clips) || !canFullscreen();
    if (live && clips) {
      cover.hidden = true;
      note.hidden = true;
      $('liveBadge').hidden = !channel.live;  // only a channel that runs live says so
      state.player.setChannel(channel);
    } else {
      state.player.setChannel(null);
      if (channel.poster) cover.src = channel.poster;
      cover.hidden = !channel.poster;
      $('liveBadge').hidden = true;
      note.hidden = false;
      note.textContent = live ? 'Clips are on their way' : `Starts at ${fmt(airing.start)}`;
    }
    renderClock();
  }

  async function refresh() {
    try {
      await load();
    } catch (error) {  // (a bare `catch {` needs Chromium 66)
      return;  // keep showing what we have
    }
    const empty = state.channels.length === 0;
    $('empty').hidden = !empty;
    $('hero').hidden = empty;
    $('guide').hidden = empty;
    if (empty) return;
    renderFilters();
    renderGrid();
    renderHero();
  }

  state.player = new LivePlayer($('player'), { muted: true });
  $('mute').addEventListener('click', () => setMuted(!state.muted));
  $('fs').addEventListener('click', toggleFullscreen);
  document.addEventListener('fullscreenchange', renderFullscreen);
  document.addEventListener('webkitfullscreenchange', renderFullscreen);
  refresh();
  setInterval(renderClock, 1000);
  setInterval(() => {
    renderGrid();
    renderHero();
  }, 30000);
  setInterval(refresh, 60000);  // new clips and channels
})();
