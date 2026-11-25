// Plays a channel live: the clip that is on now, from where it is now, then the next ones without a gap. Two stacked
// <video> elements take turns: while one plays, the other loads the next clip, and they swap when it ends. The clock is
// the server's (skew), so every viewer sees the same moment of the loop.
/* global Live */
(function () {
  // TV browsers (Samsung Tizen, LG webOS) decode one video at a time, so there one <video> plays the clips in turn, with a
  // short load between clips, instead of two taking turns.
  const ONE_VIDEO = /SMART-TV|SmartTV|Tizen|Web0S|webOS|NetCast|HbbTV/i.test(navigator.userAgent);

  class LivePlayer {
    constructor(container, { muted = true, skew = 0, onClip = () => {} } = {}) {
      this.container = container;
      this.skew = skew;
      this.muted = muted;
      this.onClip = onClip;
      this.channel = null;
      this.generation = 0;
      this.videos = ONE_VIDEO ? [this.makeVideo()] : [this.makeVideo(), this.makeVideo()];
      this.active = 0;
      this.poster = document.createElement('img');
      this.poster.alt = '';
      this.poster.hidden = true;
      container.append(this.poster, ...this.videos);
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden && this.channel) this.sync();
      });
    }

    makeVideo() {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.setAttribute('playsinline', '');
      video.addEventListener('ended', () => {
        if (video === this.videos[this.active]) this.advance();
      });
      return video;
    }

    now() {
      return Date.now() + this.skew;
    }

    setChannel(channel) {
      const same = this.channel && channel && this.channel.id === channel.id &&
        this.channel.clips.map((c) => c.id).join() === channel.clips.map((c) => c.id).join();
      this.channel = channel;
      if (!same) this.sync();
    }

    setMuted(muted) {
      this.muted = muted;
      this.videos[this.active].muted = muted;
    }

    // Starts the clip that is on now, from where it is now.
    sync() {
      const generation = ++this.generation;
      const position = this.channel && Live.livePosition(this.channel.clips, this.now());
      for (const video of this.videos) {
        video.pause();
        video.classList.remove('on');
      }
      if (!position) {
        this.showPoster();
        return;
      }
      this.poster.hidden = true;
      this.position = position;
      for (const other of this.videos) {  // shown while a clip loads, instead of black
        if (this.channel.poster) other.poster = this.channel.poster;
        else other.removeAttribute('poster');
      }
      const video = this.videos[this.active];
      const startedAt = this.now();
      this.load(video, position.clip.url, () => {
        if (generation !== this.generation) return;
        // the loop kept going while the clip loaded
        video.currentTime = Math.min(position.offset + (this.now() - startedAt) / 1000, Math.max(0, position.clip.duration - 0.05));
        this.show(video);
      });
      this.preloadNext();
      this.onClip(position);
    }

    load(video, url, ready) {
      if (video.dataset.url === url && video.readyState >= 1) {
        ready();
        return;
      }
      video.dataset.url = url;
      video.src = url;
      video.addEventListener('loadedmetadata', ready, { once: true });
      video.load();
    }

    show(video) {
      video.muted = this.muted;
      video.play().catch(() => {});
      video.classList.add('on');
      for (const other of this.videos) if (other !== video) other.classList.remove('on');
    }

    preloadNext() {
      if (this.videos.length < 2) return;
      const { clips, index } = this.position;
      const next = clips[(index + 1) % clips.length];
      const other = this.videos[1 - this.active];
      if (other.dataset.url !== next.url) {
        other.dataset.url = next.url;
        other.src = next.url;
        other.load();
      }
    }

    // The clip ended: the preloaded one takes over, unless the loop has moved on meanwhile (a background tab).
    advance() {
      const { clips, index } = this.position;
      const expected = Live.livePosition(this.channel.clips, this.now());
      const nextIndex = (index + 1) % clips.length;
      if (this.videos.length < 2 || !expected || expected.clip.id !== clips[nextIndex].id || expected.offset > 3) {
        this.sync();
        return;
      }
      this.active = 1 - this.active;
      const video = this.videos[this.active];
      video.currentTime = 0;
      this.show(video);
      this.videos[1 - this.active].pause();
      this.position = Object.assign({}, expected, { index: nextIndex, clip: clips[nextIndex] });  // (object spread needs Chromium 60)
      this.preloadNext();
      this.onClip(this.position);
    }

    showPoster() {
      this.position = null;
      if (this.channel && this.channel.poster) {
        this.poster.src = this.channel.poster;
        this.poster.hidden = false;
      } else {
        this.poster.hidden = true;
      }
      this.onClip(null);
    }
  }

  self.LivePlayer = LivePlayer;
})();
