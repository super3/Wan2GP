// Loaded first on every page, and written in ES5 so even old TV browsers run it. Samsung TVs from before 2023 run
// Chromium 56-85 (Tizen 4-6.5): flexbox `gap` needs Chromium 84, so on older ones the pages fall back to margins
// (html.no-flexgap in app.css). And a TV has no developer tools, so when a script fails the page says why, at the bottom.
(function () {
  var root = document.documentElement;

  var probe = document.createElement('div');
  probe.style.cssText = 'display:flex;flex-direction:column;row-gap:1px;position:absolute;visibility:hidden';
  probe.appendChild(document.createElement('div'));
  probe.appendChild(document.createElement('div'));
  document.body.appendChild(probe);
  if (probe.scrollHeight !== 1) root.className += ' no-flexgap';
  document.body.removeChild(probe);

  function report(message) {
    var box = document.getElementById('compat-error');
    if (!box) {
      box = document.createElement('div');
      box.id = 'compat-error';
      box.className = 'compat-error';
      document.body.appendChild(box);
    }
    var browser = (navigator.userAgent.match(/Tizen [\d.]+|Web0S|webOS|Chrome\/\d+/g) || []).join(' ');
    box.textContent = 'This browser hit an error: ' + message + (browser ? ' [' + browser + ']' : '');
  }
  window.addEventListener('error', function (event) {
    if (!event.message) return;  // a failed image or video load, not a script error
    var where = event.filename ? ' (' + event.filename.split('/').pop() + ':' + event.lineno + ')' : '';
    report(event.message + where);
  });
  window.addEventListener('unhandledrejection', function (event) {
    var reason = event.reason;
    report(String(reason && reason.message ? reason.message : reason));
  });
})();
