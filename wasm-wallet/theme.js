// theme.js: the saved theme, set before the page paints so it doesn't flash.
// Dark unless light was chosen (the key the classic page uses too).
(function () {
    var light = false;
    try { light = localStorage.getItem('mds-theme') === 'light'; } catch (_) {}
    document.documentElement.setAttribute('data-theme', light ? 'light' : 'dark');
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', light ? '#ffffff' : '#0a0a0a');
})();
