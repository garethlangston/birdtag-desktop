/* Fork branding (single source of truth: analyzer/branding.json).
 *
 * Loaded first by visualizer.html and culling.html. Reads branding.json from the
 * local static server, then:
 *  - replaces the upstream product name ("Project Kestrel" / "Kestrel") in visible
 *    text, titles and alt text — including dialogs and toasts added later;
 *  - when upstream_services is false, hides the UI for upstream's hosted services
 *    (Perch, Cloud Compute, account sign-in, donations), whose names are not
 *    licensed to forks (see assets/LICENSE).
 */
(function () {
  var BRAND = { product_name: 'Birdtag Desktop', short_name: 'Birdtag', upstream_services: false };
  try {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', 'branding.json', false);  // sync: must be ready before other scripts run
    xhr.send(null);
    if (xhr.status >= 200 && xhr.status < 300) {
      var data = JSON.parse(xhr.responseText);
      for (var k in data) { if (k.charAt(0) !== '_') BRAND[k] = data[k]; }
    }
  } catch (e) { /* defaults */ }
  window.BRAND = BRAND;

  var FULL = /Project Kestrel/g;
  var SHORT = /\bKestrel\b/g;  // capitalised word only: ".kestrel" folders / kestrel:* stay
  function rebrand(s) {
    return s.replace(FULL, BRAND.product_name).replace(SHORT, BRAND.short_name);
  }
  window.brandText = rebrand;

  function fixNode(node) {
    if (node.nodeType === 3) {
      var v = node.nodeValue;
      if (v && v.indexOf('Kestrel') !== -1) node.nodeValue = rebrand(v);
      return;
    }
    if (node.nodeType !== 1) return;
    var tag = node.tagName;
    if (tag === 'SCRIPT' || tag === 'STYLE') return;
    ['title', 'alt', 'aria-label', 'placeholder'].forEach(function (a) {
      var v = node.getAttribute && node.getAttribute(a);
      if (v && v.indexOf('Kestrel') !== -1) node.setAttribute(a, rebrand(v));
    });
    for (var c = node.firstChild; c; c = c.nextSibling) fixNode(c);
  }

  if (!BRAND.upstream_services) {
    var css = [
      '#accountBtn', '#donateBtnMain', '#donateDlg', '#analyzeDestCloud', '#cloudQueuePanel',
      '#perchUploadsPanel', '.folder-perch-btn', '#cloudAccountDlg', '#analyzeDlgCloudEstPanel', '#openFeedback',
    ].join(',') + '{display:none !important}';
    var st = document.createElement('style');
    st.textContent = css;
    (document.head || document.documentElement).appendChild(st);
  }

  function start() {
    document.title = rebrand(document.title || BRAND.product_name);
    fixNode(document.body);
    new MutationObserver(function (muts) {
      muts.forEach(function (m) {
        if (m.type === 'characterData') fixNode(m.target);
        m.addedNodes && m.addedNodes.forEach(fixNode);
      });
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
