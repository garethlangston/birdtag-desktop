/* Fork branding (single source of truth: analyzer/branding.json).
 *
 * Loaded first by visualizer.html and culling.html. It:
 *  - exposes window.BRAND plus the globals BRAND_NAME / BRAND_SHORT, which the UI
 *    scripts use in their messages;
 *  - fills static markup once at load: <span data-brand="name|short">,
 *    data-brand-text="… {name} …" (element text, e.g. <title>) and
 *    data-brand-tpl-<attr>="… {short} …" (title/alt/aria-label/placeholder);
 *  - hides UI for upstream's hosted services (Perch, Cloud Compute, account,
 *    donations, feedback) when upstream_services is false.
 *
 * It deliberately does NOT rewrite arbitrary page text: species such as
 * "Common Kestrel" must display unchanged (fixed in 6.3.0-uk.2).
 */
var BRAND = { product_name: 'Birdtag Desktop', short_name: 'Birdtag', upstream_services: false };
(function () {
  try {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', 'branding.json', false);  // sync: other scripts read BRAND_* at load
    xhr.send(null);
    if (xhr.status >= 200 && xhr.status < 300) {
      var data = JSON.parse(xhr.responseText);
      for (var k in data) { if (k.charAt(0) !== '_') BRAND[k] = data[k]; }
    }
  } catch (e) { /* keep defaults */ }
  window.BRAND = BRAND;
})();
var BRAND_NAME = BRAND.product_name;
var BRAND_SHORT = BRAND.short_name;

(function () {
  if (!BRAND.upstream_services) {
    var css = [
      '#accountBtn', '#donateBtnMain', '#donateDlg', '#analyzeDestCloud', '#cloudQueuePanel',
      '#perchUploadsPanel', '.folder-perch-btn', '#cloudAccountDlg', '#analyzeDlgCloudEstPanel', '#openFeedback',
    ].join(',') + '{display:none !important}';
    var st = document.createElement('style');
    st.textContent = css;
    (document.head || document.documentElement).appendChild(st);
  }

  function fill(tplText) {
    return String(tplText).split('{name}').join(BRAND_NAME).split('{short}').join(BRAND_SHORT);
  }
  function apply() {
    var i, el, nodes = document.querySelectorAll('[data-brand]');
    for (i = 0; i < nodes.length; i++) {
      el = nodes[i];
      el.textContent = el.getAttribute('data-brand') === 'short' ? BRAND_SHORT : BRAND_NAME;
    }
    nodes = document.querySelectorAll('[data-brand-text]');
    for (i = 0; i < nodes.length; i++) nodes[i].textContent = fill(nodes[i].getAttribute('data-brand-text'));
    ['title', 'alt', 'aria-label', 'placeholder'].forEach(function (a) {
      var list = document.querySelectorAll('[data-brand-tpl-' + a + ']');
      for (var j = 0; j < list.length; j++) list[j].setAttribute(a, fill(list[j].getAttribute('data-brand-tpl-' + a)));
    });
  }
  window.applyBranding = apply;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply);
  else apply();
})();
