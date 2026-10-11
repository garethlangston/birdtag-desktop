// node check.js <brand.js path> : renders species cells + a "model missing" status line
const { JSDOM } = require('jsdom');
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const html = `<!doctype html><html><head><title data-brand-text="Culling — {name}">x</title></head><body>
<span data-brand="name">?</span><button id="b" data-brand-tpl-title="Open {short} settings">s</button>
<div id="sp"></div><div id="status"></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', url: 'http://localhost/' });
const w = dom.window;
w.eval(src);
w.document.dispatchEvent(new w.Event('DOMContentLoaded'));
w.document.getElementById('sp').innerHTML =
  '<td class="species">Common Kestrel</td><td>Lesser Kestrel</td><td>Kestrel</td>';
const t0 = Date.now();
// Settings status line when EVA-02 is missing (lists the legacy folder):
w.document.getElementById('status').textContent =
  'UK/Europe/World bird model not found. Looked in: C:\\Users\\G\\AppData\\Local\\BirdtagDesktop\\models_extra; C:\\Users\\G\\AppData\\Local\\ProjectKestrel\\models_extra';
setTimeout(() => {
  console.log(JSON.stringify({
    species: w.document.getElementById('sp').textContent,
    brandSpan: w.document.querySelector('[data-brand]').textContent,
    title: w.document.title, btnTitle: w.document.getElementById('b').title,
    statusSettledMs: Date.now() - t0,
  }));
  process.exit(0);
}, 50);
setTimeout(() => { console.log('HUNG: event loop starved >3s'); process.exit(2); }, 3000).unref();
