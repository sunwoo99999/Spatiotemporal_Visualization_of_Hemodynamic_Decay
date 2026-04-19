/**
 * Capture hero_t2dm.png from glass_brain.html
 * - Serves viz/ via local HTTP to avoid CORS issues with brain_data.json
 * - Waits for WebGL scene to load and reaches T2DM t_peak window
 * - Saves to viz/hero_t2dm.png
 */
const puppeteer = require('puppeteer');
const http      = require('http');
const fs        = require('fs');
const path      = require('path');

const VIZ_DIR   = path.join(__dirname, '..', 'viz');
const OUT_FILE  = path.join(VIZ_DIR, 'hero_t2dm.png');
const PORT      = 7771;
const WIDTH     = 1920;
const HEIGHT    = 1080;

// ── tiny static file server ──────────────────────────────────────────────────
const MIME = {
  '.html': 'text/html', '.js': 'application/javascript',
  '.json': 'application/json', '.png': 'image/png',
  '.css': 'text/css',
};
const server = http.createServer((req, res) => {
  const filePath = path.join(VIZ_DIR, req.url === '/' ? 'glass_brain.html' : req.url);
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  });
});

(async () => {
  // Start server
  await new Promise(r => server.listen(PORT, r));
  console.log(`[server] http://localhost:${PORT}`);

  const browser = await puppeteer.launch({
    headless: false,           // real GPU window → proper WebGL blending
    defaultViewport: { width: WIDTH, height: HEIGHT },
    args: [
      `--window-size=${WIDTH},${HEIGHT}`,
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-webgl',
      '--ignore-gpu-blocklist',
      '--enable-gpu-rasterization',
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT });

  console.log('[page] Navigating to glass_brain.html ...');
  await page.goto(`http://localhost:${PORT}/glass_brain.html`, { waitUntil: 'networkidle0', timeout: 60000 });

  // Switch to T2DM group
  await page.evaluate(() => setGroup('t2dm'));
  console.log('[page] Switched to T2DM group');

  // Set hero camera angle
  await page.evaluate(() => setHeroView());
  console.log('[page] Hero view set');

  // T2DM ARRIVALS: V1arr=2.0 + V1tPeak=7.209 → simTime peak = 9.21 s
  //                V2arr=3.5 + V2tPeak=6.966 → simTime peak = 10.47 s
  // Capture near V1 peak so V1,V2 are both bright simultaneously
  await page.evaluate(() => {
    paused = false;
    brainSim = 8.5;   // run-up just before V1 peak (9.21s)
    simSpeed = 0.18;
  });

  // Let animation run ~4 s real time → ~0.72 s sim → lands at ~9.22 s (V1 peak)
  await new Promise(r => setTimeout(r, 4000));

  // Freeze right at T2DM V1 peak
  await page.evaluate(() => {
    brainSim = 9.21;
    simSpeed = 0;
  });

  // One more RAF cycle to flush the frozen frame to the canvas
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
  await new Promise(r => setTimeout(r, 600));

  // Capture full page (3D canvas + HTML UI panels)
  const pngBuf = await page.screenshot({ type: 'png' });
  fs.writeFileSync(OUT_FILE, pngBuf);

  console.log(`[done] Saved → ${OUT_FILE}`);
  await browser.close();
  server.close();
})().catch(err => {
  console.error(err);
  server.close();
  process.exit(1);
});
