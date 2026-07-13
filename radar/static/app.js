/* Slovenia Radar — multi-panel Leaflet dashboard.
   Talks to the FastAPI backend with RELATIVE urls so it works under /radar. */
'use strict';

const API = 'api';                       // relative -> /radar/api when proxied
const MOMENTS = [
  { id: 'DBZH', label: 'Reflectivity' },
  { id: 'RATE', label: 'Rainfall rate' },
  { id: 'ET', label: 'Echo tops' },
  { id: 'VIL', label: 'VIL (hail/severe)' },
  { id: 'VRADH', label: 'Velocity' },
  { id: 'AZSHEAR', label: 'Azimuthal shear (rotation)' },
  { id: 'TH', label: 'Total power' },
  { id: 'RHOHV', label: 'ρHV particle uniformity' },
  { id: 'ZDR', label: 'ZDR (particle shape)' },
];
const ANALYSIS_MOMENTS = [
  { id: 'NOWCAST', label: 'Nowcast +30min' },
  { id: 'ACCUM', label: 'Rain accumulation 1h' },
  { id: 'MOTION', label: 'Storm motion (arrows)' },
  { id: 'WIND', label: 'Dual-Doppler wind' },
];
const MOMENT_LABEL = Object.fromEntries(
  [...MOMENTS, ...ANALYSIS_MOMENTS].map(m => [m.id, m.label]));
const COLUMN = new Set(['ET', 'VIL']);            // vertical products, no tilt
const CMAXABLE = new Set(['DBZH', 'TH', 'RATE']); // support the CMAX tilt option
const SCALAR_ANALYSIS = new Set(['ACCUM', 'NOWCAST', 'WIND']); // PNG field, no tilt
const ARROW = new Set(['MOTION', 'WIND']);        // draw vector arrows
const NOTILT = new Set([...COLUMN, ...SCALAR_ANALYSIS, 'MOTION']);
const DARK_TILES = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

const state = {
  layout: 4,
  frames: [],          // [{key, iso}]
  index: 0,
  tilts: [],           // [{index, elevation}]
  sources: [],         // select options [{value,label}]
  gridBounds: null,    // [[s,w],[n,e]]
  playing: false,
  timer: null,
  speed: 350,
  archiveMode: false,  // 24h flagship archive vs live 90-min window
  liveFrames: [],      // the live timeline, saved while in archive mode
};

const panels = [];     // {el, map, overlay, cfg:{source,moment,tilt}, legendCache}
let syncing = false;

/* ---------- boot ---------- */
async function boot() {
  const [src, tl] = await Promise.all([
    fetch(`${API}/sources`).then(r => r.json()),
    fetch(`${API}/timeline?node=silis`).then(r => r.json()),
  ]);

  state.gridBounds = src.grid.bounds;
  state.siteById = Object.fromEntries(
    src.sources.map(s => [s.id, { name: s.name, lat: s.lat, lon: s.lon, country: s.country }])
  );
  state.momentsByNode = Object.fromEntries(src.sources.map(s => [s.id, s.moments || []]));
  state.frames = tl.frames;
  state.liveFrames = tl.frames;
  state.index = Math.max(0, state.frames.length - 1);
  state.tilts = tl.sweeps;
  state.flagship = `comp:${src.default_composite.join(',')}`;   // archived product source

  // Build source options: composite first, then each active radar.
  const active = src.sources.filter(s => s.active);
  state.sources = [
    { value: `comp:${src.default_composite.join(',')}`, label: '◆ Composite (SI+HR)' },
    ...active.map(s => ({
      value: `single:${s.id}`,
      label: `${s.name}${s.country !== 'SI' ? ' (' + s.country + ')' : ''}`,
    })),
  ];

  const defaults = [
    { source: state.sources[0].value, moment: 'DBZH', tilt: 'cmax' },
    {},
    {},
    {},
  ];

  const tpl = document.getElementById('panelTpl');
  const grid = document.getElementById('grid');
  for (let i = 0; i < 4; i++) {
    const el = tpl.content.firstElementChild.cloneNode(true);
    grid.appendChild(el);
    panels.push(buildPanel(el, defaults[i].source ? defaults[i] : { source: state.sources[0].value, moment: 'DBZH', tilt: 'cmax' }));
  }
  syncMaps();
  wireControls();
  updateTimelineUI();
  refreshAll(true);   // initial load: set tiles + legends + warm the cache
  setLayout(1);
}

/* ---------- panel construction ---------- */
function buildPanel(el, cfg) {
  const map = L.map(el.querySelector('.map'), {
    center: [46.15, 14.95], zoom: 8, zoomControl: false,
    attributionControl: true, preferCanvas: true,
  });
  L.tileLayer(DARK_TILES, {
    subdomains: 'abcd', maxZoom: 12,
    attribution: '&copy; OpenStreetMap &copy; CARTO',
  }).addTo(map);
  const overlay = L.imageOverlay('', state.gridBounds,
    { opacity: 0.85, className: 'radar-overlay', interactive: false }).addTo(map);
  const markers = L.layerGroup().addTo(map);
  const arrows = L.layerGroup().addTo(map);

  // loading / no-data indicator layered over the map
  const loading = document.createElement('div');
  loading.className = 'map-loading';
  loading.innerHTML = '<div class="spinner"></div><span></span>';
  el.querySelector('.map').appendChild(loading);

  const panel = { el, map, overlay, markers, arrows, loading, cfg: { ...cfg }, legendCache: {} };

  // Reveal the tile only once it has actually rendered; flag missing data.
  overlay.on('load', () => markReady(panel));
  overlay.on('error', () => showNoData(panel));

  // dropdowns
  const selSrc = el.querySelector('.sel-source');
  state.sources.forEach(s => selSrc.add(new Option(s.label, s.value)));
  selSrc.value = cfg.source;
  const selMom = el.querySelector('.sel-moment');
  const selTilt = el.querySelector('.sel-tilt');
  selTilt.add(new Option('CMAX', 'cmax'));
  selTilt.add(new Option('All tilts', 'col'));
  state.tilts.forEach(t => selTilt.add(new Option(`${t.elevation.toFixed(1)}°`, t.index)));

  panel.refreshMoments = () => {
    const avail = availableMoments(panel.cfg.source);
    selMom.innerHTML = '';
    avail.forEach(m => selMom.add(new Option(m.label, m.id)));
    if (!avail.some(m => m.id === panel.cfg.moment)) panel.cfg.moment = avail[0].id;
    selMom.value = panel.cfg.moment;
    syncTiltState(panel, selTilt);
  };

  selSrc.onchange = () => {
    panel.cfg.source = selSrc.value;
    panel.refreshMoments();
    updatePanel(panel, true);
  };
  selMom.onchange = () => { panel.cfg.moment = selMom.value; syncTiltState(panel, selTilt); updatePanel(panel, true); };
  selTilt.onchange = () => {
    panel.cfg.tilt = selTilt.value === 'cmax' ? 'cmax' : +selTilt.value;
    updatePanel(panel, true);
  };
  panel.refreshMoments();

  // hover inspector
  const readout = el.querySelector('.hover-readout');
  map.on('mousemove', throttle(e => queryValue(panel, e.latlng, readout), 120));
  map.on('mouseout', () => { readout.textContent = 'move over map…'; });

  return panel;
}

/* ---------- tile + legend updates ---------- */
function isCmax(cfg) { return cfg.tilt === 'cmax' && CMAXABLE.has(cfg.moment); }

function applyTiltParams(p, cfg) {
  if (COLUMN.has(cfg.moment)) return;      // column products: backend uses all tilts
  if (isCmax(cfg)) { p.set('cmax', 'true'); p.set('sweep', '0'); }
  else { p.set('sweep', cfg.tilt === 'cmax' ? '0' : cfg.tilt); }
}

function availableMoments(sourceVal) {
  const [kind, ids] = sourceVal.split(':');
  const nodeIds = kind === 'comp' ? ids.split(',') : [ids];
  const set = new Set();
  nodeIds.forEach(id => (state.momentsByNode[id] || []).forEach(x => set.add(x)));
  const list = MOMENTS.filter(m => set.has(m.id));
  // Analysis products: always offer motion/accum/nowcast; WIND needs a pair.
  const analysis = ANALYSIS_MOMENTS.filter(m => m.id !== 'WIND' || nodeIds.length >= 2);
  return (list.length ? list : MOMENTS).concat(analysis);
}

/* Enable/disable the tilt selector for the current moment. */
function syncTiltState(panel, selTilt) {
  const m = panel.cfg.moment;
  if (NOTILT.has(m)) {                        // column + analysis products
    selTilt.disabled = true;
    panel.cfg.tilt = 'col';
    selTilt.value = 'col';
    return;
  }
  selTilt.disabled = false;
  if (panel.cfg.tilt === 'col') panel.cfg.tilt = CMAXABLE.has(m) ? 'cmax' : 0;
  if (!CMAXABLE.has(m) && panel.cfg.tilt === 'cmax') panel.cfg.tilt = 0;
  selTilt.value = panel.cfg.tilt === 'cmax' ? 'cmax' : String(panel.cfg.tilt);
}

/* Shared quantity/param builder for tile + value requests. MOTION uses a faint
   reflectivity backdrop under its arrows. */
function fieldParams(p, cfg) {
  if (cfg.moment === 'MOTION') {
    p.set('quantity', 'DBZH'); p.set('cmax', 'true'); p.set('sweep', '0');
  } else {
    p.set('quantity', cfg.moment);
    if (cfg.moment === 'ACCUM') p.set('window', '60');
    else if (cfg.moment === 'NOWCAST') p.set('lead', '30');
    else if (!SCALAR_ANALYSIS.has(cfg.moment)) applyTiltParams(p, cfg);
  }
  const [kind, ids] = cfg.source.split(':');
  if (kind === 'comp') p.set('nodes', ids); else p.set('node', ids);
}

function tileUrl(cfg, key) {
  const p = new URLSearchParams({ time: key });
  fieldParams(p, cfg);
  return `${API}/tile.png?${p}`;
}

function targetOpacity(panel) { return panel.cfg.moment === 'MOTION' ? 0.4 : 0.85; }

function showLoader(panel, label) {
  panel.loading.className = 'map-loading show';
  panel.loading.querySelector('span').textContent = `Rendering ${label}…`;
  panel.overlay.setOpacity(0);           // hide the previous product immediately
}
function markReady(panel) {
  panel.loading.className = 'map-loading';
  panel.overlay.setOpacity(targetOpacity(panel));
}
function showNoData(panel) {
  panel.loading.className = 'map-loading show nodata';
  panel.loading.querySelector('span').textContent = 'No data for this product/time';
  panel.overlay.setOpacity(0);
}

function updatePanel(panel, cfgChanged = false) {
  const key = state.frames[state.index]?.key;
  if (!key) return;
  const label = MOMENT_LABEL[panel.cfg.moment] || panel.cfg.moment;
  if (cfgChanged) showLoader(panel, label);     // switching product: hide old, spin
  panel.overlay.setUrl(tileUrl(panel.cfg, key));
  updateArrows(panel);
  if (cfgChanged) { updateLegend(panel); updateMarkers(panel); preloadPanel(panel); }
}

/* Storm-motion / dual-Doppler wind arrows (drawn client-side from vectors). */
async function updateArrows(panel) {
  panel.arrows.clearLayers();
  if (!ARROW.has(panel.cfg.moment)) return;
  const key = state.frames[state.index]?.key;
  if (!key) return;
  const [kind, ids] = panel.cfg.source.split(':');
  let url;
  if (panel.cfg.moment === 'WIND') {
    url = `${API}/windvectors?time=${key}&nodes=${ids}&step=45`;
  } else {
    const q = kind === 'comp' ? `nodes=${ids}` : `node=${ids}`;
    url = `${API}/motion?time=${key}&${q}&lag=15`;
  }
  const token = (panel._arrowTok = (panel._arrowTok || 0) + 1);
  try {
    const d = await fetch(url).then(r => r.json());
    if (token !== panel._arrowTok) return;   // superseded
    drawArrows(panel, d.vectors || [], panel.cfg.moment === 'WIND' ? '#ffffff' : '#ffe14d');
  } catch (e) { /* ignore */ }
}

function drawArrows(panel, vectors, color) {
  const SECS = 1200;   // arrow length = 20-minute projected displacement
  vectors.forEach(v => {
    const dlat = v.v / 111320 * SECS;
    const dlon = v.u / (111320 * Math.cos(v.lat * Math.PI / 180)) * SECS;
    const mag = Math.hypot(dlat, dlon);
    if (mag < 1e-6) return;
    const end = [v.lat + dlat, v.lon + dlon];
    L.polyline([[v.lat, v.lon], end], { color, weight: 1.6, opacity: 0.95 })
      .addTo(panel.arrows).bindTooltip(`${v.speed} m/s`, { sticky: true });
    const ux = dlat / mag, uy = dlon / mag;   // unit direction
    for (const a of [Math.PI * 0.82, -Math.PI * 0.82]) {   // barbs
      const rx = ux * Math.cos(a) - uy * Math.sin(a);
      const ry = ux * Math.sin(a) + uy * Math.cos(a);
      const bl = mag * 0.32;
      L.polyline([end, [end[0] + rx * bl, end[1] + ry * bl]],
        { color, weight: 1.6, opacity: 0.95 }).addTo(panel.arrows);
    }
  });
}

/* Radar-site markers. For velocity (radar-relative) they're emphasised with a
   permanent label so it's clear which radar the readings belong to. */
function updateMarkers(panel) {
  panel.markers.clearLayers();
  const [kind, ids] = panel.cfg.source.split(':');
  const nodeIds = kind === 'comp' ? ids.split(',') : [ids];
  const isVel = panel.cfg.moment === 'VRADH';
  nodeIds.forEach(id => {
    const s = state.siteById[id];
    if (!s || s.lat == null) return;
    L.circleMarker([s.lat, s.lon], {
      radius: isVel ? 6 : 4,
      color: isVel ? '#ffd23f' : '#39b3ff',
      weight: 2, fillColor: '#0b0e13', fillOpacity: 0.9,
    }).addTo(panel.markers).bindTooltip(
      isVel ? `📡 ${s.name}` : s.name,
      { permanent: isVel, direction: 'top', className: 'radar-tip', offset: [0, -4] }
    );
  });
}

function refreshAll(cfgChanged = false) { panels.forEach(p => updatePanel(p, cfgChanged)); }

async function updateLegend(panel) {
  const q = panel.cfg.moment === 'MOTION' ? 'DBZH' : panel.cfg.moment;
  const box = panel.el.querySelector('.legend');
  if (!panel.legendCache[q]) {
    panel.legendCache[q] = await fetch(`${API}/legend?quantity=${q}`).then(r => r.json());
  }
  const lg = panel.legendCache[q];
  const n = lg.stops.length;
  const vals = lg.stops.map(s => s.value);
  const min = vals[0], max = vals[vals.length - 1];
  // Linear scale positions by value; log/skewed scale (rain rate) by even steps.
  const pos = lg.scale === 'log'
    ? i => (i / (n - 1) * 100)
    : i => ((vals[i] - min) / (max - min) * 100);
  const grad = lg.stops.map((s, i) => `${s.color} ${pos(i).toFixed(1)}%`).join(', ');
  box.innerHTML =
    `<span class="lbl">${min}</span>` +
    `<span class="bar" style="background:linear-gradient(90deg,${grad})"></span>` +
    `<span class="lbl">${max} ${lg.units}</span>`;
}

/* ---------- hover value query ---------- */
let hoverAbort = null;
async function queryValue(panel, latlng, readout) {
  const key = state.frames[state.index]?.key;
  if (!key) return;
  const p = new URLSearchParams({
    time: key, lat: latlng.lat.toFixed(4), lon: latlng.lng.toFixed(4),
  });
  fieldParams(p, panel.cfg);
  if (hoverAbort) hoverAbort.abort();
  hoverAbort = new AbortController();
  try {
    const d = await fetch(`${API}/value?${p}`, { signal: hoverAbort.signal }).then(r => r.json());
    if (!d.in_range) { readout.innerHTML = 'out of range'; return; }
    const parts = [d.value === null ? '—' : `<b>${d.value} ${d.units || ''}</b>`];
    if (d.beam_height_m != null) parts.push(`${d.beam_height_m} m`);
    if (d.source) parts.push(d.source);
    readout.innerHTML = parts.join(' · ');
  } catch (e) { /* aborted */ }
}

/* ---------- playback ---------- */
function setIndex(i) {
  state.index = (i + state.frames.length) % state.frames.length;
  refreshAll();
  updateTimelineUI();
}
function play() {
  state.playing = true;
  document.getElementById('playBtn').textContent = '⏸';
  clearInterval(state.timer);
  state.timer = setInterval(() => setIndex(state.index + 1), state.speed);
}
function pause() {
  state.playing = false;
  document.getElementById('playBtn').textContent = '▶';
  clearInterval(state.timer);
}

/* preload frames of one panel into browser cache for smooth looping */
function preloadPanel(panel) {
  state.frames.forEach(f => { const im = new Image(); im.src = tileUrl(panel.cfg, f.key); });
}

/* ---------- controls / layout ---------- */
function wireControls() {
  document.getElementById('layoutGroup').addEventListener('click', e => {
    if (e.target.dataset.layout) setLayout(+e.target.dataset.layout);
  });
  document.getElementById('playBtn').onclick = () => state.playing ? pause() : play();
  document.getElementById('stepFwd').onclick = () => { pause(); setIndex(state.index + 1); };
  document.getElementById('stepBack').onclick = () => { pause(); setIndex(state.index - 1); };
  document.getElementById('speedSel').onchange = e => {
    state.speed = +e.target.value; if (state.playing) play();
  };
  const slider = document.getElementById('timeSlider');
  slider.oninput = e => { pause(); setIndex(+e.target.value); };
  document.getElementById('archiveBtn').onclick = toggleArchive;
}

/* Switch between the live 90-min window and the persisted 24h reflectivity
   archive. In archive mode every panel shows the flagship composite CMAX
   reflectivity (the only product kept for the full day). */
async function toggleArchive() {
  pause();
  const btn = document.getElementById('archiveBtn');
  state.archiveMode = !state.archiveMode;
  btn.classList.toggle('active', state.archiveMode);
  if (state.archiveMode) {
    const a = await fetch(`${API}/archive`).then(r => r.json());
    if (!a.frames.length) {
      state.archiveMode = false; btn.classList.remove('active');
      alert('No archive yet — it fills in as new frames arrive (up to 24h).');
      return;
    }
    state.frames = a.frames;
    panels.forEach(p => { p.cfg = { source: state.flagship, moment: 'DBZH', tilt: 'cmax' }; });
  } else {
    state.frames = state.liveFrames;
  }
  state.index = state.frames.length - 1;
  panels.forEach(p => { if (p.refreshMoments) p.refreshMoments(); });
  refreshAll(true);
  updateTimelineUI();
}

function setLayout(n) {
  state.layout = n;
  document.getElementById('grid').className = `layout-${n}`;
  document.querySelectorAll('#layoutGroup button').forEach(b =>
    b.classList.toggle('active', +b.dataset.layout === n));
  setTimeout(() => panels.forEach(p => p.map.invalidateSize()), 60);
}

function updateTimelineUI() {
  const slider = document.getElementById('timeSlider');
  slider.max = Math.max(0, state.frames.length - 1);
  slider.value = state.index;
  const f = state.frames[state.index];
  if (!f) return;
  const d = new Date(f.iso);
  document.getElementById('frameTime').textContent =
    d.toLocaleString([], { hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short' });
  const ageMin = Math.round((Date.now() - d.getTime()) / 60000);
  document.getElementById('frameAge').textContent =
    `${d.toISOString().slice(11, 16)}Z · ${ageMin} min ago`;
  const ticks = document.getElementById('ticks');
  if (ticks.childElementCount === 0 && state.frames.length) {
    const first = new Date(state.frames[0].iso), last = new Date(f.iso);
    ticks.innerHTML =
      `<span>${first.toISOString().slice(11, 16)}Z</span>` +
      `<span>${last.toISOString().slice(11, 16)}Z</span>`;
  }
}

/* ---------- map sync ---------- */
function syncMaps() {
  panels.forEach(src => {
    src.map.on('move zoom', () => {
      if (syncing) return;
      syncing = true;
      const c = src.map.getCenter(), z = src.map.getZoom();
      panels.forEach(p => { if (p !== src) p.map.setView(c, z, { animate: false }); });
      syncing = false;
    });
  });
}

/* ---------- utils ---------- */
function throttle(fn, ms) {
  let last = 0, t = null;
  return (...a) => {
    const now = Date.now();
    if (now - last >= ms) { last = now; fn(...a); }
    else { clearTimeout(t); t = setTimeout(() => { last = Date.now(); fn(...a); }, ms); }
  };
}

boot();
