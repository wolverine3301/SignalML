/* NetViz — dependency-free SVG neural-net schematic with animated pulses.
 *
 * Deliberately a SCHEMATIC: blocks are architecture components, node lattices are
 * decorative (bounded counts), only the topology is truthful. Pulses are meant to
 * be bound to real signals (training steps/sec, active model, stage progress) —
 * cosmetic-only mode is `ambient` for demos.
 *
 * Usage:
 *   const viz = NetViz.render(container, spec, {ambient: false});
 *   spec = {
 *     blocks: [{id, label, sub?, cols: [3,5,3], accent?, skip?: true}],
 *     links:  [{from, to, label?}],
 *   }
 *   viz.pulse("from>to")        one pulse along a link (block ids joined by ">")
 *   viz.setRate("from>to", hz)  sustained pulse rate (0 stops)
 *   viz.setActive(blockId, on)  glow a block (e.g. the model currently training)
 *   viz.destroy()
 *
 * Layout: blocks flow left-to-right, links are cubic curves between block edges.
 * A block with `skip: true` draws U-Net-style skip arcs between mirrored columns.
 * One rAF loop per instance; pauses when the document is hidden.
 */
"use strict";

const NetViz = (() => {
  const NS = "http://www.w3.org/2000/svg";
  // geometry defaults; override per-render via opts.geometry (e.g. denser lattices)
  const GEO = {
    nodeR: 3.5,   // node radius
    colGap: 26,   // px between lattice columns
    rowGap: 16,   // px between lattice rows
    blockPad: 16, // lattice inset inside the frame
    blockGap: 56, // px between blocks
    labelH: 30,   // room reserved for the block label
  };

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function blockSize(block, geo) {
    const maxRows = Math.max(...block.cols);
    return {
      w: geo.blockPad * 2 + (block.cols.length - 1) * geo.colGap,
      h: geo.blockPad * 2 + (maxRows - 1) * geo.rowGap + geo.labelH,
    };
  }

  function render(container, spec, opts = {}) {
    const accent = opts.accent || "#3987e5";
    const geo = { ...GEO, ...(opts.geometry || {}) };
    const blocks = new Map();
    let x = 0, maxH = 0;
    for (const b of spec.blocks) {
      const size = blockSize(b, geo);
      blocks.set(b.id, { ...b, x, ...size });
      x += size.w + geo.blockGap;
      maxH = Math.max(maxH, size.h);
    }
    const width = x - geo.blockGap;
    const height = maxH + 8;
    for (const b of blocks.values()) b.y = (height - b.h) / 2; // vertical centering

    const svg = el("svg", {
      viewBox: `-4 -4 ${width + 8} ${height + 8}`,
      class: "netviz", role: "img",
      "aria-label": "architecture schematic",
    });
    container.appendChild(svg);

    // soft glow filter for pulses and active blocks
    const defs = el("defs", {}, svg);
    const filter = el("filter", { id: "nvglow", x: "-80%", y: "-80%",
                                  width: "260%", height: "260%" }, defs);
    el("feGaussianBlur", { stdDeviation: 2.2, result: "b" }, filter);
    const merge = el("feMerge", {}, filter);
    el("feMergeNode", { in: "b" }, merge);
    el("feMergeNode", { in: "SourceGraphic" }, merge);

    const linkLayer = el("g", {}, svg);
    const blockLayer = el("g", {}, svg);
    const pulseLayer = el("g", {}, svg);

    // ---- blocks: frame, label, node lattice, intra-block edges, skip arcs ----
    const nodePos = (b, ci, ri) => {
      const rows = b.cols[ci];
      const x0 = b.x + geo.blockPad + ci * geo.colGap;
      const y0 = b.y + geo.labelH + geo.blockPad +
        ((Math.max(...b.cols) - rows) * geo.rowGap) / 2 + ri * geo.rowGap;
      return [x0, y0];
    };

    for (const b of blocks.values()) {
      const g = el("g", { class: "nv-block", "data-block": b.id }, blockLayer);
      b.frame = el("rect", {
        x: b.x, y: b.y, width: b.w, height: b.h, rx: 10,
        class: "nv-frame", stroke: b.accent || "rgba(255,255,255,0.14)",
      }, g);
      el("text", { x: b.x + b.w / 2, y: b.y + 18, class: "nv-label",
                   "text-anchor": "middle" }, g).textContent = b.label;
      if (b.sub) {
        el("text", { x: b.x + b.w / 2, y: b.y + b.h - 6, class: "nv-sub",
                     "text-anchor": "middle" }, g).textContent = b.sub;
      }
      // intra-block edges (faint, bipartite between adjacent columns)
      for (let ci = 0; ci < b.cols.length - 1; ci++) {
        for (let ri = 0; ri < b.cols[ci]; ri++) {
          for (let rj = 0; rj < b.cols[ci + 1]; rj++) {
            const [x1, y1] = nodePos(b, ci, ri);
            const [x2, y2] = nodePos(b, ci + 1, rj);
            el("line", { x1, y1, x2, y2, class: "nv-edge" }, g);
          }
        }
      }
      // U-Net skip arcs between mirrored columns
      if (b.skip) {
        const n = b.cols.length;
        for (let ci = 0; ci < Math.floor(n / 2); ci++) {
          const mirror = n - 1 - ci;
          if (mirror <= ci) continue;
          const [x1, y1] = nodePos(b, ci, 0);
          const [x2] = nodePos(b, mirror, 0);
          const lift = 14 + ci * 6;
          el("path", {
            d: `M ${x1} ${y1 - 6} C ${x1} ${y1 - lift}, ${x2} ${y1 - lift}, ${x2} ${y1 - 6}`,
            class: "nv-skip", stroke: b.accent || accent,
          }, g);
        }
      }
      // nodes, with a slow phase-offset shimmer
      for (let ci = 0; ci < b.cols.length; ci++) {
        for (let ri = 0; ri < b.cols[ci]; ri++) {
          const [cx, cy] = nodePos(b, ci, ri);
          const node = el("circle", { cx, cy, r: geo.nodeR, class: "nv-node",
                                      fill: b.accent || accent }, g);
          node.style.animationDelay = `${((ci * 7 + ri * 13) % 20) / 10}s`;
        }
      }
    }

    // ---- links between blocks (cubic curves, right edge -> left edge) ----
    const links = new Map();
    for (const l of spec.links) {
      const a = blocks.get(l.from), b = blocks.get(l.to);
      if (!a || !b) throw new Error(`netviz link references unknown block: ${l.from}>${l.to}`);
      const x1 = a.x + a.w, y1 = a.y + geo.labelH + (a.h - geo.labelH) / 2;
      const x2 = b.x, y2 = b.y + geo.labelH + (b.h - geo.labelH) / 2;
      const mid = (x1 + x2) / 2;
      const path = el("path", {
        d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
        class: "nv-link",
      }, linkLayer);
      const key = `${l.from}>${l.to}`;
      links.set(key, { path, len: path.getTotalLength(), rate: 0, acc: 0, pulses: [] });
      if (l.label) {
        el("text", { x: mid, y: Math.min(y1, y2) - 6, class: "nv-sub",
                     "text-anchor": "middle" }, linkLayer).textContent = l.label;
      }
    }

    // ---- pulse engine: one rAF loop, pooled circles, hidden-tab pause ----
    const pool = [];
    const spawn = (link) => {
      const dot = pool.pop() ||
        el("circle", { r: 3, class: "nv-pulse", filter: "url(#nvglow)" }, pulseLayer);
      dot.setAttribute("fill", accent);
      dot.style.display = "";
      link.pulses.push({ dot, t: 0 });
    };

    let raf = null, last = 0;
    const tick = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000) || 0.016;
      last = now;
      for (const link of links.values()) {
        if (link.rate > 0) {
          link.acc += dt * link.rate;
          while (link.acc >= 1) { link.acc -= 1; spawn(link); }
        }
        for (let i = link.pulses.length - 1; i >= 0; i--) {
          const p = link.pulses[i];
          p.t += dt * 0.9; // pulse travel: ~1.1s per link
          if (p.t >= 1) {
            p.dot.style.display = "none";
            pool.push(p.dot);
            link.pulses.splice(i, 1);
            continue;
          }
          const pt = link.path.getPointAtLength(p.t * link.len);
          p.dot.setAttribute("cx", pt.x);
          p.dot.setAttribute("cy", pt.y);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    const start = () => { if (raf == null) { last = performance.now(); raf = requestAnimationFrame(tick); } };
    const stop = () => { if (raf != null) { cancelAnimationFrame(raf); raf = null; } };
    const onVis = () => (document.hidden ? stop() : start());
    document.addEventListener("visibilitychange", onVis);
    start();

    if (opts.ambient) {
      for (const link of links.values()) link.rate = 0.5 + Math.random() * 0.5;
    }

    return {
      svg,
      pulse(key, count = 1) {
        const link = links.get(key);
        if (link) for (let i = 0; i < count; i++) spawn(link);
      },
      setRate(key, hz) {
        const link = links.get(key);
        if (link) link.rate = Math.max(0, hz);
      },
      setActive(blockId, on = true) {
        const b = blocks.get(blockId);
        if (b) b.frame.classList.toggle("nv-active", on);
      },
      destroy() {
        stop();
        document.removeEventListener("visibilitychange", onVis);
        svg.remove();
      },
    };
  }

  return { render };
})();

/* Base styles — inject once so the module is drop-in. Colors ride on
   --nv-accent / --nv-ink custom properties set by the host page. */
(() => {
  if (document.getElementById("netviz-style")) return;
  const style = document.createElement("style");
  style.id = "netviz-style";
  style.textContent = `
    .netviz { width: 100%; height: auto; display: block; }
    .netviz .nv-frame { fill: rgba(255,255,255,0.025); stroke-width: 1; }
    .netviz .nv-frame.nv-active { stroke-width: 2;
      filter: url(#nvglow); }
    .netviz .nv-label { fill: var(--nv-ink, #c3c2b7); font: 600 11px system-ui;
      letter-spacing: 0.08em; text-transform: uppercase; }
    .netviz .nv-sub { fill: var(--nv-muted, #898781); font: 400 9px system-ui; }
    .netviz .nv-edge { stroke: var(--nv-ink, #c3c2b7); stroke-opacity: 0.07;
      stroke-width: 0.6; }
    .netviz .nv-skip { fill: none; stroke-opacity: 0.35; stroke-width: 1;
      stroke-dasharray: 3 3; }
    .netviz .nv-link { fill: none; stroke: var(--nv-ink, #c3c2b7);
      stroke-opacity: 0.25; stroke-width: 1.2; }
    .netviz .nv-node { opacity: 0.55; animation: nv-shimmer 4s ease-in-out infinite; }
    .netviz .nv-pulse { opacity: 0.95; }
    @keyframes nv-shimmer { 0%, 100% { opacity: 0.35; } 50% { opacity: 0.85; } }
    @media (prefers-reduced-motion: reduce) {
      .netviz .nv-node { animation: none; }
    }
  `;
  document.head.appendChild(style);
})();
