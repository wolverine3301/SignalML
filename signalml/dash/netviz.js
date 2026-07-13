/* NetViz — dependency-free SVG neural-net schematic with animated pulses.
 *
 * Deliberately a SCHEMATIC: blocks are architecture components, node lattices are
 * decorative (bounded counts), only the topology is truthful. Pulses are meant to
 * be bound to real signals (training steps/sec, active model, stage progress) —
 * cosmetic-only mode is `ambient` for demos.
 *
 * Usage:
 *   const viz = NetViz.render(container, spec, opts);
 *   spec = {
 *     blocks: [{id, label, sub?, cols: [3,5,3], accent?, skip?: true, lane?: 1}],
 *     links:  [{from, to, label?}],
 *   }
 *   opts = {
 *     orientation: "horizontal" (default) | "vertical",
 *     geometry: {nodeR, rowGap, colGap, blockPad, blockGap, laneGap, labelH},
 *     ambient: false, accent: "#3987e5",
 *   }
 *   viz.pulse("from>to")        one pulse along a link (block ids joined by ">")
 *   viz.setRate("from>to", hz)  sustained pulse rate (0 stops)
 *   viz.setActive(blockId, on)  glow a block (e.g. the model currently training)
 *   viz.destroy()
 *
 * Layout: main-flow blocks run along the orientation axis; a block with `lane: 1`
 * sits BESIDE the next main-flow block (side-feeder, e.g. a speaker embedding
 * injected into the acoustic model). `skip: true` draws U-Net skip arcs between
 * mirrored layers. One rAF loop per instance; pauses when the document is hidden.
 */
"use strict";

const NetViz = (() => {
  const NS = "http://www.w3.org/2000/svg";
  // geometry defaults; override per-render via opts.geometry (e.g. denser lattices)
  const GEO = {
    nodeR: 3.5,   // node radius
    colGap: 26,   // px between layers (along the flow axis)
    rowGap: 16,   // px between nodes within a layer (across the flow axis)
    blockPad: 16, // lattice inset inside the frame
    blockGap: 56, // px between blocks along the flow
    laneGap: 28,  // px between the main flow and a side lane
    labelH: 30,   // room reserved for the block label
  };

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function blockSize(block, geo, vert) {
    const across = (Math.max(...block.cols) - 1) * geo.rowGap;   // within a layer
    const along = (block.cols.length - 1) * geo.colGap;          // layer stacking
    return vert
      ? { w: geo.blockPad * 2 + across, h: geo.blockPad * 2 + along + geo.labelH }
      : { w: geo.blockPad * 2 + along, h: geo.blockPad * 2 + across + geo.labelH };
  }

  function render(container, spec, opts = {}) {
    const accent = opts.accent || "#3987e5";
    const vert = opts.orientation === "vertical";
    const geo = { ...GEO, ...(opts.geometry || {}) };
    const blocks = new Map();
    for (const b of spec.blocks) blocks.set(b.id, { ...b, ...blockSize(b, geo, vert) });

    // ---- layout: main flow along the axis, lane blocks beside their anchor ----
    const mains = spec.blocks.filter((b) => !b.lane).map((b) => blocks.get(b.id));
    const lanes = spec.blocks.filter((b) => b.lane).map((b) => blocks.get(b.id));
    let flow = 0, mainAcross = 0;
    for (const b of mains) {
      b.flow = flow;
      flow += (vert ? b.h : b.w) + geo.blockGap;
      mainAcross = Math.max(mainAcross, vert ? b.w : b.h);
    }
    const flowTotal = flow - geo.blockGap;
    for (const b of mains) {  // center main blocks in the across axis
      if (vert) { b.y = b.flow; b.x = (mainAcross - b.w) / 2; }
      else { b.x = b.flow; b.y = (mainAcross - b.h) / 2; }
    }
    let laneAcross = 0;
    for (const b of lanes) {  // a lane block sits beside the NEXT main block in order
      const idx = spec.blocks.indexOf(spec.blocks.find((s) => s.id === b.id));
      const anchor = spec.blocks.slice(idx + 1).map((s) => blocks.get(s.id))
        .find((s) => !s.lane) || mains[mains.length - 1];
      if (vert) {
        b.x = mainAcross + geo.laneGap;
        b.y = anchor.y + Math.max(0, (anchor.h - b.h) / 2);
        laneAcross = Math.max(laneAcross, geo.laneGap + b.w);
      } else {
        b.y = mainAcross + geo.laneGap;
        b.x = anchor.x + Math.max(0, (anchor.w - b.w) / 2);
        laneAcross = Math.max(laneAcross, geo.laneGap + b.h);
      }
    }
    const width = vert ? mainAcross + laneAcross : flowTotal;
    const height = vert ? flowTotal : mainAcross + laneAcross;

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

    // node position: layer ci, node ri within the layer (centered across)
    const nodePos = (b, ci, ri) => {
      const pad = (Math.max(...b.cols) - b.cols[ci]) * geo.rowGap / 2;
      return vert
        ? [b.x + geo.blockPad + pad + ri * geo.rowGap,
           b.y + geo.labelH + geo.blockPad + ci * geo.colGap]
        : [b.x + geo.blockPad + ci * geo.colGap,
           b.y + geo.labelH + geo.blockPad + pad + ri * geo.rowGap];
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
      // intra-block edges (faint, bipartite between adjacent layers)
      for (let ci = 0; ci < b.cols.length - 1; ci++) {
        for (let ri = 0; ri < b.cols[ci]; ri++) {
          for (let rj = 0; rj < b.cols[ci + 1]; rj++) {
            const [x1, y1] = nodePos(b, ci, ri);
            const [x2, y2] = nodePos(b, ci + 1, rj);
            el("line", { x1, y1, x2, y2, class: "nv-edge" }, g);
          }
        }
      }
      // U-Net skip arcs between mirrored layers (top edge horiz / left edge vert)
      if (b.skip) {
        const n = b.cols.length;
        for (let ci = 0; ci < Math.floor(n / 2); ci++) {
          const mirror = n - 1 - ci;
          if (mirror <= ci) continue;
          const [ax, ay] = nodePos(b, ci, 0);
          const [bx, by] = nodePos(b, mirror, 0);
          const lift = 14 + ci * 6;
          const d = vert
            ? `M ${ax - 6} ${ay} C ${ax - lift} ${ay}, ${bx - lift} ${by}, ${bx - 6} ${by}`
            : `M ${ax} ${ay - 6} C ${ax} ${ay - lift}, ${bx} ${by - lift}, ${bx} ${by - 6}`;
          el("path", { d, class: "nv-skip", stroke: b.accent || accent }, g);
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

    // ---- links: along-flow = exit/entry faces; lane links = facing side edges ----
    const midOf = (b) => [b.x + b.w / 2, b.y + geo.labelH + (b.h - geo.labelH) / 2];
    const links = new Map();
    for (const l of spec.links) {
      const a = blocks.get(l.from), b = blocks.get(l.to);
      if (!a || !b) throw new Error(`netviz link references unknown block: ${l.from}>${l.to}`);
      let d;
      if (vert && a.y + a.h <= b.y + 1) {          // downward chain link
        const x1 = a.x + a.w / 2, y1 = a.y + a.h;
        const x2 = b.x + b.w / 2, y2 = b.y;
        const mid = (y1 + y2) / 2;
        d = `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;
      } else if (!vert && a.x + a.w <= b.x + 1) {  // rightward chain link
        const [, y1] = midOf(a), [, y2] = midOf(b);
        const x1 = a.x + a.w, x2 = b.x;
        const mid = (x1 + x2) / 2;
        d = `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
      } else if (vert) {                            // side lane -> main (horizontal)
        const [, y1] = midOf(a), [, y2] = midOf(b);
        const [x1, x2] = a.x > b.x ? [a.x, b.x + b.w] : [a.x + a.w, b.x];
        const mid = (x1 + x2) / 2;
        d = `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
      } else {                                      // side lane -> main (vertical)
        const [x1] = midOf(a), [x2] = midOf(b);
        const [y1, y2] = a.y > b.y ? [a.y, b.y + b.h] : [a.y + a.h, b.y];
        const mid = (y1 + y2) / 2;
        d = `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;
      }
      const path = el("path", { d, class: "nv-link" }, linkLayer);
      const key = `${l.from}>${l.to}`;
      links.set(key, { path, len: path.getTotalLength(), rate: 0, acc: 0, pulses: [] });
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
