"""Karaoke A/B page: one recording, two alignments, the current word lit up in each.

Built to judge forced alignments by ear (docs/notes/chunked_alignment.md): press play
and watch which column lights the word actually being sung. Any pair of TextGrid sets
works - two aligners, two settings, an aligner vs hand labels.

    python -m uv run python scripts/karaoke_compare.py --out <dir> \
        --a <textgrids A> --b <textgrids B> --audio <audio dir> \
        [--label-a "Whole song"] [--label-b "Phrase by phrase"] \
        [--titles titles.json] [--flac] [--title "Alignment check"]

Each item is an id found in both A and B, as ``<dir>/<id>.TextGrid`` or
``<dir>/<id>/<id>.TextGrid``; its audio is ``<audio>/<id>.wav`` or
``<audio>/<id>/<id>.wav``. ``--titles`` maps id -> display name (default: the id).
The page is self-contained: ``<out>/index.html`` plus ``<out>/audio/``. Open it from a
local web server (``python -m http.server`` in <out>) - some viewers block audio on
file:// pages.

Per-item numbers, when a phones tier exists: share of phones at the 30 ms MFA frame
floor, runs of >= 4 floor phones in a row, the longest single phone - the symptoms of
an alignment that gave up (docs/notes/chunked_alignment.md).
"""

# ruff: noqa: E501 - the page template below is minified HTML/CSS/JS

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SILENCE = {"", "sil", "sp", "<eps>"}
FLOOR_SEC = 0.031


def _find(root: Path, item: str, exts: tuple[str, ...]) -> Path | None:
    for ext in exts:
        for p in (root / f"{item}{ext}", root / item / f"{item}{ext}"):
            if p.exists():
                return p
    return None


def _ids(root: Path) -> set[str]:
    return {p.stem for p in root.rglob("*.TextGrid")}


def read_tiers(path: Path) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """(words, phones) from an MFA-style TextGrid; tier names may carry a speaker
    prefix ("spk - words")."""
    from praatio import textgrid as ptg

    tg = ptg.openTextgrid(str(path), includeEmptyIntervals=False)

    def tier(suffix: str) -> list[tuple[float, float, str]]:
        name = next((n for n in tg.tierNames if n.lower().endswith(suffix)), None)
        if name is None:
            return []
        return [(e.start, e.end, e.label) for e in tg.getTier(name).entries
                if e.label.strip() not in SILENCE]

    return tier("words"), tier("phones")


def phone_stats(phones: list[tuple[float, float, str]]) -> dict | None:
    if not phones:
        return None
    du = [e - s for s, e, _ in phones]
    run = runs4 = 0
    for d in du:
        run = run + 1 if d <= FLOOR_SEC else 0
        runs4 += run == 4
    return {"floor": round(sum(d <= FLOOR_SEC for d in du) / len(du), 4), "runs4": runs4,
            "longest": round(max(du), 2), "phones": len(du)}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#fafafa;--fg:#222;--muted:#666;--line:#ddd;--hi:#ffd54a;--a:#1f5fa8;--b:#1e8a4c;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#9a9a9a;--line:#333;--hi:#8a6d00;--a:#6ea4e6;--b:#5cc48a;--card:#1f1f1f}}
body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
select{font-size:15px;padding:4px;max-width:100%}audio{width:100%;margin-top:8px}
.stats{font-size:13px;color:var(--muted);margin-top:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px 16px}
@media (max-width:640px){.cols{grid-template-columns:1fr}.col{height:40vh!important}}
.col{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;height:calc(100vh - 220px);overflow:auto;line-height:1.9}
.col h3{margin:0 0 8px;font-size:15px}.col.a h3{color:var(--a)}.col.b h3{color:var(--b)}
.w{padding:2px 3px;border-radius:4px;cursor:pointer}.w.on{background:var(--hi);color:#000}
.gap{display:block;height:6px}p.help{margin:6px 0 0;font-size:13px;color:var(--muted);max-width:900px}
</style></head><body><header>
<select id="pick" aria-label="Recording"></select>
<p class="help">__HELP__</p>
<audio id="au" controls preload="auto"></audio><div class="stats" id="stats"></div></header>
<div class="cols"><div class="col a"><h3>__LA__</h3><div id="wa"></div></div>
<div class="col b"><h3>__LB__</h3><div id="wb"></div></div></div>
<script>
const ITEMS=__DATA__, LA=__LAJ__, LB=__LBJ__;
const pick=document.getElementById('pick'),au=document.getElementById('au');
ITEMS.forEach((s,i)=>pick.add(new Option(s.title,i)));
let els={},last={a:null,b:null};
function fmt(s,n){return s?`floor ${(s.floor*100).toFixed(0)}% · stacked runs ${s.runs4} · longest phone ${s.longest.toFixed(1)}s · words ${n}`:`words ${n}`}
function render(box,words){box.innerHTML='';const out=[];let prev=0;
 words.forEach(([s,e,w])=>{if(s-prev>1.2)box.appendChild(Object.assign(document.createElement('span'),{className:'gap'}));
  const el=document.createElement('span');el.className='w';el.textContent=w+' ';el.title=`${s.toFixed(2)}–${e.toFixed(2)}s`;
  el.onclick=()=>{au.currentTime=s;au.play()};box.appendChild(el);out.push([s,e,el]);prev=e});return out}
function load(i){const it=ITEMS[i];au.src=it.audio;last={a:null,b:null};
 els={a:render(document.getElementById('wa'),it.a),b:render(document.getElementById('wb'),it.b)};
 document.getElementById('stats').innerHTML=`<b style="color:var(--a)">${LA}:</b> ${fmt(it.stats_a,it.a.length)}<br><b style="color:var(--b)">${LB}:</b> ${fmt(it.stats_b,it.b.length)}`}
function tick(){const t=au.currentTime;for(const k of ['a','b']){const hit=(els[k]||[]).find(([s,e])=>t>=s&&t<e);
 const el=hit?hit[2]:null;if(el!==last[k]){if(last[k])last[k].classList.remove('on');if(el){el.classList.add('on');
 const box=el.parentElement.parentElement,r=el.getBoundingClientRect(),b=box.getBoundingClientRect();
 if(r.top<b.top+40||r.bottom>b.bottom-40)box.scrollTop+=r.top-b.top-b.height/3}last[k]=el}}requestAnimationFrame(tick)}
pick.onchange=()=>load(pick.value);load(0);tick();
</script></body></html>"""

HELP = ("Play, and watch which side lights up the word actually being sung. "
        "Click any word to jump there.")


def build(out: Path, a: Path, b: Path, audio: Path, label_a: str, label_b: str,
          titles: dict[str, str], flac: bool, title: str, help_text: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    items, skipped = [], []
    for item in sorted(_ids(a) & _ids(b)):
        ta, tb = _find(a, item, (".TextGrid",)), _find(b, item, (".TextGrid",))
        src = _find(audio, item, (".wav", ".flac", ".mp3", ".ogg"))
        if not (ta and tb and src):
            skipped.append(item)
            continue
        if flac and src.suffix == ".wav":
            import soundfile as sf

            y, sr = sf.read(str(src), dtype="float32")
            dst = out / "audio" / f"{item}.flac"
            sf.write(str(dst), y, sr, subtype="PCM_16")
        else:
            dst = out / "audio" / f"{item}{src.suffix}"
            shutil.copyfile(src, dst)
        wa, pa = read_tiers(ta)
        wb, pb = read_tiers(tb)
        items.append({
            "id": item, "title": titles.get(item, item), "audio": f"audio/{dst.name}",
            "a": [[round(s, 3), round(e, 3), w] for s, e, w in wa],
            "b": [[round(s, 3), round(e, 3), w] for s, e, w in wb],
            "stats_a": phone_stats(pa), "stats_b": phone_stats(pb),
        })
    html = (PAGE.replace("__TITLE__", title).replace("__HELP__", help_text)
            .replace("__LA__", label_a).replace("__LB__", label_b)
            .replace("__LAJ__", json.dumps(label_a)).replace("__LBJ__", json.dumps(label_b))
            .replace("__DATA__", json.dumps(items, ensure_ascii=False)))
    (out / "index.html").write_text(html, encoding="utf-8")
    stats = {i["id"]: {"a": i["stats_a"], "b": i["stats_b"]} for i in items}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="TextGrids, left column")
    ap.add_argument("--b", type=Path, required=True, help="TextGrids, right column")
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--titles", type=Path, help="JSON {id: display title}")
    ap.add_argument("--flac", action="store_true", help="store audio as 16-bit FLAC")
    ap.add_argument("--title", default="Alignment check")
    ap.add_argument("--help-text", default=HELP)
    args = ap.parse_args()
    titles = json.loads(args.titles.read_text(encoding="utf-8")) if args.titles else {}
    skipped = build(args.out, args.a, args.b, args.audio, args.label_a, args.label_b,
                    titles, args.flac, args.title, args.help_text)
    print(f"wrote {args.out / 'index.html'}")
    if skipped:
        print(f"skipped (missing TextGrid or audio): {skipped}")


if __name__ == "__main__":
    main()
