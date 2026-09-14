#!/usr/bin/env python3
"""build one self contained dashboard/index.html from reports/*.json.

standard library only. no external js or css, one inline script for the light/dark toggle. opens from file:// offline.
light is the default. add ?theme=dark to the url or press the dark button for dark.

usage
  python3 dashboard/build.py                  build, skipping reports whose name starts with _
  python3 dashboard/build.py --include-samples  also include _sample-*.json
  python3 dashboard/build.py --serve           build then serve dashboard/ on http://localhost:8765
"""

import argparse
import datetime as dt
import html
import json
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "dashboard" / "index.html"

GYM_ORDER = ["kernel-rocm", "nki-sim", "swe", "cve", "k8s", "iac", "ml"]
GYM_LABEL = {
    "kernel-rocm": "kernel (rocm)",
    "nki-sim": "kernel (nki, simulator)",
    "swe": "swe",
    "cve": "cve",
    "k8s": "k8s",
    "iac": "iac",
    "ml": "ml",
}
CHECK_ORDER = ["correctness", "cheat-unhardened", "cheat-hardened", "headroom", "ground-truth"]
STATUS_WORD = {"pass": "pass", "warn": "warn", "fail": "fail"}

# colours follow the dataviz skill reference palette. status colours are the
# reserved status set (good, warning, critical) and never reused for series.
CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --hairline: #e1e0d9;
  --ring: rgba(11,11,11,0.10);
  --good: #0ca30c;
  --warn: #fab219;
  --bad: #d03b3b;
  --track: #eeede8;
  --gap-bar: #2a78d6;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --hairline: #2c2c2a;
  --ring: rgba(255,255,255,0.10);
  --track: #2c2c2a;
  --gap-bar: #3987e5;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--page);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 22px;
  line-height: 1.35;
  padding: 28px 36px 48px;
}
header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--hairline);
  padding-bottom: 18px;
  margin-bottom: 28px;
}
header h1 { font-size: 44px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
header .meta { color: var(--ink-2); font-size: 22px; font-variant-numeric: tabular-nums; }
header .meta span { margin-left: 22px; }
header .meta button {
  margin-left: 22px; font: inherit; font-size: 18px; color: var(--ink-2);
  background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 4px 12px; cursor: pointer;
}
.badge { display: inline-block; vertical-align: middle; font-size: 15px; font-weight: 600; color: var(--ink-2);
  border: 1px solid var(--ring); border-radius: 6px; padding: 2px 8px; margin-left: 12px; letter-spacing: 0.02em; }
.grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 26px;
}
@media (max-width: 1500px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 950px)  { .grid { grid-template-columns: 1fr; } body { padding: 18px; } }
.card {
  background: var(--surface);
  border: 1px solid var(--ring);
  border-radius: 14px;
  padding: 26px 28px 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.card.missing { justify-content: center; align-items: center; min-height: 420px; color: var(--muted); }
.card.missing h2 { color: var(--ink); }
.card h2 { font-size: 36px; margin: 0; font-weight: 700; line-height: 1.1; }
.card .task { font-size: 24px; color: var(--ink); margin: 6px 0 0; }
.card .hw { font-size: 20px; color: var(--ink-2); margin: 4px 0 0; }
.card .hw .when { color: var(--muted); margin-left: 16px; font-size: 17px; }
.checks { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 14px; }
.check {
  display: grid;
  grid-template-columns: 30px 1fr;
  column-gap: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--hairline);
}
.dot {
  width: 22px; height: 22px; border-radius: 50%;
  margin-top: 6px;
  box-shadow: 0 0 0 2px var(--surface);
}
.dot.pass { background: var(--good); }
.dot.warn { background: var(--warn); }
.dot.fail { background: var(--bad); }
.check .name { font-size: 19px; color: var(--muted); text-transform: lowercase; letter-spacing: 0.01em; }
.check .name b { color: var(--ink-2); font-weight: 600; margin-right: 10px; }
.check .picture { font-size: 27px; font-weight: 600; line-height: 1.25; margin: 2px 0 4px; }
.check .detail { font-size: 19px; color: var(--ink-2); line-height: 1.35; }
.check .req { font-size: 17px; color: var(--muted); margin-top: 4px; }
.scores {
  border-top: 1px solid var(--hairline);
  padding-top: 16px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px 22px;
}
.tile .label { font-size: 18px; color: var(--muted); }
.tile .value { font-size: 38px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .value small { font-size: 18px; font-weight: 500; color: var(--ink-2); margin-left: 6px; }
.cheat { grid-column: 1 / -1; }
.bars { display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }
.bar-row { display: grid; grid-template-columns: 130px 1fr 90px; align-items: center; gap: 12px; font-size: 20px; }
.bar-row .who { color: var(--ink-2); }
.bar-row .val { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; font-size: 24px; }
.track { height: 24px; background: var(--track); border-radius: 0 4px 4px 0; position: relative; }
.fill { height: 100%; border-radius: 0 4px 4px 0; min-width: 3px; }
.fill.bad { background: var(--bad); }
.fill.good { background: var(--good); }
.drop { grid-column: 1 / -1; font-size: 19px; color: var(--ink-2); }
.gap { grid-column: 1 / -1; }
.gap .pair { display: flex; gap: 26px; align-items: baseline; }
.gap .pair .value { font-size: 30px; }
.gap .pair .value.measured { color: var(--ink); }
.legend { display: flex; gap: 22px; font-size: 18px; color: var(--ink-2); margin-top: 4px; flex-wrap: wrap; }
.legend i { display: inline-block; width: 14px; height: 14px; border-radius: 50%; vertical-align: -2px; margin-right: 6px; }
footer { margin-top: 28px; color: var(--muted); font-size: 17px; }
"""

JS = """<script>
(function () {
  var root = document.documentElement, btn = document.getElementById("theme"), saved = null;
  try { saved = localStorage.getItem("theme"); } catch (err) {}
  var q = new URLSearchParams(location.search).get("theme");
  function apply(t) {
    if (t === "dark") { root.setAttribute("data-theme", "dark"); btn.textContent = "light"; }
    else { root.removeAttribute("data-theme"); btn.textContent = "dark"; }
    try { localStorage.setItem("theme", t); } catch (err) {}
  }
  apply(q || saved || "light");
  btn.addEventListener("click", function () { apply(root.getAttribute("data-theme") === "dark" ? "light" : "dark"); });
})();
</script>
"""


def fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, (int, float)):
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"
    return str(x)


def e(x):
    return html.escape(str(x), quote=True)


def when(iso):
    """render an ISO 8601 stamp as YYYY-MM-DD HH:MM zone, or pass the raw text through."""
    if not iso:
        return "n/a"
    try:
        t = dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)
    if t.tzinfo is None:
        return t.strftime("%Y-%m-%d %H:%M")
    return t.strftime("%Y-%m-%d %H:%M %Z").replace("UTC+00:00", "UTC")


def load_reports(include_samples):
    found = {}
    skipped = []
    for p in sorted(REPORTS.glob("*.json")):
        if p.name.startswith("_") and not include_samples:
            skipped.append(p.name)
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # a broken report should not sink the page
            print(f"skip {p.name}: {exc}", file=sys.stderr)
            continue
        gym = data.get("gym") or p.stem.lstrip("_").replace("sample-", "")
        data["_file"] = p.name
        data["_sample"] = p.name.startswith("_")
        found[gym] = data
    return found, skipped


def ordered_checks(checks):
    by_id = {c.get("id"): c for c in checks if isinstance(c, dict)}
    out = [by_id[i] for i in CHECK_ORDER if i in by_id]
    out += [c for c in checks if isinstance(c, dict) and c.get("id") not in CHECK_ORDER]
    return out


def render_check(idx, c):
    status = c.get("status", "warn")
    if status not in STATUS_WORD:
        status = "warn"
    nums = c.get("numbers") or {}
    tip = ", ".join(f"{k} {fmt(v)}" for k, v in nums.items())
    req = ""
    if c.get("requirement") and os.environ.get("GYM_AUDIT_REQUIREMENTS") == "1":
        req = f'\n          <div class="req">requirement. {e(c["requirement"])}</div>'
    return f"""
      <li class="check" title="{e(tip)}">
        <span class="dot {status}" aria-hidden="true"></span>
        <div>
          <div class="name"><b>{e(STATUS_WORD[status])}</b>{idx}. {e(c.get('name', c.get('id', '')))}</div>
          <div class="picture">{e(c.get('picture', ''))}</div>
          <div class="detail">{e(c.get('detail', ''))}</div>{req}
        </div>
      </li>"""


def bar(who, value, cls):
    v = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    pct = f"{v * 100:.1f}%"
    return f"""
        <div class="bar-row" title="{e(who)} {e(fmt(value))}">
          <span class="who">{e(who)}</span>
          <div class="track"><div class="fill {cls}" style="width:{pct}"></div></div>
          <span class="val">{e(fmt(value))}</span>
        </div>"""


def render_scores(s):
    golden = s.get("golden")
    cm = s.get("correct_model")
    un = s.get("zero_work_agent_unhardened")
    hd = s.get("zero_work_agent_hardened")
    gp = s.get("gap_projected")
    gm = s.get("gap_measured")

    drop = ""
    if un is not None and hd is not None:
        drop = f"cheat score drops from {fmt(un)} to {fmt(hd)} once the grader is hardened"

    gap = ""
    if gp is not None or gm is not None:
        halves = ""
        if gp is not None:
            halves += f'<span class="value">{e(fmt(gp, 2))}<small>x projected</small></span>'
        if gm is not None:
            halves += f'<span class="value measured">{e(fmt(gm, 2))}<small>x measured</small></span>'
        label = "headroom gap, projected vs measured" if gp is not None and gm is not None else "headroom gap"
        gap = f"""
      <div class="tile gap">
        <div class="label">{label}</div>
        <div class="pair">{halves}</div>
      </div>"""

    return f"""
    <div class="scores">
      <div class="tile"><div class="label">golden</div><div class="value">{e(fmt(golden))}</div></div>
      <div class="tile"><div class="label">correct model</div><div class="value">{e(fmt(cm))}</div></div>
      <div class="tile cheat">
        <div class="label">zero work agent, plain grader then hardened grader</div>
        <div class="bars">{bar("unhardened", un, "bad")}{bar("hardened", hd, "good")}
        </div>
      </div>
      <div class="drop">{e(drop)}</div>{gap}
    </div>"""


def render_card(gym, r):
    if r is None:
        return f"""
  <section class="card missing">
    <h2>{e(GYM_LABEL.get(gym, gym))}</h2>
    <p>no report yet</p>
  </section>"""
    checks = ordered_checks(r.get("checks") or [])
    items = "".join(render_check(i + 1, c) for i, c in enumerate(checks))
    tag = '<span class="badge">sample data</span>' if r.get("_sample") else ""
    return f"""
  <section class="card">
    <div>
      <h2>{e(GYM_LABEL.get(gym, gym))}{tag}</h2>
      <p class="task">{e(r.get('task', ''))}</p>
      <p class="hw">{e(r.get('hardware', ''))}<span class="when">reported {e(when(r.get('timestamp')))}</span></p>
    </div>
    <ul class="checks">{items}
    </ul>
    {render_scores(r.get('scores') or {})}
  </section>"""


def build(include_samples):
    reports, skipped = load_reports(include_samples)
    gyms = GYM_ORDER + sorted(g for g in reports if g not in GYM_ORDER)
    cards = "".join(render_card(g, reports.get(g)) for g in gyms)
    now = dt.datetime.now().astimezone()
    stamps = [r.get("timestamp") for r in reports.values() if r.get("timestamp")]
    latest = max(stamps) if stamps else "no reports"
    n_have = sum(1 for g in GYM_ORDER if g in reports)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gym audit report card</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>one report card, seven gyms</h1>
  <div class="meta">{n_have} of {len(GYM_ORDER)} reports<span>latest report {e(when(latest) if stamps else latest)}</span><span>built {e(now.strftime('%Y-%m-%d %H:%M %Z'))}</span><button type="button" id="theme">dark</button></div>
</header>
<div class="legend"><span><i style="background:var(--good)"></i>pass</span><span><i style="background:var(--warn)"></i>warn</span><span><i style="background:var(--bad)"></i>fail</span><span>bars are the zero work agent's score, red on the plain grader, green on the hardened grader</span></div>
<main class="grid">{cards}
</main>
<footer>built by dashboard/build.py from reports/*.json. {len(reports)} report file(s) read{', ' + str(len(skipped)) + ' sample file(s) skipped' if skipped else ''}.</footer>
{JS}
</body>
</html>
"""
    OUT.write_text(page)
    return reports, skipped


def serve(port):
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUT.parent))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        print(f"serving {OUT.parent} at http://localhost:{port}/  (ctrl c to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-samples", action="store_true", help="include reports whose file name starts with _")
    ap.add_argument("--serve", action="store_true", help="build then serve dashboard/ on localhost")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    reports, skipped = build(args.include_samples)
    print(f"wrote {OUT}  ({len(reports)} report(s): {', '.join(sorted(reports)) or 'none'})")
    if skipped:
        print(f"skipped {len(skipped)} sample file(s): {', '.join(skipped)}  (pass --include-samples to include)")
    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
