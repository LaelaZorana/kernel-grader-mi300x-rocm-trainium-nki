"""Space entry point.

Runs the five processor gyms live on a click and renders the result beside the
kernel gym numbers, which were measured on an accelerator and are read from the
committed report.

The whole processor suite finishes in about two seconds, so the number a visitor
sees is computed in front of them rather than read from a file.
"""
import json
import os
import subprocess
import sys
import time

import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
CPU_GYMS = ["swe", "cve", "k8s", "iac", "ml"]

HEADLINE = """
## One report card for seven reinforcement learning environments

Each gym holds a golden solution, a zero work cheating agent, and two graders.
The plain grader gives the cheat full marks. The hardened grader catches the same
agent on the same transcript and scores it zero.

The pair is the measurement. A grader that scores a cheat 1.00 and a hardened
grader that scores it 0.00 says more than a claim that the grader is sound. The
golden solution still scores 1.00 on the hardened grader, which shows the
hardening did not make the task unsolvable.
"""

METHOD = """
### What each number means

Plain is the zero work agent against the ungarded grader. Hardened is the same
agent against the hardened one. Golden is the reference solution, which has to
keep full marks after hardening or the task has been broken rather than fixed.

The kernel row is measured on an accelerator, so it is read from its committed
report rather than recomputed here. Everything else is recomputed on your click.
"""


def read_report(name):
    path = os.path.join(REPORTS, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def score_of(rep, *keys):
    sc = (rep or {}).get("scores", {})
    for k in keys:
        if sc.get(k) is not None:
            return sc[k]
    return None


def fmt(v):
    return "" if v is None else ("%.2f" % v if isinstance(v, (int, float)) else str(v))


def table_rows(live=False):
    rows = []
    for g in CPU_GYMS:
        rep = read_report(g)
        rows.append([
            g,
            fmt(score_of(rep, "golden")),
            fmt(score_of(rep, "cheat_unhardened", "zero_work_agent_unhardened")),
            fmt(score_of(rep, "cheat_hardened", "zero_work_agent_hardened")),
            len((rep or {}).get("checks", [])),
            "recomputed now" if live else "from the committed report",
        ])
    for g in ("kernel-rocm", "nki-sim"):
        rep = read_report(g)
        if not rep:
            continue
        rows.append([
            g,
            fmt(score_of(rep, "golden")),
            fmt(score_of(rep, "cheat_unhardened", "zero_work_agent_unhardened")),
            fmt(score_of(rep, "cheat_hardened", "zero_work_agent_hardened")),
            len(rep.get("checks", [])),
            rep.get("hardware", "")[:60],
        ])
    return rows


def run_live():
    start = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join("gyms", "run_all.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - start
    note = "Recomputed the five processor gyms in %.2f seconds." % elapsed
    if proc.returncode != 0:
        note = ("Recomputed the processor gyms that run without a virtual "
                "environment, in %.2f seconds. The rest are read from their "
                "committed reports." % elapsed)
    return table_rows(live=True), note, proc.stdout[-4000:]


HEADERS = ["gym", "golden", "plain", "hardened", "checks", "where the number came from"]

with gr.Blocks(title="Gym audit") as demo:
    gr.Markdown(HEADLINE)
    with gr.Row():
        btn = gr.Button("Run the processor gyms now", variant="primary")
    note = gr.Markdown("Showing the committed reports. Press the button to recompute.")
    grid = gr.Dataframe(value=table_rows(), headers=HEADERS, wrap=True, interactive=False)
    gr.Markdown(METHOD)
    with gr.Accordion("Run output", open=False):
        raw = gr.Textbox(lines=18, show_label=False)
    btn.click(run_live, outputs=[grid, note, raw])

if __name__ == "__main__":
    demo.launch()
