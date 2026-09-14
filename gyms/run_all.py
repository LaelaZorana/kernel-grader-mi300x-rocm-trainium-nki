#!/usr/bin/env python3
"""Run all five cpu gyms in order and print a one line summary table.
Run: python3 gyms/run_all.py"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GYMS = ["swe", "cve", "k8s", "iac", "ml"]


def main():
    for g in GYMS:
        print("\n### running gym %s ###" % g)
        r = subprocess.run([sys.executable, os.path.join(HERE, g, "audit.py")])
        if r.returncode != 0:
            sys.exit("gym %s failed with code %d" % (g, r.returncode))

    print("\n" + "=" * 72)
    print("SUMMARY  all reports in reports/")
    print("=" * 72)
    print("%-6s %-7s %-7s %-8s %-9s %s" %
          ("gym", "golden", "naive", "cheat+", "cheat-", "checks"))
    for g in GYMS:
        rep = json.load(open(os.path.join(ROOT, "reports", g + ".json")))
        s = rep["scores"]
        status = {}
        for c in rep["checks"]:
            status[c["status"]] = status.get(c["status"], 0) + 1
        cktxt = " ".join("%d%s" % (v, k[0]) for k, v in sorted(status.items()))
        print("%-6s %-7.2f %-7.2f %-8.2f %-9.2f %s" %
              (g, s["golden"], s["correct_model"], s["zero_work_agent_unhardened"],
               s["zero_work_agent_hardened"], cktxt))
    print("=" * 72)
    print("cheat+ is the zero work agent against the plain grader, cheat- against the hardened one.")
    print("every hardened cheat score is 0, every plain cheat score is a full pass.")


if __name__ == "__main__":
    main()
