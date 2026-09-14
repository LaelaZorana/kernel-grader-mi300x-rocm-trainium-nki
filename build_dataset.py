"""Turn the committed reports into a dataset the viewer can render.

Two files come out. gyms.jsonl carries one row per gym, which is the summary a
reader sees first. checks.jsonl carries one row per check across every gym,
which is the detail behind each score.

Both are flat, so the dataset viewer renders them without any loading script.

Every row carries a source and a licence field. A card can be deleted and a
readme can be rewritten, but a row that is copied into someone else's file
carries its origin with it, and so does any subset of the rows.
"""
import json
import os
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
OUT = os.path.join(ROOT, "dataset")

# Stamped into every row of every file this script writes.
SOURCE = "LaelaZorana/kernel-grader-mi300x-rocm-trainium-nki"
CREATOR = "Laela Zorana"
LICENCE = "CC-BY-4.0"


def stamped(row):
    """Return the row with its origin on it, origin first so it reads first."""
    out = {"source": SOURCE, "created_by": CREATOR, "license": LICENCE}
    out.update(row)
    return out


def score(scores, *keys):
    for k in keys:
        if scores.get(k) is not None:
            return scores[k]
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    gyms, checks = [], []

    for path in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        sc = rep.get("scores", {})
        name = rep.get("gym") or os.path.splitext(os.path.basename(path))[0]

        gyms.append({
            "gym": name,
            "task": rep.get("task"),
            "hardware": rep.get("hardware"),
            "golden": score(sc, "golden"),
            "correct_model": score(sc, "correct_model"),
            "cheat_plain_grader": score(sc, "zero_work_agent_unhardened", "cheat_unhardened"),
            "cheat_hardened_grader": score(sc, "zero_work_agent_hardened", "cheat_hardened"),
            "n_checks": len(rep.get("checks", [])),
            "measured_at": rep.get("timestamp"),
        })

        for c in rep.get("checks", []):
            checks.append({
                "gym": name,
                "hardware": rep.get("hardware"),
                "check_id": c.get("id"),
                "check_name": c.get("name"),
                "status": c.get("status"),
                "picture": c.get("picture"),
                "detail": c.get("detail"),
                "numbers": json.dumps(c.get("numbers")) if c.get("numbers") else None,
            })

    for fname, rows in (("gyms.jsonl", gyms), ("checks.jsonl", checks)):
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(stamped(r), ensure_ascii=False) + "\n")
        print("%-14s %4d rows" % (fname, len(rows)))

    return gyms, checks


if __name__ == "__main__":
    main()
