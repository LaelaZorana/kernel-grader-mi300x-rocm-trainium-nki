"""Shared helpers for every gym. Writes the report json and prints the card."""
import json
import os
import datetime

# optional per check "requirement" tags are written only when this env var is 1.
# the public reports and the dashboard carry no requirement text by default.
EMIT_REQUIREMENTS = os.environ.get("GYM_AUDIT_REQUIREMENTS") == "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
REQUIRED = ["correctness", "cheat-unhardened", "cheat-hardened", "headroom", "ground-truth"]
CHECK_FIELDS = {"id", "name", "status", "picture", "detail", "numbers", "requirement"}


def check(cid, name, status, picture, detail, numbers, requirement=None):
    """requirement is an optional tag for the check. It is only written to the
    report when GYM_AUDIT_REQUIREMENTS=1 is set."""
    assert status in ("pass", "fail", "warn")
    c = {"id": cid, "name": name, "status": status, "picture": picture,
         "detail": detail, "numbers": numbers}
    if requirement is not None:
        assert isinstance(requirement, str) and requirement.strip(), "requirement must be a non empty string"
        if EMIT_REQUIREMENTS:
            c["requirement"] = requirement
    return c


def attach_requirements(checks, mapping):
    """Tag each check with the requirement it answers. Every check id must have
    an entry, so a new check cannot slip in without one. The tag is only written
    when GYM_AUDIT_REQUIREMENTS=1 is set."""
    missing = [c["id"] for c in checks if c["id"] not in mapping]
    assert not missing, "no requirement tag for checks %s" % missing
    if EMIT_REQUIREMENTS:
        for c in checks:
            c["requirement"] = mapping[c["id"]]
    return checks


def finish(gym, task, scores, checks, hardware="cpu"):
    ids = [c["id"] for c in checks[:5]]
    assert ids == REQUIRED, "first five check ids must be %s, got %s" % (REQUIRED, ids)
    for c in checks:
        extra = set(c) - CHECK_FIELDS
        assert not extra, "check %s has fields outside the schema: %s" % (c["id"], sorted(extra))
    report = {"gym": gym, "task": task, "hardware": hardware,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scores": scores, "checks": checks}
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, gym + ".json")
    _guard_measured_report(path, hardware)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print_card(report, path)
    return report



def _simulated(hw):
    """True when a hardware string describes a stand in rather than a card."""
    h = (hw or "").strip().lower()
    return (not h) or h == "cpu" or "dry" in h or "simulat" in h


def _guard_measured_report(path, new_hardware):
    """Refuse to replace a measured report with a simulated one.

    A simulated run computes every histogram for real and catches every cheat,
    so it is worth running anywhere. Its timings are modelled. Writing one over
    a report produced on a card destroys the only copy of a number that cost
    card time, and the loss is silent because both files look alike.

    Set GYM_AUDIT_FORCE_OVERWRITE=1 to overwrite anyway.
    """
    if os.environ.get("GYM_AUDIT_FORCE_OVERWRITE") == "1":
        return
    if not _simulated(new_hardware) or not os.path.exists(path):
        return
    try:
        with open(path) as f:
            prev = json.load(f).get("hardware", "")
    except Exception:
        return
    if _simulated(prev):
        return
    raise SystemExit(
        "refusing to overwrite a measured report.\n"
        "  file      %s\n"
        "  measured  %s\n"
        "  this run  %s\n"
        "  A simulated run models its timings, so writing here would lose the\n"
        "  measured number. Set GYM_AUDIT_FORCE_OVERWRITE=1 to overwrite."
        % (path, prev, new_hardware or "cpu"))


def print_card(r, path):
    s = r["scores"]
    print("=" * 72)
    print("GYM %s  (%s)" % (r["gym"], r["hardware"]))
    print("task: %s" % r["task"])
    print("-" * 72)
    print("scores  golden %.2f | correct naive %.2f | cheat plain %.2f | cheat hardened %.2f"
          % (s["golden"], s["correct_model"], s["zero_work_agent_unhardened"],
             s["zero_work_agent_hardened"]))
    print("        gap projected %s | gap measured %s" % (s["gap_projected"], s["gap_measured"]))
    print("-" * 72)
    for c in r["checks"]:
        print("[%-4s] %-18s %s" % (c["status"].upper(), c["id"], c["name"]))
        print("       picture: %s" % c["picture"])
        print("       detail : %s" % c["detail"])
        print("       numbers: %s" % json.dumps(c["numbers"]))
        if c.get("requirement"):
            print("       requirement: %s" % c["requirement"])
    print("-" * 72)
    print("report written to %s" % path)
    print("=" * 72)
