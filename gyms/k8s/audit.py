#!/usr/bin/env python3
"""k8s gym audit. A CronJob manifest task with a stale ground truth. The plain
grader encodes a pre 1.21 rule and marks a correct manifest wrong. The corrected
grader is pinned to v1.28. A grep only grader is fooled by an invalid schema.
Run: python3 gyms/k8s/audit.py"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C  # noqa: E402

# ---- three candidate manifests ----

GOLDEN = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: report-roller
spec:
  schedule: "*/5 * * * *"
  startingDeadlineSeconds: 200
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 3
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: roller
              image: reports:1.4
"""

# naive but correct: valid schema, sane fields, but leaves concurrencyPolicy at
# the Allow default instead of choosing Forbid for a job that must not overlap.
NAIVE = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: report-roller
spec:
  schedule: "*/5 * * * *"
  startingDeadlineSeconds: 200
  concurrencyPolicy: Allow
  jobTemplate:
    spec:
      backoffLimit: 3
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: roller
              image: reports:1.4
"""

# cheat: every keyword the grep grader looks for is present as a string, but the
# schedule is not a real cron expression and concurrencyPolicy is not a real enum
# value. A grep passes, a schema check does not.
CHEAT = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: report-roller
spec:
  schedule: "every 5 minutes please"
  startingDeadlineSeconds: "soon"
  concurrencyPolicy: Yes
  jobTemplate:
    spec:
      template:
        spec:
          containers: []
"""

VALID_POLICIES = {"Allow", "Forbid", "Replace"}


def grep_grader(text):
    """Plain grader. Greps the raw text for the field names. No parsing."""
    need = ["startingDeadlineSeconds", "concurrencyPolicy", "schedule"]
    return 1.0 if all(k in text for k in need) else 0.0


def stale_correctness_grader(spec):
    """Plain correctness rule, encodes pre 1.21 folklore. It insists on
    startingDeadlineSeconds <= 100 because it believes more than 100 missed
    starts stalls the CronJob forever. On 1.28 this rule is wrong and it marks a
    correct manifest as failing."""
    sd = spec.get("startingDeadlineSeconds")
    if not isinstance(sd, int):
        return 0.0
    return 1.0 if sd <= 100 else 0.0


def v128_correctness_grader(spec):
    """Corrected rule, pinned to Kubernetes v1.28. The controller counts missed
    starts inside the startingDeadlineSeconds window only, warns
    TooManyMissedTimes past 100 and still runs the latest schedule, so a larger
    deadline is fine. Source named below, not fetched. Requires an int deadline
    and a concurrencyPolicy of Forbid for a non overlapping job."""
    sd = spec.get("startingDeadlineSeconds")
    cp = spec.get("concurrencyPolicy")
    if not isinstance(sd, int) or sd <= 0:
        return 0.0
    return 1.0 if cp == "Forbid" else 0.0


def schema_valid(spec):
    """Strict schema validator, no kubectl. Checks the CronJob shape the API
    server would reject on, field by field."""
    errs = []
    if spec.get("apiVersion") != "batch/v1":
        errs.append("apiVersion must be batch/v1")
    if spec.get("kind") != "CronJob":
        errs.append("kind must be CronJob")
    s = spec.get("spec", {})
    sched = s.get("schedule")
    if not isinstance(sched, str) or not is_cron(sched):
        errs.append("schedule is not a 5 field cron expression")
    cp = s.get("concurrencyPolicy")
    if cp not in VALID_POLICIES:
        errs.append("concurrencyPolicy %r not in %s" % (cp, sorted(VALID_POLICIES)))
    sd = s.get("startingDeadlineSeconds")
    if not isinstance(sd, int) or isinstance(sd, bool):
        errs.append("startingDeadlineSeconds must be an integer")
    jt = s.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec", {})
    conts = jt.get("containers")
    if not isinstance(conts, list) or len(conts) == 0:
        errs.append("jobTemplate needs at least one container")
    return (len(errs) == 0), errs


def is_cron(text):
    fields = text.split()
    if len(fields) != 5:
        return False
    for f in fields:
        for token in f.replace("*", "0").replace("/", ",").replace("-", ",").split(","):
            if token == "":
                continue
            if not token.isdigit():
                return False
    return True


def spec_of(manifest):
    return yaml.safe_load(manifest)["spec"]


def full_grader(manifest):
    """Hardened. Parse, validate the schema strictly, then apply the v1.28 rule.
    A grep pass with a broken schema scores 0."""
    doc = yaml.safe_load(manifest)
    ok, errs = schema_valid(doc)
    if not ok:
        return 0.0, errs
    return v128_correctness_grader(doc["spec"]), []


def main():
    g_grep = grep_grader(GOLDEN)
    g_full, g_errs = full_grader(GOLDEN)
    g_stale = stale_correctness_grader(spec_of(GOLDEN))
    n_full, _ = full_grader(NAIVE)
    c_grep = grep_grader(CHEAT)
    c_full, c_errs = full_grader(CHEAT)

    checks = [
        C.check("correctness", "golden passes the v1.28 grader",
                "pass" if g_full == 1.0 else "fail",
                "the corrected grader accepted the golden manifest and the stale grader rejected the same file",
                "golden sets a valid cron schedule, concurrencyPolicy Forbid and a 200 second "
                "deadline. The corrected grader accepts it. The stale grader would reject it.",
                {"golden_v128": g_full, "golden_stale_rule": g_stale, "schema_errors": g_errs}),
        C.check("cheat-unhardened", "keyword stuffing fools grep grader",
                "pass" if c_grep == 1.0 else "fail",
                "the cheat carried the three field names as strings, so the grep only grader passed a schedule written as prose",
                "cheat contains the strings startingDeadlineSeconds, concurrencyPolicy and schedule, "
                "so the grep only grader passes it even though the schedule is prose and the policy "
                "is the yaml boolean Yes.",
                {"cheat_grep_score": c_grep}),
        C.check("cheat-hardened", "schema validator catches the cheat",
                "pass" if c_full == 0.0 else "fail",
                "the hardened grader parsed the yaml and found a schedule that is prose, a policy set to the boolean Yes and an empty container list",
                "the hardened grader parses the yaml and validates each field. schedule is not a "
                "cron expression, concurrencyPolicy Yes is not in {Allow,Forbid,Replace} and there "
                "are no containers, so it scores 0.",
                {"cheat_hardened_score": c_full, "schema_errors": c_errs}),
        C.check("headroom", "naive correct manifest scores below golden",
                "pass" if n_full < g_full else "fail",
                "the naive manifest validates but leaves concurrencyPolicy at Allow, so overlapping runs are permitted and it scores below the golden",
                "naive is schema valid but leaves concurrencyPolicy at Allow, so overlapping runs "
                "are permitted and the v1.28 rule scores it below the golden.",
                {"naive_v128": n_full, "golden_v128": g_full}),
        C.check("ground-truth", "stale rule corrected to v1.28 behaviour",
                "pass" if (g_stale == 0.0 and g_full == 1.0) else "fail",
                "the pre 1.21 claim that 100 missed starts stalls the job forever is stale, and on v1.28 the controller warns and runs the latest",
                "the pre 1.21 claim that more than 100 missed starts stalls the CronJob forever is "
                "stale. On v1.28 the controller warns TooManyMissedTimes and runs the latest. Source "
                "kubernetes/kubernetes pkg/controller/cronjob/utils.go, not fetched, named only.",
                {"stale_rule_marks_golden": g_stale, "v128_rule_marks_golden": g_full,
                 "source": "kubernetes/kubernetes pkg/controller/cronjob/utils.go, v1.28"}),
        C.check("stale-vs-corrected", "wrong ground truth demonstrated",
                "warn" if g_stale != g_full else "pass",
                "the stale grader scored the golden 0.0 and the corrected grader scored the same file 1.0",
                "the stale grader scores the golden %.1f and the corrected grader scores it %.1f, "
                "which is the whole point of the wrong ground truth demo." % (g_stale, g_full),
                {"stale": g_stale, "corrected": g_full}),
    ]
    # requirement tags, one per check. written only when GYM_AUDIT_REQUIREMENTS=1
    C.attach_requirements(checks, {
        "correctness": "manifest correctness, and failure-mode troubleshooting",
        "cheat-unhardened": "Experience authoring and reviewing manifests / Helm charts",
        "cheat-hardened": "manifest correctness, and failure-mode troubleshooting",
        "headroom": "quality, correctness, and production-readiness of Kubernetes tasks",
        "ground-truth": "Deep understanding of cluster internals",
        "stale-vs-corrected": "failure-mode troubleshooting",
    })
    scores = {"golden": g_full, "correct_model": n_full,
              "zero_work_agent_unhardened": c_grep, "zero_work_agent_hardened": c_full,
              "gap_projected": None, "gap_measured": round(g_full - n_full, 3)}
    C.finish("k8s", "write a v1.28 CronJob manifest with correct startingDeadlineSeconds and concurrencyPolicy",
             scores, checks)


if __name__ == "__main__":
    main()
