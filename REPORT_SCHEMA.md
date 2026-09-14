# gym audit report schema, shared by every gym in this demo

Every gym writes `reports/<gym>.json` with exactly this shape.

```json
{
  "gym": "kernel-rocm | swe | cve | k8s | iac | ml",
  "task": "one line, what the agent is asked to do",
  "hardware": "AMD Instinct MI300X VF, ROCm 7.2.4  |  cpu",
  "timestamp": "ISO 8601",
  "scores": {
    "golden": 1.0,
    "correct_model": 0.0 to 1.0,
    "zero_work_agent_unhardened": 0.0 to 1.0,
    "zero_work_agent_hardened": 0.0 to 1.0,
    "gap_projected": null or float,
    "gap_measured": null or float
  },
  "checks": [
    {
      "id": "short-slug",
      "name": "plain words, under 8 words",
      "status": "pass | fail | warn",
      "picture": "one sentence a non technical person can picture",
      "detail": "one or two sentences with the mechanism",
      "numbers": {"any": "measured values used"},
      requirement: "optional. a short tag naming the skill this check exercises. written only when GYM_AUDIT_REQUIREMENTS=1"
    }
  ]
}
```

Required check ids for every gym, in this order.
1. correctness        does the golden pass its own tests
2. cheat-unhardened   a zero work agent against the plain grader, must show the score it got
3. cheat-hardened     the same agent against the hardened grader, must be 0
4. headroom           can the task still rank models for the kernel gym that is the measured gap, and elsewhere it asks whether a naive but correct solution scores below golden
5. ground-truth       the reference was checked against the upstream source at the pinned version, name the source
Extra checks may follow.

The requirement field is optional. Each gym keeps a mapping from check id to a short
tag naming the skill the check exercises, and _common.py asserts every check has one.
The tag is only written into the report when the env var GYM_AUDIT_REQUIREMENTS=1 is set,
so the checked in reports carry no requirement field. _common.py rejects any other extra
field on a check.
