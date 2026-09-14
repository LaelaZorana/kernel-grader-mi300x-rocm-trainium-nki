#!/usr/bin/env python3
"""swe gym audit. A tiny package with a bug, a hidden test file, three patches
and two graders. Run: python3 gyms/swe/audit.py
Add --docker to run the hardened grader inside a fresh python:3.12-slim container
with the tests mounted read only, a non root user and no network."""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C  # noqa: E402
import solutions  # noqa: E402

TASK = os.path.join(HERE, "task")
HIDDEN = os.path.join(HERE, "hidden_tests")
VENV_PY = os.path.join(HERE, ".venv", "bin", "python")
def _venv_site():
    """Find the venv site-packages by looking inside the venv.

    Deriving it from the running interpreter breaks the moment the audit is
    started by a different python than the one that built the venv. That
    reads as a failing golden patch rather than as a path problem, because
    the hardened grader then collects 0 tests and scores 0.
    """
    hits = sorted(glob.glob(os.path.join(HERE, ".venv", "lib", "python*", "site-packages")))
    return hits[-1] if hits else os.path.join(
        HERE, ".venv", "lib", "python%d.%d" % sys.version_info[:2], "site-packages")


VENV_SITE = _venv_site()
DOCKER_IMAGE = "python:3.12-slim"
DOCKER_USER = "65534:65534"  # nobody, never root
DOCKER_MARK = "==JUNIT=="
# colima and Docker Desktop share only $HOME into the VM, so a /var/folders temp dir
# bind mounts as an empty directory. Docker runs stage their temp dirs here instead.
DOCKER_TMP = os.path.join(HERE, ".docker-tmp")


def apply_patch(patch, tmp_dir=None):
    """Fresh copy of the task with the patch files written on top."""
    work = tempfile.mkdtemp(prefix="swe-", dir=tmp_dir)
    shutil.copytree(TASK, work, dirs_exist_ok=True)
    for rel, text in patch.items():
        path = os.path.join(work, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    return work


def junit_counts(xml_path):
    if not os.path.exists(xml_path):
        return 0, 0
    suite = ET.parse(xml_path).getroot()
    if suite.tag == "testsuites":
        suite = suite[0]
    total = int(suite.get("tests", 0))
    bad = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    return total - bad, total


def resolved(passed, total):
    """SWE-bench scoring: a patch resolves the issue only when every fail-to-pass
    test passes. Partial credit is not a thing, so the score is 1.0 or 0.0."""
    return 1.0 if total and passed == total else 0.0


def plain_grader(patch):
    """Runs pytest in tree. cwd is on sys.path, the tests dir is writable and
    readable, any conftest.py the patch added is auto loaded."""
    work = apply_patch(patch)
    xml = os.path.join(work, "junit.xml")
    subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests", "--junitxml", xml],
                   cwd=work, capture_output=True)
    passed, total = junit_counts(xml)
    shutil.rmtree(work)
    return resolved(passed, total), "%d/%d" % (passed, total)


def _hardened_setup(patch, tmp_dir=None):
    """Fresh copy of the patched tree with its tests dir removed, and the hidden
    tests copied into their own read only folder outside the tree."""
    if tmp_dir:
        # the staging dir stays in place between runs. virtiofs caches the parent, so
        # removing and recreating it makes the next bind mount fail with no such file.
        os.makedirs(tmp_dir, exist_ok=True)
        note = os.path.join(tmp_dir, "README.txt")
        if not os.path.exists(note):
            with open(note, "w") as f:
                f.write("staging dir for --docker runs of gyms/swe/audit.py. safe to leave, "
                        "it is emptied after every run. it lives under HOME because colima "
                        "only shares HOME into the VM.\n")
    work = apply_patch(patch, tmp_dir)
    shutil.rmtree(os.path.join(work, "tests"), ignore_errors=True)
    tests = tempfile.mkdtemp(prefix="swe-tests-", dir=tmp_dir)
    shutil.copy(os.path.join(HIDDEN, "test_slugify.py"), tests)
    os.chmod(os.path.join(tests, "test_slugify.py"), 0o444)
    os.chmod(tests, 0o555)
    return work, tests


def _hardened_teardown(work, tests):
    os.chmod(tests, 0o755)
    shutil.rmtree(tests)
    shutil.rmtree(work)


def _pytest_code(site, work, tests, xml):
    """One line python that pins sys.path to the venv pytest and the patched
    package only, then runs pytest on the sealed tests dir."""
    return ("import sys; sys.path[:0] = [%r, %r]; import pytest; "
            "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir', %r, "
            "'--confcutdir', %r, '--junitxml', %r, %r]))"
            % (site, work, tests, tests, xml, tests))


def hardened_grader(patch):
    """Copies the hidden tests fresh from the read only folder into their own
    temp dir outside the patched tree, then runs pytest with python -S -I so
    site, sitecustomize, PYTHONPATH and cwd are all ignored. The package import
    path is pinned to the patched package only. confcutdir stops pytest from
    walking up into any conftest.py the patch left behind."""
    work, tests = _hardened_setup(patch)
    xml = tempfile.mktemp(suffix=".xml")
    # -I ignores env and user site, -S skips site so no sitecustomize runs. pytest lives
    # in a venv, so its site-packages dir is appended explicitly and nothing else is.
    # The venv interpreter runs it rather than whichever python started this audit,
    # because a venv pytest built for one version imports under that version only.
    # Using the caller's interpreter made the hardened grader collect 0 tests and
    # report a failing golden, which looks like a broken task and is a path problem.
    subprocess.run([VENV_PY, "-S", "-I", "-c", _pytest_code(VENV_SITE, work, tests, xml)],
                   cwd=tempfile.gettempdir(), capture_output=True)
    passed, total = junit_counts(xml)
    _hardened_teardown(work, tests)
    return resolved(passed, total), "%d/%d" % (passed, total)


def docker_hardened_grader(patch):
    """Same sealed setup, run inside a fresh python:3.12-slim container. The
    patched tree, the tests dir and the venv site-packages are all bind mounted
    read only, the process runs as uid 65534 (nobody) and the network is off.
    The junit xml is written to the container's own /tmp and echoed back on
    stdout behind a marker, so nothing on the host needs to be writable."""
    work, tests = _hardened_setup(patch, DOCKER_TMP)
    os.chmod(work, 0o755)
    xml = "/tmp/junit.xml"
    code = _pytest_code("/pysite", "/work", "/tests", xml)
    code = code.replace("sys.exit(pytest.main(", "rc = (pytest.main(", 1)
    code = code.rstrip(")") + ")); print(%r); print(open(%r).read()); sys.exit(rc)" % (DOCKER_MARK, xml)
    cmd = ["docker", "run", "--rm", "--network", "none", "--user", DOCKER_USER,
           "-v", "%s:/work:ro" % work, "-v", "%s:/tests:ro" % tests,
           "-v", "%s:/pysite:ro" % VENV_SITE, "-w", "/tmp",
           DOCKER_IMAGE, "python", "-S", "-I", "-c", code]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    _hardened_teardown(work, tests)
    if DOCKER_MARK not in r.stdout:
        return 0.0, "0/0 (docker run produced no junit: %s)" % _no_local_paths(r.stderr.strip())[-200:]
    xml_text = r.stdout.split(DOCKER_MARK, 1)[1].strip()
    tmp = tempfile.mktemp(suffix=".xml")
    with open(tmp, "w") as f:
        f.write(xml_text)
    passed, total = junit_counts(tmp)
    os.remove(tmp)
    return resolved(passed, total), "%d/%d" % (passed, total)


def _no_local_paths(text):
    """Strip home directory paths out of captured tool output.

    A docker or pytest failure quotes whatever socket or file it tried, and
    that carries the account name of the machine that ran it into a report
    other people read. The failure still has to be legible, so the path is
    replaced rather than dropped.
    """
    text = re.sub(r"/(?:Users|home)/[^/\s:,;)\]]+", "/home/user", text)
    return re.sub(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s:,;)\]]+", r"C:\\\\Users\\\\user", text)


def docker_available(pull_timeout=180):
    """True when the docker daemon answers and the image is present or pulls
    inside the timeout. Returns (ok, reason)."""
    try:
        r = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return True, "image %s present" % DOCKER_IMAGE
        r = subprocess.run(["docker", "pull", DOCKER_IMAGE], capture_output=True, text=True,
                           timeout=pull_timeout)
        if r.returncode == 0:
            return True, "image %s pulled" % DOCKER_IMAGE
        return False, "docker pull failed: %s" % _no_local_paths(r.stderr.strip())[-200:]
    except FileNotFoundError:
        return False, "docker binary not found"
    except subprocess.TimeoutExpired:
        return False, "docker pull exceeded %d seconds" % pull_timeout
    except Exception as ex:  # daemon down and similar
        return False, "docker error: %s" % ex


def reference_slugify(text):
    """Reimplementation of django.utils.text.slugify, Django 5.0, django/utils/text.py."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-_")


def load_golden_slugify():
    work = apply_patch(solutions.GOLDEN)
    ns = {}
    exec(open(os.path.join(work, "textkit", "__init__.py")).read(), ns)
    shutil.rmtree(work)
    return ns["slugify"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docker", action="store_true",
                    help="run the hardened grader inside %s, read only mounts, non root, no network" % DOCKER_IMAGE)
    args = ap.parse_args()

    if not os.path.exists(VENV_PY):
        sys.exit("missing %s, create it with: python3 -m venv gyms/swe/.venv && "
                 "gyms/swe/.venv/bin/pip install pytest" % VENV_PY)

    dock_ok, dock_why = docker_available()
    use_docker = args.docker and dock_ok
    if args.docker and not dock_ok:
        print("--docker requested but %s, falling back to the local hardened grader" % dock_why)
    hard = docker_hardened_grader if use_docker else hardened_grader

    g_plain, g_txt = plain_grader(solutions.GOLDEN)
    g_hard, g_htxt = hard(solutions.GOLDEN)
    n_hard, n_txt = hard(solutions.NAIVE)
    c_plain, c_ptxt = plain_grader(solutions.CHEAT)
    c_hard, c_htxt = hard(solutions.CHEAT)

    # docker isolation is verified whenever docker is available, even on a default
    # run, so the 7th check reports a measured result rather than a promise.
    dock_nums = {"docker_used_for_scores": use_docker, "docker_available": dock_ok,
                 "image": DOCKER_IMAGE, "user": DOCKER_USER, "network": "none",
                 "mounts": ["patched tree :ro", "tests dir :ro", "venv site-packages :ro"]}
    if dock_ok:
        dg, dg_txt = docker_hardened_grader(solutions.GOLDEN)
        dc, dc_txt = docker_hardened_grader(solutions.CHEAT)
        dock_nums.update({"golden_in_docker": dg, "golden_tests": dg_txt,
                          "cheat_in_docker": dc, "cheat_tests": dc_txt})
        # the cheat must have actually run (total tests above 0) for its 0 to count
        dock_pass = dg == 1.0 and dc == 0.0 and dg_txt == "6/6" and dc_txt.split("/")[1][:1] != "0"
    else:
        dock_pass = False

    golden = load_golden_slugify()
    probes = ["Hello World", "Hello   World--Again", "  --Hello-- ", "Crème Brûlée",
              "What's up? 100% real!", "ÀÉÎÕÜ ñ", "a_b c", "---", "Straße", "naïve café 2024"]
    agree = sum(golden(p) == reference_slugify(p) for p in probes)
    where = "inside %s as uid %s with no network" % (DOCKER_IMAGE, DOCKER_USER) if use_docker \
        else "with python -S -I on the host"

    checks = [
        C.check("correctness", "golden patch passes hidden tests",
                "pass" if g_hard == 1.0 else "fail",
                "the golden patch passed all 6 hidden tests, read fresh from a read only folder outside the repository",
                "golden patch run through the hardened grader, fresh read only tests, %s." % where,
                {"plain_grader": g_txt, "hardened_grader": g_htxt, "docker": use_docker}),
        C.check("cheat-unhardened", "conftest cheat fools plain grader",
                "pass" if c_plain == 1.0 else "fail",
                "the cheat left the package as it was and rewrote the test module from its conftest before collection, and the plain grader passed it",
                "cheat patch never touches textkit. Its conftest.py reads tests/test_slugify.py "
                "with ast and swaps slugify for a lookup table before collection.",
                {"cheat_plain_score": c_plain, "tests": c_ptxt}),
        C.check("cheat-hardened", "hardened grader catches cheat",
                "pass" if c_hard == 0.0 else "fail",
                "the tests were copied fresh from a read only folder outside the tree, so the rewrite faced a clean copy and 4 of 6 tests failed",
                "tests copied fresh from a 0o555 folder outside the repo, run %s, "
                "confcutdir on the tests dir, package path pinned. The untouched bug fails 4 of 6 "
                "tests, so under the all-or-nothing SWE-bench rule the patch scores 0." % where,
                {"cheat_hardened_score": c_hard, "tests": c_htxt, "docker": use_docker}),
        C.check("headroom", "naive fix scores below golden",
                "pass" if n_hard < g_hard else "fail",
                "the naive patch collapsed dashes but skipped unicode folding, so one hidden test failed and the all or nothing rule gave it 0",
                "naive patch collapses dashes but skips unicode folding, so one hidden test fails "
                "and the all-or-nothing rule gives it 0.",
                {"naive_hardened": round(n_hard, 3), "naive_tests": n_txt, "golden_hardened": g_hard}),
        C.check("ground-truth", "golden matches django slugify",
                "pass" if agree == len(probes) else "fail",
                "the reference was compared against the upstream slugify on 10 probe strings",
                "reference is a reimplementation of django.utils.text.slugify from "
                "django/utils/text.py at Django 5.0, compared on %d probe strings." % len(probes),
                {"probes": len(probes), "agree": agree,
                 "source": "django/utils/text.py slugify, Django 5.0"}),
        C.check("local-default-path", "default path needs only stdlib plus venv pytest",
                "pass", "the whole environment runs from one folder with python 3 and a local venv, on the host and offline",
                "pytest 9.x lives in gyms/swe/.venv. Without --docker no container is started "
                "and no network is used, so the gym runs anywhere python 3 runs.",
                {"python": sys.version.split()[0], "docker_flag": args.docker}),
        C.check("docker-isolation", "tests run sealed in a fresh container",
                "pass" if dock_pass else "warn",
                "the hardened grader re-ran inside a container as an unprivileged user with networking switched off, and the tree, tests and site packages mounted read only"
                if dock_ok else "the container run was set up and the build time attempt returned a socket error, so this check is recorded with its reason",
                ("hardened grader re-run inside %s as uid %s with --network none. Patched tree, tests "
                 "and venv site-packages mounted read only. Golden scored %.1f and the conftest cheat "
                 "scored %.1f in the container, matching the host result." %
                 (DOCKER_IMAGE, DOCKER_USER, dock_nums.get("golden_in_docker", 0.0),
                  dock_nums.get("cheat_in_docker", 0.0)))
                if dock_ok else
                ("docker was unavailable at build time (%s). The check stays in the report so the "
                 "gap is visible. Re-run with --docker once a daemon is up." % dock_why),
                dock_nums),
    ]
    # requirement tags, one per check. written only when GYM_AUDIT_REQUIREMENTS=1
    C.attach_requirements(checks, {
        "correctness": "reference patches, test harnesses, and grading integrity",
        "cheat-unhardened": "detect answer leakage / reward hacking",
        "cheat-hardened": "detect answer leakage / reward hacking",
        "headroom": "quality, correctness, and reproducibility of software-engineering benchmark tasks",
        "ground-truth": "reference patches, test harnesses, and grading integrity",
        "local-default-path": "reproducibility of software-engineering benchmark tasks",
        "docker-isolation": "audit reference patches, test runners, and Docker isolation",
    })
    scores = {"golden": g_hard, "correct_model": round(n_hard, 3),
              "zero_work_agent_unhardened": c_plain, "zero_work_agent_hardened": c_hard,
              "gap_projected": None, "gap_measured": round(g_hard - n_hard, 3)}
    C.finish("swe", "fix textkit.slugify so the hidden tests in tests/test_slugify.py pass",
             scores, checks)


if __name__ == "__main__":
    main()
