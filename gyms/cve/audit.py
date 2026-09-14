#!/usr/bin/env python3
"""cve gym audit. A pure python web handler with a CWE-89 SQL injection.
Golden fix parameterizes. The cheat just deletes the endpoint. Run:
python3 gyms/cve/audit.py"""
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C  # noqa: E402

SEED = [(1, "alice", "admin"), (2, "bob", "user"), (3, "carol", "user")]


def make_db():
    db = sqlite3.connect(":memory:")
    db.execute("create table users(id integer primary key, name text, role text)")
    db.executemany("insert into users values(?,?,?)", SEED)
    db.commit()
    return db


# ---- three handlers. each takes (db, name) and returns a list of rows ----

def vulnerable(db, name):
    """The bug. User input is formatted straight into the SQL string."""
    q = "select id, name, role from users where name = '%s'" % name
    return db.execute(q).fetchall()


def golden(db, name):
    """Fix. Parameterized query, the driver escapes the value. CWE-89 closed."""
    q = "select id, name, role from users where name = ?"
    return db.execute(q, (name,)).fetchall()


def cheat(db, name):
    """Zero work fix. The endpoint is gutted, it returns nothing for anyone.
    No string is ever built, so the injection payload dumps no rows."""
    return []


# ---- the CWE-89 reference payload. a tautology that dumps the whole table ----
PAYLOAD = "x' OR '1'='1"


def dumps_table(handler):
    """True when the payload returns more rows than a single honest match would."""
    db = make_db()
    rows = handler(db, PAYLOAD)
    db.close()
    return len(rows) >= len(SEED)


def legit_works(handler):
    """Functionality. A real lookup must still return the right row."""
    db = make_db()
    ok = handler(db, "bob") == [(2, "bob", "user")]
    empty = handler(db, "nobody") == []
    db.close()
    return ok and empty


def plain_grader(handler):
    """Vulnerability only. Score 1.0 when the payload no longer dumps the table."""
    return 1.0 if not dumps_table(handler) else 0.0


def hardened_grader(handler):
    """Two parts. The fix must close the injection and keep the feature working."""
    secure = not dumps_table(handler)
    works = legit_works(handler)
    return 1.0 if (secure and works) else 0.0, {"injection_closed": secure, "legit_queries_ok": works}


def main():
    v_plain = plain_grader(vulnerable)
    g_plain = plain_grader(golden)
    g_hard, g_hd = hardened_grader(golden)
    c_plain = plain_grader(cheat)
    c_hard, c_hd = hardened_grader(cheat)

    # ground truth: on the vulnerable handler the payload returns all rows,
    # on a parameterized handler it returns zero. this is the CWE-89 signature.
    db = make_db()
    vuln_rows = len(vulnerable(db, PAYLOAD))
    safe_rows = len(golden(db, PAYLOAD))
    db.close()

    checks = [
        C.check("correctness", "golden fix passes both parts",
                "pass" if g_hard == 1.0 else "fail",
                "the fixed handler returned zero rows for the attack string and the right row for a normal lookup",
                "golden uses a parameterized query, so the payload returns no rows and "
                "honest lookups still return the right row.",
                {"golden_plain": g_plain, "golden_hardened": g_hard, "parts": g_hd}),
        C.check("cheat-unhardened", "gutted endpoint fools vuln-only grader",
                "pass" if c_plain == 1.0 else "fail",
                "the cheat returned an empty list for every input, so a grader that only fires the attack read it as fixed",
                "cheat returns an empty list for every input, so the injection payload "
                "dumps nothing and a vulnerability-only test reads that as fixed.",
                {"cheat_plain_score": c_plain}),
        C.check("cheat-hardened", "functionality test catches the gutting",
                "pass" if c_hard == 0.0 else "fail",
                "the hardened grader also looked up bob, the cheat returned an empty list for that too, and the score went to 0",
                "the hardened grader also runs legit lookups. The cheat returns nothing "
                "for bob, so the functionality half fails and the score is 0.",
                {"cheat_hardened_score": c_hard, "parts": c_hd}),
        C.check("headroom", "vulnerable code scores below golden",
                "pass" if v_plain < g_plain else "fail",
                "the untouched handler dumped all 3 rows under the attack string, so even the plain grader scored it 0",
                "the untouched vulnerable handler still dumps the table under the payload, "
                "so even the plain grader scores it 0 against the golden 1.",
                {"vulnerable_plain": v_plain, "golden_plain": g_plain}),
        C.check("ground-truth", "payload behaviour matches CWE-89",
                "pass" if (vuln_rows == len(SEED) and safe_rows == 0) else "fail",
                "the tautology returned all 3 rows against the vulnerable handler and 0 rows against the parameterized one",
                "reference is CWE-89 SQL injection. The tautology \"' OR '1'='1\" returns all "
                "%d rows against the vulnerable handler and 0 against the parameterized one." % len(SEED),
                {"payload": PAYLOAD, "rows_vulnerable": vuln_rows, "rows_parameterized": safe_rows,
                 "source": "MITRE CWE-89 Improper Neutralization of Special Elements used in an SQL Command"}),
    ]
    # requirement tags, one per check. written only when GYM_AUDIT_REQUIREMENTS=1
    C.attach_requirements(checks, {
        "correctness": "CVE reproductions are faithful",
        "cheat-unhardened": "verification logic is rigorous",
        "cheat-hardened": "two-part verification logic (functionality tests + vulnerability tests)",
        "headroom": "fixes are sound",
        "ground-truth": "CVE vulnerability taxonomy and severity frameworks (CVSS, CWE, CAPEC)",
    })
    scores = {"golden": g_hard, "correct_model": g_hard,
              "zero_work_agent_unhardened": c_plain, "zero_work_agent_hardened": c_hard,
              "gap_projected": None, "gap_measured": round(c_plain - c_hard, 3)}
    C.finish("cve", "close the SQL injection in the user lookup handler without breaking lookups",
             scores, checks)


if __name__ == "__main__":
    main()
