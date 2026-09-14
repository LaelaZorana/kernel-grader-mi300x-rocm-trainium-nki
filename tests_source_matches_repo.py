"""Guard against the stale name fault.

A repository can be renamed after its data is written. When that happens every
row still points at the old name, and so does the citation block, so a reader
who copies the citation credits a name that no longer resolves. This test
compares the source field carried inside the data against the name declared in
the builder, and fails when they drift apart.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def declared_source():
    src = open(os.path.join(HERE, "build_dataset.py"), encoding="utf-8").read()
    m = re.search(r'^SOURCE = "([^"]+)"', src, re.M)
    assert m, "build_dataset.py has no SOURCE constant"
    return m.group(1)


def test_every_row_carries_the_declared_source():
    want = declared_source()
    checked = 0
    for path in sorted(glob.glob(os.path.join(HERE, "dataset", "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                row = json.loads(line)
                got = row.get("source")
                assert got == want, "%s line %d carries %r, expected %r" % (
                    os.path.basename(path), i, got, want)
                checked += 1
    assert checked > 0, "no rows were checked"
    print("rows checked %d, all carrying %s" % (checked, want))


def test_the_citation_url_matches_the_declared_source():
    want = declared_source()
    card = open(os.path.join(HERE, "dataset", "README.md"), encoding="utf-8").read()
    urls = re.findall(r"huggingface\.co/datasets/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", card)
    assert urls, "the dataset card carries no repository url"
    for u in urls:
        assert u == want, "the card points at %r, the data says %r" % (u, want)


def test_the_checksums_match_the_files_on_disk():
    import hashlib
    recorded = json.load(open(os.path.join(HERE, "dataset", "checksums.json"), encoding="utf-8"))
    for name, v in recorded.items():
        p = os.path.join(HERE, "dataset", name)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert got == v["sha256"], "%s changed since the checksum was written" % name
        assert sum(1 for _ in open(p, encoding="utf-8")) == v["rows"], "%s row count drifted" % name
