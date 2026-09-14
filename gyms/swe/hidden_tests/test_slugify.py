from textkit import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_collapse_spaces():
    assert slugify("Hello   World") == "hello-world"


def test_collapse_dashes():
    assert slugify("Hello   World--Again") == "hello-world-again"


def test_strip_edges():
    assert slugify("  --Hello-- ") == "hello"


def test_unicode_fold():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_punctuation():
    assert slugify("What's up? 100% real!") == "whats-up-100-real"
