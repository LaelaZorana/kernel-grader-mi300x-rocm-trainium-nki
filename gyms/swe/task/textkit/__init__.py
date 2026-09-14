"""textkit, a tiny text helper package. Issue #7: slugify leaves repeated dashes."""
import re


def slugify(text):
    # bug: spaces become dashes one for one, runs are never collapsed,
    # edges are never trimmed, and accents are dropped along with the letter
    text = text.lower()
    text = text.replace(" ", "-")
    return re.sub(r"[^a-z0-9-]", "", text)
