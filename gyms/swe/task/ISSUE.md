# Issue 7, slugify leaves repeated dashes and drops accented letters

Calling `slugify("Hello   World--Again")` returns `hello---world--again` and should return `hello-world-again`.

Runs of spaces and dashes must collapse to one dash, and dashes at either end must go.

Calling `slugify("Crème Brûlée")` returns `crme-brle` and should return `creme-brulee`.

Hidden tests live in `tests/test_slugify.py`.
