# Contributing

Contributions are welcome when they preserve the target-style/reference-structure separation and do not weaken font-engineering verification.

## Before submitting

1. Create a focused branch and keep changes small enough to review.
2. Do not commit fonts, trained weights, generated fonts, caches or virtual environments.
3. Run `python -m compileall -q .` and `python selftest.py` in the project environment.
4. Keep all runtime and launcher messages in English.
5. Update tests and documentation for behavior, configuration or file-format changes.
6. Preserve checkpoint compatibility or clearly document and version an intentional break.

## Third-party material

Do not paste or vendor upstream code, model weights, datasets or fonts merely because a paper or repository is cited. For any third-party material, document its exact source and revision, verify redistribution permission, preserve all required notices and update `METHOD_REFERENCES.md` and `THIRD_PARTY_NOTICES.md`.

## Data-flow rule

Training must use target-font self-reconstruction only. The reference font may provide generation structure and output coverage only. A contribution that introduces reference paths into training supervision will not be accepted.

Unless explicitly stated otherwise, submitted contributions are licensed under Apache License 2.0 in accordance with the project's `LICENSE`.
