# `tests/` — pytest suite

Run with `python -m pytest tests/ -v` from the repo root.

| File | Covers |
|------|--------|
| `conftest.py` | Shared pytest config (currently minimal). |
| `test_phone_template.py` | Marker-position math, paper-size handling, and a round-trip render-then-detect check that catches PDF generation regressions. |
| `test_phone_preprocess.py` | ArUco detection, homography, and the marker-not-found / scale-mismatch error paths. |
| `test_web_jobs.py` | `JobManager` smoke tests — job creation, phase-A/phase-B dispatch, SSE replay, redo with `skip_trace`, TTL sweep, and the shutdown sentinel. Uses a mocked `run_pipeline` so the SAM2 model never loads. |

The trace + layout + Fusion modules are integration-tested by hand —
no unit tests yet.
