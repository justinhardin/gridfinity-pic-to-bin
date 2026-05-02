"""FastAPI web wrapper for the pic-to-bin pipeline.

Exposes the same functionality as the ``pic-to-bin`` CLI through a browser
UI: drag-drop photos, fill a form, watch progress, preview the layout, choose
proceed or re-do, then download the artifacts.

Run with::

    pic-to-bin-web --port 8000
"""

from pic_to_bin.web.jobs import JobManager, JobState, JobStatus  # noqa: F401
