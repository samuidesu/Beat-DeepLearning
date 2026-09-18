"""Make stdout able to print Romanian.

On Windows, a console that is not already UTF-8 uses the ANSI code page
(cp1252 here). Printing a Romanian sentence then raises

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u021b'

and the script dies -- on `print`, not on anything that matters. Every script
in this project displays target text, so each one calls this first.

errors="replace" rather than "strict": a console font that cannot draw a
character should substitute it, not abort the run.
"""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 when they are not already."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if stream is None or encoding == "utf8":
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Redirected to a pipe that cannot be reconfigured: leave it alone.
            pass
