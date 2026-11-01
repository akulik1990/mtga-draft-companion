import os
import tempfile

# Must precede any companion/src import: upstream fixes its paths from cwd at import time.
os.environ.setdefault("MTGA_COMPANION_DATA", tempfile.mkdtemp(prefix="companion-test-"))

import companion  # noqa: E402,F401
