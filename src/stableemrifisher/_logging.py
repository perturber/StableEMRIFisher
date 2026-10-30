"""Set up the "stableemrifisher" logger used by every module of the package.

Modules get it with logging.getLogger("stableemrifisher"). It prints INFO and
above to stdout, through tqdm.write so that messages appear above any active
progress bar instead of breaking it. For full debugging output, set its level
to DEBUG:

    logging.getLogger("stableemrifisher").setLevel("DEBUG")
"""

import logging
import sys

from tqdm import tqdm

LOGGER_NAME = "stableemrifisher"

logger = logging.getLogger(LOGGER_NAME)


class _TqdmStreamHandler(logging.StreamHandler):
    """Stream handler that writes through tqdm.write."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


# attach the handler only once, even if this module is reloaded
if not any(getattr(h, "_stableemrifisher", False) for h in logger.handlers):
    _handler = _TqdmStreamHandler(sys.stdout)
    _handler._stableemrifisher = True
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
