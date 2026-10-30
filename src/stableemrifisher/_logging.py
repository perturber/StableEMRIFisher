"""Set up the "stableemrifisher" logger used by every module of the package.

Modules get it with logging.getLogger("stableemrifisher"). It prints INFO and
above to stdout; StableEMRIFisher(stats_for_nerds=True) switches it to DEBUG.
"""

import logging
import sys

LOGGER_NAME = "stableemrifisher"

logger = logging.getLogger(LOGGER_NAME)

# attach the handler only once, even if this module is reloaded
if not any(getattr(h, "_stableemrifisher", False) for h in logger.handlers):
    _handler = logging.StreamHandler(sys.stdout)
    _handler._stableemrifisher = True
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def set_verbose(verbose):
    """Log DEBUG messages if `verbose`, otherwise INFO and above."""
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
