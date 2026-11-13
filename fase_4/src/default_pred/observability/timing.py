import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """Holds the elapsed time (ms) of a timed block."""

    label: str
    elapsed_ms: float = 0.0


@contextmanager
def timed(label: str) -> Iterator[Timer]:
    """Measure and log the duration of a block, even if it raises."""
    timer = Timer(label=label)
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s finished in %.2f ms", label, timer.elapsed_ms)
