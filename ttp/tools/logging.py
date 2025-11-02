import time
from contextlib import contextmanager
from torchtitan.tools.logging import logger


@contextmanager
def timer(msg_format="{delta}", **fmt):
    try:
        start = time.perf_counter_ns()
        yield
    finally:
        delta_ns = time.perf_counter_ns() - start
        if delta_ns > 5e9:
            delta = f"{delta_ns / 1e9:.3f} (seconds)"
        elif delta_ns > 5e6:
            delta = f"{delta_ns / 1e6:.3f} (milliseconds)"
        elif delta_ns > 5e3:
            delta = f"{delta_ns / 1e3:.3f} (microseconds)"
        else:
            delta = f"{delta_ns / 1.0:.3f} (nanoseconds)"
        logger.info(msg_format.format(delta=delta, **fmt))
