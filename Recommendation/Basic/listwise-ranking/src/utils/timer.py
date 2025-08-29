from __future__ import annotations
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.time()
    yield
    dur = time.time() - start
    print(f"[TIMER] {name}: {dur:.3f}s")