"""Pytest configuration and test utilities.

Ensures repository root is on sys.path so that the `src` package
(with `__init__.py`) can be imported without per-test hacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
