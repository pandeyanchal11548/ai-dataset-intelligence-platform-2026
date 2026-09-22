"""Helpers for turning pandas / numpy objects into plain JSON-safe Python."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe primitives.

    NaN / +-inf become None (strict JSON has no representation for them),
    timestamps become ISO-8601 strings, numpy scalars become Python scalars.
    """
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, (int, str)):
        return obj
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return str(obj)