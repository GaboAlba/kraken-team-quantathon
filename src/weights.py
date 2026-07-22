"""Weight schemes for the graph edges (transmission lines).

An edge's weight represents how relevant it is to cut it (place a protection
element) in the event of a fault: the higher the weight, the more critical the
line is for the fault-zone partition.

All schemes are positive functions ``fn(voltage, length_m) -> float``,
interchangeable through the ``SCHEMES`` registry. The default scheme is ``kv``
(voltage level), documented and justified in ``Docs/desiciones.md``.
"""

from __future__ import annotations

# Reference voltage (the highest in the national system) used for normalization.
_KV_REF = 230.0


def _kv(voltage: float, length_m: float) -> float:
    """Weight = voltage level in kV (230 or 138). Default scheme."""
    return float(voltage)


def _kv_normalized(voltage: float, length_m: float) -> float:
    """Weight = voltage normalized to the (0, 1] range relative to 230 kV."""
    return float(voltage) / _KV_REF


def _kv_over_length(voltage: float, length_m: float) -> float:
    """Weight = normalized voltage divided by the length in km.

    Proxy for electrical coupling: short, high-voltage lines couple more.
    Guards against null or missing lengths.
    """
    length_km = max(float(length_m or 0.0) / 1000.0, 1e-3)
    return (float(voltage) / _KV_REF) / length_km


SCHEMES = {
    "kv": _kv,
    "kv_normalized": _kv_normalized,
    "kv_over_length": _kv_over_length,
}

DEFAULT_SCHEME = "kv"
