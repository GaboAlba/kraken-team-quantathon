"""Weight schemes for the graph edges (transmission lines).

An edge's weight represents how relevant it is to cut it (place a protection
element) in the event of a fault: the higher the weight, the more critical the
line is for the fault-zone partition.

All schemes are functions ``fn(voltage, length_m, gens_u=(), gens_v=()) ->
float`` registered in ``SCHEMES``. ``gens_u``/``gens_v`` are the generator
lists attached to the edge endpoints (see ``graph.assign_generators``); the
voltage-only schemes ignore them. The default scheme is ``generation``
(generator-aware), documented and justified in ``Docs/desiciones.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

# Reference voltage (the highest in the national system) used for normalization.
_KV_REF = 230.0


def _kv(voltage: float, length_m: float, gens_u: Sequence = (),
        gens_v: Sequence = ()) -> float:
    """Weight = voltage level in kV (230 or 138)."""
    return float(voltage)


def _kv_normalized(voltage: float, length_m: float, gens_u: Sequence = (),
                   gens_v: Sequence = ()) -> float:
    """Weight = voltage normalized to the (0, 1] range relative to 230 kV."""
    return float(voltage) / _KV_REF


def _kv_over_length(voltage: float, length_m: float, gens_u: Sequence = (),
                    gens_v: Sequence = ()) -> float:
    """Weight = normalized voltage divided by the length in km.

    Proxy for electrical coupling: short, high-voltage lines couple more.
    Guards against null or missing lengths.
    """
    length_km = max(float(length_m or 0.0) / 1000.0, 1e-3)
    return (float(voltage) / _KV_REF) / length_km


def _generation(voltage: float, length_m: float, gens_u: Sequence = (),
                gens_v: Sequence = ()) -> float:
    """Generator-aware criticality weight for a line ``u``--``v``.

    Starts from a base of 1 (substations only) and adjusts for the generation
    connected near each endpoint::

        weight = 1
               + n_gen(u) + n_gen(v)                 # +1 per nearby generator
               - 0.5   if BOTH endpoints have a generator
               - sum_{k=1..T} 0.5**k                  # thermal penalty (halving)
               + voltage / 1000                       # kV contribution

    ``T`` is the number of fossil-thermal generators associated with the line
    (across both endpoints). The thermal penalty is a halving series, so the
    k-th thermal generator subtracts ``0.5**k`` (0.5, 0.25, 0.125, ...) for a
    total of ``1 - 2**-T``. Higher-voltage and generator-dense lines are more
    critical; thermal (non-sustainable) generation reduces the weight.

    Note: unlike the voltage schemes, this weight is not guaranteed positive --
    a heavy thermal penalty on a low-voltage line with few generators can drive
    it below zero.
    """
    n_u, n_v = len(gens_u), len(gens_v)
    weight = 1.0 + n_u + n_v
    if n_u > 0 and n_v > 0:
        weight -= 0.5

    thermal = sum(1 for g in gens_u if g.get("thermal")) \
        + sum(1 for g in gens_v if g.get("thermal"))
    for k in range(1, thermal + 1):
        weight -= 0.5 ** k

    weight += float(voltage or 0.0) / 1000.0
    return weight


SCHEMES = {
    "kv": _kv,
    "kv_normalized": _kv_normalized,
    "kv_over_length": _kv_over_length,
    "generation": _generation,
}

DEFAULT_SCHEME = "generation"
