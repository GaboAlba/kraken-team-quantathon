"""Weight schemes for the graph edges (transmission lines).

An edge's weight represents how relevant it is to cut it (place a protection
element) in the event of a fault: the higher the weight, the more critical the
line is for the fault-zone partition.

All schemes are functions ``fn(voltage, length_m, gens_u=(), gens_v=()) ->
float`` registered in ``SCHEMES``. ``gens_u``/``gens_v`` are the generator
lists attached to the edge endpoints (see ``graph.assign_generators``); the
voltage-only schemes ignore them. The default scheme is ``generation_inverted``
(generator-aware; the sign-inverted ``generation`` weight, so critical lines
score highest and positive), documented in ``docs/qubo.md``.
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

    Starts from a base of 1 and applies signed modifiers so the most critical
    lines score *lowest* (cheapest to cut): ``-1`` per nearby generator, minus
    each generator's ``power_norm``, ``+0.5`` if both endpoints have a generator,
    a halving thermal penalty (``+0.5**k`` for the k-th fossil-thermal generator),
    and ``-voltage/1000``. Not guaranteed positive. Generators without a
    ``power_norm`` contribute 0. See ``docs/qubo.md`` for the full formula.
    """
    n_u, n_v = len(gens_u), len(gens_v)
    weight = 1.0 - n_u - n_v
    if n_u > 0 and n_v > 0:
        weight += 0.5

    weight -= sum(g.get("power_norm", 0.0) for g in gens_u) \
        + sum(g.get("power_norm", 0.0) for g in gens_v)

    thermal = sum(1 for g in gens_u if g.get("thermal")) \
        + sum(1 for g in gens_v if g.get("thermal"))
    for k in range(1, thermal + 1):
        weight += 0.5 ** k

    weight -= float(voltage or 0.0) / 1000.0
    return weight


def _generation_inverted(voltage: float, length_m: float, gens_u: Sequence = (),
                         gens_v: Sequence = ()) -> float:
    """Sign-inverted ``generation`` weight (``-_generation``).

    The ``generation`` weights are mostly negative, with the *most critical*
    lines being the most negative. Negating them makes the weights mostly
    *positive*, with the most critical lines scoring the *highest*. This is meant
    for a **minimize-cut** QUBO objective: minimizing the total weight of the cut
    lines makes the fault-zone boundary avoid the high-weight (critical) lines
    and settle on the cheapest (least critical) ones. See ``docs/qubo.md``.
    """
    return -_generation(voltage, length_m, gens_u, gens_v)


SCHEMES = {
    "kv": _kv,
    "kv_normalized": _kv_normalized,
    "kv_over_length": _kv_over_length,
    "generation": _generation,
    "generation_inverted": _generation_inverted,
}

DEFAULT_SCHEME = "generation_inverted"
