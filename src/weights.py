"""Esquemas de peso para las aristas del grafo (líneas de transmisión).

El peso de una arista representa qué tan relevante es poder cortarla (colocar un
elemento de protección) ante una falla: a mayor peso, más crítica la línea para
la partición en zonas de falla.

Todos los esquemas son funciones ``fn(voltaje, length_m) -> float`` positivas,
intercambiables mediante el registro ``SCHEMES``. El esquema por defecto es
``kv`` (nivel de tensión), documentado y justificado en ``Docs/desiciones.md``.
"""

from __future__ import annotations

# Tensión de referencia (la más alta del sistema nacional) para normalizar.
_KV_REF = 230.0


def _kv(voltaje: float, length_m: float) -> float:
    """Peso = nivel de tensión en kV (230 o 138). Esquema por defecto."""
    return float(voltaje)


def _kv_normalized(voltaje: float, length_m: float) -> float:
    """Peso = tensión normalizada al rango (0, 1] respecto de 230 kV."""
    return float(voltaje) / _KV_REF


def _kv_over_length(voltaje: float, length_m: float) -> float:
    """Peso = tensión normalizada dividida por la longitud en km.

    Proxy de acoplamiento eléctrico: líneas cortas y de alta tensión acoplan más.
    Se protege contra longitudes nulas o ausentes.
    """
    length_km = max(float(length_m or 0.0) / 1000.0, 1e-3)
    return (float(voltaje) / _KV_REF) / length_km


SCHEMES = {
    "kv": _kv,
    "kv_normalized": _kv_normalized,
    "kv_over_length": _kv_over_length,
}

DEFAULT_SCHEME = "kv"
