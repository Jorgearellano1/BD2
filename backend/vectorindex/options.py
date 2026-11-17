"""
Opciones de CREATE INDEX ... USING IVF.

Sintaxis:

    CREATE INDEX idx_vec ON multimedia(embedding)
      USING IVF WITH (lists = 100, metric = 'COSINE');

Opciones aceptadas:
    lists   numero de celdas de Voronoi (partitiones), por defecto 25.
    metric  metrica de distancia: 'L2', 'COSINE', 'IP' o 'DTW'.
    probes  numero de celdas consultadas por defecto (SET ivf.probes).
    iterations  iteraciones de K-Means al construir el indice.
    seed   semilla determinista del K-Means.
"""

from __future__ import annotations

from vectorindex.metrics import Metric, metric_by_name

DEFAULT_LISTS = 25
DEFAULT_PROBES = 1
DEFAULT_METRIC = Metric.COSINE
DEFAULT_ITERATIONS = 10
DEFAULT_SEED = 42

_ALLOWED = {'lists', 'metric', 'probes', 'iterations', 'seed'}


class VectorOptionsError(ValueError):
    """Opcion de indice vectorial invalida."""


def normalize_vector_options(
    options: dict[str, str | int] | None,
) -> dict[str, str | int]:
    """Valida y normaliza las opciones de WITH (...) de un indice IVF."""
    if options is None:
        options = {}

    unknown = set(options) - _ALLOWED

    if unknown:
        raise VectorOptionsError(
            f"Opcion(es) no soportada(s) para un indice IVF: "
            f"{', '.join(sorted(unknown))}. "
            f"Opciones validas: {', '.join(sorted(_ALLOWED))}."
        )

    result: dict[str, str | int] = {
        'lists': DEFAULT_LISTS,
        'metric': DEFAULT_METRIC.value,
        'probes': DEFAULT_PROBES,
        'iterations': DEFAULT_ITERATIONS,
        'seed': DEFAULT_SEED,
    }

    for key, value in options.items():
        key = key.lower()

        if key == 'metric':
            result['metric'] = metric_by_name(str(value)).value
            continue

        if key in ('lists', 'probes', 'iterations', 'seed'):
            if isinstance(value, bool) or not isinstance(value, int):
                raise VectorOptionsError(
                    f"La opcion '{key}' debe ser un entero."
                )

            if key == 'lists' and value <= 0:
                raise VectorOptionsError(
                    "La opcion 'lists' debe ser un entero positivo."
                )

            if key == 'probes' and value <= 0:
                raise VectorOptionsError(
                    "La opcion 'probes' debe ser un entero positivo."
                )

            if key == 'iterations' and value <= 0:
                raise VectorOptionsError(
                    "La opcion 'iterations' debe ser un entero positivo."
                )

            result[key] = value
            continue

    return result
