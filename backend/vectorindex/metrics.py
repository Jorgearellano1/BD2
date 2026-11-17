"""
Metricas de similitud para el modulo multimedia.

Metricas disponibles:
    L2      distancia euclidiana        operador SQL '<->'
    COSINE  distancia coseno (1 - cos)  operador SQL '<=>'
    IP      producto interno negado      operador SQL '<#>'
    DTW     alineamiento temporal        (series de audio, sin operador SQL)

La distancia coseno es 1 - coseno y el producto interno se niega, de modo que
en todas las metricas menor distancia significa mayor similitud.

DTW y su cota inferior LB_Keogh se usan para series temporales (audio) con
el esquema Filter-and-Refine: LB_Keogh nunca sobreestima la distancia real,
por lo que podar con ella no pierde falsos negativos.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

from vectorindex.vectors import Vector


class Metric(str, Enum):
    """Metricas soportadas por el modulo vectorial."""

    L2 = 'L2'
    COSINE = 'COSINE'
    IP = 'IP'
    DTW = 'DTW'


# Operadores SQL de similitud vectorial.
OPERATOR_METRIC: dict[str, Metric] = {
    '<->': Metric.L2,
    '<=>': Metric.COSINE,
    '<#>': Metric.IP,
}

METRIC_OPERATOR: dict[Metric, str] = {
    Metric.L2: '<->',
    Metric.COSINE: '<=>',
    Metric.IP: '<#>',
}


class MetricError(ValueError):
    """Metrica desconocida o incompatible con la consulta."""


def metric_by_name(name: str) -> Metric:
    """Convierte el texto de la opcion WITH (metric = 'COSINE') en Metric."""
    try:
        return Metric(name.strip().upper())
    except ValueError as exc:
        raise MetricError(
            f"Metrica '{name}' no soportada. Opciones: "
            f"{', '.join(sorted(metric.value for metric in Metric))}"
        ) from exc


def metric_by_operator(operator: str) -> Metric:
    """Convierte un operador SQL ('<=>') en la metrica correspondiente."""
    if operator not in OPERATOR_METRIC:
        raise MetricError(
            f"Operador vectorial '{operator}' no soportado. Opciones: "
            f"{', '.join(sorted(OPERATOR_METRIC))}"
        )

    return OPERATOR_METRIC[operator]


def l2_distance(left: Vector, right: Vector) -> float:
    """Distancia euclidiana entre dos vectores."""
    if len(left) != len(right):
        raise MetricError(
            f"No se pueden comparar vectores de dimensiones "
            f"{len(left)} y {len(right)}."
        )

    total = 0.0

    for a, b in zip(left, right):
        difference = a - b
        total += difference * difference

    return math.sqrt(total)


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Coseno del angulo entre dos vectores."""
    if len(left) != len(right):
        raise MetricError(
            f"No se pueden comparar vectores de dimensiones "
            f"{len(left)} y {len(right)}."
        )

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0

    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def cosine_distance(left: Vector, right: Vector) -> float:
    """Distancia coseno: 1 - similitud coseno."""
    return 1.0 - cosine_similarity(left, right)


def inner_product(left: Vector, right: Vector) -> float:
    """Producto interno entre dos vectores."""
    if len(left) != len(right):
        raise MetricError(
            f"No se pueden comparar vectores de dimensiones "
            f"{len(left)} y {len(right)}."
        )

    return sum(a * b for a, b in zip(left, right))


def distance(metric: Metric, left: Vector, right: Vector) -> float:
    """Aplica la metrica indicada a un par de vectores."""
    if metric is Metric.L2:
        return l2_distance(left, right)

    if metric is Metric.COSINE:
        return cosine_distance(left, right)

    if metric is Metric.IP:
        # Producto interno negado: menor distancia = mayor similitud.
        return -inner_product(left, right)

    if metric is Metric.DTW:
        return dtw_distance(left, right)

    raise MetricError(f"Metrica no soportada: {metric}")


def _point_cost(left, right) -> float:
    """Costo cuadratico entre dos puntos (escalares o vectores)."""
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            raise MetricError('DTW no puede comparar un escalar con un vector.')

        if len(left) != len(right):
            raise MetricError(
                f'DTW requiere puntos de igual dimension '
                f'({len(left)} y {len(right)}).'
            )

        return sum((a - b) ** 2 for a, b in zip(left, right))

    return (left - right) ** 2


def dtw_distance(left: Sequence, right: Sequence) -> float:
    """
    Distancia de alineamiento temporal dinamico entre dos series.

    Cada punto de la serie puede ser un escalar (energia, un coeficiente) o un
    vector (un MFCC por ventana). Se usa la recursion clasica con dos filas.
    """
    n = len(left)
    m = len(right)

    if n == 0 or m == 0:
        return float('inf')

    previous: list[float] = [float('inf')] * (m + 1)
    previous[0] = 0.0

    for i in range(1, n + 1):
        current = [float('inf')] * (m + 1)

        for j in range(1, m + 1):
            cost = _point_cost(left[i - 1], right[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])

        previous = current

    return math.sqrt(previous[m])


def _envelope(query: Sequence, radius: int) -> tuple[list, list]:
    """Envolvente superior e inferior de la consulta por ventana."""
    n = len(query)
    first = query[0]

    if isinstance(first, (tuple, list)):
        dimension = len(first)
        lower = [[math.inf] * dimension for _ in range(n)]
        upper = [[-math.inf] * dimension for _ in range(n)]

        for index in range(n):
            window = query[max(0, index - radius):min(n, index + radius + 1)]

            for point in window:
                for axis in range(dimension):
                    lower[index][axis] = min(lower[index][axis], point[axis])
                    upper[index][axis] = max(upper[index][axis], point[axis])

        return lower, upper

    lower: list[float] = []
    upper: list[float] = []

    for index in range(n):
        window = query[max(0, index - radius):min(n, index + radius + 1)]
        lower.append(min(window))
        upper.append(max(window))

    return lower, upper


def lb_keogh(query: Sequence, candidate: Sequence, radius: int) -> float:
    """
    Cota inferior de LB_Keogh para la distancia DTW.

    Admite series de escalares y series de vectores (MFCC por ventana). El
    valor obtenido nunca supera la distancia DTW real, por lo que sirve para
    podar candidatos sin falsos negativos (Filter-and-Refine).
    """
    n = len(query)

    if n == 0:
        return 0.0

    if len(candidate) != n:
        raise MetricError(
            f"LB_Keogh requiere series de igual longitud "
            f"({n} y {len(candidate)})."
        )

    lower, upper = _envelope(query, radius)
    total = 0.0
    first = query[0]

    if isinstance(first, (tuple, list)):
        dimension = len(first)

        for index in range(n):
            for axis in range(dimension):
                value = candidate[index][axis]

                if value < lower[index][axis]:
                    total += (lower[index][axis] - value) ** 2
                elif value > upper[index][axis]:
                    total += (value - upper[index][axis]) ** 2
    else:
        for index in range(n):
            value = candidate[index]

            if value < lower[index]:
                total += (lower[index] - value) ** 2
            elif value > upper[index]:
                total += (value - upper[index]) ** 2

    return math.sqrt(total)
