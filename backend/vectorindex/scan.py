"""
Busqueda exacta sobre vectores en memoria.

Es la linea base del modulo vectorial: recorre todos los vectores de la tabla
y conserva los k mejores con una cola de prioridad. Sirve como verdad base
(ground truth) de los experimentos de busqueda aproximada y como ruta de
acceso cuando la consulta fuerza un escaneo secuencial.
"""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from vectorindex.metrics import Metric, dtw_distance, lb_keogh
from vectorindex.vectors import Vector

RID = tuple[int, int]


class ScanError(RuntimeError):
    """Error durante la busqueda exacta."""


def _distance(metric: Metric, query: Vector, vector: Vector) -> float:
    if metric is Metric.DTW:
        return dtw_distance(query, vector)

    if len(query) != len(vector):
        raise ScanError(
            f'No se pueden comparar vectores de dimensiones '
            f'{len(query)} y {len(vector)}.'
        )

    if metric is Metric.L2:
        total = 0.0

        for a, b in zip(query, vector):
            difference = a - b
            total += difference * difference

        return math.sqrt(total)

    if metric is Metric.COSINE:
        dot = 0.0
        left = 0.0
        right = 0.0

        for a, b in zip(query, vector):
            dot += a * b
            left += a * a
            right += b * b

        if left == 0.0 or right == 0.0:
            return 1.0

        return 1.0 - dot / (math.sqrt(left) * math.sqrt(right))

    total = 0.0

    for a, b in zip(query, vector):
        total += a * b

    return -total


def exact_knn(
    records: Iterable[tuple[RID, Vector]],
    query: Vector,
    k: int,
    metric: Metric = Metric.COSINE,
    allowed_rids: set[RID] | None = None,
) -> list[tuple[RID, float]]:
    """
    k-NN lineal exacto: conserva los k vectores mas cercanos.

    Devuelve [(RID, distancia)] ordenados de menor a mayor distancia.
    """
    if k <= 0:
        return []

    heap: list[tuple[float, RID]] = []

    for rid, vector in records:
        if allowed_rids is not None and rid not in allowed_rids:
            continue

        value = _distance(metric, query, vector)

        # Max-heap de tamano k: la raiz guarda la peor distancia.
        if len(heap) < k:
            heapq.heappush(heap, (-value, rid))
        elif value < -heap[0][0]:
            heapq.heapreplace(heap, (-value, rid))

    return [(rid, -negative) for negative, rid in sorted(
        heap, key=lambda item: (-item[0], item[1]),
    )]


def exact_range(
    records: Iterable[tuple[RID, Vector]],
    query: Vector,
    radius: float,
    metric: Metric = Metric.L2,
    allowed_rids: set[RID] | None = None,
) -> list[tuple[RID, float]]:
    """
    Busqueda secuencial por rango: todos los vectores a distancia <= radio.

    El radio se expresa en la misma unidad de la metrica: distancia L2 para
    '<->' y distancia coseno para '<=>'.
    """
    results: list[tuple[RID, float]] = []

    for rid, vector in records:
        if allowed_rids is not None and rid not in allowed_rids:
            continue

        value = _distance(metric, query, vector)

        if value <= radius:
            results.append((rid, value))

    return sorted(results, key=lambda item: (item[1], item[0]))


def filter_and_refine(
    records: Iterable[tuple[RID, Vector]],
    query: Vector,
    k: int,
    radius: int = 3,
    allowed_rids: set[RID] | None = None,
) -> list[tuple[RID, float]]:
    """
    DTW con poda LB_Keogh (Filter-and-Refine).

    Primero se calcula la cota inferior LB_Keogh de cada candidato; si supera
    la distancia del k-esimo mejor encontrado hasta el momento, el candidato
    se descarta sin evaluar la matriz de DTW completa. La cota nunca
    sobreestima la distancia real, por lo que no se pierden falsos negativos.
    """
    if k <= 0:
        return []

    heap: list[tuple[float, RID]] = []

    for rid, vector in records:
        if allowed_rids is not None and rid not in allowed_rids:
            continue

        bound = lb_keogh(query, vector, radius)

        if len(heap) == k and bound > -heap[0][0]:
            continue

        value = dtw_distance(query, vector)

        # Max-heap de tamano k: la raiz guarda la peor distancia.
        if len(heap) < k:
            heapq.heappush(heap, (-value, rid))
        elif value < -heap[0][0]:
            heapq.heapreplace(heap, (-value, rid))

    return [(rid, -negative) for negative, rid in sorted(
        heap, key=lambda item: (-item[0], item[1]),
    )]
