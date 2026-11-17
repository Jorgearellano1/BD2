"""
K-Means determinista para cuantizacion vectorial.

Se usa en dos lugares:
    - Codebook de BoVW: agrupa descriptores locales en K palabras visuales.
    - Particionamiento IVF: divide el espacio en C celdas de Voronoi.

La inicializacion es k-means++ y las iteraciones son de Lloyd. El generador
aleatorio es determinista (semilla fija), de modo que repetir una
construccion produce el mismo codebook y la misma particion.

Solo biblioteca estandar.
"""

from __future__ import annotations

import math
import random

from vectorindex.vectors import Vector


class KMeansError(ValueError):
    """Parametros invalidos o datos insuficientes para agrupar."""


def _squared_distance(left: Vector, right: Vector) -> float:
    total = 0.0

    for a, b in zip(left, right):
        difference = a - b
        total += difference * difference

    return total


def _nearest_centroid(point: Vector, centroids: list[Vector]) -> int:
    """Centroide mas cercano; usa |c|^2 - 2 c.x + |x|^2 para ahorrar trabajo."""
    best = 0
    best_value = math.inf

    for index, centroid in enumerate(centroids):
        value = 0.0

        for a, b in zip(point, centroid):
            value += (a - b) * (a - b)

        if value < best_value:
            best_value = value
            best = index

    return best


def _initial_centroids(points: list[Vector], k: int, seed: int) -> list[Vector]:
    """k-means++: el primer centroide es aleatorio y los siguientes se eligen
    con probabilidad proporcional al cuadrado de la distancia al mas cercano."""
    rng = random.Random(seed)

    first = rng.randrange(len(points))
    centroids: list[Vector] = [tuple(points[first])]

    distances = [_squared_distance(point, centroids[0]) for point in points]

    while len(centroids) < k:
        total = sum(distances)

        if total <= 0.0:
            # Todos los puntos coinciden: se completa con el primero.
            centroids.append(tuple(points[0]))
            distances = [0.0] * len(points)
            continue

        threshold = rng.random() * total
        accumulated = 0.0
        chosen = len(points) - 1

        for index, value in enumerate(distances):
            accumulated += value

            if accumulated >= threshold:
                chosen = index
                break

        centroids.append(tuple(points[chosen]))

        for index, point in enumerate(points):
            distance = _squared_distance(point, centroids[-1])

            if distance < distances[index]:
                distances[index] = distance

    return centroids


def kmeans(
    points: list[Vector],
    k: int,
    iterations: int = 10,
    seed: int = 42,
) -> tuple[list[Vector], list[int]]:
    """
    Agrupa los vectores en k grupos.

    Devuelve (centroids, asignaciones) donde asignaciones[i] es el indice
    del centroide asignado a points[i].
    """
    if not points:
        raise KMeansError("K-Means necesita al menos un vector.")

    if k <= 0:
        raise KMeansError("K-Means necesita k > 0.")

    if k > len(points):
        raise KMeansError(
            f"K-Means necesita al menos k vectores (k={k}, "
            f"vectores={len(points)})."
        )

    dimension = len(points[0])
    centroids = _initial_centroids(points, k, seed)
    assignments = [0] * len(points)

    for _ in range(max(1, iterations)):
        changed = False

        for index, point in enumerate(points):
            group = _nearest_centroid(point, centroids)

            if group != assignments[index]:
                assignments[index] = group
                changed = True

        sums = [[0.0] * dimension for _ in range(k)]
        counts = [0] * k

        for point, group in zip(points, assignments):
            counts[group] += 1

            for axis in range(dimension):
                sums[group][axis] += point[axis]

        for group in range(k):
            if counts[group] == 0:
                continue

            centroids[group] = tuple(
                sums[group][axis] / counts[group] for axis in range(dimension)
            )

        if not changed:
            break

    return centroids, assignments


def quantize(point: Vector, centroids: list[Vector]) -> int:
    """Indice del centroide mas cercano a un vector."""
    if not centroids:
        raise KMeansError("No hay centroides para cuantizar.")

    return _nearest_centroid(point, centroids)
