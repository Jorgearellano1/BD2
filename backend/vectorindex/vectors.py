"""
Literales y utilidades sobre vectores densos.

Un vector es una tupla de floats de dimension fija. El motor los persiste
dentro del registro HeapFile con un formato struct de la forma '128f', de modo
que la dimension queda grabada en el esquema de la tabla.

Literales SQL:
    '[0.12, -0.34, 0.56]'  ->  (0.12, -0.34, 0.56)

Solo se usa la biblioteca estandar: el modulo no depende de numpy ni de
ninguna libreria de algebra lineal.
"""

from __future__ import annotations

import math
import struct

Vector = tuple[float, ...]


class VectorError(ValueError):
    """Error de formato o de dimension de un vector."""


def parse_vector(text: str) -> Vector:
    """Convierte un literal '[a, b, c]' en una tupla de floats."""
    cleaned = text.strip()

    if cleaned.startswith('[') and cleaned.endswith(']'):
        cleaned = cleaned[1:-1]

    if not cleaned.strip():
        return ()

    values: list[float] = []

    for part in cleaned.split(','):
        part = part.strip()

        if not part:
            raise VectorError(
                f"Literal vectorial invalido: '{text}' (componente vacio)"
            )

        try:
            values.append(float(part))
        except ValueError as exc:
            raise VectorError(
                f"Literal vectorial invalido: '{part}' no es un numero"
            ) from exc

    return tuple(values)


def format_vector(vector: Vector, precision: int = 6) -> str:
    """Devuelve el literal SQL de un vector."""
    return '[' + ','.join(f'{value:.{precision}f}' for value in vector) + ']'


def vector_to_bytes(vector: Vector) -> bytes:
    """Serializa un vector como D float32 (formato del registro HeapFile)."""
    return struct.pack(f'={len(vector)}f', *vector)


def vector_from_bytes(data: bytes, dimension: int) -> Vector:
    """Reconstruye un vector a partir de su bloque de float32."""
    if len(data) != 4 * dimension:
        raise VectorError(
            f'Bloque vectorial de {len(data)} bytes para dimension {dimension}.'
        )

    return struct.unpack(f'={dimension}f', data)


def l2_norm(vector: Vector) -> float:
    """Norma euclidiana de un vector."""
    return math.sqrt(sum(value * value for value in vector))


def normalize(vector: Vector) -> Vector:
    """Vector unitario; un vector nulo se devuelve sin cambios."""
    norm = l2_norm(vector)

    if norm == 0.0:
        return tuple(vector)

    return tuple(value / norm for value in vector)


def check_dimension(vector: Vector, dimension: int, context: str = 'vector') -> None:
    """Verifica que el vector tenga la dimension esperada."""
    if len(vector) != dimension:
        raise VectorError(
            f"El {context} tiene {len(vector)} componentes y se esperaban "
            f"{dimension}."
        )
