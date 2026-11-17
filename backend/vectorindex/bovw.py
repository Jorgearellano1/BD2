"""
Paradigma clasico: Bag of Visual/Audio Words.

El pipeline es:

    1. Descriptores locales invariantes (SIFT 128D o MFCC por ventanas).
    2. Cuantizacion vectorial con K-Means: se construye un codebook de K
       palabras visuales (o acusticas).
    3. Cada objeto se representa como un histograma de frecuencias de esas
       palabras, ponderado con TF-IDF.
    4. El histograma se indexa sobre el indice invertido: cada
       palabra visual es un termino y su frecuencia en la imagen es el TF.

El histograma se serializa como un documento de texto con una repeticion por
palabra visual ('v0012 v0012 v0007'), de modo que el indice invertido
existente lo indexa sin cambios y la similitud coseno del modulo textual
sirve como ranking BoVW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vectorindex.kmeans import kmeans
from vectorindex.vectors import Vector

WORD_PREFIX = 'v'


class BovwError(ValueError):
    """Error en la construccion o uso del codebook visual."""


@dataclass
class Codebook:
    """Vocabulario visual (o acustico) construido con K-Means."""

    centroids: list[Vector]
    dimension: int

    @property
    def size(self) -> int:
        """Cantidad de palabras del codebook."""
        return len(self.centroids)

    def assign(self, descriptor: Vector) -> int:
        """Palabra visual mas cercana al descriptor."""
        if len(descriptor) != self.dimension:
            raise BovwError(
                f"El descriptor tiene {len(descriptor)} componentes y el "
                f'codebook espera {self.dimension}.'
            )

        best = 0
        best_value = math.inf

        for index, centroid in enumerate(self.centroids):
            value = 0.0

            for a, b in zip(descriptor, centroid):
                difference = a - b
                value += difference * difference

            if value < best_value:
                best_value = value
                best = index

        return best

    def assign_many(self, descriptors: list[Vector]) -> list[int]:
        """Palabra visual de cada descriptor."""
        return [self.assign(descriptor) for descriptor in descriptors]


def build_codebook(
    descriptors: list[Vector],
    words: int = 100,
    iterations: int = 10,
    seed: int = 42,
) -> Codebook:
    """Construye el codebook con K-Means sobre los descriptores locales."""
    if not descriptors:
        raise BovwError('No hay descriptores para construir el codebook.')

    centroids, _ = kmeans(
        descriptors, words, iterations=iterations, seed=seed,
    )

    return Codebook(centroids=centroids, dimension=len(descriptors[0]))


def histogram(descriptors: list[Vector], codebook: Codebook) -> dict[int, int]:
    """Histograma de palabras visuales de un objeto."""
    counts: dict[int, int] = {}

    for descriptor in descriptors:
        word = codebook.assign(descriptor)
        counts[word] = counts.get(word, 0) + 1

    return counts


def word_token(word: int) -> str:
    """Termino del indice invertido que representa una palabra visual."""
    return f'{WORD_PREFIX}{word:04d}'


def histogram_document(counts: dict[int, int]) -> str:
    """
    Serializa un histograma como documento para el indice invertido.

    Cada palabra visual aparece tantas veces como su frecuencia, de modo que
    el TF del indice invertido es el valor del histograma.
    """
    tokens: list[str] = []

    for word in sorted(counts):
        tokens.extend([word_token(word)] * counts[word])

    return ' '.join(tokens)


def parse_document(document: str) -> dict[int, int]:
    """Reconstruye un histograma a partir del documento serializado."""
    counts: dict[int, int] = {}

    for token in document.split():
        token = token.strip().lower()

        if not token.startswith(WORD_PREFIX):
            continue

        try:
            word = int(token[len(WORD_PREFIX):])
        except ValueError:
            continue

        counts[word] = counts.get(word, 0) + 1

    return counts


def tfidf_weights(
    counts: dict[int, int],
    document_frequency: dict[int, int],
    num_documents: int,
) -> dict[int, float]:
    """
    Ponderacion TF-IDF clasica de BoVW.

        tf   = frecuencia de la palabra / numero de descriptores
        idf  = log((N + 1) / (df + 1)) + 1
        peso = tf * idf

    El indice invertido usa su propia formula de TF-IDF para el
    ranking; esta funcion sirve para la evaluacion offline (Precision@K) y
    para comparar ambos paradigmas.
    """
    total = sum(counts.values())

    if total == 0:
        return {}

    weights: dict[int, float] = {}

    for word, count in counts.items():
        df = document_frequency.get(word, 0)
        idf = math.log((num_documents + 1.0) / (df + 1.0)) + 1.0
        weights[word] = (count / total) * idf

    return weights


def l2_normalize_weights(weights: dict[int, float]) -> dict[int, float]:
    """Normaliza los pesos TF-IDF con la norma euclidiana."""
    norm = math.sqrt(sum(value * value for value in weights.values()))

    if norm == 0.0:
        return dict(weights)

    return {word: value / norm for word, value in weights.items()}


def cosine(weights: dict[int, float], other: dict[int, float]) -> float:
    """Similitud coseno entre dos histogramas ponderados."""
    numerator = 0.0
    left = 0.0
    right = 0.0

    for word, value in weights.items():
        left += value * value
        numerator += value * other.get(word, 0.0)

    for value in other.values():
        right += value * value

    if left == 0.0 or right == 0.0:
        return 0.0

    return numerator / (math.sqrt(left) * math.sqrt(right))
