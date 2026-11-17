"""
Extraccion de caracteristicas multimedia.

El modulo reconoce tres fuentes de descriptores:

    - Dataset multimedia (data/datasets/multimedia): descriptores SIFT 128D
      extraidos con OpenCV y embeddings densos de ResNet-18. Los archivos se
      leen con struct, sin depender de las librerias que los generaron.
    - Imagenes sueltas: descriptores locales de 128D calculados con Pillow
      (histograma de orientaciones de gradiente por celdas). Sirve para
      funcionar sin el dataset precalculado.
    - Audio WAV: coeficientes cepstrales de Mel (MFCC) calculados con el
      modulo wave de la biblioteca estandar.

Ninguna de estas rutas interviene en el almacenamiento: el motor solo guarda
los vectores resultantes.
"""

from __future__ import annotations

import math
import os
import struct
import wave
from dataclasses import dataclass

from vectorindex.vectors import Vector

SIFT_MAGIC = b'BD2SIFT1'
EMBEDDINGS_MAGIC = b'BD2EMB01'

# Numero de celdas del descriptor local de imagen (4x4) y bins de orientacion
# (8) por celda: 16 * 8 = 128 dimensiones.
GRID = 4
ORIENTATION_BINS = 8
LOCAL_DESCRIPTOR_DIM = GRID * GRID * ORIENTATION_BINS


class FeatureError(RuntimeError):
    """Error leyendo un dataset o extrayendo una caracteristica."""


@dataclass
class EmbeddingRow:
    """Fila del manifiesto de embeddings."""

    id: int
    category: str
    filename: str


def load_sift_descriptors(path: str) -> list[list[Vector]]:
    """
    Lee los descriptores SIFT del dataset.

    Formato: magic 'BD2SIFT1', uint32 num_images, uint32 dim, y por imagen
    uint32 count seguido de count*dim valores uint8 (rango 0..255).
    """
    if not os.path.exists(path):
        raise FeatureError(f'No existe el dataset de descriptores: {path}')

    with open(path, 'rb') as handle:
        raw = handle.read()

    if len(raw) < 16:
        raise FeatureError('El dataset de descriptores esta truncado.')

    magic, num_images, dimension = struct.unpack_from('<8sII', raw, 0)

    if magic != SIFT_MAGIC:
        raise FeatureError('El dataset de descriptores no tiene el formato esperado.')

    offset = 16
    documents: list[list[Vector]] = []

    for _ in range(num_images):
        (count,) = struct.unpack_from('<I', raw, offset)
        offset += 4

        if offset + count * dimension > len(raw):
            raise FeatureError('El dataset de descriptores esta truncado.')

        descriptors: list[Vector] = []

        for _ in range(count):
            values = raw[offset:offset + dimension]
            offset += dimension
            descriptors.append(tuple(value / 255.0 for value in values))

        documents.append(descriptors)

    return documents


def load_embeddings(path: str) -> list[Vector]:
    """
    Lee los embeddings densos del dataset.

    Formato: magic 'BD2EMB01', uint32 num_images, uint32 dim y luego
    num_images*dim float32 en orden de fila.
    """
    if not os.path.exists(path):
        raise FeatureError(f'No existe el dataset de embeddings: {path}')

    with open(path, 'rb') as handle:
        raw = handle.read()

    if len(raw) < 16:
        raise FeatureError('El dataset de embeddings esta truncado.')

    magic, num_images, dimension = struct.unpack_from('<8sII', raw, 0)

    if magic != EMBEDDINGS_MAGIC:
        raise FeatureError('El dataset de embeddings no tiene el formato esperado.')

    expected = 16 + num_images * dimension * 4

    if len(raw) < expected:
        raise FeatureError('El dataset de embeddings esta truncado.')

    vectors: list[Vector] = []

    for index in range(num_images):
        start = 16 + index * dimension * 4
        values = struct.unpack_from(f'<{dimension}f', raw, start)
        vectors.append(values)

    return vectors


def load_embedding_manifest(path: str) -> list[EmbeddingRow]:
    """Lee el manifiesto CSV de los embeddings (id, category, filename)."""
    if not os.path.exists(path):
        raise FeatureError(f'No existe el manifiesto de embeddings: {path}')

    rows: list[EmbeddingRow] = []

    with open(path, 'r', encoding='utf-8') as handle:
        header = handle.readline().strip().split(',')

        for line in handle:
            line = line.strip()

            if not line:
                continue

            parts = line.split(',')

            if len(parts) < 3:
                continue

            rows.append(
                EmbeddingRow(int(parts[0]), parts[1], parts[2]),
            )

    return rows


def load_arff_vectors(path: str) -> list[tuple[str, Vector]]:
    """
    Lee un ARFF con atributos numericos y una clase nominal.

    Devuelve pares (clase, vector). Se usa para el dataset tabular de granos,
    que sirve como coleccion de vectores de 16 dimensiones para las pruebas
    de busqueda exacta y ANN.
    """
    if not os.path.exists(path):
        raise FeatureError(f'No existe el archivo ARFF: {path}')

    vectors: list[tuple[str, Vector]] = []
    in_data = False

    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith('%'):
                continue

            if line.lower() == '@data':
                in_data = True
                continue

            if not in_data:
                continue

            parts = [part.strip() for part in line.split(',')]

            if len(parts) < 2:
                continue

            label = parts[-1]
            values = tuple(float(part) for part in parts[:-1])
            vectors.append((label, values))

    return vectors


def image_descriptors(path: str, dimension: int = LOCAL_DESCRIPTOR_DIM) -> list[Vector]:
    """
    Descriptores locales de una imagen (128D por defecto).

    Se divide la imagen en una rejilla GRID x GRID; en cada celda se calcula
    un histograma de ORIENTATION_BINS gradientes (magnitude ponderada). Es
    el equivalente barato y local de un SIFT: por eso el codebook BoVW
    necesita cuantizar y no puede comparar pixeles crudos.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise FeatureError(
            'La extraccion de descriptores de imagen requiere Pillow '
            '(pip install pillow).'
        ) from exc

    if dimension != LOCAL_DESCRIPTOR_DIM:
        raise FeatureError(
            f'El descriptor local tiene {LOCAL_DESCRIPTOR_DIM} dimensiones '
            f'y se pidieron {dimension}.'
        )

    with Image.open(path) as image:
        gray = image.convert('L').resize((64, 64))
        pixels = list(gray.getdata())

    size = 64
    cell = size // GRID
    descriptors: list[Vector] = []

    for row in range(GRID):
        for column in range(GRID):
            histogram = [0.0] * ORIENTATION_BINS

            for y in range(row * cell, (row + 1) * cell - 1):
                for x in range(column * cell, (column + 1) * cell - 1):
                    index = y * size + x
                    gx = float(pixels[index + 1] - pixels[index - 1])
                    gy = float(pixels[index + size] - pixels[index - size])
                    magnitude = math.sqrt(gx * gx + gy * gy)

                    if magnitude == 0.0:
                        continue

                    angle = math.atan2(gy, gx) + math.pi
                    bin_index = int(angle / (2 * math.pi) * ORIENTATION_BINS)
                    bin_index = min(ORIENTATION_BINS - 1, max(0, bin_index))
                    histogram[bin_index] += magnitude

            norm = math.sqrt(sum(value * value for value in histogram))

            if norm > 0.0:
                histogram = [value / norm for value in histogram]

            descriptors.append(tuple(histogram))

    return descriptors


def mel_filterbank(
    num_filters: int = 26,
    num_fft: int = 512,
    low_hz: float = 20.0,
    high_hz: float = 8000.0,
    sample_rate: int = 16000,
) -> list[list[float]]:
    """Banca de filtros triangulares de Mel (implementacion propia)."""
    high_hz = min(high_hz, sample_rate / 2.0)

    def mel(frequency: float) -> float:
        return 2595.0 * math.log10(1.0 + frequency / 700.0)

    def hz(mel_value: float) -> float:
        return 700.0 * (10.0 ** (mel_value / 2595.0) - 1.0)

    mel_points = [
        mel(low_hz) + (mel(high_hz) - mel(low_hz)) * index / (num_filters + 1)
        for index in range(num_filters + 2)
    ]
    hz_points = [hz(value) for value in mel_points]
    bins = num_fft // 2 + 1
    filterbank: list[list[float]] = []

    for index in range(1, num_filters + 1):
        left, center, right = hz_points[index - 1], hz_points[index], hz_points[index + 1]
        row: list[float] = []

        for bin_index in range(bins):
            frequency = bin_index * sample_rate / num_fft

            if frequency <= left or frequency > right:
                row.append(0.0)
            elif frequency <= center:
                row.append((frequency - left) / (center - left) if center > left else 0.0)
            else:
                row.append((right - frequency) / (right - center) if right > center else 0.0)

        filterbank.append(row)

    return filterbank


def _dct(vector: list[float], num_ceps: int) -> list[float]:
    """Transformada discreta del coseno (tipo II) sobre un bloque de energia."""
    n = len(vector)
    output: list[float] = []

    for k in range(num_ceps):
        total = 0.0

        for index in range(n):
            total += vector[index] * math.cos(
                math.pi * k * (index + 0.5) / n
            )

        output.append(total * math.sqrt(2.0 / n))

    return output


def mfcc_frames_from_wav(
    path: str,
    num_ceps: int = 13,
    frame_size: int = 512,
    hop: int = 256,
) -> list[Vector]:
    """
    Coeficientes cepstrales de Mel (MFCC) por ventana de un archivo WAV.

    Cada ventana produce un descriptor local de num_ceps dimensiones, que es
    la entrada del pipeline BoAW. Se usa el modulo wave de la biblioteca
    estandar y una implementacion propia de FFT, banco de filtros de Mel y DCT.
    """
    if not os.path.exists(path):
        raise FeatureError(f'No existe el archivo de audio: {path}')

    with wave.open(path, 'rb') as handle:
        frames = handle.getnframes()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(frames)

    if width != 2:
        raise FeatureError('El audio debe ser PCM de 16 bits.')

    samples = struct.unpack(f'<{len(raw) // 2}h', raw)

    if channels > 1:
        mono = [
            sum(samples[index:index + channels]) / channels
            for index in range(0, len(samples), channels)
        ]
    else:
        mono = list(samples)

    if len(mono) < frame_size:
        mono = mono + [0] * (frame_size - len(mono))

    filterbank = mel_filterbank(sample_rate=rate)
    bins = frame_size // 2 + 1
    output: list[Vector] = []

    for start in range(0, len(mono) - frame_size + 1, hop):
        frame = mono[start:start + frame_size]
        spectrum = _fft(frame)[:bins]
        energies: list[float] = []

        for row in filterbank:
            total = 0.0

            for bin_index, value in enumerate(spectrum):
                power = value.real * value.real + value.imag * value.imag
                total += row[bin_index] * power

            energies.append(math.log(total + 1e-10))

        output.append(tuple(_dct(energies, num_ceps)))

    if not output:
        return [tuple([0.0] * num_ceps)]

    return output


def mfcc_from_wav(
    path: str,
    num_ceps: int = 13,
    frame_size: int = 512,
    hop: int = 256,
) -> Vector:
    """MFCC medio de todas las ventanas de un archivo WAV."""
    frames = mfcc_frames_from_wav(path, num_ceps, frame_size, hop)
    total = len(frames)

    return tuple(
        sum(frame[index] for frame in frames) / total
        for index in range(len(frames[0]))
    )


def _fft(samples: list[float]) -> list[complex]:
    """FFT radix-2 (potencia de dos) implementada de forma directa."""
    n = len(samples)

    if n == 0:
        return []

    if n & (n - 1) != 0:
        raise FeatureError('La FFT requiere un numero de muestras potencia de dos.')

    values = [complex(sample, 0.0) for sample in samples]

    j = 0

    for i in range(1, n):
        bit = n >> 1

        while j & bit:
            j ^= bit
            bit >>= 1

        j ^= bit

        if i < j:
            values[i], values[j] = values[j], values[i]

    length = 2

    while length <= n:
        angle = -2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))

        for start in range(0, n, length):
            factor = complex(1.0, 0.0)

            for offset in range(length // 2):
                even = values[start + offset]
                odd = factor * values[start + offset + length // 2]
                values[start + offset] = even + odd
                values[start + offset + length // 2] = even - odd
                factor *= step

        length <<= 1

    return values
