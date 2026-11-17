"""
Indice vectorial aproximado IVF en disco.

IVF (Inverted File Flat) particiona el espacio vectorial en C celdas de
Voronoi obtenidas con K-Means y guarda, para cada celda, la lista de los
vectores asignados. Una consulta visita solo las celdas cuyos centroides
estan mas cerca del vector de consulta (probes) y evita el resto.

Estructura fisica (directorio del indice):

    centroids.ctr
        pagina 0        cabecera: magic, B, dimension, numero de celdas,
                        metrica, numero de vectores y marca de desactualizado.
        paginas 1..P    entradas <cluster_id u32><head_page i64><centroide f32*dim>

    lists.lst
        pagina 0        cabecera: magic, B, dimension, numero de celdas.
        paginas         listas paginadas: <next_page i64><count u16> seguido de
                        <page_id i64><slot i64><vector f32*dim>

Cada celda es una cadena de paginas (next_page) que termina en -1. Todas las
lecturas y escrituras se hacen con seek + read/write sobre bloques de B
bytes, igual que el resto de estructuras del motor.
"""

from __future__ import annotations

import heapq
import math
import os
import shutil
import struct
from dataclasses import dataclass
from time import perf_counter

from vectorindex.kmeans import kmeans
from vectorindex.metrics import Metric, metric_by_name
from vectorindex.pagedfile import PagedFile, PagedFileError
from vectorindex.vectors import Vector

CENTROIDS_FILE = 'centroids.ctr'
LISTS_FILE = 'lists.lst'

CENTROIDS_MAGIC = b'BD2IVF01'
LISTS_MAGIC = b'BD2IVFL1'

# magic, page_size, dim, lists, metric, num_vectors, dirty
CENTROIDS_HEADER = struct.Struct('<8sIIIIQI')
# magic, page_size, dim, lists
LISTS_HEADER = struct.Struct('<8sIII')
# next_page, count
LIST_PAGE_HEADER = struct.Struct('<qH')
# cluster_id, head_page
CENTROID_ENTRY = struct.Struct('<Iq')

METRIC_CODE = {
    Metric.L2: 1,
    Metric.COSINE: 2,
    Metric.IP: 3,
    Metric.DTW: 4,
}

CODE_METRIC = {code: metric for metric, code in METRIC_CODE.items()}


class IvfError(RuntimeError):
    """Error al construir, abrir o consultar el indice IVF."""


@dataclass
class IvfBuildStats:
    """Metricas de la construccion del indice."""

    num_vectors: int
    lists: int
    dimension: int
    pages_written: int
    pages_read: int
    bytes_centroids: int
    bytes_lists: int
    elapsed_ms: float


@dataclass
class IvfSearchStats:
    """Metricas de una consulta sobre el indice."""

    cells_probed: int
    candidates: int
    pages_read: int
    elapsed_ms: float


class IvfIndex:
    """Indice vectorial aproximado IVF persistido en disco."""

    def __init__(
        self,
        directory: str,
        dimension: int,
        metric: Metric = Metric.COSINE,
        page_size: int = 4096,
    ) -> None:
        self.directory = directory
        self.dimension = dimension
        self.metric = metric
        self.page_size = page_size

        self.lists = 0
        self.num_vectors = 0
        self.dirty = False

        self._centroids: PagedFile | None = None
        self._lists_file: PagedFile | None = None

    # ------------------------------------------------------------------
    # Rutas
    # ------------------------------------------------------------------

    @property
    def centroids_path(self) -> str:
        return os.path.join(self.directory, CENTROIDS_FILE)

    @property
    def lists_path(self) -> str:
        return os.path.join(self.directory, LISTS_FILE)

    # ------------------------------------------------------------------
    # Construccion
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        directory: str,
        vectors: list[tuple[tuple[int, int], Vector]],
        lists: int = 25,
        metric: Metric = Metric.COSINE,
        page_size: int = 4096,
        iterations: int = 10,
        seed: int = 42,
    ) -> tuple['IvfIndex', IvfBuildStats]:
        """
        Construye el indice a partir de los vectores de la tabla.

        vectors es una lista de (RID, vector) con RID = (page_id, slot).
        """
        if not vectors:
            raise IvfError('No hay vectores para indexar.')

        dimension = len(vectors[0][1])

        if any(len(vector) != dimension for _, vector in vectors):
            raise IvfError(
                'Todos los vectores del indice deben tener la misma dimension.'
            )

        if lists <= 0:
            raise IvfError('El numero de celdas (lists) debe ser positivo.')

        if lists > len(vectors):
            raise IvfError(
                f'El numero de celdas ({lists}) no puede superar la cantidad '
                f'de vectores ({len(vectors)}).'
            )

        if 16 + 4 * dimension + LIST_PAGE_HEADER.size > page_size:
            raise IvfError(
                f'El tamano de pagina {page_size} no permite almacenar un '
                f'vector de {dimension} dimensiones.'
            )

        if CENTROID_ENTRY.size + 4 * dimension + 2 > page_size:
            raise IvfError(
                f'El tamano de pagina {page_size} no permite almacenar un '
                f'centroide de {dimension} dimensiones.'
            )

        start = perf_counter()

        work = directory + '.building'

        if os.path.exists(work):
            shutil.rmtree(work)

        os.makedirs(work)

        points = [vector for _, vector in vectors]
        centroids, assignments = kmeans(
            points, lists, iterations=iterations, seed=seed,
        )

        cells: list[list[tuple[tuple[int, int], Vector]]] = [
            [] for _ in range(lists)
        ]

        for item, group in zip(vectors, assignments):
            cells[group].append(item)

        lists_file = PagedFile(os.path.join(work, LISTS_FILE), page_size, 'w+b')
        lists_file.write_page(
            0, LISTS_HEADER.pack(LISTS_MAGIC, page_size, dimension, lists),
        )

        head_pages: list[int] = []
        next_page = 1
        entry_size = 16 + 4 * dimension
        per_page = max(1, (page_size - LIST_PAGE_HEADER.size) // entry_size)

        for cell in cells:
            if not cell:
                head_pages.append(-1)
                continue

            # Las paginas de la celda se escriben encadenadas; la primera es
            # la cabeza y las siguientes se referencian con next_page.
            page_ids: list[int] = []

            for start_index in range(0, len(cell), per_page):
                chunk = cell[start_index:start_index + per_page]
                page_ids.append(next_page)
                next_page += 1

            head_pages.append(page_ids[0])

            for position, page_id in enumerate(page_ids):
                chunk = cell[position * per_page:(position + 1) * per_page]
                following = page_ids[position + 1] if position + 1 < len(page_ids) else -1
                payload = bytearray(LIST_PAGE_HEADER.pack(following, len(chunk)))

                for rid, vector in chunk:
                    payload += struct.pack('<qq', rid[0], rid[1])
                    payload += struct.pack(f'<{dimension}f', *vector)

                lists_file.write_page(page_id, bytes(payload))

        lists_file.flush()
        pages_written = lists_file.writes
        lists_file.close()

        centroids_file = PagedFile(os.path.join(work, CENTROIDS_FILE), page_size, 'w+b')
        centroids_file.write_page(
            0,
            CENTROIDS_HEADER.pack(
                CENTROIDS_MAGIC, page_size, dimension, lists,
                METRIC_CODE[metric], len(vectors), 0,
            ),
        )

        entry_size = CENTROID_ENTRY.size + 4 * dimension
        per_page = max(1, (page_size - 2) // entry_size)
        centroids_page = 1
        entries: list[bytes] = []
        used = 2

        for cluster_id, centroid in enumerate(centroids):
            entry = CENTROID_ENTRY.pack(cluster_id, head_pages[cluster_id])
            entry += struct.pack(f'<{dimension}f', *centroid)

            if used + len(entry) > page_size:
                centroids_file.write_page(
                    centroids_page, struct.pack('<H', len(entries)) + b''.join(entries),
                )
                centroids_page += 1
                entries = []
                used = 2

            entries.append(entry)
            used += len(entry)

        if entries:
            centroids_file.write_page(
                centroids_page, struct.pack('<H', len(entries)) + b''.join(entries),
            )
            centroids_page += 1

        centroids_file.flush()
        pages_written += centroids_file.writes
        centroids_file.close()

        if os.path.exists(directory):
            old = directory + '.old'

            if os.path.exists(old):
                shutil.rmtree(old)

            os.replace(directory, old)
            os.replace(work, directory)
            shutil.rmtree(old)
        else:
            os.replace(work, directory)

        index = cls(directory, dimension, metric, page_size)
        index.lists = lists
        index.num_vectors = len(vectors)
        index.open()

        stats = IvfBuildStats(
            num_vectors=len(vectors),
            lists=lists,
            dimension=dimension,
            pages_written=pages_written,
            pages_read=0,
            bytes_centroids=os.path.getsize(index.centroids_path),
            bytes_lists=os.path.getsize(index.lists_path),
            elapsed_ms=(perf_counter() - start) * 1000.0,
        )

        return index, stats

    # ------------------------------------------------------------------
    # Apertura y estado
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Abre el indice leyendo solo la cabecera."""
        if not os.path.exists(self.centroids_path):
            raise FileNotFoundError(
                f'No existe el indice IVF: {self.directory}'
            )

        self._centroids = PagedFile(self.centroids_path, self.page_size, 'r+b')
        self._lists_file = PagedFile(self.lists_path, self.page_size, 'rb')

        header = self._centroids.read_page(0)
        magic, page_size, dimension, lists, metric, num_vectors, dirty = \
            CENTROIDS_HEADER.unpack_from(header, 0)

        if magic != CENTROIDS_MAGIC:
            raise IvfError('El indice IVF no tiene el formato esperado.')

        if page_size != self.page_size:
            raise IvfError(
                f'El indice fue construido con B={page_size} y se abrio con '
                f'B={self.page_size}.'
            )

        if dimension != self.dimension:
            raise IvfError(
                f'El indice IVF tiene dimension {dimension} y se abrio con '
                f'{self.dimension}.'
            )

        self.lists = lists
        self.num_vectors = num_vectors
        self.dirty = bool(dirty)

    def mark_dirty(self) -> None:
        """Marca el indice como desactualizado (1 escritura de pagina)."""
        if self.dirty or self._centroids is None:
            return

        self.dirty = True
        header = CENTROIDS_HEADER.pack(
            CENTROIDS_MAGIC, self.page_size, self.dimension, self.lists,
            METRIC_CODE[self.metric], self.num_vectors, 1,
        )
        self._centroids.write_page(0, header)
        self._centroids.flush()

    def close(self) -> None:
        for paged in (self._centroids, self._lists_file):
            if paged is not None:
                paged.close()

        self._centroids = None
        self._lists_file = None

    @property
    def page_reads(self) -> int:
        total = 0

        if self._centroids is not None:
            total += self._centroids.reads

        if self._lists_file is not None:
            total += self._lists_file.reads

        return total

    @property
    def page_writes(self) -> int:
        if self._centroids is None:
            return 0

        return self._centroids.writes

    def disk_size(self) -> dict[str, int]:
        return {
            CENTROIDS_FILE: os.path.getsize(self.centroids_path),
            LISTS_FILE: os.path.getsize(self.lists_path),
        }

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def _read_centroids(self) -> list[tuple[int, int, Vector]]:
        """Lee las paginas de centroides: (cluster_id, head_page, centroide)."""
        if self._centroids is None:
            raise IvfError('El indice IVF no esta abierto.')

        entries: list[tuple[int, int, Vector]] = []
        total_pages = self._centroids.num_pages()

        # La pagina 0 es la cabecera; las siguientes contienen los centroides.
        for page_no in range(1, total_pages):
            page = self._centroids.read_page(page_no)
            (count,) = struct.unpack_from('<H', page, 0)
            position = 2

            for _ in range(count):
                cluster_id, head_page = CENTROID_ENTRY.unpack_from(page, position)
                position += CENTROID_ENTRY.size
                values = struct.unpack_from(
                    f'<{self.dimension}f', page, position,
                )
                position += 4 * self.dimension
                entries.append((cluster_id, head_page, values))

        return entries

    def search(
        self,
        query: Vector,
        k: int,
        probes: int = 1,
        allowed_rids: set[tuple[int, int]] | None = None,
    ) -> tuple[list[tuple[tuple[int, int], float]], IvfSearchStats]:
        """
        Busca los k vectores mas cercanos al vector de consulta.

        Solo se visitan las 'probes' celdas con centroides mas cercanos. Si
        allowed_rids no es None, solo se consideran esos RID (prefiltrado
        relacional).
        """
        if self._centroids is None or self._lists_file is None:
            raise IvfError('El indice IVF no esta abierto.')

        if k <= 0:
            return [], IvfSearchStats(0, 0, self.page_reads, 0.0)

        start = perf_counter()
        entries = self._read_centroids()

        if not entries:
            return [], IvfSearchStats(0, 0, self.page_reads, 0.0)

        ranked = sorted(
            entries,
            key=lambda item: self._distance_to(query, item[2]),
        )

        selected = ranked[:max(1, min(probes, len(ranked)))]

        heap: list[tuple[float, tuple[int, int]]] = []
        candidates = 0

        for _, head_page, _ in selected:
            if head_page < 0:
                continue

            page_no = head_page

            while page_no >= 0:
                page = self._lists_file.read_page(page_no)
                next_page, count = LIST_PAGE_HEADER.unpack_from(page, 0)
                position = LIST_PAGE_HEADER.size

                for _ in range(count):
                    page_id, slot = struct.unpack_from('<qq', page, position)
                    position += 16
                    values = struct.unpack_from(
                        f'<{self.dimension}f', page, position,
                    )
                    position += 4 * self.dimension

                    rid = (page_id, slot)

                    if allowed_rids is not None and rid not in allowed_rids:
                        continue

                    candidates += 1
                    value = self._distance_to(query, values)

                    # Max-heap de tamano k: la raiz guarda la peor distancia
                    # (por eso se almacena el valor negado).
                    if len(heap) < k:
                        heapq.heappush(heap, (-value, rid))
                    elif value < -heap[0][0]:
                        heapq.heapreplace(heap, (-value, rid))

                page_no = next_page

        results = [(rid, -negative) for negative, rid in sorted(
            heap, key=lambda item: (-item[0], item[1]),
        )]
        stats = IvfSearchStats(
            cells_probed=len(selected),
            candidates=candidates,
            pages_read=self.page_reads,
            elapsed_ms=(perf_counter() - start) * 1000.0,
        )

        return results, stats

    def _distance_to(self, query: Vector, vector: Vector) -> float:
        """Distancia entre la consulta y un vector del indice."""
        if self.metric is Metric.L2:
            total = 0.0

            for a, b in zip(query, vector):
                difference = a - b
                total += difference * difference

            return math.sqrt(total)

        if self.metric is Metric.COSINE:
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

        if self.metric is Metric.IP:
            total = 0.0

            for a, b in zip(query, vector):
                total += a * b

            return -total

        if self.metric is Metric.DTW:
            from vectorindex.metrics import dtw_distance

            return dtw_distance(query, vector)

        raise IvfError(f"Metrica no soportada: {self.metric}")


def read_metric_code(code: int) -> Metric:
    """Convierte el codigo de metrica de la cabecera en su enum."""
    if code not in CODE_METRIC:
        raise IvfError(f'Codigo de metrica desconocido: {code}')

    return CODE_METRIC[code]


def metric_code(metric: Metric) -> int:
    """Codigo numerico de una metrica en la cabecera."""
    if metric not in METRIC_CODE:
        raise IvfError(f'Metrica no soportada: {metric}')

    return METRIC_CODE[metric]


__all__ = [
    'CENTROIDS_FILE',
    'IvfBuildStats',
    'IvfError',
    'IvfIndex',
    'IvfSearchStats',
    'LISTS_FILE',
    'PagedFileError',
    'metric_by_name',
    'metric_code',
    'read_metric_code',
]
