from __future__ import annotations

from threading import Lock
from time import perf_counter
from uuid import uuid4
from collections import OrderedDict

from fastapi import HTTPException
from engine.metadata import IndexKind, StorageKind
from storage.heapFile import HeapFile
from storage.sequentialFile import SequentialFile

from engine.catalog import Catalog
from engine.session import Session
from parser.ast import (
    BinaryExp,
    BinaryOp,
    IdExp,
    KnnExp,
    LiteralExp,
    SelectStm,
    UnaryExp,
    WithinRadiusExp,
)
from parser.parser import Parser
from parser.scanner import Scanner

from api import serializers


def _geo_indices(columns: list[str]) -> tuple[int, int] | None:
    names = [str(column).lower() for column in columns]
    latitude = next(
        (index for index, name in enumerate(names) if name in {'latitud', 'latitude', 'lat'}),
        None,
    )
    longitude = next(
        (index for index, name in enumerate(names) if name in {'longitud', 'longitude', 'lon', 'lng'}),
        None,
    )
    return (latitude, longitude) if latitude is not None and longitude is not None else None


def _label_index(columns: list[str], rows: list[list], geo: tuple[int, int]) -> int | None:
    skip = set(geo)
    for index in range(1, len(columns)):
        if index in skip:
            continue
        if any(
            isinstance(row[index], str) and row[index].strip()
            for row in rows[:20]
            if len(row) > index
        ):
            return index
    return None


class DatabaseService:
    """
    Punto de entrada del API hacia el motor.

    Serializa las consultas SQL, ejecuta el Program con una Session y
    convierte el ExecutionOutput a JSON listo para el consumidor. Un lock
    serializa las sentencias para que dos peticiones no muten a la vez el
    catalogo ni los archivos fisicos.
    """

    def __init__(
        self,
        catalog: Catalog,
        page_size: int,
        sort_buffer_size: int = 128 * 1024 * 1024,
        level_memory_limit: int = 64 * 1024 * 1024,
    ) -> None:
        self.catalog = catalog
        self.page_size = page_size
        self.sort_buffer_size = sort_buffer_size
        self.level_memory_limit = level_memory_limit
        self._lock = Lock()
        self._results = OrderedDict()

    def execute_sql(self, sql: str, page_size: int = 50) -> dict:
        with self._lock:
            start = perf_counter()
            try:
                program = Parser(Scanner(sql)).parse_program()
            except (SyntaxError, ValueError) as exc:
                raise HTTPException(400, str(exc)) from exc
            parsed = perf_counter()
            if not program.statements:
                raise HTTPException(400, 'Escribe al menos una sentencia SQL.')
            output = Session(
                self.catalog, self.page_size, self.sort_buffer_size, self.level_memory_limit,
            ).execute(program)
            payload = serializers.frontend_output(
                output, program, (parsed - start) * 1000, (perf_counter() - parsed) * 1000,
            )
            payload['spatial'] = self._spatial_metadata(program)
            return self._remember(payload, page_size)

    def _remember(self, payload: dict, page_size: int) -> dict:
        result_id = uuid4().hex
        payload['result_id'] = result_id
        payload['total_rows'] = len(payload['rows'])
        payload['total_pages'] = max(1, (len(payload['rows']) + page_size - 1) // page_size)
        self._results[result_id] = (payload, page_size)
        # Bound retained snapshots; pagination never executes SQL again.
        while len(self._results) > 4:
            self._results.popitem(last=False)
        return self._page(result_id, 1)

    def _page(self, result_id: str, page: int) -> dict:
        if result_id not in self._results:
            raise HTTPException(404, 'El resultado expiró. Ejecuta nuevamente la consulta.')
        payload, size = self._results[result_id]
        if page < 1 or page > payload['total_pages']:
            raise HTTPException(400, 'Página fuera de rango.')
        start = (page - 1) * size
        return {**payload, 'page': page, 'rows': payload['rows'][start:start + size]}

    def result_page(self, result_id: str, page: int) -> dict:
        with self._lock:
            return self._page(result_id, page)

    def result_map(self, result_id: str, limit: int = 2000) -> dict:
        with self._lock:
            if result_id not in self._results:
                raise HTTPException(
                    404,
                    'El resultado expiró. Ejecuta nuevamente la consulta.',
                )

            payload, _ = self._results[result_id]
            columns = list(payload.get('columns') or [])
            rows = payload.get('rows') or []
            geo = _geo_indices(columns)

            if geo is None:
                return {
                    'points': [],
                    'shown': 0,
                    'total': len(rows),
                    'truncated': False,
                    'columns': columns,
                    'spatial': payload.get('spatial'),
                }

            latitude_index, longitude_index = geo
            label_index = _label_index(columns, rows, geo)
            points = []
            total = 0

            for row in rows:
                try:
                    latitude = float(row[latitude_index])
                    longitude = float(row[longitude_index])
                except (IndexError, TypeError, ValueError):
                    continue

                total += 1
                if len(points) >= limit:
                    continue

                label = ''
                if label_index is not None and row[label_index] is not None:
                    label = str(row[label_index])[:80]

                points.append([latitude, longitude, label])

            return {
                'points': points,
                'shown': len(points),
                'total': total,
                'truncated': total > len(points),
                'columns': columns,
                'lat_column': columns[latitude_index],
                'lon_column': columns[longitude_index],
                'spatial': payload.get('spatial'),
            }

    def _spatial_metadata(self, program) -> dict | None:
        statement = next(
            (
                item
                for item in reversed(program.statements)
                if isinstance(item, SelectStm)
            ),
            None,
        )

        if statement is None or statement.where is None:
            return None

        radius = self._find_ast(statement.where, WithinRadiusExp)
        if radius is not None:
            return {
                'kind': 'radius',
                'center': {
                    'lat': radius.center_lat,
                    'lon': radius.center_lon,
                },
                'radius_km': radius.radius_km,
                'latitude': radius.latitude,
                'longitude': radius.longitude,
            }

        knn = self._find_ast(statement.where, KnnExp)
        if knn is not None:
            return {
                'kind': 'knn',
                'center': {
                    'lat': knn.center_lat,
                    'lon': knn.center_lon,
                },
                'k': knn.k,
                'latitude': knn.latitude,
                'longitude': knn.longitude,
            }

        index = next(
            (
                item
                for item in self.catalog.get_table_indexes(statement.table)
                if item.kind is IndexKind.RTREE and item.spatial_columns
            ),
            None,
        )
        if index is None or index.spatial_columns is None:
            return None

        latitude_column, longitude_column = index.spatial_columns
        bounds = {
            latitude_column: [None, None],
            longitude_column: [None, None],
        }
        self._collect_bounds(statement.where, bounds)

        latitude_bounds = bounds[latitude_column]
        longitude_bounds = bounds[longitude_column]

        if (
            latitude_bounds[0] is None
            or latitude_bounds[1] is None
            or longitude_bounds[0] is None
            or longitude_bounds[1] is None
        ):
            return None

        return {
            'kind': 'rect',
            'bbox': {
                'lat_min': latitude_bounds[0],
                'lat_max': latitude_bounds[1],
                'lon_min': longitude_bounds[0],
                'lon_max': longitude_bounds[1],
            },
            'latitude': latitude_column,
            'longitude': longitude_column,
        }

    @staticmethod
    def _find_ast(node, target):
        if isinstance(node, target):
            return node

        if isinstance(node, BinaryExp):
            return (
                DatabaseService._find_ast(node.left, target)
                or DatabaseService._find_ast(node.right, target)
            )

        if isinstance(node, UnaryExp):
            return DatabaseService._find_ast(node.value, target)

        return None

    @staticmethod
    def _collect_bounds(node, bounds: dict[str, list[float | None]]) -> None:
        if isinstance(node, BinaryExp) and node.op is BinaryOp.AND:
            DatabaseService._collect_bounds(node.left, bounds)
            DatabaseService._collect_bounds(node.right, bounds)
            return

        if not isinstance(node, BinaryExp):
            return

        left = node.left
        right = node.right
        operator = node.op

        if isinstance(right, IdExp) and isinstance(left, LiteralExp):
            left, right = right, left
            operator = {
                BinaryOp.GT: BinaryOp.LT,
                BinaryOp.GE: BinaryOp.LE,
                BinaryOp.LT: BinaryOp.GT,
                BinaryOp.LE: BinaryOp.GE,
            }.get(operator, operator)

        if not isinstance(left, IdExp) or not isinstance(right, LiteralExp):
            return

        if left.value not in bounds:
            return

        value = right.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return

        value = float(value)

        if operator in (BinaryOp.GT, BinaryOp.GE, BinaryOp.EQ):
            current = bounds[left.value][0]
            bounds[left.value][0] = value if current is None else max(current, value)

        if operator in (BinaryOp.LT, BinaryOp.LE, BinaryOp.EQ):
            current = bounds[left.value][1]
            bounds[left.value][1] = value if current is None else min(current, value)

    def _storage(self, entry):
        kind = SequentialFile if entry.storage is StorageKind.SEQUENTIAL else HeapFile
        return kind(entry.metadata, entry.file_path, self.page_size)

    def tables(self) -> dict:
        with self._lock:
            tables = []
            for entry in self.catalog.tables():
                storage = self._storage(entry)
                try:
                    tables.append({
                        **serializers.table_to_dict(entry), 'records': storage.count(),
                        'indexes': [serializers.index_to_dict(index) for index in self.catalog.get_table_indexes(entry.name)],
                    })
                finally:
                    storage.close()
            return {'tables': tables, 'page_size': self.page_size}

    def indexes(self) -> dict:
        with self._lock:
            return serializers.catalog_indexes(self.catalog)

    def reorganize(self, table: str) -> dict:
        with self._lock:
            if not self.catalog.table_exists(table):
                raise HTTPException(404, 'La tabla no existe.')
            entry = self.catalog.get_table(table)
            if entry.storage is not StorageKind.SEQUENTIAL:
                raise HTTPException(400, 'Solo se pueden reorganizar tablas SEQUENTIAL.')
            storage = self._storage(entry)
            try:
                start = perf_counter()
                storage.reset_stats()
                storage.reorganize()
                elapsed = (perf_counter() - start) * 1000
                payload = {
                    'success': True, 'columns': [], 'rows': [], 'table': table,
                    'message': 'Tabla reorganizada correctamente.', 'plan': 'SequentialReorganize',
                    'affected_rows': storage.count(), 'statements': 1,
                    'metrics': {'disk_reads': storage.page_reads, 'disk_writes': storage.page_writes,
                                'parse_ms': 0, 'execution_ms': elapsed},
                }
                return self._remember(payload, 50)
            finally:
                storage.close()
