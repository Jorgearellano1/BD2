from __future__ import annotations

from threading import Lock
from time import perf_counter
from uuid import uuid4
from collections import OrderedDict

from fastapi import HTTPException
from engine.metadata import StorageKind
from storage.heapFile import HeapFile
from storage.sequentialFile import SequentialFile

from engine.catalog import Catalog
from engine.session import Session
from parser.parser import Parser
from parser.scanner import Scanner

from api import serializers


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
