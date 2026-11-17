from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from engine.bitmap import Bitmap
from engine.catalog import Catalog
from engine.conditions import evaluate_expression
from engine.metadata import (
    CleanRecord,
    IndexKind,
    IndexMetadata,
    RawRecord,
    RID,
    StorageKind,
    TableEntry,
    TableMetadata,
)
from engine.plan import (
    AddUniquePlan,
    BitmapAndPlan,
    BitmapHeapScanPlan,
    BitmapIndexRangeScanPlan,
    BitmapIndexScanPlan,
    BitmapOrPlan,
    CreateIndexPlan,
    CreateTablePlan,
    DeletePlan,
    EmptyResultPlan,
    FilterPlan,
    FilterRidsPlan,
    HeapFetchPlan,
    IndexRangeScanPlan,
    IndexScanPlan,
    InsertPlan,
    PlanNode,
    ProgramPlan,
    ProjectPlan,
    SeqScanPlan,
    SeqScanRidsPlan,
    SequentialDeletePlan,
    SequentialRangeScanPlan,
    StatementInfo,
)
from engine.plan_visitor import PlanVisitor
from indexes.bplusTree import BPlusTree
from indexes.ExtendibleHash import ExtendibleHash
from rtree.rtree import RTree 
from rtree.rtreeNode import MBR
from indexes.index import Index
from storage.heapFile import HeapFile
from storage.sequentialFile import SequentialFile
from storage.storageFile import StorageFile


class ExecutorError(Exception):
    """Error de ejecucion fisica o de consistencia."""


class RollbackError(RuntimeError):
    """Fallo al revertir una sentencia."""


@dataclass(frozen=True)
class CounterSnapshot:
    reads: int
    writes: int


@dataclass
class OperationMetrics:
    resource_type: str
    resource_name: str
    operation: str
    elapsed_ms: float
    pages_read: int
    pages_written: int
    entries_affected: int = 0


@dataclass
class NodeMetrics:
    node_id: int
    node_type: str
    elapsed_ms: float
    self_time_ms: float
    rows_out: int
    pages_read: int
    pages_written: int
    self_pages_read: int
    self_pages_written: int
    operations: list[OperationMetrics] = field(default_factory=list)
    children: list["NodeMetrics"] = field(default_factory=list)


@dataclass
class ExecutionValue:
    records: list[RawRecord] | list[CleanRecord] | None = None
    rids: list[RID] | None = None
    bitmap: Bitmap | None = None
    affected_rows: int | None = None
    output_columns: list[str] | None = None
    records_clean: bool = False


@dataclass
class UndoAction:
    description: str
    undo: Callable[[], None]


class RollbackJournal:
    def __init__(self) -> None:
        self.actions: list[UndoAction] = []

    def add(self, description: str, undo: Callable[[], None]) -> None:
        self.actions.append(UndoAction(description, undo))

    def rollback(self) -> None:
        errors: list[str] = []
        for action in reversed(self.actions):
            try:
                action.undo()
            except Exception as exc:
                errors.append(f"{action.description}: {exc}")
        self.actions.clear()
        if errors:
            raise RollbackError("Fallo el rollback:\n" + "\n".join(errors))

    def commit(self) -> None:
        self.actions.clear()


@dataclass
class FileSnapshot:
    original_path: str
    snapshot_path: str


@dataclass
class StatementResult:
    statement_id: int
    statement_type: str
    columns: list[str]
    rows: list[CleanRecord]
    affected_rows: int | None
    message: str | None = None


@dataclass
class StatementExplain:
    statement_id: int
    statement_type: str
    root: NodeMetrics | None
    success: bool = True
    error: str | None = None
    error_type: str | None = None


@dataclass
class ExecutionOutput:
    explains: list[StatementExplain]
    results: list[StatementResult]
    success: bool = True
    error: str | None = None
    error_type: str | None = None
    failed_statement_id: int | None = None


@dataclass
class ExecutionContext:
    catalog: Catalog
    page_size: int
    sort_buffer_size: int = 128 * 1024 * 1024
    level_memory_limit: int = 64 * 1024 * 1024

    storage_cache: dict[str, StorageFile] = field(default_factory=dict)
    index_cache: dict[str, Index | RTree] = field(default_factory=dict)

    next_node_id: int = 1
    node_metrics: dict[int, NodeMetrics] = field(default_factory=dict)

    current_statement_id: int | None = None
    transaction: RollbackJournal | None = None

    io_reads: int = 0
    io_writes: int = 0


@dataclass
class _NodeFrame:
    node_id: int
    node_type: str
    start: float
    reads_before: int
    writes_before: int
    operations: list[OperationMetrics] = field(default_factory=list)
    children: list[NodeMetrics] = field(default_factory=list)


class Executor(PlanVisitor[ExecutionValue]):
    """
    Ejecuta un ProgramPlan de forma determinista sobre las estructuras
    fisicas, manteniendo el Heap y todos sus indices consistentes.

    Los accesos entregan RawRecord. ProjectPlan y el cierre de SELECT
    producen CleanRecord.
    """

    def __init__(self, context: ExecutionContext) -> None:
        self.ctx = context
        self._stack: list[_NodeFrame] = []
        self._last_metrics: NodeMetrics | None = None

    # Programa

    def execute(self, program: ProgramPlan) -> ExecutionOutput:
        explains: list[StatementExplain] = []
        results: list[StatementResult] = []

        try:
            for statement, info in zip(program.statements, program.statement_infos):
                explain, result = self.execute_statement(statement, info)
                explains.append(explain)

                if result is None:
                    return ExecutionOutput(
                        explains,
                        results,
                        success=False,
                        error=explain.error,
                        error_type=explain.error_type,
                        failed_statement_id=explain.statement_id,
                    )

                results.append(result)

            return ExecutionOutput(explains, results)
        finally:
            self.close()

    def execute_statement(
        self,
        statement: PlanNode,
        info: StatementInfo,
    ) -> tuple[StatementExplain, StatementResult | None]:
        self.ctx.current_statement_id = info.statement_id
        self.ctx.transaction = RollbackJournal()
        self._last_metrics = None

        try:
            value = statement.accept(self)
        except Exception as exc:
            error = exc
            try:
                self.ctx.transaction.rollback()
            except RollbackError as rb:
                error = ExecutorError(f"{exc}; {rb}")

            explain = StatementExplain(
                info.statement_id,
                info.statement_type,
                self._last_metrics,
                False,
                str(error),
                type(error).__name__,
            )
            return explain, None

        self.ctx.transaction.commit()
        explain = StatementExplain(
            info.statement_id,
            info.statement_type,
            self._last_metrics,
            True,
            None,
        )
        return explain, self._statement_result(info, value)

    def visit_program(self, node: ProgramPlan) -> ExecutionValue:
        output = self.execute(node)
        return ExecutionValue(affected_rows=len(output.results))

    # Accesos

    def visit_seq_scan(self, node: SeqScanPlan) -> ExecutionValue:
        storage = self._get_storage(node.table)

        def body() -> ExecutionValue:
            records = list(self._iter_raw_records(storage))
            if node.predicate is not None:
                meta = self._table_meta(node.table)
                records = [
                    record
                    for record in records
                    if evaluate_expression(record, node.predicate, meta)
                ]
            return ExecutionValue(records=records, records_clean=False)

        return self._execute_node(
            node,
            "SeqScanPlan",
            lambda: self._measure(
                storage,
                self._resource_type(storage),
                node.table,
                "seq_scan",
                body,
                entries_fn=lambda value: len(value.records or []),
            ),
            rows_out_fn=lambda value: len(value.records or []),
        )

    def visit_sequential_range_scan(
        self,
        node: SequentialRangeScanPlan,
    ) -> ExecutionValue:
        storage = self._get_storage(node.table)

        def body() -> ExecutionValue:
            records = storage.range_search_raw(
                node.column,
                node.range.lower,
                node.range.upper,
                node.range.include_lower,
                node.range.include_upper,
            )
            return ExecutionValue(records=records, records_clean=False)

        return self._execute_node(
            node,
            "SequentialRangeScanPlan",
            lambda: self._measure(
                storage,
                self._resource_type(storage),
                node.table,
                "range_search",
                body,
                entries_fn=lambda value: len(value.records or []),
            ),
            rows_out_fn=lambda value: len(value.records or []),
        )

    def visit_seq_scan_rids(self, node: SeqScanRidsPlan) -> ExecutionValue:
        storage = self._get_storage(node.table)

        def body() -> ExecutionValue:
            rids: list[RID] = []
            for page_records in storage.iter_pages_with_records():
                for rid, _ in page_records:
                    rids.append(rid)
            return ExecutionValue(rids=rids)

        return self._execute_node(
            node,
            "SeqScanRidsPlan",
            lambda: self._measure(
                storage,
                self._resource_type(storage),
                node.table,
                "seq_scan_rids",
                body,
                entries_fn=lambda value: len(value.rids or []),
            ),
            rows_out_fn=lambda value: len(value.rids or []),
        )

    def visit_index_scan(self, node: IndexScanPlan) -> ExecutionValue:
        index = self._get_index(node.index)

        def body() -> ExecutionValue:
            if node.index_kind is IndexKind.RTREE:
                if not isinstance(node.key, MBR):
                    raise ExecutorError(
                        "La búsqueda R-Tree requiere un MBR como consulta."
                    )

                rids = index.search_intersects(node.key)

            else:
                rids = index.search_all(node.key)

            return ExecutionValue(rids=rids)

        return self._execute_node(
            node,
            "IndexScanPlan",
            lambda: self._measure(
                index,
                self._index_resource_type(index),
                node.index,
                "search",
                body,
                entries_fn=lambda value: len(value.rids or []),
            ),
            rows_out_fn=lambda value: len(value.rids or []),
        )

    def visit_index_range_scan(
        self,
        node: IndexRangeScanPlan,
    ) -> ExecutionValue:

        index = self._get_index(node.index)

        if isinstance(index, RTree):
            raise ExecutorError(
                "El R-Tree no admite búsquedas escalares por rango. "
                "Utiliza una consulta espacial mediante MBR."
            )

        def body() -> ExecutionValue:
            rids = index.range_search(
                node.range.lower,
                node.range.upper,
                node.range.include_lower,
                node.range.include_upper,
            )
            return ExecutionValue(rids=rids)

        return self._execute_node(
            node,
            "IndexRangeScanPlan",
            lambda: self._measure(
                index,
                self._index_resource_type(index),
                node.index,
                "range_search",
                body,
                entries_fn=lambda value: len(value.rids or []),
            ),
            rows_out_fn=lambda value: len(value.rids or []),
        )

    def visit_bitmap_index_scan(
        self,
        node: BitmapIndexScanPlan,
    ) -> ExecutionValue:
        index = self._get_index(node.index)

        if isinstance(index, RTree):
            raise ExecutorError(
                "El R-Tree no admite búsquedas Bitmap por igualdad. "
                "Utiliza una consulta espacial mediante MBR."
            )

        def body() -> ExecutionValue:
            bitmap = Bitmap()

            for rid in index.search_all(node.key):
                bitmap.add(rid)

            return ExecutionValue(bitmap=bitmap)

        return self._execute_node(
            node,
            "BitmapIndexScanPlan",
            lambda: self._measure(
                index,
                self._index_resource_type(index),
                node.index,
                "search",
                body,
                entries_fn=lambda value: self._bitmap_size(value.bitmap),
            ),
            rows_out_fn=lambda value: self._bitmap_size(value.bitmap),
        )

    def visit_bitmap_index_range_scan(
        self,
        node: BitmapIndexRangeScanPlan,
    ) -> ExecutionValue:
        index = self._get_index(node.index)

        if isinstance(index, RTree):
            raise ExecutorError(
                "El R-Tree no admite búsquedas Bitmap por rango escalar. "
                "Utiliza una consulta espacial mediante MBR."
            )

        def body() -> ExecutionValue:
            bitmap = Bitmap()

            rids = index.range_search(
                node.range.lower,
                node.range.upper,
                node.range.include_lower,
                node.range.include_upper,
            )

            for rid in rids:
                bitmap.add(rid)

            return ExecutionValue(bitmap=bitmap)

        return self._execute_node(
            node,
            "BitmapIndexRangeScanPlan",
            lambda: self._measure(
                index,
                self._index_resource_type(index),
                node.index,
                "range_search",
                body,
                entries_fn=lambda value: self._bitmap_size(value.bitmap),
            ),
            rows_out_fn=lambda value: self._bitmap_size(value.bitmap),
        )

    def visit_bitmap_and(self, node: BitmapAndPlan) -> ExecutionValue:
        def body() -> ExecutionValue:
            values = [child.accept(self) for child in node.children]
            present = [value.bitmap for value in values if value.bitmap is not None]
            if not present:
                return ExecutionValue(bitmap=Bitmap())
            result = present[0]
            for bitmap in present[1:]:
                result = result & bitmap
            return ExecutionValue(bitmap=result)

        return self._execute_node(
            node,
            "BitmapAndPlan",
            body,
            rows_out_fn=lambda value: self._bitmap_size(value.bitmap),
        )

    def visit_bitmap_or(self, node: BitmapOrPlan) -> ExecutionValue:
        def body() -> ExecutionValue:
            values = [child.accept(self) for child in node.children]
            present = [value.bitmap for value in values if value.bitmap is not None]
            if not present:
                return ExecutionValue(bitmap=Bitmap())
            result = present[0]
            for bitmap in present[1:]:
                result = result | bitmap
            return ExecutionValue(bitmap=result)

        return self._execute_node(
            node,
            "BitmapOrPlan",
            body,
            rows_out_fn=lambda value: self._bitmap_size(value.bitmap),
        )

    def visit_heap_fetch(self, node: HeapFetchPlan) -> ExecutionValue:
        storage = self._get_storage(node.table)

        def body() -> ExecutionValue:
            child = node.child.accept(self)
            if child.rids is not None:
                records = self._measure(
                    storage,
                    self._resource_type(storage),
                    node.table,
                    "read_many",
                    lambda: storage.read_many(list(child.rids)),
                    entries_fn=lambda value: len(value),
                )
            elif child.bitmap is not None:
                records = self._measure(
                    storage,
                    self._resource_type(storage),
                    node.table,
                    "read_bitmap",
                    lambda: storage.read_bitmap(child.bitmap),
                    entries_fn=lambda value: len(value),
                )
            else:
                records = []
            return ExecutionValue(records=records, records_clean=False)

        return self._execute_node(
            node,
            "HeapFetchPlan",
            body,
            rows_out_fn=lambda value: len(value.records or []),
        )

    def visit_bitmap_heap_scan(
        self,
        node: BitmapHeapScanPlan,
    ) -> ExecutionValue:
        storage = self._get_storage(node.table)

        def body() -> ExecutionValue:
            child = node.child.accept(self)
            if child.bitmap is not None:
                records = self._measure(
                    storage,
                    self._resource_type(storage),
                    node.table,
                    "read_bitmap",
                    lambda: storage.read_bitmap(child.bitmap),
                    entries_fn=lambda value: len(value),
                )
            else:
                records = []
            return ExecutionValue(records=records, records_clean=False)

        return self._execute_node(
            node,
            "BitmapHeapScanPlan",
            body,
            rows_out_fn=lambda value: len(value.records or []),
        )

    # Residual y proyeccion

    def visit_filter(self, node: FilterPlan) -> ExecutionValue:
        def body() -> ExecutionValue:
            child = node.child.accept(self)
            meta = self._table_meta(self._table_of(node))
            result = [
                record
                for record in list(child.records or [])
                if evaluate_expression(record, node.predicate, meta)
            ]
            return ExecutionValue(
                records=result,
                records_clean=child.records_clean,
            )

        return self._execute_node(
            node,
            "FilterPlan",
            body,
            rows_out_fn=lambda value: len(value.records or []),
        )

    def visit_filter_rids(self, node: FilterRidsPlan) -> ExecutionValue:
        storage = self._get_storage(node.table)

        def body() -> ExecutionValue:
            child = node.child.accept(self)
            if child.rids is not None:
                rids = list(child.rids)
            elif child.bitmap is not None:
                rids = list(child.bitmap.iter_rids())
            else:
                rids = []

            if not rids:
                return ExecutionValue(rids=[])

            meta = self._table_meta(node.table)
            pairs = self._measure(
                storage,
                self._resource_type(storage),
                node.table,
                "read_many",
                lambda: storage.read_many_with_rids(rids),
                entries_fn=lambda value: len(value),
            )
            survivors = [
                rid
                for rid, record in pairs
                if evaluate_expression(record, node.predicate, meta)
            ]
            return ExecutionValue(rids=survivors)

        return self._execute_node(
            node,
            "FilterRidsPlan",
            body,
            rows_out_fn=lambda value: len(value.rids or []),
        )

    def visit_project(self, node: ProjectPlan) -> ExecutionValue:
        def body() -> ExecutionValue:
            child = node.child.accept(self)
            records = list(child.records or [])
            if not records:
                return ExecutionValue(
                    records=[],
                    output_columns=list(node.columns),
                    records_clean=True,
                )
            meta = self._table_meta(self._table_of(node))
            indexes = [meta.col_index(column) for column in node.columns]
            projected: list[CleanRecord] = []
            for record in records:
                cleaned = meta.clean_tuple(record)
                if cleaned is None:
                    continue
                projected.append(tuple(cleaned[index] for index in indexes))
            return ExecutionValue(
                records=projected,
                output_columns=list(node.columns),
                records_clean=True,
            )

        return self._execute_node(
            node,
            "ProjectPlan",
            body,
            rows_out_fn=lambda value: len(value.records or []),
        )

    def visit_empty_result(self, node: EmptyResultPlan) -> ExecutionValue:
        return self._execute_node(
            node,
            "EmptyResultPlan",
            lambda: ExecutionValue(records=[], affected_rows=0),
            rows_out_fn=lambda value: 0,
        )

    # Operaciones de datos

    def visit_insert(self, node: InsertPlan) -> ExecutionValue:
        entry = self.ctx.catalog.get_table(node.table)
        meta = entry.metadata
        record = meta.normalize_tuple(list(node.values))

        if entry.storage is StorageKind.SEQUENTIAL:
            storage = self._get_storage(node.table)

            def body() -> ExecutionValue:
                self._measure(
                    storage,
                    "SequentialFile",
                    node.table,
                    "insert",
                    lambda: storage.insert(record),
                    entries_fn=lambda value: 1,
                )
                return ExecutionValue(affected_rows=1)

            return self._execute_node(
                node, "InsertPlan", body, rows_out_fn=lambda value: 1
            )

        storage = self._get_storage(node.table)
        indexes = self._table_indexes(node.table)

        def body() -> ExecutionValue:
            # Validar restricciones de unicidad en índices escalares.
            for info in indexes:
                if info.kind is IndexKind.RTREE:
                    continue

                if not meta.is_unique_column(info.column):
                    continue

                index = self._get_index(info.name)
                key = record[meta.col_index(info.column)]

                if index.search(key) is not None:
                    raise ExecutorError(f"Clave duplicada: {info.column}")

            rid = self._measure(
                storage,
                "HeapFile",
                node.table,
                "insert",
                lambda: storage.insert(record),
                entries_fn=lambda value: 1,
            )

            inserted: list[tuple[object, object, RID, bool]] = []

            try:
                for info in indexes:
                    index = self._get_index(info.name)

                    if info.kind is IndexKind.RTREE:
                        if info.spatial_columns is None:
                            raise ExecutorError(
                                f"El índice R-Tree '{info.name}' "
                                "no tiene columnas espaciales."
                            )

                        latitude_column, longitude_column = info.spatial_columns

                        latitude = float(
                            record[meta.col_index(latitude_column)]
                        )
                        longitude = float(
                            record[meta.col_index(longitude_column)]
                        )

                        mbr = MBR(
                            longitude,
                            latitude,
                            longitude,
                            latitude,
                        )

                        ok = self._measure(
                            index,
                            "RTree",
                            info.name,
                            "insert",
                            lambda index=index, mbr=mbr, rid=rid:
                                index.insert(mbr, rid),
                            entries_fn=lambda value: 1,
                        )

                        if not ok:
                            raise ExecutorError(
                                f"No se pudo insertar en el índice R-Tree "
                                f"'{info.name}'."
                            )

                        inserted.append((index, mbr, rid, True))

                    else:
                        key = record[meta.col_index(info.column)]

                        ok = self._measure(
                            index,
                            self._index_resource_type(index),
                            info.name,
                            "insert",
                            lambda index=index, key=key, rid=rid:
                                index.insert(key, rid),
                            entries_fn=lambda value: 1,
                        )

                        if not ok:
                            raise ExecutorError(
                                f"Clave duplicada: {info.column}"
                            )

                        inserted.append((index, key, rid, False))

            except Exception:
                for index, key_or_mbr, inserted_rid, is_rtree in reversed(inserted):
                    if is_rtree:
                        index.delete(key_or_mbr, inserted_rid)
                    else:
                        index.delete(key_or_mbr, inserted_rid)

                storage.remove(rid)
                raise

            return ExecutionValue(affected_rows=1)

        return self._execute_node(
            node, "InsertPlan", body, rows_out_fn=lambda value: 1
        )

    def visit_delete(self, node: DeletePlan) -> ExecutionValue:
        def body() -> ExecutionValue:
            child = node.child.accept(self)

            if child.rids is not None:
                rids = list(child.rids)
            elif child.bitmap is not None:
                rids = list(child.bitmap.iter_rids())
            else:
                rids = []

            if not rids:
                return ExecutionValue(records=[], affected_rows=0)

            storage = self._get_storage(node.table)
            meta = self._table_meta(node.table)
            indexes = self._table_indexes(node.table)

            pairs = self._measure(
                storage,
                "HeapFile",
                node.table,
                "read_many",
                lambda: storage.read_many_with_rids(rids),
                entries_fn=lambda value: len(value),
            )

            target_rids = [rid for rid, _ in pairs]
            records = [record for _, record in pairs]

            removed_entries: list[tuple[object, object, RID, bool]] = []

            try:
                for info in indexes:
                    index = self._get_index(info.name)

                    if info.kind is IndexKind.RTREE:
                        if info.spatial_columns is None:
                            raise ExecutorError(
                                f"El índice R-Tree '{info.name}' "
                                "no tiene columnas espaciales."
                            )

                        latitude_column, longitude_column = info.spatial_columns

                        latitude_index = meta.col_index(latitude_column)
                        longitude_index = meta.col_index(longitude_column)

                        for rid, record in zip(target_rids, records):
                            latitude = float(record[latitude_index])
                            longitude = float(record[longitude_index])

                            mbr = MBR(
                                longitude,
                                latitude,
                                longitude,
                                latitude,
                            )

                            self._measure(
                                index,
                                "RTree",
                                info.name,
                                "delete",
                                lambda index=index, mbr=mbr, rid=rid:
                                    index.delete(mbr, rid),
                                entries_fn=lambda value: 1,
                            )

                            removed_entries.append((index, mbr, rid, True))

                    else:
                        column_index = meta.col_index(info.column)

                        for rid, record in zip(target_rids, records):
                            key = record[column_index]

                            self._measure(
                                index,
                                self._index_resource_type(index),
                                info.name,
                                "delete",
                                lambda index=index, key=key, rid=rid:
                                    index.delete(key, rid),
                                entries_fn=lambda value: 1,
                            )

                            removed_entries.append((index, key, rid, False))

            except Exception:
                for index, key_or_mbr, rid, _ in reversed(removed_entries):
                    index.insert(key_or_mbr, rid)
                raise

            snapshot = self._snapshot_file(storage.filepath)

            try:
                deleted = self._measure(
                    storage,
                    "HeapFile",
                    node.table,
                    "remove_many",
                    lambda: storage.remove_many(target_rids),
                    entries_fn=lambda value: len(value),
                )

            except Exception:
                self._restore_storage_snapshot(node.table, snapshot)

                for index, key_or_mbr, rid, _ in reversed(removed_entries):
                    index.insert(key_or_mbr, rid)

                raise

            self._discard_snapshot(snapshot)

            return ExecutionValue(
                records=deleted,
                affected_rows=len(deleted),
            )

        return self._execute_node(
            node,
            "DeletePlan",
            body,
            rows_out_fn=lambda value: value.affected_rows or 0,
        )

    def visit_sequential_delete(
        self,
        node: SequentialDeletePlan,
    ) -> ExecutionValue:
        storage = self._get_storage(node.table)
        snapshot = self._snapshot_file(storage.filepath)

        def body() -> ExecutionValue:
            try:
                if node.predicate is None:
                    deleted = self._measure(
                        storage,
                        "SequentialFile",
                        node.table,
                        "reset",
                        lambda: self._reset_sequential(storage),
                        entries_fn=lambda value: len(value),
                    )
                else:
                    deleted = self._measure(
                        storage,
                        "SequentialFile",
                        node.table,
                        "delete_all",
                        lambda: storage.delete_all(node.predicate),
                        entries_fn=lambda value: len(value),
                    )
            except Exception:
                self._restore_storage_snapshot(node.table, snapshot)
                raise

            self._discard_snapshot(snapshot)
            return ExecutionValue(records=deleted, affected_rows=len(deleted))

        return self._execute_node(
            node,
            "SequentialDeletePlan",
            body,
            rows_out_fn=lambda value: value.affected_rows or 0,
        )

    @staticmethod
    def _reset_sequential(storage: SequentialFile):
        records = list(storage.iter_records())
        storage.reset()
        return records

    # Definicion de estructuras

    def visit_create_table(self, node: CreateTablePlan) -> ExecutionValue:
        metadata = node.metadata
        if metadata is None:
            if node.csv_path is None:
                raise ExecutorError(
                    f"CREATE TABLE '{node.name}' sin columnas y sin CSV"
                )
            metadata = self._infer_csv_metadata(
                node.name,
                node.csv_path,
                node.delimiter,
                node.primary_key,
            )

        file_path = self.ctx.catalog.table_path(node.name, node.storage)
        if os.path.exists(file_path):
            raise ExecutorError(f"El archivo ya existe: {file_path}")

        unique_columns = [
            column["name"]
            for column in metadata.columns
            if metadata.is_unique_column(column["name"])
        ]

        created_files: list[str] = [file_path]
        index_objects: list[tuple[IndexMetadata, Index]] = []
        holder: dict[str, StorageFile | None] = {"storage": None}

        def body() -> ExecutionValue:
            storage: StorageFile

            if node.storage is StorageKind.SEQUENTIAL:
                storage = SequentialFile(
                    metadata, file_path, self.ctx.page_size
                )
                holder["storage"] = storage
                if node.csv_path is not None:
                    self._measure(
                        storage,
                        "SequentialFile",
                        node.name,
                        "bulk_load",
                        lambda: storage.rebulk_load(node.csv_path, node.delimiter),
                        entries_fn=lambda value: storage.count(),
                    )
            elif node.csv_path is not None and unique_columns:
                heap = HeapFile(metadata, file_path, self.ctx.page_size)
                holder["storage"] = heap
                index_paths = [
                    self.ctx.catalog.index_path(
                        node.name, column, IndexKind.BTREE
                    )
                    for column in unique_columns
                ]
                created_files.extend(index_paths)

                heap_before = self._snapshot(heap)
                start = perf_counter()
                try:
                    built = BPlusTree.bulk_load_multiple(
                        metadata,
                        heap,
                        unique_columns,
                        self.ctx.page_size,
                        node.csv_path,
                        index_paths,
                        delimiter=node.delimiter,
                        sort_buffer_size=self.ctx.sort_buffer_size,
                        level_memory_limit=self.ctx.level_memory_limit,
                    )
                except ValueError as exc:
                    raise ExecutorError(
                        f"No se pudo cargar el CSV en '{node.name}': {exc}"
                    ) from exc
                elapsed = (perf_counter() - start) * 1000.0
                heap_change = self._delta(heap_before, self._snapshot(heap))
                self._record_operation(
                    "HeapFile",
                    node.name,
                    "bulk_load",
                    elapsed,
                    heap_change.reads,
                    heap_change.writes,
                    heap.count(),
                )
                self.ctx.io_reads += heap_change.reads
                self.ctx.io_writes += heap_change.writes

                for column, index in zip(unique_columns, built):
                    self._record_operation(
                        "BPlusTree",
                        self._canonical_btree_name(node.name, column),
                        "build",
                        0.0,
                        index.page_reads,
                        index.page_writes,
                        heap.count(),
                    )
                    self.ctx.io_reads += index.page_reads
                    self.ctx.io_writes += index.page_writes
                self._register_built_indexes(
                    node.name, unique_columns, built, index_objects
                )
                storage = heap
            elif node.csv_path is not None:
                heap = HeapFile(metadata, file_path, self.ctx.page_size)
                holder["storage"] = heap
                self._measure(
                    heap,
                    "HeapFile",
                    node.name,
                    "bulk_load",
                    lambda: heap.rebulk_load(node.csv_path, node.delimiter),
                    entries_fn=lambda value: heap.count(),
                )
                storage = heap
            else:
                storage = HeapFile(metadata, file_path, self.ctx.page_size)
                holder["storage"] = storage

            if (
                node.storage is StorageKind.HEAP
                and unique_columns
                and not index_objects
            ):
                for column in unique_columns:
                    index_path = self.ctx.catalog.index_path(
                        node.name, column, IndexKind.BTREE
                    )
                    created_files.append(index_path)
                    start = perf_counter()
                    index = BPlusTree.build_from_storage(
                        metadata,
                        storage,
                        column,
                        self.ctx.page_size,
                        index_path,
                        sort_buffer_size=self.ctx.sort_buffer_size,
                        level_memory_limit=self.ctx.level_memory_limit,
                    )
                    elapsed = (perf_counter() - start) * 1000.0
                    self._record_operation(
                        "BPlusTree",
                        self._canonical_btree_name(node.name, column),
                        "build",
                        elapsed,
                        index.page_reads,
                        index.page_writes,
                        storage.count(),
                    )
                    self.ctx.io_reads += index.page_reads
                    self.ctx.io_writes += index.page_writes
                    info = IndexMetadata(
                        self._canonical_btree_name(node.name, column),
                        node.name,
                        column,
                        IndexKind.BTREE,
                        index_path,
                    )
                    index_objects.append((info, index))

            holder["storage"] = storage
            return ExecutionValue(affected_rows=0)

        try:
            value = self._execute_node(
                node, "CreateTablePlan", body, rows_out_fn=lambda value: 0
            )
        except Exception:
            self._discard_created(
                node.name, holder["storage"], index_objects, created_files
            )
            raise

        entry = TableEntry(node.name, metadata, node.storage, file_path)
        try:
            self.ctx.catalog.add_table(entry)
            for info, _ in index_objects:
                self.ctx.catalog.add_index(info)
        except Exception:
            for info, _ in index_objects:
                if self.ctx.catalog.index_exists(info.name):
                    self.ctx.catalog.remove_index(info.name)
            if self.ctx.catalog.table_exists(node.name):
                self.ctx.catalog.remove_table(node.name)
            self._discard_created(
                node.name, holder["storage"], index_objects, created_files
            )
            raise

        storage = holder["storage"]
        if storage is not None:
            self.ctx.storage_cache[node.name] = storage
        for info, index in index_objects:
            self.ctx.index_cache[info.name] = index

        return value

    

    
    def visit_create_index(self, node: CreateIndexPlan) -> ExecutionValue:
        if node.index_kind is IndexKind.RTREE:
            if (
                node.spatial_columns is None
                or len(node.spatial_columns) != 2
            ):
                raise ExecutorError(
                    "CREATE INDEX R-Tree requiere dos columnas espaciales."
                )

            existing = self.ctx.catalog.find_index(
                node.table, node.spatial_columns, node.index_kind
            )
        else:
            existing = self.ctx.catalog.find_index(
                node.table, node.column, node.index_kind
            )

        if existing is not None:
            if node.index_kind is IndexKind.RTREE:
                columns_description = (
                    f"{node.spatial_columns[0]}, {node.spatial_columns[1]}"
                )
            else:
                columns_description = node.column

            raise ExecutorError(
                f"Ya existe un indice {node.index_kind.value} "
                f"'{existing.name}' para las columnas "
                f"'{columns_description}' de la tabla '{node.table}'."
            )

        if node.index_kind is IndexKind.RTREE:
            latitude_column, longitude_column = node.spatial_columns

            entry = self.ctx.catalog.get_table(node.table)

            for column in node.spatial_columns:
                try:
                    column_index = entry.metadata.col_index(column)
                except ValueError as exc:
                    raise ExecutorError(
                        f"La columna espacial '{column}' no existe."
                    ) from exc

                column_type = entry.metadata.columns[column_index]["type"]

                if column_type not in ("INT", "FLOAT", "DOUBLE"):
                    raise ExecutorError(
                        f"La columna '{column}' debe ser numerica."
                    )

            if entry.storage is not StorageKind.HEAP:
                raise ExecutorError(
                    "R-Tree solamente puede construirse sobre tablas HEAP."
                )

            index_path = self.ctx.catalog.index_path(
                node.table,
                node.spatial_columns,
                node.index_kind,
            )

        else:
            latitude_column = None
            longitude_column = None

            index_path = self.ctx.catalog.index_path(
                node.table,
                node.column,
                node.index_kind,
            )

        if os.path.exists(index_path):
            raise ExecutorError(
                f"El archivo de indice ya existe: {index_path}"
            )

        entry = self.ctx.catalog.get_table(node.table)
        heap = self._get_storage(node.table)

        holder: dict[str, object | None] = {"index": None}

        def build_index():
            if node.index_kind is IndexKind.RTREE:
                return RTree.build_from_storage(
                    entry.metadata,
                    heap,
                    (latitude_column, longitude_column),
                    self.ctx.page_size,
                    index_path,
                )

            if node.index_kind is IndexKind.HASH:
                return ExtendibleHash.build_from_storage(
                    entry.metadata,
                    heap,
                    node.column,
                    self.ctx.page_size,
                    index_path,
                )

            if node.index_kind is IndexKind.BTREE:
                return BPlusTree.build_from_storage(
                    entry.metadata,
                    heap,
                    node.column,
                    self.ctx.page_size,
                    index_path,
                    sort_buffer_size=self.ctx.sort_buffer_size,
                    level_memory_limit=self.ctx.level_memory_limit,
                )

            raise ExecutorError(
                f"Tipo de indice no soportado: {node.index_kind}"
            )

        def body() -> ExecutionValue:
            heap_before = self._snapshot(heap)
            start = perf_counter()

            index = build_index()

            elapsed = (perf_counter() - start) * 1000.0
            heap_change = self._delta(
                heap_before,
                self._snapshot(heap),
            )

            if node.index_kind in (IndexKind.HASH, IndexKind.RTREE):
                index_reads = index.page_reads
                index_writes = index.page_writes
            else:
                index_reads = max(
                    0, index.page_reads - heap_change.reads
                )
                index_writes = max(
                    0, index.page_writes - heap_change.writes
                )

            self._record_operation(
                self._index_resource_type(index),
                node.name,
                "build",
                elapsed,
                index_reads,
                index_writes,
                heap.count(),
            )

            self.ctx.io_reads += index_reads + heap_change.reads
            self.ctx.io_writes += index_writes + heap_change.writes

            holder["index"] = index

            return ExecutionValue(affected_rows=0)

        try:
            value = self._execute_node(
                node,
                "CreateIndexPlan",
                body,
                rows_out_fn=lambda value: 0,
            )
        except Exception:
            self._remove_index_file(holder["index"], index_path)
            raise

        index = holder["index"]

        info = IndexMetadata(
            node.name,
            node.table,
            node.column,
            node.index_kind,
            index_path,
            spatial_columns=node.spatial_columns,
        )

        try:
            self.ctx.catalog.add_index(info)
        except Exception:
            self._remove_index_file(index, index_path)
            raise

        assert index is not None
        self.ctx.index_cache[node.name] = index

        return value



    def visit_add_unique(self, node: AddUniquePlan) -> ExecutionValue:
        entry = self.ctx.catalog.get_table(node.table)

        existing = self.ctx.catalog.find_index(
            node.table, node.column, IndexKind.BTREE
        )
        if existing is not None:
            raise ExecutorError(
                f"Ya existe un indice BPlusTree '{existing.name}' para la "
                f"columna '{node.column}' de la tabla '{node.table}'"
            )

        index_path = self.ctx.catalog.index_path(
            node.table, node.column, IndexKind.BTREE
        )
        if os.path.exists(index_path):
            raise ExecutorError(f"El archivo de indice ya existe: {index_path}")

        updated_metadata = self._metadata_with_unique(entry.metadata, node.column)
        heap = self._get_storage(node.table)
        holder: dict[str, Index | None] = {"index": None}

        def body() -> ExecutionValue:
            start = perf_counter()
            try:
                index = BPlusTree.build_from_storage(
                    updated_metadata,
                    heap,
                    node.column,
                    self.ctx.page_size,
                    index_path,
                    sort_buffer_size=self.ctx.sort_buffer_size,
                    level_memory_limit=self.ctx.level_memory_limit,
                )
            except ValueError as exc:
                raise ExecutorError(
                    f"No se pudo aplicar UNIQUE en '{node.column}': {exc}"
                ) from exc
            elapsed = (perf_counter() - start) * 1000.0
            name = self._canonical_btree_name(node.table, node.column)
            self._record_operation(
                "BPlusTree",
                name,
                "build",
                elapsed,
                index.page_reads,
                index.page_writes,
                heap.count(),
            )
            self.ctx.io_reads += index.page_reads
            self.ctx.io_writes += index.page_writes
            holder["index"] = index
            return ExecutionValue(affected_rows=0)

        try:
            value = self._execute_node(
                node, "AddUniquePlan", body, rows_out_fn=lambda value: 0
            )
        except Exception:
            self._remove_index_file(holder["index"], index_path)
            raise

        index = holder["index"]
        name = self._canonical_btree_name(node.table, node.column)
        info = IndexMetadata(
            name, node.table, node.column, IndexKind.BTREE, index_path
        )
        updated_entry = TableEntry(
            node.table, updated_metadata, entry.storage, entry.file_path
        )
        try:
            self.ctx.catalog.add_index(info)
            self.ctx.catalog.update_table(updated_entry)
        except Exception:
            if self.ctx.catalog.index_exists(name):
                self.ctx.catalog.remove_index(name)
            self._remove_index_file(index, index_path)
            raise

        assert index is not None
        self.ctx.index_cache[name] = index
        return value

    # Auxiliares de contexto

    def _get_storage(self, table: str) -> StorageFile:
        cached = self.ctx.storage_cache.get(table)
        if cached is not None:
            return cached

        entry = self.ctx.catalog.get_table(table)
        if entry.storage is StorageKind.SEQUENTIAL:
            storage: StorageFile = SequentialFile(
                entry.metadata, entry.file_path, self.ctx.page_size
            )
        else:
            storage = HeapFile(
                entry.metadata, entry.file_path, self.ctx.page_size
            )

        self.ctx.storage_cache[table] = storage
        return storage

    def _get_index(self, name: str) -> Index | RTree:
        cached = self.ctx.index_cache.get(name)
        if cached is not None:
            return cached

        meta = self.ctx.catalog.get_index(name)
        entry = self.ctx.catalog.get_table(meta.table)
        heap = self._get_storage(meta.table)

        if meta.kind is IndexKind.RTREE:
            if meta.spatial_columns is None:
                raise ExecutorError(
                    f"El índice R-Tree '{name}' no tiene columnas espaciales."
                )

            index = RTree(
                entry.metadata,
                meta.spatial_columns,
                heap,
                self.ctx.page_size,
                meta.file_path,
            )

            try:
                index.open()
            except FileNotFoundError as exc:
                raise ExecutorError(
                    f"No se pudo abrir el índice R-Tree '{name}': {exc}"
                ) from exc

        elif meta.kind is IndexKind.HASH:
            index = ExtendibleHash(
                entry.metadata,
                meta.column,
                heap,
                self.ctx.page_size,
                meta.file_path,
            )

            try:
                index.open()
            except FileNotFoundError as exc:
                raise ExecutorError(
                    f"No se pudo abrir el índice HASH '{name}': {exc}"
                ) from exc

        elif meta.kind is IndexKind.BTREE:
            index = BPlusTree(
                entry.metadata,
                meta.column,
                heap,
                self.ctx.page_size,
                meta.file_path,
                self.ctx.sort_buffer_size,
                self.ctx.level_memory_limit,
            )

        else:
            raise ExecutorError(
                f"Tipo de índice no soportado: {meta.kind}"
            )

        self.ctx.index_cache[name] = index
        return index

    def _table_indexes(self, table: str) -> list[IndexMetadata]:
        return [
            info
            for info in self.ctx.catalog.get_table_indexes(table)
            if info.kind in (
                IndexKind.BTREE,
                IndexKind.HASH,
                IndexKind.RTREE,
            )
        ]

    def _table_meta(self, table: str) -> TableMetadata:
        return self.ctx.catalog.get_table(table).metadata

    def _table_of(self, node: PlanNode) -> str:
        direct = getattr(node, "table", None)
        if isinstance(direct, str):
            return direct

        if isinstance(
            node,
            (
                IndexScanPlan,
                IndexRangeScanPlan,
                BitmapIndexScanPlan,
                BitmapIndexRangeScanPlan,
            ),
        ):
            return self.ctx.catalog.get_index(node.index).table

        if isinstance(node, (FilterPlan, ProjectPlan, DeletePlan)):
            return self._table_of(node.child)

        if isinstance(node, (BitmapAndPlan, BitmapOrPlan)):
            for child in node.children:
                return self._table_of(child)

        raise ExecutorError(
            f"No se pudo resolver la tabla del plan {type(node).__name__}"
        )

    def _iter_raw_records(self, storage: StorageFile):
        if isinstance(storage, HeapFile):
            for page_records in storage.iter_pages_with_records():
                for _, record in page_records:
                    yield record
        else:
            yield from storage.iter_records()

    def close(self) -> None:
        for index in self.ctx.index_cache.values():
            try:
                index.close()
            except Exception:
                pass
        self.ctx.index_cache.clear()

        for storage in self.ctx.storage_cache.values():
            try:
                storage.close()
            except Exception:
                pass
        self.ctx.storage_cache.clear()

    # Auxiliares de metadata y esquema

    @staticmethod
    def _index_resource_type(index: Index) -> str:
        return type(index).__name__

    @staticmethod
    def _canonical_btree_name(table: str, column: str) -> str:
        return f"{table}_{column}_bplustree"

    @staticmethod
    def _metadata_with_unique(
        metadata: TableMetadata,
        column: str,
    ) -> TableMetadata:
        columns = []
        for current in metadata.columns:
            copied = dict(current)
            if copied["name"] == column:
                copied["unique"] = True
            columns.append(copied)
        return TableMetadata(metadata.name, columns, metadata.primary_key)

    def _register_built_indexes(
        self,
        table: str,
        columns: list[str],
        built: list[BPlusTree],
        index_objects: list[tuple[IndexMetadata, Index]],
    ) -> None:
        for column, index in zip(columns, built):
            index_path = self.ctx.catalog.index_path(
                table, column, IndexKind.BTREE
            )
            info = IndexMetadata(
                self._canonical_btree_name(table, column),
                table,
                column,
                IndexKind.BTREE,
                index_path,
            )
            index_objects.append((info, index))

    def _discard_created(
        self,
        table: str,
        storage: StorageFile | None,
        index_objects: list[tuple[IndexMetadata, Index]],
        created_files: list[str],
    ) -> None:
        for _, index in index_objects:
            try:
                index.close()
            except Exception:
                pass
        if storage is not None:
            try:
                storage.close()
            except Exception:
                pass
        self.ctx.storage_cache.pop(table, None)
        for path in created_files:
            if os.path.exists(path):
                os.remove(path)

    def _remove_index_file(
        self,
        index: Index | RTree | None,
        path: str,
    ) -> None:
        if index is not None:
            try:
                index.close()
            except Exception:
                pass

        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _infer_csv_metadata(
        name: str,
        csv_path: str,
        delimiter: str,
        primary_key: str | None,
    ) -> TableMetadata:
        if not os.path.exists(csv_path):
            raise ExecutorError(f"El CSV no existe: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source, delimiter=delimiter or ",")
            try:
                header = next(reader)
            except StopIteration:
                raise ExecutorError("El CSV esta vacio: falta la cabecera")

            header = [value.strip() for value in header]

            if any(value == "" for value in header):
                raise ExecutorError("La cabecera del CSV tiene columnas vacias")

            if len(set(header)) != len(header):
                raise ExecutorError("La cabecera del CSV tiene columnas repetidas")

            if primary_key is None:
                raise ExecutorError(
                    "CREATE TABLE FROM CSV requiere PRIMARY KEY"
                )

            if primary_key not in header:
                raise ExecutorError(
                    f"La PRIMARY KEY '{primary_key}' no existe en la cabecera "
                    f"del CSV"
                )

            all_int = [True] * len(header)
            all_float = [True] * len(header)
            max_length = [0] * len(header)
            has_rows = False

            for row in reader:
                if not row:
                    continue

                has_rows = True

                for index in range(len(header)):
                    value = row[index].strip() if index < len(row) else ""

                    if value == "":
                        all_int[index] = False
                        all_float[index] = False
                    else:
                        if re.fullmatch(r"[+-]?\d+", value) is None:
                            all_int[index] = False

                        try:
                            float(value)
                        except ValueError:
                            all_float[index] = False

                    max_length[index] = max(
                        max_length[index],
                        len(value.encode("utf-8"))
                    )

            if not has_rows:
                raise ExecutorError("El CSV no tiene filas de datos")

            columns = []

            for index, column_name in enumerate(header):
                if all_int[index]:
                    columns.append(
                        {
                            "name": column_name,
                            "type": "INT",
                            "size": None,
                            "unique": False,
                        }
                    )

                elif all_float[index]:
                    columns.append(
                        {
                            "name": column_name,
                            "type": "DOUBLE",
                            "size": None,
                            "unique": False,
                        }
                    )

                else:
                    columns.append(
                        {
                            "name": column_name,
                            "type": "CHAR",
                            "size": max(1, max_length[index]),
                            "unique": False,
                        }
                    )

        return TableMetadata(name, columns, primary_key)
    # Auxiliares de metricas

    @staticmethod
    def _snapshot(resource) -> CounterSnapshot:
        return CounterSnapshot(
            reads=resource.page_reads,
            writes=resource.page_writes,
        )

    @staticmethod
    def _delta(
        before: CounterSnapshot,
        after: CounterSnapshot,
    ) -> CounterSnapshot:
        return CounterSnapshot(
            reads=after.reads - before.reads,
            writes=after.writes - before.writes,
        )

    @staticmethod
    def _bitmap_size(bitmap: Bitmap | None) -> int:
        if bitmap is None:
            return 0
        return sum(1 for _ in bitmap.iter_rids())

    @staticmethod
    def _resource_type(storage: StorageFile) -> str:
        return type(storage).__name__

    def _record_operation(
        self,
        resource_type: str,
        resource_name: str,
        operation: str,
        elapsed_ms: float,
        pages_read: int,
        pages_written: int,
        entries_affected: int,
    ) -> None:
        if not self._stack:
            return
        self._stack[-1].operations.append(
            OperationMetrics(
                resource_type,
                resource_name,
                operation,
                elapsed_ms,
                pages_read,
                pages_written,
                entries_affected,
            )
        )

    def _measure(
        self,
        resource,
        resource_type: str,
        resource_name: str,
        operation: str,
        fn: Callable[[], object],
        entries_fn: Callable[[object], int] | None = None,
    ):
        before = self._snapshot(resource)
        start = perf_counter()
        value = fn()
        elapsed = (perf_counter() - start) * 1000.0
        after = self._snapshot(resource)
        change = self._delta(before, after)
        entries = entries_fn(value) if entries_fn is not None else 0

        self._record_operation(
            resource_type,
            resource_name,
            operation,
            elapsed,
            change.reads,
            change.writes,
            entries,
        )
        self.ctx.io_reads += change.reads
        self.ctx.io_writes += change.writes
        return value

    def _execute_node(
        self,
        node: PlanNode,
        node_type: str,
        fn: Callable[[], ExecutionValue],
        rows_out_fn: Callable[[ExecutionValue], int] | None = None,
    ) -> ExecutionValue:
        node_id = self.ctx.next_node_id
        self.ctx.next_node_id += 1

        frame = _NodeFrame(
            node_id,
            node_type,
            perf_counter(),
            self.ctx.io_reads,
            self.ctx.io_writes,
        )
        self._stack.append(frame)

        try:
            value = fn()
        except BaseException:
            self._stack.pop()
            raise

        self._stack.pop()

        elapsed = (perf_counter() - frame.start) * 1000.0
        pages_read = self.ctx.io_reads - frame.reads_before
        pages_written = self.ctx.io_writes - frame.writes_before

        child_time = sum(child.elapsed_ms for child in frame.children)
        child_reads = sum(child.pages_read for child in frame.children)
        child_writes = sum(child.pages_written for child in frame.children)
        rows_out = rows_out_fn(value) if rows_out_fn is not None else 0

        metrics = NodeMetrics(
            node_id,
            node_type,
            max(0.0, elapsed),
            max(0.0, elapsed - child_time),
            rows_out,
            max(0, pages_read),
            max(0, pages_written),
            max(0, pages_read - child_reads),
            max(0, pages_written - child_writes),
            frame.operations,
            frame.children,
        )

        if self._stack:
            self._stack[-1].children.append(metrics)
        else:
            self.ctx.node_metrics[metrics.node_id] = metrics
            self._last_metrics = metrics

        return value

    # Auxiliares de archivos

    def _snapshot_file(self, path: str) -> FileSnapshot:
        parent = os.path.dirname(path) or "."
        fd, snapshot_path = tempfile.mkstemp(
            prefix="exec_snapshot_", suffix=".tmp", dir=parent
        )
        os.close(fd)
        shutil.copy2(path, snapshot_path)
        return FileSnapshot(path, snapshot_path)

    @staticmethod
    def _discard_snapshot(snapshot: FileSnapshot) -> None:
        if os.path.exists(snapshot.snapshot_path):
            os.remove(snapshot.snapshot_path)

    def _restore_storage_snapshot(
        self,
        table: str,
        snapshot: FileSnapshot,
    ) -> None:
        storage = self.ctx.storage_cache.pop(table, None)
        if storage is not None:
            try:
                storage.close()
            except Exception:
                pass

        shutil.copy2(snapshot.snapshot_path, snapshot.original_path)
        if os.path.exists(snapshot.snapshot_path):
            os.remove(snapshot.snapshot_path)

        self._get_storage(table)

    # Resultado de sentencia

    def _statement_result(
        self,
        info,
        value: ExecutionValue,
    ) -> StatementResult:
        if info.statement_type == "SELECT":
            records = list(value.records or [])
            if records and not value.records_clean and info.table is not None:
                meta = self._table_meta(info.table)
                rows = [meta.clean_tuple(record) for record in records]
            else:
                rows = records
            rows = [row for row in rows if row is not None]

            if info.select_columns is not None:
                columns = list(info.select_columns)
            elif info.table is not None:
                columns = [
                    column["name"]
                    for column in self._table_meta(info.table).columns
                ]
            else:
                columns = list(value.output_columns or [])

            return StatementResult(
                info.statement_id,
                info.statement_type,
                columns,
                rows,
                None,
            )

        if info.statement_type == "INSERT":
            affected = (
                value.affected_rows if value.affected_rows is not None else 0
            )
            return StatementResult(
                info.statement_id, info.statement_type, [], [], affected
            )

        if info.statement_type == "DELETE":
            return StatementResult(
                info.statement_id,
                info.statement_type,
                [],
                [],
                value.affected_rows or 0,
            )

        return StatementResult(
            info.statement_id, info.statement_type, [], [], 0
        )
