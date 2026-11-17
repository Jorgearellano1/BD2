from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from engine.plan import (
    PlanNode,
    ProgramPlan,
    CreateTablePlan,
    CreateIndexPlan,
    InsertPlan,
    AddUniquePlan,
    DeletePlan,
    SequentialDeletePlan,
    SeqScanPlan,
    SeqScanRidsPlan,
    IndexScanPlan,
    IndexRangeScanPlan,
    SequentialRangeScanPlan,
    BitmapIndexScanPlan,
    BitmapIndexRangeScanPlan,
    BitmapAndPlan,
    BitmapOrPlan,
    HeapFetchPlan,
    BitmapHeapScanPlan,
    FilterPlan,
    FilterRidsPlan,
    EmptyResultPlan,
    ProjectPlan,
    TextIndexScanPlan,
    TextFetchPlan,
    FullTextScanPlan,
    SortPlan,
    LimitPlan,
    TextProjectPlan,
)


R = TypeVar("R")


class PlanVisitor(ABC, Generic[R]):
    """
    Visitor abstracto para recorrer el arbol de ejecucion.

    Cada metodo visit_* corresponde a un tipo concreto de PlanNode.

    R representa el tipo de resultado que produce el visitor.
    Por ejemplo, ExecutorVisitor puede usar:

        PlanVisitor[ExecutionValue]
    """

    @abstractmethod
    def visit_program(self, node: ProgramPlan) -> R:
        """Visita un ProgramPlan."""
        pass

    @abstractmethod
    def visit_create_table(self, node: CreateTablePlan) -> R:
        """Visita un CreateTablePlan."""
        pass

    @abstractmethod
    def visit_create_index(self, node: CreateIndexPlan) -> R:
        """Visita un CreateIndexPlan."""
        pass

    @abstractmethod
    def visit_insert(self, node: InsertPlan) -> R:
        """Visita un InsertPlan."""
        pass

    @abstractmethod
    def visit_add_unique(self, node: AddUniquePlan) -> R:
        """Visita un AddUniquePlan."""
        pass

    @abstractmethod
    def visit_delete(self, node: DeletePlan) -> R:
        """Visita un DeletePlan."""
        pass

    @abstractmethod
    def visit_seq_scan(self, node: SeqScanPlan) -> R:
        """Visita un SeqScanPlan."""
        pass

    @abstractmethod
    def visit_sequential_range_scan(
        self,
        node: SequentialRangeScanPlan,
    ) -> R:
        """Visita un SequentialRangeScanPlan."""
        pass

    @abstractmethod
    def visit_sequential_delete(self, node: SequentialDeletePlan) -> R:
        """Visita un SequentialDeletePlan."""
        pass

    @abstractmethod
    def visit_seq_scan_rids(self, node: SeqScanRidsPlan) -> R:
        """Visita un SeqScanRidsPlan."""
        pass

    @abstractmethod
    def visit_index_scan(self, node: IndexScanPlan) -> R:
        """Visita un IndexScanPlan."""
        pass

    @abstractmethod
    def visit_index_range_scan(self, node: IndexRangeScanPlan) -> R:
        """Visita un IndexRangeScanPlan."""
        pass

    @abstractmethod
    def visit_bitmap_index_scan(self, node: BitmapIndexScanPlan) -> R:
        """Visita un BitmapIndexScanPlan."""
        pass

    @abstractmethod
    def visit_bitmap_index_range_scan(
        self,
        node: BitmapIndexRangeScanPlan,
    ) -> R:
        """Visita un BitmapIndexRangeScanPlan."""
        pass

    @abstractmethod
    def visit_bitmap_and(self, node: BitmapAndPlan) -> R:
        """Visita un BitmapAndPlan."""
        pass

    @abstractmethod
    def visit_bitmap_or(self, node: BitmapOrPlan) -> R:
        """Visita un BitmapOrPlan."""
        pass

    @abstractmethod
    def visit_heap_fetch(self, node: HeapFetchPlan) -> R:
        """Visita un HeapFetchPlan."""
        pass

    @abstractmethod
    def visit_bitmap_heap_scan(self, node: BitmapHeapScanPlan) -> R:
        """Visita un BitmapHeapScanPlan."""
        pass

    @abstractmethod
    def visit_filter(self, node: FilterPlan) -> R:
        """Visita un FilterPlan."""
        pass

    @abstractmethod
    def visit_filter_rids(self, node: FilterRidsPlan) -> R:
        """Visita un FilterRidsPlan."""
        pass

    @abstractmethod
    def visit_empty_result(self, node: EmptyResultPlan) -> R:
        """Visita un EmptyResultPlan."""
        pass

    @abstractmethod
    def visit_project(self, node: ProjectPlan) -> R:
        """Visita un ProjectPlan."""
        pass

    # Modulo textual. No son abstractos para no obligar a otros visitors.

    def visit_text_index_scan(self, node: TextIndexScanPlan) -> R:
        raise NotImplementedError("TextIndexScanPlan no soportado")

    def visit_text_fetch(self, node: TextFetchPlan) -> R:
        raise NotImplementedError("TextFetchPlan no soportado")

    def visit_full_text_scan(self, node: FullTextScanPlan) -> R:
        raise NotImplementedError("FullTextScanPlan no soportado")

    def visit_sort(self, node: SortPlan) -> R:
        raise NotImplementedError("SortPlan no soportado")

    def visit_limit(self, node: LimitPlan) -> R:
        raise NotImplementedError("LimitPlan no soportado")

    def visit_text_project(self, node: TextProjectPlan) -> R:
        raise NotImplementedError("TextProjectPlan no soportado")

