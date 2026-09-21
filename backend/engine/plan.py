from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar
from engine.conditions import Expression
from engine.metadata import (
    IndexKind,
    Key,
    StorageKind,
    TableMetadata,
)


if TYPE_CHECKING:
    from engine.plan_visitor import PlanVisitor

R = TypeVar("R")


class PlanNode:
    """Nodo base del arbol de ejecucion."""

    def accept(self, visitor: PlanVisitor[R]) -> R:
        raise NotImplementedError


@dataclass(frozen=True)
class StatementInfo:
    """Informacion original de una sentencia para el Executor."""

    statement_id: int
    statement_type: str
    table: str | None
    select_columns: list[str] | None


@dataclass
class ProgramPlan(PlanNode):
    """Plan para ejecutar varias sentencias."""

    statements: list[PlanNode]
    statement_infos: list[StatementInfo] = field(default_factory=list)

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_program(self)


@dataclass
class CreateTablePlan(PlanNode):
    """Plan para crear una tabla."""

    name: str
    storage: StorageKind
    metadata: TableMetadata | None
    csv_path: str | None
    delimiter: str
    primary_key: str | None

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_create_table(self)


@dataclass
class CreateIndexPlan(PlanNode):
    """Plan para crear un indice."""

    name: str
    table: str
    column: str
    index_kind: IndexKind

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_create_index(self)


@dataclass
class InsertPlan(PlanNode):
    """Plan para insertar un registro."""

    table: str
    values: tuple[Key, ...]

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_insert(self)


@dataclass
class AddUniquePlan(PlanNode):
    """Plan para agregar UNIQUE."""

    table: str
    column: str

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_add_unique(self)


@dataclass
class DeletePlan(PlanNode):
    """
    Plan para eliminar registros.

    child:
        Plan que produce los RIDs finales que deben eliminarse.

    Resultado:
        El DELETE devuelve los RawRecord eliminados.
    """

    table: str
    child: PlanNode

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_delete(self)

@dataclass
class SequentialDeletePlan(PlanNode):
    """Elimina registros directamente mediante SequentialFile."""

    table: str
    predicate: Expression | None

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_sequential_delete(self)


@dataclass
class SequentialRangeScanPlan(PlanNode):
    table: str
    column: str
    range: RangeSpec

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_sequential_range_scan(self)


@dataclass
class SeqScanPlan(PlanNode):
    """
    Recorrido secuencial de una tabla.

    predicate=None:
        scan completo.

    predicate=Expression:
        scan secuencial aplicando esa expresion completa.
    """

    table: str
    predicate: Expression | None = None

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_seq_scan(self)


@dataclass
class SeqScanRidsPlan(PlanNode):
    """
    Recorrido secuencial de una tabla que produce RIDs.

    Se utiliza como access path para DELETE cuando no se utiliza
    un indice.

    Resultado:
        RIDs de los registros recorridos.
    """

    table: str

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_seq_scan_rids(self)


@dataclass
class IndexScanPlan(PlanNode):
    """
    Busqueda por igualdad mediante un indice.

    El resultado conceptual son RIDs de registros candidatos.
    """

    table: str
    index: str
    index_kind: IndexKind
    key: Key

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_index_scan(self)


@dataclass(frozen=True)
class RangeSpec:
    """Define los limites de un rango."""

    lower: Key | None
    upper: Key | None
    include_lower: bool
    include_upper: bool


@dataclass
class IndexRangeScanPlan(PlanNode):
    """
    Busqueda de rango mediante un indice.

    El resultado conceptual son RIDs de registros candidatos.
    """

    table: str
    index: str
    range: RangeSpec

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_index_range_scan(self)


@dataclass
class BitmapIndexScanPlan(PlanNode):
    """
    Busqueda por igualdad para generar un bitmap.

    El bitmap representa un conjunto de RIDs.
    No recupera todavia los registros del Heap.
    """

    table: str
    index: str
    index_kind: IndexKind
    key: Key

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_bitmap_index_scan(self)


@dataclass
class BitmapIndexRangeScanPlan(PlanNode):
    """
    Busqueda de rango para generar un bitmap.

    El bitmap representa un conjunto de RIDs.
    """

    table: str
    index: str
    range: RangeSpec

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_bitmap_index_range_scan(self)


@dataclass
class BitmapAndPlan(PlanNode):
    """
    Interseccion de varios bitmaps.

    El resultado sigue siendo un conjunto de RIDs.
    """

    children: list[PlanNode]

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_bitmap_and(self)


@dataclass
class BitmapOrPlan(PlanNode):
    """
    Union de varios bitmaps.

    El resultado sigue siendo un conjunto de RIDs.
    """

    children: list[PlanNode]

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_bitmap_or(self)


@dataclass
class HeapFetchPlan(PlanNode):
    """
    Recupera registros a partir de RIDs.

    Se utiliza en SELECT antes de Filter y Project.
    """

    table: str
    child: PlanNode

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_heap_fetch(self)


@dataclass
class BitmapHeapScanPlan(PlanNode):
    """
    Recupera registros a partir de un bitmap de RIDs.

    Se utiliza en SELECT antes de Filter y Project.
    """

    table: str
    child: PlanNode

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_bitmap_heap_scan(self)


@dataclass
class FilterPlan(PlanNode):
    """
    Evalua una condicion sobre registros completos.

    No trabaja directamente con RIDs.
    """

    predicate: Expression
    child: PlanNode

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_filter(self)


@dataclass
class FilterRidsPlan(PlanNode):
    """
    Filtra candidatos identificados por RID.

    child:
        Plan que produce RIDs o un bitmap de RIDs.

    predicate:
        Condicion que debe evaluarse sobre el registro asociado
        a cada RID.

    Resultado:
        RIDs de los registros que cumplen la condicion.
    """

    table: str
    predicate: Expression
    child: PlanNode

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_filter_rids(self)


@dataclass
class EmptyResultPlan(PlanNode):
    """
    Representa un resultado vacio.

    Se utiliza cuando una expresion se reduce logicamente
    a FALSE.

    Resultado:
        No produce registros ni RIDs.
    """

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_empty_result(self)


@dataclass
class ProjectPlan(PlanNode):
    """
    Selecciona las columnas de salida.

    Solo corresponde a SELECT.
    """

    columns: list[str]
    child: PlanNode

    def accept(self, visitor: PlanVisitor[R]) -> R:
        return visitor.visit_project(self)