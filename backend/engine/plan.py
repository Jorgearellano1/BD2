from __future__ import annotations
from rtree.rtreeNode import MBR
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
    name: str
    table: str
    column: str
    index_kind: IndexKind
    spatial_columns: tuple[str, str] | None = None
    options: dict | None = None

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
    Búsqueda mediante un índice.

    B+ Tree y Hash:
        key contiene una clave escalar.

    R-Tree:
        key contiene un MBR que representa la región espacial
        que se desea consultar.
    """

    table: str
    index: str
    index_kind: IndexKind
    key: Key | MBR

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


# ---------------------------------------------------------------------------
# Modulo textual
# ---------------------------------------------------------------------------

@dataclass
class TextIndexScanPlan(PlanNode):
    """
    Busqueda en el indice invertido.

    Evalua la consulta booleana con mezclas lineales y, si rank=True, calcula
    CosineScore y conserva los top_k mejores con un heap.

    strategy:
        INDEX_ONLY  : no hay filtro relacional.
        PRE_FILTER  : 'prefilter' (B+ / Hash) produce los RIDs permitidos
                      antes de puntuar.
        POST_FILTER : el filtro relacional se evalua despues, sobre los
                      registros candidatos (ver TextFetchPlan).

    Resultado: RIDs en orden de relevancia (o de doc_id) y sus scores.
    """

    table: str
    index: str
    column: str
    query: str
    rank: bool
    top_k: int | None
    descending: bool
    strategy: str
    prefilter: PlanNode | None = None

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_text_index_scan(self)


@dataclass
class TextFetchPlan(PlanNode):
    """
    Recupera del Heap los registros de los RIDs candidatos conservando su
    orden y su score. Si hay 'residual' (post-filtering) lo evalua y, con
    'limit', se detiene en cuanto tiene 'limit' registros validos.
    """

    table: str
    child: PlanNode
    residual: Expression | None = None
    limit: int | None = None

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_text_fetch(self)


@dataclass
class FullTextScanPlan(PlanNode):
    """
    Linea base sin indice: recorre todo el Heap, preprocesa cada documento y
    aplica la misma consulta y el mismo scoring que el indice invertido.
    """

    table: str
    column: str
    query: str
    options: dict
    rank: bool
    top_k: int | None
    descending: bool
    residual: Expression | None = None

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_full_text_scan(self)


@dataclass
class SortPlan(PlanNode):
    """ORDER BY sobre una columna de la tabla (ordenamiento estable)."""

    column: str
    descending: bool
    child: PlanNode

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_sort(self)


@dataclass
class LimitPlan(PlanNode):
    """LIMIT n: conserva los primeros n registros."""

    limit: int
    child: PlanNode

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_limit(self)


@dataclass
class TextProjectPlan(PlanNode):
    """Proyeccion que intercala COSINE_SCORE() AS alias entre las columnas."""

    columns: list[str]
    score_alias: str
    score_position: int
    child: PlanNode

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_text_project(self)


# ---------------------------------------------------------------------------
# Modulo vectorial
# ---------------------------------------------------------------------------

@dataclass
class VectorScanPlan(PlanNode):
    """
    Busqueda de similitud sobre una columna VECTOR.

    index:
        Nombre del indice IVF o None para escaneo secuencial exacto.

    strategy:
        INDEX_ONLY   sin filtro relacional.
        PRE_FILTER   el filtro relacional produce los RID permitidos antes
                     de visitar las celdas del IVF.
        POST_FILTER  el filtro relacional se evalua despues de la busqueda.
        SEQ_SCAN     no hay indice: k-NN lineal exacto.
    """

    table: str
    column: str
    index: str | None
    query: str
    operator: str
    top_k: int | None
    strategy: str
    prefilter: PlanNode | None = None

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_vector_scan(self)


@dataclass
class VectorFetchPlan(PlanNode):
    """Recupera del Heap los RID del IVF y conserva el orden por distancia."""

    table: str
    column: str
    query: str
    operator: str
    alias: str
    child: PlanNode
    residual: Expression | None = None
    limit: int | None = None

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_vector_fetch(self)


@dataclass
class VectorProjectPlan(PlanNode):
    """Proyeccion que intercala la distancia vectorial entre las columnas."""

    table: str
    columns: list[str] | None
    alias: str
    position: int
    child: PlanNode

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_vector_project(self)


@dataclass
class SetPlan(PlanNode):
    """SET de una variable de sesion (ivf.probes)."""

    name: str
    value: int | float | str

    def accept(self, visitor: "PlanVisitor[R]") -> R:
        return visitor.visit_set(self)

