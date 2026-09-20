from __future__ import annotations

from dataclasses import dataclass

from engine.conditions import Expression
from engine.metadata import (
    IndexKind,
    Key,
    StorageKind,
    TableMetadata,
)


class PlanNode:
    """Nodo base del arbol de ejecucion."""

    pass


@dataclass
class ProgramPlan(PlanNode):
    """Plan para ejecutar varias sentencias."""

    statements: list[PlanNode]


@dataclass
class CreateTablePlan(PlanNode):
    """Plan para crear una tabla."""

    name: str
    storage: StorageKind
    metadata: TableMetadata | None
    csv_path: str | None
    delimiter: str
    primary_key: str | None


@dataclass
class CreateIndexPlan(PlanNode):
    """Plan para crear un indice."""

    name: str
    table: str
    column: str
    index_kind: IndexKind


@dataclass
class InsertPlan(PlanNode):
    """Plan para insertar un registro."""

    table: str
    values: tuple[Key, ...]


@dataclass
class AddUniquePlan(PlanNode):
    """Plan para agregar UNIQUE."""

    table: str
    column: str


@dataclass
class DeletePlan(PlanNode):
    """
    Plan para eliminar registros.

    child:
        Plan de acceso que localiza los registros candidatos.
        Puede trabajar con RIDs, bitmaps o registros, dependiendo
        del tipo de acceso y almacenamiento.

    residual:
        Condicion del WHERE que no fue resuelta por el indice.
        Debe evaluarse sobre los registros candidatos antes de
        realizar el borrado.

    Resultado:
        El DELETE devuelve los RawRecord que fueron eliminados.
        Los RIDs se usan internamente para realizar el borrado.
    """

    table: str
    child: PlanNode
    residual: Expression | None = None


@dataclass
class SeqScanPlan(PlanNode):
    """
    Recorrido secuencial de una tabla.

    SELECT:
        produce registros para Filter/Project.

    DELETE:
        sirve como fuente de candidatos para DeletePlan.
    """

    table: str


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


@dataclass
class BitmapIndexRangeScanPlan(PlanNode):
    """
    Busqueda de rango para generar un bitmap.

    El bitmap representa un conjunto de RIDs.
    """

    table: str
    index: str
    range: RangeSpec


@dataclass
class BitmapAndPlan(PlanNode):
    """
    Interseccion de varios bitmaps.

    El resultado sigue siendo un conjunto de RIDs.
    """

    children: list[PlanNode]


@dataclass
class BitmapOrPlan(PlanNode):
    """
    Union de varios bitmaps.

    El resultado sigue siendo un conjunto de RIDs.
    """

    children: list[PlanNode]


@dataclass
class HeapFetchPlan(PlanNode):
    """
    Recupera registros a partir de RIDs.

    Se utiliza en SELECT antes de Filter y Project.
    """

    table: str
    child: PlanNode


@dataclass
class BitmapHeapScanPlan(PlanNode):
    """
    Recupera registros a partir de un bitmap de RIDs.

    Se utiliza en SELECT antes de Filter y Project.
    """

    table: str
    child: PlanNode


@dataclass
class FilterPlan(PlanNode):
    """
    Evalua una condicion sobre registros completos.

    No trabaja directamente con RIDs.
    """

    predicate: Expression
    child: PlanNode


@dataclass
class ProjectPlan(PlanNode):
    """
    Selecciona las columnas de salida.

    Solo corresponde a SELECT.
    """

    columns: list[str]
    child: PlanNode