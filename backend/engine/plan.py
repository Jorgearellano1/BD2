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
    """Plan para eliminar registros."""

    table: str
    child: PlanNode


@dataclass
class SeqScanPlan(PlanNode):
    """Recorrido secuencial de una tabla."""

    table: str


@dataclass
class IndexScanPlan(PlanNode):
    """Busqueda por igualdad mediante un indice."""

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
    """Busqueda de rango mediante un indice."""

    table: str
    index: str
    range: RangeSpec


@dataclass
class BitmapIndexScanPlan(PlanNode):
    """Busqueda por igualdad para generar un bitmap."""

    table: str
    index: str
    index_kind: IndexKind
    key: Key


@dataclass
class BitmapIndexRangeScanPlan(PlanNode):
    """Busqueda de rango para generar un bitmap."""

    table: str
    index: str
    range: RangeSpec


@dataclass
class BitmapAndPlan(PlanNode):
    """Interseccion de varios bitmaps."""

    children: list[PlanNode]


@dataclass
class BitmapOrPlan(PlanNode):
    """Union de varios bitmaps."""

    children: list[PlanNode]


@dataclass
class HeapFetchPlan(PlanNode):
    """Recupera registros a partir de RIDs."""

    table: str
    child: PlanNode


@dataclass
class BitmapHeapScanPlan(PlanNode):
    """Recupera registros a partir de un bitmap."""

    table: str
    child: PlanNode


@dataclass
class FilterPlan(PlanNode):
    """Evalua una condicion sobre los registros."""

    predicate: Expression
    child: PlanNode


@dataclass
class ProjectPlan(PlanNode):
    """Selecciona las columnas de salida."""

    columns: list[str]
    child: PlanNode