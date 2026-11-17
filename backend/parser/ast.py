from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from visitor import Visitor


S = TypeVar("S")
E = TypeVar("E")


# Operators

class BinaryOp(Enum):
    EQ = auto()
    NEQ = auto()
    NEQ2 = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    AND = auto()
    OR = auto()


class UnaryOp(Enum):
    NOT = auto()


class DataType(Enum):
    INT = auto()
    FLOAT = auto()
    DOUBLE = auto()
    CHAR = auto()
    VECTOR = auto()


class StorageType(Enum):
    HEAP = auto()
    SEQUENTIAL = auto()


class IndexType(Enum):
    BTREE = auto()
    HASH = auto()
    INVERTED = auto()
    RTREE = auto()
    IVF = auto()


# Expressions

class Exp:

    def accept(self, visitor: Visitor[S, E]) -> E:
        raise NotImplementedError


@dataclass
class BinaryExp(Exp):
    left: Exp
    right: Exp
    op: BinaryOp

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_binary_exp(self)


@dataclass
class UnaryExp(Exp):
    value: Exp
    op: UnaryOp

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_unary_exp(self)


@dataclass
class LiteralExp(Exp):
    value: int | float | str

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_literal_exp(self)


@dataclass
class IdExp(Exp):
    value: str

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_id_exp(self)


@dataclass
class MatchExp(Exp):
    """columna MATCH 'consulta' (recuperacion textual)."""
    column: str
    query: str

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_match_exp(self)


@dataclass
class WithinRadiusExp(Exp):
    """Predicado WITHIN_RADIUS."""
    latitude: str
    longitude: str
    center_lat: float
    center_lon: float
    radius_km: float

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_within_radius_exp(self)


@dataclass
class KnnExp(Exp):
    """Predicado KNN."""
    latitude: str
    longitude: str
    center_lat: float
    center_lon: float
    k: int

    def accept(self, visitor: Visitor[S, E]) -> E:
        return visitor.visit_knn_exp(self)


@dataclass
class VectorDistance:
    """Item de SELECT: columna <op> '[...]' AS alias."""

    column: str
    operator: str
    literal: str
    alias: str
    position: int


# Table definitions

@dataclass
class ColumnDef:
    name: str
    data_type: DataType
    size: int | None = None
    primary_key: bool = False
    unique: bool = False


@dataclass
class CsvSource:
    path: str
    delimiter: str = ","


# Statements

class Stm:

    def accept(self, visitor: Visitor[S, E]) -> S:
        raise NotImplementedError


@dataclass
class CreateTableStm(Stm):
    name: str
    columns: list[ColumnDef] | None = None
    storage: StorageType = StorageType.HEAP
    csv: CsvSource | None = None
    primary_key: str | None = None

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_create_table(self)



@dataclass
class CreateIndexStm(Stm):
    name: str
    table: str
    column: str
    index_type: IndexType
    spatial_columns: tuple[str, str] | None = None
    options: dict[str, str | int] | None = None

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_create_index(self)

@dataclass
class InsertStm(Stm):
    table: str
    values: list[Exp]

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_insert(self)


@dataclass
class AlterStm(Stm):
    table: str
    column: str

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_alter(self)


@dataclass
class SelectStm(Stm):
    columns: list[str] | None
    table: str
    force_indexes: list[str] | None = None
    force_seqscan: bool = False
    force_all_indexes: bool = False
    where: Exp | None = None
    # Extensiones del modulo textual. 'columns' conserva solo columnas de la
    # tabla; COSINE_SCORE() se describe aparte con su alias y su posicion en
    # la lista de salida.
    score_alias: str | None = None
    score_position: int | None = None
    order_by: str | None = None
    order_desc: bool = False
    limit: int | None = None
    # Item de distancia vectorial: embedding <=> '[...]' AS dist.
    vector_distance: VectorDistance | None = None

    def output_columns(self) -> list[str] | None:
        """Nombres de salida en orden, incluidos score y distancia."""
        if self.columns is None and self.score_alias is None \
                and self.vector_distance is None:
            return None

        items: list[tuple[int, str]] = list(enumerate(self.columns or []))

        if self.score_alias is not None:
            items.append((self.score_position or 0, self.score_alias))

        if self.vector_distance is not None:
            items.append(
                (self.vector_distance.position, self.vector_distance.alias)
            )

        items.sort(key=lambda item: item[0])
        return [name for _, name in items]

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_select(self)


@dataclass
class SetStm(Stm):
    """SET ivf.probes = N."""

    name: str
    value: int | float | str

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_set(self)


@dataclass
class DeleteStm(Stm):
    table: str
    force_indexes: list[str] | None = None
    force_seqscan: bool = False
    force_all_indexes: bool = False
    where: Exp | None = None

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_delete(self)


# Program

@dataclass
class Program:
    statements: list[Stm]

    def accept(self, visitor: Visitor[S, E]) -> S:
        return visitor.visit_program(self)