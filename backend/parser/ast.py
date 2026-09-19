from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from visitor import Visitor


T = TypeVar("T")


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


class StorageType(Enum):
    HEAP = auto()
    SEQUENTIAL = auto()


class IndexType(Enum):
    BTREE = auto()
    HASH = auto()


# Expressions

class Exp:

    def accept(self, visitor: Visitor[T]) -> T:
        raise NotImplementedError


@dataclass
class BinaryExp(Exp):
    left: Exp
    right: Exp
    op: BinaryOp

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_binary_exp(self)


@dataclass
class UnaryExp(Exp):
    value: Exp
    op: UnaryOp

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_unary_exp(self)


@dataclass
class LiteralExp(Exp):
    value: int | float | str

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_literal_exp(self)


@dataclass
class IdExp(Exp):
    value: str

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_id_exp(self)


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

    def accept(self, visitor: Visitor[T]) -> T:
        raise NotImplementedError


@dataclass
class CreateTableStm(Stm):
    name: str
    columns: list[ColumnDef] | None = None
    storage: StorageType = StorageType.HEAP
    csv: CsvSource | None = None
    primary_key: str | None = None

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_create_table(self)


@dataclass
class CreateIndexStm(Stm):
    name: str
    table: str
    column: str
    index_type: IndexType

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_create_index(self)


@dataclass
class InsertStm(Stm):
    table: str
    values: list[Exp]

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_insert(self)


@dataclass
class AlterStm(Stm):
    table: str
    column: str

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_alter(self)


@dataclass
class SelectStm(Stm):
    columns: list[str] | None
    table: str
    force_indexes: list[str] | None = None
    force_seqscan: bool = False
    where: Exp | None = None

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_select(self)


@dataclass
class DeleteStm(Stm):
    table: str
    where: Exp | None = None

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_delete(self)


# Program

@dataclass
class Program:
    statements: list[Stm]

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_program(self)