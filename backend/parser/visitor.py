from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .ast import (
    Program,
    CreateTableStm,
    CreateIndexStm,
    InsertStm,
    AlterStm,
    SelectStm,
    DeleteStm,
    BinaryExp,
    UnaryExp,
    LiteralExp,
    IdExp,
)


S = TypeVar("S")
E = TypeVar("E")


class Visitor(ABC, Generic[S, E]):

    @abstractmethod
    def visit_program(self, node: Program) -> S:
        pass

    @abstractmethod
    def visit_create_table(self, node: CreateTableStm) -> S:
        pass

    @abstractmethod
    def visit_create_index(self, node: CreateIndexStm) -> S:
        pass

    @abstractmethod
    def visit_insert(self, node: InsertStm) -> S:
        pass

    @abstractmethod
    def visit_alter(self, node: AlterStm) -> S:
        pass

    @abstractmethod
    def visit_select(self, node: SelectStm) -> S:
        pass

    @abstractmethod
    def visit_delete(self, node: DeleteStm) -> S:
        pass

    @abstractmethod
    def visit_binary_exp(self, node: BinaryExp) -> E:
        pass

    @abstractmethod
    def visit_unary_exp(self, node: UnaryExp) -> E:
        pass

    @abstractmethod
    def visit_literal_exp(self, node: LiteralExp) -> E:
        pass

    @abstractmethod
    def visit_id_exp(self, node: IdExp) -> E:
        pass
