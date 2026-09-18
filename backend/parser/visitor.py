from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ast import (
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


T = TypeVar("T")


class Visitor(ABC, Generic[T]):

    @abstractmethod
    def visit_program(self, node: Program) -> T:
        pass

    @abstractmethod
    def visit_create_table(self, node: CreateTableStm) -> T:
        pass

    @abstractmethod
    def visit_create_index(self, node: CreateIndexStm) -> T:
        pass

    @abstractmethod
    def visit_insert(self, node: InsertStm) -> T:
        pass

    @abstractmethod
    def visit_alter(self, node: AlterStm) -> T:
        pass

    @abstractmethod
    def visit_select(self, node: SelectStm) -> T:
        pass

    @abstractmethod
    def visit_delete(self, node: DeleteStm) -> T:
        pass

    @abstractmethod
    def visit_binary_exp(self, node: BinaryExp) -> T:
        pass

    @abstractmethod
    def visit_unary_exp(self, node: UnaryExp) -> T:
        pass

    @abstractmethod
    def visit_literal_exp(self, node: LiteralExp) -> T:
        pass

    @abstractmethod
    def visit_id_exp(self, node: IdExp) -> T:
        pass