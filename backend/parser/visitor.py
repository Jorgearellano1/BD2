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
    MatchExp,
    WithinRadiusExp,
    KnnExp,
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

    def visit_match_exp(self, node: MatchExp) -> E:
        """MATCH textual. No es abstracto para no romper visitors previos."""
        raise NotImplementedError(
            f"{type(self).__name__} no soporta MATCH"
        )

    def visit_within_radius_exp(self, node: WithinRadiusExp) -> E:
        """WITHIN_RADIUS opcional."""
        raise NotImplementedError(
            f"{type(self).__name__} no soporta WITHIN_RADIUS"
        )

    def visit_knn_exp(self, node: KnnExp) -> E:
        """KNN opcional."""
        raise NotImplementedError(
            f"{type(self).__name__} no soporta KNN"
        )

    def visit_set(self, node) -> S:
        """SET de configuracion de sesion. Opcional para visitors previos."""
        raise NotImplementedError(
            f"{type(self).__name__} no soporta SET"
        )
