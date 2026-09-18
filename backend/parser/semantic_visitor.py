from visitor import Visitor

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


class SemanticVisitor(Visitor[None]):

    def visit_program(self, node: Program) -> None:
        table_names: set[str] = set()
        index_names: set[str] = set()

        for statement in node.statements:

            if isinstance(statement, CreateTableStm):
                if statement.name in table_names:
                    raise ValueError(
                        f"Error semantico: la tabla "
                        f"'{statement.name}' esta repetida"
                    )

                table_names.add(statement.name)

            elif isinstance(statement, CreateIndexStm):
                if statement.name in index_names:
                    raise ValueError(
                        f"Error semantico: el indice "
                        f"'{statement.name}' esta repetido"
                    )

                index_names.add(statement.name)

            statement.accept(self)

    def visit_create_table(self, node: CreateTableStm) -> None:
        if node.columns is None:
            return

        column_names: set[str] = set()
        primary_key_count = 0

        for column in node.columns:

            if column.name in column_names:
                raise ValueError(
                    f"Error semantico: la columna "
                    f"'{column.name}' esta repetida"
                )

            column_names.add(column.name)

            if column.primary_key:
                primary_key_count += 1

            if column.data_type.name == "CHAR":
                if column.size <= 0:
                    raise ValueError(
                        f"Error semantico: CHAR debe tener "
                        f"un tamaño mayor que 0"
                    )

        if primary_key_count > 1:
            raise ValueError(
                f"Error semantico: la tabla "
                f"'{node.name}' tiene mas de una PRIMARY KEY"
            )

    def visit_create_index(self, node: CreateIndexStm) -> None:
        pass

    def visit_insert(self, node: InsertStm) -> None:
        pass

    def visit_alter(self, node: AlterStm) -> None:
        pass

    def visit_select(self, node: SelectStm) -> None:
        pass

    def visit_delete(self, node: DeleteStm) -> None:
        pass

    def visit_binary_exp(self, node: BinaryExp) -> None:
        pass

    def visit_unary_exp(self, node: UnaryExp) -> None:
        pass

    def visit_literal_exp(self, node: LiteralExp) -> None:
        pass

    def visit_id_exp(self, node: IdExp) -> None:
        pass