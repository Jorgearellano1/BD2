from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    ColumnDef,
    DataType,
    StorageType,
    IndexType,
    BinaryOp,
)

from engine.metadata import (
    IndexKind,
    StorageKind,
    TableMetadata,
)

if TYPE_CHECKING:
    from engine.catalog import Catalog
    from engine.metadata import IndexMetadata


# Operadores de comparacion (se excluyen AND y OR).
_COMPARISON_OPS = {
    BinaryOp.EQ,
    BinaryOp.NEQ,
    BinaryOp.NEQ2,
    BinaryOp.LT,
    BinaryOp.LE,
    BinaryOp.GT,
    BinaryOp.GE,
}

# Tipos numericos soportados por el almacenamiento.
_NUMERIC_TYPES = {'INT', 'FLOAT'}

# Tipos soportados por el almacenamiento.
_SUPPORTED_TYPES = {'INT', 'FLOAT', 'CHAR'}


@dataclass
class _TableInfo:
    """Estado semantico de una tabla declarada dentro del Program."""
    metadata: TableMetadata | None
    storage: StorageKind
    primary_key: str | None
    pending_unique: set[str]


@dataclass
class _IndexInfo:
    """Estado semantico de un indice declarado dentro del Program."""
    table: str
    column: str
    kind: IndexKind


def _storage_kind(storage: StorageType) -> StorageKind:
    if storage is StorageType.SEQUENTIAL:
        return StorageKind.SEQUENTIAL
    return StorageKind.HEAP


def _index_kind(index_type: IndexType) -> IndexKind:
    if index_type is IndexType.HASH:
        return IndexKind.HASH
    return IndexKind.BTREE


def _table_meta_from_columns(
    name: str,
    columns: list[ColumnDef],
    primary_key: str | None,
) -> TableMetadata:
    built: list[dict] = []
    for column in columns:
        built.append({
            'name': column.name,
            'type': column.data_type.name,
            'size': column.size,
            'unique': bool(column.unique),
        })
    return TableMetadata(name, built, primary_key)


def _table_meta_with_unique(
    meta: TableMetadata,
    column_name: str,
) -> TableMetadata:
    """Copia el esquema marcando una columna como UNIQUE."""
    columns: list[dict] = []
    for column in meta.columns:
        copied = dict(column)
        if copied['name'] == column_name:
            copied['unique'] = True
        columns.append(copied)
    return TableMetadata(meta.name, columns, meta.primary_key)


class SemanticVisitor(Visitor[None]):

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._new_tables: dict[str, _TableInfo] = {}
        self._new_indexes: dict[str, _IndexInfo] = {}
        self._current_table: TableMetadata | None = None

    def visit_program(self, node: Program) -> None:
        self._new_tables = {}
        self._new_indexes = {}
        self._current_table = None

        for statement in node.statements:
            statement.accept(self)

    def visit_create_table(self, node: CreateTableStm) -> None:
        if self._table_exists(node.name):
            raise ValueError(
                f"Error semantico: la tabla '{node.name}' ya existe"
            )

        storage = _storage_kind(node.storage)

        if node.columns is None:
            # Esquema inferido del CSV: el Executor lo conoce al cargarlo.
            self._new_tables[node.name] = _TableInfo(
                metadata=None,
                storage=storage,
                primary_key=node.primary_key,
                pending_unique=set(),
            )
            return

        column_names: set[str] = set()
        primary_key_count = 0

        for column in node.columns:
            if column.name in column_names:
                raise ValueError(
                    f"Error semantico: la columna '{column.name}' esta "
                    f"repetida en la tabla '{node.name}'"
                )
            column_names.add(column.name)

            if column.primary_key:
                primary_key_count += 1

            self._check_column_type(column)

        if primary_key_count > 1:
            raise ValueError(
                f"Error semantico: la tabla '{node.name}' tiene mas de una "
                f"PRIMARY KEY"
            )

        if node.primary_key is not None and node.primary_key not in column_names:
            raise ValueError(
                f"Error semantico: la PRIMARY KEY '{node.primary_key}' no "
                f"existe en la tabla '{node.name}'"
            )

        self._new_tables[node.name] = _TableInfo(
            metadata=_table_meta_from_columns(
                node.name, node.columns, node.primary_key
            ),
            storage=storage,
            primary_key=node.primary_key,
            pending_unique=set(),
        )

    def _check_column_type(self, column: ColumnDef) -> None:
        type_name = column.data_type.name

        if type_name not in _SUPPORTED_TYPES:
            raise ValueError(
                f"Error semantico: tipo '{type_name}' no soportado en la "
                f"columna '{column.name}'"
            )

        if type_name == 'CHAR':
            if column.size is None or column.size <= 0:
                raise ValueError(
                    f"Error semantico: CHAR debe tener un tamano mayor que 0 "
                    f"en la columna '{column.name}'"
                )

    def visit_create_index(self, node: CreateIndexStm) -> None:
        if not self._table_exists(node.table):
            raise ValueError(
                f"Error semantico: la tabla '{node.table}' no existe"
            )

        if self._get_table_storage(node.table) is StorageKind.SEQUENTIAL:
            raise ValueError(
                f"Error semantico: no se permiten indices sobre la tabla "
                f"SEQUENTIAL '{node.table}'"
            )

        metadata = self._get_table_meta(node.table)
        if metadata is not None:
            self._require_column(metadata, node.column)

        if self._index_exists(node.name):
            raise ValueError(
                f"Error semantico: el indice '{node.name}' ya existe"
            )

        self._new_indexes[node.name] = _IndexInfo(
            table=node.table,
            column=node.column,
            kind=_index_kind(node.index_type),
        )

    def visit_insert(self, node: InsertStm) -> None:
        if not self._table_exists(node.table):
            raise ValueError(
                f"Error semantico: la tabla '{node.table}' no existe"
            )

        metadata = self._get_table_meta(node.table)
        if metadata is None:
            return

        if len(node.values) != len(metadata.columns):
            raise ValueError(
                f"Error semantico: INSERT en '{node.table}' espera "
                f"{len(metadata.columns)} valores y recibio "
                f"{len(node.values)}"
            )

        for column, value in zip(metadata.columns, node.values):
            if not isinstance(value, LiteralExp):
                raise ValueError(
                    f"Error semantico: valor no valido en INSERT sobre "
                    f"'{node.table}'"
                )
            if not self._literal_compatible(column, value.value):
                raise ValueError(
                    f"Error semantico: tipo incompatible para la columna "
                    f"'{column['name']}' de '{node.table}'"
                )

    def visit_alter(self, node: AlterStm) -> None:
        if not self._table_exists(node.table):
            raise ValueError(
                f"Error semantico: la tabla '{node.table}' no existe"
            )

        metadata = self._get_table_meta(node.table)

        if metadata is None:
            info = self._new_tables.get(node.table)
            if info is None:
                return
            if node.column in info.pending_unique:
                raise ValueError(
                    f"Error semantico: la columna '{node.column}' ya tiene "
                    f"UNIQUE en '{node.table}'"
                )
            info.pending_unique.add(node.column)
            return

        self._require_column(metadata, node.column)

        if metadata.is_unique_column(node.column):
            raise ValueError(
                f"Error semantico: la columna '{node.column}' ya tiene "
                f"UNIQUE en '{node.table}'"
            )

        self._new_tables[node.table] = _TableInfo(
            metadata=_table_meta_with_unique(metadata, node.column),
            storage=self._get_table_storage(node.table),
            primary_key=metadata.primary_key,
            pending_unique=set(),
        )

    def visit_select(self, node: SelectStm) -> None:
        if not self._table_exists(node.table):
            raise ValueError(
                f"Error semantico: la tabla '{node.table}' no existe"
            )

        metadata = self._get_table_meta(node.table)

        if node.columns is not None and metadata is not None:
            for column in node.columns:
                self._require_column(metadata, column)

        if node.force_indexes is not None:
            for index_name in node.force_indexes:
                info = self._get_index_info(index_name)
                if info is None:
                    raise ValueError(
                        f"Error semantico: el indice '{index_name}' no existe"
                    )
                if info.table != node.table:
                    raise ValueError(
                        f"Error semantico: el indice '{index_name}' no "
                        f"pertenece a la tabla '{node.table}'"
                    )

        self._validate_where(metadata, node.where)

    def visit_delete(self, node: DeleteStm) -> None:
        if not self._table_exists(node.table):
            raise ValueError(
                f"Error semantico: la tabla '{node.table}' no existe"
            )

        metadata = self._get_table_meta(node.table)
        self._validate_where(metadata, node.where)

    def _validate_where(
        self,
        metadata: TableMetadata | None,
        where,
    ) -> None:
        if where is None or metadata is None:
            return

        self._current_table = metadata
        try:
            where.accept(self)
        finally:
            self._current_table = None

    def visit_binary_exp(self, node: BinaryExp) -> None:
        if node.op is BinaryOp.AND or node.op is BinaryOp.OR:
            node.left.accept(self)
            node.right.accept(self)
            return

        if node.op in _COMPARISON_OPS:
            self._validate_comparison(node)

    def visit_unary_exp(self, node: UnaryExp) -> None:
        node.value.accept(self)

    def visit_literal_exp(self, node: LiteralExp) -> None:
        pass

    def visit_id_exp(self, node: IdExp) -> None:
        pass

    def _validate_comparison(self, node: BinaryExp) -> None:
        table = self._current_table
        if table is None:
            return

        left = node.left
        right = node.right

        if isinstance(left, IdExp) and isinstance(right, IdExp):
            self._require_column(table, left.value)
            self._require_column(table, right.value)
            left_type = self._column_type(table, left.value)
            right_type = self._column_type(table, right.value)
            if not self._types_compatible(left_type, right_type):
                raise ValueError(
                    f"Error semantico: comparacion incompatible entre "
                    f"'{left.value}' y '{right.value}'"
                )
            return

        if isinstance(left, IdExp) and isinstance(right, LiteralExp):
            self._validate_column_literal(table, left.value, right.value)
            return

        if isinstance(right, IdExp) and isinstance(left, LiteralExp):
            self._validate_column_literal(table, right.value, left.value)
            return

        # Literal contra literal: permitido.

    def _validate_column_literal(
        self,
        table: TableMetadata,
        column_name: str,
        value,
    ) -> None:
        self._require_column(table, column_name)
        column = table.columns[table.col_index(column_name)]
        if not self._literal_compatible(column, value):
            raise ValueError(
                f"Error semantico: tipo incompatible en la columna "
                f"'{column_name}' de '{table.name}'"
            )

 
    # Helpers de contexto (overlay -> catalogo)
    def _table_exists(self, name: str) -> bool:
        if name in self._new_tables:
            return True
        return self.catalog.table_exists(name)

    def _get_table_meta(self, name: str) -> TableMetadata | None:
        if name in self._new_tables:
            return self._new_tables[name].metadata
        return self.catalog.get_table(name).metadata

    def _get_table_storage(self, name: str) -> StorageKind:
        if name in self._new_tables:
            return self._new_tables[name].storage
        return self.catalog.get_table(name).storage

    def _index_exists(self, name: str) -> bool:
        if name in self._new_indexes:
            return True
        return self.catalog.index_exists(name)

    def _get_index_info(self, name: str) -> _IndexInfo | IndexMetadata | None:
        if name in self._new_indexes:
            return self._new_indexes[name]
        if self.catalog.index_exists(name):
            return self.catalog.get_index(name)
        return None

    # Helpers de esquema
    def _require_column(self, metadata: TableMetadata, name: str) -> None:
        try:
            metadata.col_index(name)
        except ValueError:
            raise ValueError(
                f"Error semantico: la columna '{name}' no existe en la tabla "
                f"'{metadata.name}'"
            )

    @staticmethod
    def _column_type(metadata: TableMetadata, name: str) -> str:
        return metadata.columns[metadata.col_index(name)]['type'].upper()

    @staticmethod
    def _types_compatible(left: str, right: str) -> bool:
        if left in _NUMERIC_TYPES and right in _NUMERIC_TYPES:
            return True
        return left == right

    @staticmethod
    def _literal_compatible(column, value) -> bool:
        type_name = column['type'].upper()

        if type_name in _NUMERIC_TYPES:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        if type_name == 'CHAR':
            return isinstance(value, str)

        return False
