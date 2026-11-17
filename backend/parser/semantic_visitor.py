from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .visitor import Visitor

from .ast import (
    Program,
    CreateTableStm,
    CreateIndexStm,
    InsertStm,
    AlterStm,
    SelectStm,
    DeleteStm,
    SetStm,
    BinaryExp,
    UnaryExp,
    LiteralExp,
    IdExp,
    MatchExp,
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
_SUPPORTED_TYPES = {'INT', 'FLOAT', 'CHAR', 'VECTOR'}


class SemanticError(ValueError):
    """Error semantico asociado a una sentencia concreta del Program."""

    def __init__(self, statement_id: int, message: str) -> None:
        super().__init__(message)
        self.statement_id = statement_id


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
    if index_type is IndexType.INVERTED:
        return IndexKind.INVERTED
    if index_type is IndexType.RTREE:
        return IndexKind.RTREE
    return IndexKind.BTREE


def _collect_matches(expression) -> list[MatchExp]:
    found: list[MatchExp] = []

    def visit(node) -> None:
        if isinstance(node, MatchExp):
            found.append(node)
        elif isinstance(node, BinaryExp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnaryExp):
            visit(node.value)

    visit(expression)
    return found


def _top_level_conjuncts(expression) -> list:
    if isinstance(expression, BinaryExp) and expression.op is BinaryOp.AND:
        return _top_level_conjuncts(expression.left) + _top_level_conjuncts(expression.right)
    return [expression]


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


class SemanticVisitor(Visitor[None, None]):

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._new_tables: dict[str, _TableInfo] = {}
        self._new_indexes: dict[str, _IndexInfo] = {}
        self._current_table: TableMetadata | None = None

    def visit_program(self, node: Program) -> None:
        self._new_tables = {}
        self._new_indexes = {}
        self._current_table = None

        for statement_id, statement in enumerate(node.statements, start=1):
            try:
                statement.accept(self)
            except SemanticError:
                raise
            except ValueError as exc:
                raise SemanticError(statement_id, str(exc)) from exc

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

            if (
                storage is StorageKind.SEQUENTIAL
                and column.unique
                and not column.primary_key
            ):
                raise ValueError(
                    f"Error semantico: la tabla SEQUENTIAL '{node.name}' no "
                    f"admite restricciones UNIQUE en columnas"
                )

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

        if type_name == 'VECTOR':
            if column.size is None or column.size <= 0:
                raise ValueError(
                    f"Error semantico: VECTOR debe tener una dimension mayor "
                    f"que 0 en la columna '{column.name}'"
                )

            if column.primary_key or column.unique:
                raise ValueError(
                    f"Error semantico: la columna VECTOR '{column.name}' no "
                    f"admite PRIMARY KEY ni UNIQUE"
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

            vector_column = self._column_type(metadata, node.column) == 'VECTOR'

            if vector_column and node.index_type not in (
                IndexType.IVF, IndexType.INVERTED, IndexType.RTREE,
            ):
                raise ValueError(
                    f"Error semantico: la columna VECTOR '{node.column}' solo "
                    f"admite indices IVF"
                )

        if self._index_exists(node.name):
            raise ValueError(
                f"Error semantico: el indice '{node.name}' ya existe"
            )

        if node.index_type is IndexType.INVERTED:
            from textindex.options import normalize_text_options
            if metadata is not None:
                if self._column_type(metadata, node.column) != 'CHAR':
                    raise ValueError(
                        f"Error semantico: el indice INVERTED requiere una "
                        f"columna CHAR y '{node.column}' no lo es"
                    )
            normalize_text_options(node.options)
        elif node.index_type is IndexType.IVF:
            from vectorindex.options import normalize_vector_options
            if metadata is not None:
                if self._column_type(metadata, node.column) != 'VECTOR':
                    raise ValueError(
                        f"Error semantico: el indice IVF requiere una "
                        f"columna VECTOR y '{node.column}' no lo es"
                    )
            normalize_vector_options(node.options)
        elif node.options:
            raise ValueError(
                "Error semantico: WITH (...) no se admite en este tipo de indice"
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

        if self._get_table_storage(node.table) is StorageKind.SEQUENTIAL:
            raise ValueError(
                f"Error semantico: la tabla SEQUENTIAL '{node.table}' no "
                f"admite restricciones UNIQUE"
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

        self._validate_where(metadata, node.where)
        self._validate_text_select(node, metadata)
        self._validate_vector_select(node, metadata)

        self._validate_force(
            node.table, node.force_indexes, node.force_seqscan, node.force_all_indexes
        )

    def visit_set(self, node: SetStm) -> None:
        if node.name != 'ivf.probes':
            raise ValueError(
                f"Error semantico: variable de sesion no soportada: "
                f"'{node.name}'"
            )

        if not isinstance(node.value, int) or isinstance(node.value, bool) \
                or node.value <= 0:
            raise ValueError(
                "Error semantico: ivf.probes debe ser un entero positivo"
            )

    def visit_delete(self, node: DeleteStm) -> None:
        if not self._table_exists(node.table):
            raise ValueError(
                f"Error semantico: la tabla '{node.table}' no existe"
            )

        metadata = self._get_table_meta(node.table)
        if node.where is not None and _collect_matches(node.where):
            raise ValueError(
                "Error semantico: MATCH solo se admite en SELECT"
            )
        self._validate_where(metadata, node.where)

        self._validate_force(
            node.table, node.force_indexes, node.force_seqscan, node.force_all_indexes
        )

    def _validate_text_select(self, node: SelectStm, metadata) -> None:
        """Reglas de MATCH, COSINE_SCORE, ORDER BY y LIMIT."""
        matches = _collect_matches(node.where) if node.where is not None else []
        if len(matches) > 1:
            raise ValueError(
                "Error semantico: solo se admite un MATCH por consulta"
            )
        if matches and not any(
            isinstance(item, MatchExp) for item in _top_level_conjuncts(node.where)
        ):
            raise ValueError(
                "Error semantico: MATCH debe combinarse con el resto del WHERE "
                "solo mediante AND (no dentro de OR ni de NOT)"
            )
        if node.score_alias is not None and not matches:
            raise ValueError(
                "Error semantico: COSINE_SCORE() requiere un MATCH en el WHERE"
            )
        if node.score_alias is not None and metadata is not None:
            try:
                metadata.col_index(node.score_alias)
            except ValueError:
                pass
            else:
                raise ValueError(
                    f"Error semantico: el alias '{node.score_alias}' coincide "
                    f"con una columna de '{metadata.name}'"
                )
        if node.order_by is not None and node.order_by != node.score_alias:
            if metadata is not None and not (
                node.vector_distance is not None
                and node.order_by == node.vector_distance.alias
            ):
                self._require_column(metadata, node.order_by)
        if node.limit is not None and node.limit < 0:
            raise ValueError("Error semantico: LIMIT debe ser no negativo")

    def _validate_vector_select(self, node: SelectStm, metadata) -> None:
        """Reglas de la distancia vectorial en la lista de salida."""
        distance = node.vector_distance

        if distance is None:
            return

        if self._get_table_storage(node.table) is StorageKind.SEQUENTIAL:
            raise ValueError(
                "Error semantico: las consultas vectoriales requieren una "
                f"tabla HEAP y '{node.table}' es SEQUENTIAL"
            )

        if node.score_alias is not None:
            raise ValueError(
                "Error semantico: COSINE_SCORE() y una distancia vectorial "
                "no pueden combinarse en la misma consulta"
            )

        if metadata is not None:
            self._require_column(metadata, distance.column)
            column = metadata.columns[metadata.col_index(distance.column)]

            if column['type'].upper() != 'VECTOR':
                raise ValueError(
                    f"Error semantico: la distancia vectorial requiere una "
                    f"columna VECTOR y '{distance.column}' no lo es"
                )

            from vectorindex.vectors import VectorError, parse_vector

            try:
                vector = parse_vector(distance.literal)
            except VectorError as exc:
                raise ValueError(f"Error semantico: {exc}") from exc

            dimension = column.get('dim')

            if dimension is not None and len(vector) != dimension:
                raise ValueError(
                    f"Error semantico: el vector de consulta tiene "
                    f"{len(vector)} componentes y la columna "
                    f"'{distance.column}' espera {dimension}"
                )

            try:
                metadata.col_index(distance.alias)
            except ValueError:
                pass
            else:
                raise ValueError(
                    f"Error semantico: el alias '{distance.alias}' coincide "
                    f"con una columna de '{metadata.name}'"
                )

        if node.order_by is not None and node.order_by != distance.alias:
            if metadata is not None:
                self._require_column(metadata, node.order_by)

        if node.order_by == distance.alias and node.order_desc:
            raise ValueError(
                "Error semantico: la distancia vectorial solo se ordena ASC"
            )

        if node.limit is not None and node.limit < 0:
            raise ValueError("Error semantico: LIMIT debe ser no negativo")

    def _validate_force(
        self,
        table: str,
        force_indexes: list[str] | None,
        force_seqscan: bool,
        force_all_indexes: bool,
    ) -> None:
        # SequentialFile no admite FORCE: ni indexes ni seqscan.
        if self._get_table_storage(table) is StorageKind.SEQUENTIAL:
            if force_indexes is not None or force_seqscan or force_all_indexes:
                raise ValueError(
                    f"Error semantico: la tabla SEQUENTIAL '{table}' no "
                    f"admite FORCE"
                )
            return

        if force_all_indexes:
            return

        if force_indexes is None:
            return

        seen: set[str] = set()
        for index_name in force_indexes:
            if index_name in seen:
                raise ValueError(
                    f"Error semantico: el indice '{index_name}' esta "
                    f"repetido en FORCE INDEX"
                )
            seen.add(index_name)

            info = self._get_index_info(index_name)
            if info is None:
                raise ValueError(
                    f"Error semantico: el indice '{index_name}' no existe"
                )
            if info.table != table:
                raise ValueError(
                    f"Error semantico: el indice '{index_name}' no "
                    f"pertenece a la tabla '{table}'"
                )

    def _validate_where(
        self,
        metadata: TableMetadata | None,
        where,
    ) -> None:
        if where is None:
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

    def visit_match_exp(self, node: MatchExp) -> None:
        table = self._current_table
        if table is None:
            return
        self._require_column(table, node.column)
        if self._column_type(table, node.column) != 'CHAR':
            raise ValueError(
                f"Error semantico: MATCH requiere una columna CHAR y "
                f"'{node.column}' no lo es"
            )

    def _validate_comparison(self, node: BinaryExp) -> None:
        left = node.left
        right = node.right

        if isinstance(left, LiteralExp) and isinstance(right, LiteralExp):
            self._validate_literal_literal(left, right)
            return

        table = self._current_table
        if table is None:
            return

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

    def _validate_literal_literal(
        self,
        left: LiteralExp,
        right: LiteralExp,
    ) -> None:
        left_kind = self._literal_kind(left.value)
        right_kind = self._literal_kind(right.value)
        if left_kind != right_kind or left_kind == 'unknown':
            raise ValueError(
                f"Error semantico: comparacion incompatible entre literales "
                f"{left.value!r} y {right.value!r}"
            )

    @staticmethod
    def _literal_kind(value) -> str:
        if isinstance(value, bool):
            return 'unknown'
        if isinstance(value, (int, float)):
            return 'number'
        if isinstance(value, str):
            return 'text'
        return 'unknown'

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

        if type_name == 'VECTOR':
            return isinstance(value, str)

        return False
