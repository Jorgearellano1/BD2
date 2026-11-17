from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from parser.visitor import Visitor
from rtree.rtreeNode import MBR

from parser.ast import (
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
    ColumnDef,
    StorageType,
    IndexType,
    BinaryOp,
    UnaryOp,
)

from engine.conditions import (
    Operand,
    ColumnRef,
    LiteralValue,
    Expression,
    Condition,
    LogicalExpression,
    ComparisonOperator,
    LogicalOperator,
)
from engine.metadata import (
    IndexKind,
    Key,
    StorageKind,
    TableMetadata,
)
from engine.plan import (
    PlanNode,
    ProgramPlan,
    StatementInfo,
    CreateTablePlan,
    CreateIndexPlan,
    InsertPlan,
    AddUniquePlan,
    DeletePlan,
    SequentialDeletePlan,
    SeqScanPlan,
    SeqScanRidsPlan,
    IndexScanPlan,
    IndexRangeScanPlan,
    RangeSpec,
    SequentialRangeScanPlan,
    BitmapIndexScanPlan,
    BitmapIndexRangeScanPlan,
    BitmapAndPlan,
    BitmapOrPlan,
    HeapFetchPlan,
    BitmapHeapScanPlan,
    FilterPlan,
    FilterRidsPlan,
    EmptyResultPlan,
    ProjectPlan,
    TextIndexScanPlan,
    TextFetchPlan,
    FullTextScanPlan,
    SortPlan,
    LimitPlan,
    TextProjectPlan,
)


if TYPE_CHECKING:
    from engine.catalog import Catalog
    from engine.metadata import IndexMetadata


PlannerExpression = Operand | Expression


_RANGE_OPERATORS = {
    ComparisonOperator.LT,
    ComparisonOperator.LE,
    ComparisonOperator.GT,
    ComparisonOperator.GE,
}


class PlannerError(ValueError):
    """Error producido cuando no existe un plan valido bajo FORCE."""


@dataclass
class AccessResult:
    """
    Resultado intermedio de construir un access path.

    plan:
        Operacion de acceso construida.

    used_indexes:
        Indices que realmente participan en el access path.

    residual:
        Parte del WHERE que todavia debe evaluarse sobre registros.
        Este campo solo existe durante la planificacion; luego se
        convierte en FilterPlan o FilterRidsPlan segun la sentencia.
    """

    plan: PlanNode
    used_indexes: set[str]
    residual: Expression | None


@dataclass(frozen=True)
class _Bound:
    value: Key
    inclusive: bool


@dataclass(frozen=True)
class _BooleanExpression(Expression):
    """
    Constante booleana interna del Planner.

    No forma parte del plan final ni de conditions.py. Se usa
    solamente durante la simplificacion y propagacion de constantes.
    """

    value: bool


class PlannerVisitor(Visitor[PlanNode, PlannerExpression]):
    """
    Construye planes de forma determinista a partir del AST.

    No abre almacenamiento ni indices. El Catalog solo aporta metadata.
    """

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    # Statements

    def visit_program(self, node: Program) -> PlanNode:
        statements: list[PlanNode] = []
        infos: list[StatementInfo] = []

        for statement_id, statement in enumerate(node.statements, start=1):
            statements.append(statement.accept(self))
            infos.append(self.statement_info(statement_id, statement))

        return ProgramPlan(statements=statements, statement_infos=infos)

    @staticmethod
    def statement_info(statement_id: int, statement) -> StatementInfo:
        if isinstance(statement, CreateTableStm):
            return StatementInfo(statement_id, "CREATE_TABLE", statement.name, None)
        if isinstance(statement, CreateIndexStm):
            return StatementInfo(statement_id, "CREATE_INDEX", statement.table, None)
        if isinstance(statement, InsertStm):
            return StatementInfo(statement_id, "INSERT", statement.table, None)
        if isinstance(statement, AlterStm):
            return StatementInfo(statement_id, "ALTER_TABLE", statement.table, None)
        if isinstance(statement, SelectStm):
            return StatementInfo(statement_id, "SELECT", statement.table, statement.output_columns())
        if isinstance(statement, DeleteStm):
            return StatementInfo(statement_id, "DELETE", statement.table, None)
        return StatementInfo(statement_id, "UNKNOWN", None, None)

    def visit_create_table(self, node: CreateTableStm) -> PlanNode:
        metadata = None
        if node.columns is not None:
            metadata = self._table_metadata_from_columns(
                node.name,
                node.columns,
                node.primary_key,
            )

        csv_path = node.csv.path if node.csv is not None else None
        delimiter = node.csv.delimiter if node.csv is not None else ","

        return CreateTablePlan(
            name=node.name,
            storage=self._storage_kind(node.storage),
            metadata=metadata,
            csv_path=csv_path,
            delimiter=delimiter,
            primary_key=node.primary_key,
        )

    
    def visit_create_index(self, node: CreateIndexStm) -> PlanNode:
        kind = self._index_kind(node.index_type)
        options = None
        if kind is IndexKind.INVERTED:
            from textindex.options import normalize_text_options
            options = normalize_text_options(node.options)
        return CreateIndexPlan(
            name=node.name,
            table=node.table,
            column=node.column,
            index_kind=kind,
            spatial_columns=node.spatial_columns,
            options=options,
        )


    def visit_insert(self, node: InsertStm) -> PlanNode:
        values: list[Key] = []

        for value in node.values:
            if not isinstance(value, LiteralExp):
                raise PlannerError(f"INSERT en '{node.table}' contiene un valor no literal")
            values.append(value.value)

        return InsertPlan(table=node.table, values=tuple(values),)

    def visit_alter(self, node: AlterStm) -> PlanNode:
        return AddUniquePlan(table=node.table, column=node.column,)

    def visit_select(self, node: SelectStm) -> PlanNode:
        if node.where is not None and self._find_match(node.where) is not None:
            return self._build_text_select(node)

        expression = (node.where.accept(self) if node.where is not None else None)

        if self._is_sequential_table(node.table):
            access = self._build_sequential_access(
                table=node.table,
                expression=expression,
                force_indexes=node.force_indexes,
                force_seqscan=node.force_seqscan,
                force_all_indexes=node.force_all_indexes,
                produce_rids=False,
            )
        else:
            access = self._build_access(
                table=node.table,
                expression=expression,
                force_indexes=node.force_indexes,
                force_seqscan=node.force_seqscan,
                force_all_indexes=node.force_all_indexes,
                produce_rids=False,
            )

        plan = self._records_from_access(node.table, access.plan)

        if access.residual is not None:
            plan = FilterPlan(predicate=access.residual, child=plan,)

        if node.order_by is not None:
            plan = SortPlan(column=node.order_by, descending=node.order_desc, child=plan)

        if node.limit is not None:
            plan = LimitPlan(limit=node.limit, child=plan)

        if node.columns is not None:
            plan = ProjectPlan(columns=node.columns, child=plan,)

        return plan

    # ------------------------------------------------------------------
    # SELECT con MATCH (modulo textual)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_match(expression) -> MatchExp | None:
        if isinstance(expression, MatchExp):
            return expression
        if isinstance(expression, BinaryExp):
            return (PlannerVisitor._find_match(expression.left)
                    or PlannerVisitor._find_match(expression.right))
        if isinstance(expression, UnaryExp):
            return PlannerVisitor._find_match(expression.value)
        return None

    @staticmethod
    def _split_match(expression) -> tuple[MatchExp, list]:
        """Separa el MATCH de los demas conjuntos del AND de primer nivel."""
        conjuncts: list = []

        def flatten(node) -> None:
            if isinstance(node, BinaryExp) and node.op is BinaryOp.AND:
                flatten(node.left)
                flatten(node.right)
            else:
                conjuncts.append(node)

        flatten(expression)
        match = next((item for item in conjuncts if isinstance(item, MatchExp)), None)
        if match is None:
            raise PlannerError("MATCH solo puede combinarse con AND en el WHERE")
        rest = [item for item in conjuncts if item is not match]
        if any(PlannerVisitor._find_match(item) is not None for item in rest):
            raise PlannerError("Solo se admite un MATCH por consulta")
        return match, rest

    def _text_index_for(self, node: SelectStm, column: str):
        """Indice INVERTED a usar (o None para Full Text Scan) segun FORCE."""
        index = self.catalog.find_index(node.table, column, IndexKind.INVERTED)
        if node.force_seqscan:
            return None
        if node.force_indexes is not None:
            for name in node.force_indexes:
                forced = self.catalog.get_index(name)
                if forced.kind is IndexKind.INVERTED and (index is None or forced.name != index.name):
                    raise PlannerError(
                        f"El indice '{name}' no corresponde a la columna '{column}' del MATCH"
                    )
            if index is not None and index.name in node.force_indexes:
                return index
            return None
        if node.force_all_indexes and index is None:
            raise PlannerError(
                f"FORCE INDEX(*): no existe indice INVERTED sobre '{node.table}({column})'"
            )
        return index

    def _build_text_select(self, node: SelectStm) -> PlanNode:
        match, rest = self._split_match(node.where)

        relational: Expression | None = None
        if rest:
            ast = rest[0]
            for item in rest[1:]:
                ast = BinaryExp(ast, item, BinaryOp.AND)
            relational = self._as_expression(ast.accept(self))
            constant = self._constant_boolean(relational)
            if constant is True:
                relational = None
            elif constant is False:
                plan: PlanNode = EmptyResultPlan()
                return self._text_output(node, plan)

        score_order = node.order_by is not None and node.order_by == node.score_alias
        rank = node.score_alias is not None or score_order
        column_order = node.order_by if node.order_by is not None and not score_order else None
        descending = node.order_desc if score_order else True
        # El LIMIT se empuja al Top-K del indice solo si el orden final es el
        # del score (o el orden por doc_id del modo booleano).
        top_k = node.limit if column_order is None else None

        index = None if self._is_sequential_table(node.table) else self._text_index_for(node, match.column)

        if index is None:
            existing = self.catalog.find_index(node.table, match.column, IndexKind.INVERTED)
            from textindex.options import normalize_text_options
            options = normalize_text_options(
                {k: v for k, v in (existing.options or {}).items() if k != 'block_postings'}
                if existing is not None else None
            )
            plan = FullTextScanPlan(
                table=node.table,
                column=match.column,
                query=match.query,
                options=options,
                rank=rank,
                top_k=top_k,
                descending=descending,
                residual=relational,
            )
        else:
            prefilter = None
            post = relational
            strategy = 'INDEX_ONLY'
            if relational is not None:
                strategy = 'POST_FILTER'
                relational_force = None
                if node.force_indexes is not None:
                    relational_force = [
                        name for name in node.force_indexes
                        if self.catalog.get_index(name).kind is not IndexKind.INVERTED
                    ]
                if relational_force != []:
                    access = self._build_access(
                        table=node.table,
                        expression=relational,
                        force_indexes=relational_force or None,
                        force_seqscan=False,
                        force_all_indexes=False,
                        produce_rids=True,
                    )
                    if access.used_indexes:
                        prefilter = access.plan
                        post = access.residual
                        strategy = 'PRE_FILTER'
            scan = TextIndexScanPlan(
                table=node.table,
                index=index.name,
                column=match.column,
                query=match.query,
                rank=rank,
                top_k=top_k if post is None else None,
                descending=descending,
                strategy=strategy,
                prefilter=prefilter,
            )
            plan = TextFetchPlan(
                table=node.table,
                child=scan,
                residual=post,
                limit=top_k if post is not None else None,
            )

        if column_order is not None:
            plan = SortPlan(column=column_order, descending=node.order_desc, child=plan)
            if node.limit is not None:
                plan = LimitPlan(limit=node.limit, child=plan)

        return self._text_output(node, plan)

    @staticmethod
    def _text_output(node: SelectStm, plan: PlanNode) -> PlanNode:
        if node.score_alias is not None:
            return TextProjectPlan(
                columns=list(node.columns or []),
                score_alias=node.score_alias,
                score_position=node.score_position or 0,
                child=plan,
            )
        if node.columns is not None:
            return ProjectPlan(columns=node.columns, child=plan)
        return plan

    def visit_delete(self, node: DeleteStm) -> PlanNode:
        expression = (node.where.accept(self) if node.where is not None else None)

        if self._is_sequential_table(node.table):
            access = self._build_sequential_access(
                table=node.table,
                expression=expression,
                force_indexes=node.force_indexes,
                force_seqscan=node.force_seqscan,
                force_all_indexes=node.force_all_indexes,
                produce_rids=True,
            )
            return access.plan

        access = self._build_access(
            table=node.table,
            expression=expression,
            force_indexes=node.force_indexes,
            force_seqscan=node.force_seqscan,
            force_all_indexes=node.force_all_indexes,
            produce_rids=True,
        )

        plan = access.plan

        if access.residual is not None:
            plan = FilterRidsPlan(
                table=node.table,
                predicate=access.residual,
                child=plan,
            )

        return DeletePlan(table=node.table, child=plan,)

    # Expressions: AST -> conditions.py + constant folding

    def visit_binary_exp(self, node: BinaryExp) -> PlannerExpression:
        if node.op is BinaryOp.AND:
            left = self._as_expression(node.left.accept(self))
            right = self._as_expression(node.right.accept(self))

            left_value = self._constant_boolean(left)
            right_value = self._constant_boolean(right)

            if left_value is False or right_value is False:
                return _BooleanExpression(False)

            if left_value is True:
                return right

            if right_value is True:
                return left

            # Idempotencia: A AND A -> A.
            if left == right:
                return left

            # Contradicciones seguras sobre una misma columna.
            if self._and_conflict(left, right):
                return _BooleanExpression(False)

            return LogicalExpression(
                operator=LogicalOperator.AND,
                operands=(left, right),
            )

        if node.op is BinaryOp.OR:
            left = self._as_expression(node.left.accept(self))
            right = self._as_expression(node.right.accept(self))

            left_value = self._constant_boolean(left)
            right_value = self._constant_boolean(right)

            if left_value is True or right_value is True:
                return _BooleanExpression(True)

            if left_value is False:
                return right

            if right_value is False:
                return left

            # Idempotencia: A OR A -> A.
            if left == right:
                return left

            return LogicalExpression(
                operator=LogicalOperator.OR,
                operands=(left, right),
            )

        left = self._as_operand(node.left.accept(self))
        right = self._as_operand(node.right.accept(self))

        condition = Condition(
            left=left,
            operator=self._comparison_operator(node.op),
            right=right,
        )

        condition = self._normalize_condition(condition)

        # Si ambos operandos son literales, la condicion es completamente
        # conocida en esta misma visita. No hace falta recorrerla otra vez.
        if isinstance(condition.left, LiteralValue) and isinstance(condition.right, LiteralValue):
            return _BooleanExpression(
                self._evaluate_literal_comparison(
                    condition.left.value,
                    condition.operator,
                    condition.right.value,
                )
            )

        return condition

    def visit_unary_exp(self, node: UnaryExp) -> PlannerExpression:
        if node.op is not UnaryOp.NOT:
            raise PlannerError(f"Operador unario no soportado: {node.op!r}")

        value = self._as_expression(node.value.accept(self))
        constant = self._constant_boolean(value)

        if constant is not None:
            return _BooleanExpression(not constant)

        # Simplificacion local adicional: NOT NOT A -> A.
        if (isinstance(value, LogicalExpression) and value.operator is LogicalOperator.NOT):
            if len(value.operands) != 1:
                raise PlannerError("NOT requiere exactamente un operando")
            return value.operands[0]

        return LogicalExpression(
            operator=LogicalOperator.NOT,
            operands=(value,),
        )

    def visit_literal_exp(self, node: LiteralExp) -> PlannerExpression:
        return LiteralValue(node.value)

    def visit_id_exp(self, node: IdExp) -> PlannerExpression:
        return ColumnRef(node.value)

    @staticmethod
    def _constant_boolean(expression: Expression) -> bool | None:
        if isinstance(expression, _BooleanExpression):
            return expression.value
        return None

    @staticmethod
    def _evaluate_literal_comparison(
        left: Key,
        operator: ComparisonOperator,
        right: Key,
    ) -> bool:
        if operator is ComparisonOperator.EQ:
            return left == right

        if operator is ComparisonOperator.NE:
            return left != right

        try:
            if operator is ComparisonOperator.LT:
                return left < right

            if operator is ComparisonOperator.LE:
                return left <= right

            if operator is ComparisonOperator.GT:
                return left > right

            if operator is ComparisonOperator.GE:
                return left >= right

        except TypeError as exc:
            raise PlannerError(f"No se puede evaluar la comparacion constante " f"{left!r} {operator.value} {right!r}") from exc

        raise PlannerError(f"Operador de comparacion no soportado: {operator!r}")

    # Contradicciones seguras detectadas durante la misma construccion.

    @classmethod
    def _and_conflict(cls, left: Expression, right: Expression) -> bool:
        constraints: dict[str, list[Condition]] = {}
        cls._collect_and_conditions(left, constraints)
        cls._collect_and_conditions(right, constraints)

        for conditions in constraints.values():
            if cls._conditions_conflict(conditions):
                return True

        return False

    @classmethod
    def _collect_and_conditions(
        cls,
        expression: Expression,
        out: dict[str, list[Condition]],
    ) -> None:
        if (isinstance(expression, LogicalExpression) and expression.operator is LogicalOperator.AND):
            for operand in expression.operands:
                cls._collect_and_conditions(operand, out)
            return

        if isinstance(expression, Condition):
            column, literal = cls._condition_column_literal(expression)
            if column is not None and literal is not None:
                out.setdefault(column.name, []).append(expression)

    @classmethod
    def _conditions_conflict(cls, conditions: Sequence[Condition]) -> bool:
        equalities: list[Key] = []
        ranges: list[tuple[ComparisonOperator, Key]] = []

        for condition in conditions:
            if condition.operator is ComparisonOperator.EQ:
                equalities.append(condition.right.value)
            elif condition.operator in _RANGE_OPERATORS:
                ranges.append((condition.operator, condition.right.value))

        for i, value in enumerate(equalities):
            for other in equalities[i + 1:]:
                if value != other:
                    return True

        for value in equalities:
            for operator, bound in ranges:
                if not cls._value_satisfies(value, operator, bound):
                    return True

        return cls._ranges_conflict(ranges)

    @staticmethod
    def _value_satisfies(
        value: Key,
        operator: ComparisonOperator,
        bound: Key,
    ) -> bool:
        if operator is ComparisonOperator.GT:
            return value > bound
        if operator is ComparisonOperator.GE:
            return value >= bound
        if operator is ComparisonOperator.LT:
            return value < bound
        if operator is ComparisonOperator.LE:
            return value <= bound
        return True

    @staticmethod
    def _ranges_conflict(ranges: Sequence[tuple[ComparisonOperator, Key]],) -> bool:
        lower: tuple[Key, bool] | None = None
        upper: tuple[Key, bool] | None = None

        for operator, value in ranges:
            if operator is ComparisonOperator.GT or operator is ComparisonOperator.GE:
                inclusive = operator is ComparisonOperator.GE
                if lower is None or value > lower[0]:
                    lower = (value, inclusive)
                elif value == lower[0]:
                    lower = (value, lower[1] and inclusive)
            else:
                inclusive = operator is ComparisonOperator.LE
                if upper is None or value < upper[0]:
                    upper = (value, inclusive)
                elif value == upper[0]:
                    upper = (value, upper[1] and inclusive)

        if lower is None or upper is None:
            return False

        if lower[0] > upper[0]:
            return True

        if lower[0] == upper[0]:
            return not (lower[1] and upper[1])

        return False


    # Core access construction
    def _build_access(
        self,
        table: str,
        expression: Expression | None,
        force_indexes: list[str] | None,
        force_seqscan: bool,
        force_all_indexes: bool,
        produce_rids: bool,
    ) -> AccessResult:
        """
        Aplica las reglas de FORCE y construye un solo access path.

        produce_rids:
            False -> SELECT / acceso que produce registros.
            True  -> DELETE / acceso que produce RIDs.
        """

        if force_seqscan and (force_indexes is not None or force_all_indexes):
            raise PlannerError("No se puede combinar FORCE SEQSCAN con FORCE INDEX")

        constant = (self._constant_boolean(expression) if expression is not None else None)

        # Un WHERE completamente falso termina en EmptyResultPlan y no
        # necesita ningun acceso, incluso si existia FORCE.
        if constant is False:
            return AccessResult(
                plan=EmptyResultPlan(),
                used_indexes=set(),
                residual=None,
            )

        # WHERE TRUE equivale a no tener WHERE.
        if constant is True:
            expression = None

        explicit_force = force_indexes is not None

        if force_seqscan:
            if produce_rids:
                return AccessResult(
                    plan=self._scan_plan(table, produce_rids),
                    used_indexes=set(),
                    residual=expression,
                )
            return AccessResult(
                plan=SeqScanPlan(table=table, predicate=expression),
                used_indexes=set(),
                residual=None,
            )

        if expression is None:
            if explicit_force:
                raise PlannerError(f"FORCE INDEX en '{table}' requiere una condicion " "indexable")

            if force_all_indexes:
                raise PlannerError(f"FORCE INDEX(*) en '{table}' no tiene ningun " "indice utilizable")

            return AccessResult(
                plan=self._scan_plan(table, produce_rids),
                used_indexes=set(),
                residual=None,
            )

        if explicit_force:
            allowed_indexes = self._explicit_indexes(table, force_indexes,)
            assignment = self._assign_forced_indexes(expression, allowed_indexes,)
        else:
            # SIN FORCE y FORCE INDEX(*) parten del mismo conjunto de
            # indices fisicamente asociados a la tabla.
            allowed_indexes = self.catalog.get_table_indexes(table)
            assignment = None

        result = self._build_expression_access(
            table=table,
            expression=expression,
            indexes=allowed_indexes,
            produce_rids=produce_rids,
            assignment=assignment,
        )

        if explicit_force:
            required = set(force_indexes or [])

            if result.used_indexes != required:
                missing = sorted(required - result.used_indexes)
                raise PlannerError(f"No se pudo utilizar todos los indices forzados " f"en '{table}': {', '.join(missing)}")

            return result

        if force_all_indexes and not result.used_indexes:
            raise PlannerError(f"FORCE INDEX(*) en '{table}' no tiene ningun " "indice utilizable")

        if not result.used_indexes:
            if isinstance(result.plan, EmptyResultPlan):
                return result

            if produce_rids:
                return AccessResult(
                    plan=self._scan_plan(table, produce_rids),
                    used_indexes=set(),
                    residual=expression,
                )

            return AccessResult(
                plan=SeqScanPlan(table=table, predicate=expression),
                used_indexes=set(),
                residual=None,
            )

        return result

    # SequentialFile access

    def _build_sequential_access(
        self,
        table: str,
        expression: Expression | None,
        force_indexes: list[str] | None,
        force_seqscan: bool,
        force_all_indexes: bool,
        produce_rids: bool,
    ) -> AccessResult:
        """
        Construye el acceso de una tabla con SequentialFile.

        SequentialFile no usa indices. SELECT usa SeqScanPlan(predicate) o
        SequentialRangeScanPlan sobre la PRIMARY KEY. DELETE usa
        SequentialDeletePlan(predicate).
        """

        if force_seqscan and (force_indexes is not None or force_all_indexes):
            raise PlannerError("No se puede combinar FORCE SEQSCAN con FORCE INDEX")

        constant = (self._constant_boolean(expression) if expression is not None else None)

        if constant is False:
            return AccessResult(
                plan=EmptyResultPlan(),
                used_indexes=set(),
                residual=None,
            )

        if constant is True:
            expression = None

        if force_indexes is not None:
            raise PlannerError(f"La tabla '{table}' utiliza SequentialFile y no puede utilizar FORCE INDEX")

        if force_all_indexes:
            raise PlannerError(f"FORCE INDEX(*) en '{table}' no tiene ningun indice utilizable")

        if produce_rids:
            return AccessResult(
                plan=SequentialDeletePlan(table=table, predicate=expression),
                used_indexes=set(),
                residual=None,
            )

        if force_seqscan:
            return AccessResult(
                plan=SeqScanPlan(table=table, predicate=expression),
                used_indexes=set(),
                residual=None,
            )

        return self._build_sequential_select_access(table, expression)

    def _build_sequential_select_access(
        self,
        table: str,
        expression: Expression | None,
    ) -> AccessResult:
        if expression is None:
            return AccessResult(
                plan=SeqScanPlan(table=table, predicate=None),
                used_indexes=set(),
                residual=None,
            )

        range_terms, residual_terms = self._split_pk_range(table, expression)

        if not range_terms:
            return AccessResult(
                plan=SeqScanPlan(table=table, predicate=expression),
                used_indexes=set(),
                residual=None,
            )

        primary_key = self.catalog.get_table(table).metadata.primary_key
        range_spec = self._range_spec(range_terms, table, primary_key)

        if not self._range_is_valid(range_spec):
            return AccessResult(
                plan=EmptyResultPlan(),
                used_indexes=set(),
                residual=None,
            )

        return AccessResult(
            plan=SequentialRangeScanPlan(table=table, column=primary_key, range=range_spec),
            used_indexes=set(),
            residual=self._and_expression(residual_terms),
        )

    def _split_pk_range(
        self,
        table: str,
        expression: Expression,
    ) -> tuple[list[Condition], list[Expression]]:
        """
        Separa las condiciones de rango sobre la PRIMARY KEY del resto.

        Solo un AND de condiciones simples puede producir un rango de PK.
        OR, NOT, igualdades y columnas que no son la PK quedan como residual.
        """

        primary_key = self.catalog.get_table(table).metadata.primary_key

        if primary_key is None:
            return [], [expression]

        if (isinstance(expression, LogicalExpression) and expression.operator is LogicalOperator.AND):
            terms = self._flatten_and(expression)
        else:
            terms = [expression]

        range_terms: list[Condition] = []
        residual_terms: list[Expression] = []

        for term in terms:
            if isinstance(term, Condition):
                column, literal = self._condition_column_literal(term)
                if (column is not None and literal is not None and column.name == primary_key and term.operator in _RANGE_OPERATORS):
                    range_terms.append(term)
                    continue
            residual_terms.append(term)

        return range_terms, residual_terms

    def _build_expression_access(
        self,
        table: str,
        expression: Expression,
        indexes: Sequence[IndexMetadata],
        produce_rids: bool,
        assignment: dict[Condition, IndexMetadata] | None,
    ) -> AccessResult:
        constant = self._constant_boolean(expression)

        if constant is False:
            return AccessResult(
                plan=EmptyResultPlan(),
                used_indexes=set(),
                residual=None,
            )

        if constant is True:
            return AccessResult(
                plan=self._scan_plan(table, produce_rids),
                used_indexes=set(),
                residual=None,
            )

        if isinstance(expression, Condition):
            return self._build_condition_access(
                table=table,
                condition=expression,
                indexes=indexes,
                produce_rids=produce_rids,
                assignment=assignment,
            )

        if isinstance(expression, LogicalExpression):
            if expression.operator is LogicalOperator.AND:
                return self._build_and_access(
                    table=table,
                    expression=expression,
                    indexes=indexes,
                    produce_rids=produce_rids,
                    assignment=assignment,
                )

            if expression.operator is LogicalOperator.OR:
                return self._build_or_access(
                    table=table,
                    expression=expression,
                    indexes=indexes,
                    produce_rids=produce_rids,
                    assignment=assignment,
                )

            if expression.operator is LogicalOperator.NOT:
                # NOT nunca se empuja al indice.
                return AccessResult(
                    plan=self._scan_plan(table, produce_rids),
                    used_indexes=set(),
                    residual=expression,
                )

        raise PlannerError(f"Expresion no soportada por el Planner: {type(expression)!r}")

    def _build_condition_access(
        self,
        table: str,
        condition: Condition,
        indexes: Sequence[IndexMetadata],
        produce_rids: bool,
        assignment: dict[Condition, IndexMetadata] | None,
    ) -> AccessResult:
        normalized = condition

        column, value = self._condition_column_literal(normalized)

        if column is None or value is None:
            return AccessResult(
                plan=self._scan_plan(table, produce_rids),
                used_indexes=set(),
                residual=normalized,
            )

        if assignment is not None:
            index = assignment.get(normalized)

            if index is not None:
                if normalized.operator is ComparisonOperator.EQ:
                    plan = self._build_equality_access(
                        table=table,
                        column=column.name,
                        value=value.value,
                        indexes=[index],
                    )
                else:
                    range_spec = self._build_range(
                        conditions=[normalized],
                        table=table,
                        indexes=[index],
                    )
                    plan = self._build_range_plan(
                        table=table,
                        indexes=[index],
                        range_spec=range_spec,
                    )

                return AccessResult(
                    plan=plan,
                    used_indexes={index.name},
                    residual=None,
                )

        compatible = self._compatible_indexes(
            indexes=indexes,
            table=table,
            column=column.name,
            operator=normalized.operator,
        )

        if not compatible:
            return AccessResult(
                plan=self._scan_plan(table, produce_rids),
                used_indexes=set(),
                residual=normalized,
            )

        if normalized.operator is ComparisonOperator.EQ:
            hash_indexes = [index for index in compatible if index.kind is IndexKind.HASH]

            if hash_indexes:
                selected = hash_indexes
            else:
                selected = [index for index in compatible if index.kind is IndexKind.BTREE]

            plan = self._build_equality_access(
                table=table,
                column=column.name,
                value=value.value,
                indexes=selected,
            )

            return AccessResult(
                plan=plan,
                used_indexes={index.name for index in selected},
                residual=None,
            )

        range_spec = self._build_range(
            conditions=[normalized],
            table=table,
            indexes=compatible,
        )

        plan = self._build_range_plan(
            table=table,
            indexes=compatible,
            range_spec=range_spec,
        )

        return AccessResult(
            plan=plan,
            used_indexes={index.name for index in compatible},
            residual=None,
        )

    
    def _build_and_access(
        self,
        table: str,
        expression: LogicalExpression,
        indexes: Sequence[IndexMetadata],
        produce_rids: bool,
        assignment: dict[Condition, IndexMetadata] | None,
    ) -> AccessResult:

        terms = self._flatten_and(expression)

        # 1. Buscar un indice R-Tree compatible con las dos columnas.
        for index in indexes:
            if (
                index.table != table
                or index.kind is not IndexKind.RTREE
                or index.spatial_columns is None
            ):
                continue

            latitude_column, longitude_column = index.spatial_columns

            if latitude_column == longitude_column:
                continue

            bounds = {
                latitude_column: {"lower": None, "upper": None},
                longitude_column: {"lower": None, "upper": None},
            }

            # Recoger los limites de latitud y longitud.
            for term in terms:
                if not isinstance(term, Condition):
                    continue

                column, literal = self._condition_column_literal(term)

                if column is None or literal is None:
                    continue

                if column.name not in bounds:
                    continue

                value = self._normalize_access_key(
                    table=table,
                    column=column.name,
                    value=literal.value,
                )

                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue

                value = float(value)
                column_bounds = bounds[column.name]

                if term.operator in {
                    ComparisonOperator.GT,
                    ComparisonOperator.GE,
                    ComparisonOperator.EQ,
                }:
                    current = column_bounds["lower"]

                    if current is None or value > current:
                        column_bounds["lower"] = value

                if term.operator in {
                    ComparisonOperator.LT,
                    ComparisonOperator.LE,
                    ComparisonOperator.EQ,
                }:
                    current = column_bounds["upper"]

                    if current is None or value < current:
                        column_bounds["upper"] = value

            latitude_bounds = bounds[latitude_column]
            longitude_bounds = bounds[longitude_column]

            # Necesitamos limite inferior y superior en ambos ejes.
            if (
                latitude_bounds["lower"] is None
                or latitude_bounds["upper"] is None
                or longitude_bounds["lower"] is None
                or longitude_bounds["upper"] is None
            ):
                continue

            lat_min = latitude_bounds["lower"]
            lat_max = latitude_bounds["upper"]
            lon_min = longitude_bounds["lower"]
            lon_max = longitude_bounds["upper"]

            # No construir un MBR con limites contradictorios.
            if lat_min > lat_max or lon_min > lon_max:
                continue

            # El R-Tree almacena X=longitud, Y=latitud.
            query_mbr = MBR(
                lon_min,
                lat_min,
                lon_max,
                lat_max,
            )

            spatial_plan = IndexScanPlan(
                table=table,
                index=index.name,
                index_kind=IndexKind.RTREE,
                key=query_mbr,
            )

            # Conservamos el WHERE completo como filtro residual.
            # Esto garantiza la semantica exacta de <, <=, > y >=.
            return AccessResult(
                plan=spatial_plan,
                used_indexes={index.name},
                residual=expression,
            )

        # 2. Si no hay un R-Tree aplicable, continuar con la
        #    estrategia original de HASH y BTREE.

        condition_groups: dict[str, list[tuple[int, Condition]]] = {}
        residual_terms: list[Expression] = []
        access_parts: list[AccessResult] = []

        for position, term in enumerate(terms):
            if isinstance(term, Condition):
                normalized = term
                column, value = self._condition_column_literal(normalized)

                if column is not None and value is not None:
                    condition_groups.setdefault(column.name, []).append(
                        (position, normalized)
                    )
                    continue

            if isinstance(term, LogicalExpression):
                sub_access = self._build_expression_access(
                    table=table,
                    expression=term,
                    indexes=indexes,
                    produce_rids=produce_rids,
                    assignment=assignment,
                )

                if isinstance(sub_access.plan, EmptyResultPlan):
                    return sub_access

                if sub_access.used_indexes:
                    access_parts.append(sub_access)

                if sub_access.residual is not None:
                    residual_terms.append(sub_access.residual)

                continue

            residual_terms.append(term)

        for _, grouped_conditions in condition_groups.items():
            group = self._build_condition_group_access(
                table=table,
                conditions=grouped_conditions,
                indexes=indexes,
                produce_rids=produce_rids,
                assignment=assignment,
            )

            if isinstance(group.plan, EmptyResultPlan):
                return group

            if group.used_indexes:
                access_parts.append(group)

            if group.residual is not None:
                residual_terms.append(group.residual)

        used_indexes: set[str] = set()
        access_plans: list[PlanNode] = []

        for part in access_parts:
            used_indexes.update(part.used_indexes)
            access_plans.append(part.plan)

        if not access_plans:
            plan = self._scan_plan(table, produce_rids)

        elif len(access_plans) == 1:
            plan = access_plans[0]

        else:
            bitmap_children = [
                self._as_bitmap(child)
                for child in access_plans
            ]
            plan = BitmapAndPlan(children=bitmap_children)

        residual = self._and_expression(residual_terms)

        return AccessResult(
            plan=plan,
            used_indexes=used_indexes,
            residual=residual,
        )


    def _build_condition_group_access(
        self,
        table: str,
        conditions: Sequence[tuple[int, Condition]],
        indexes: Sequence[IndexMetadata],
        produce_rids: bool,
        assignment: dict[Condition, IndexMetadata] | None,
    ) -> AccessResult:
        """
        Construye acceso para varias condiciones sobre una misma columna.

        Reglas:
        - los rangos de una misma columna se fusionan en un unico rango;
        - HASH domina a BTREE para la misma igualdad;
        - con FORCE INDEX se usa la asignacion global de igualdades;
        - si igualdad y rango tienen indices distintos, ambos participan.
        """

        indexed_positions: set[int] = set()
        used_indexes: set[str] = set()
        access_parts: list[PlanNode] = []

        equality_conditions = [(position, condition) for position, condition in conditions if condition.operator is ComparisonOperator.EQ]

        range_conditions = [
            (position, condition)
            for position, condition in conditions
            if condition.operator in {
                ComparisonOperator.LT,
                ComparisonOperator.LE,
                ComparisonOperator.GT,
                ComparisonOperator.GE,
            }
        ]

        # 1. Rangos: primero, para que un BTREE pueda quedar reservado
        #    para una condicion distinta de una igualdad.
        if range_conditions:
            first_range = range_conditions[0][1]
            column, _ = self._condition_column_literal(first_range)
            assert column is not None

            btree_indexes = [index for index in indexes if (index.table == table and index.column == column.name and index.kind is IndexKind.BTREE)]

            if btree_indexes:
                range_only = [condition for _, condition in range_conditions]
                range_spec = self._build_range(
                    conditions=range_only,
                    table=table,
                    indexes=btree_indexes,
                )

                if self._range_is_valid(range_spec):
                    range_plan = self._build_range_plan(
                        table=table,
                        indexes=btree_indexes,
                        range_spec=range_spec,
                    )
                    access_parts.append(range_plan)
                    used_indexes.update(index.name for index in btree_indexes)
                    indexed_positions.update(position for position, _ in range_conditions)
                else:
                    # La interseccion de rangos es vacia. No enviamos
                    # un rango invalido al indice: utilizamos una sola
                    # restriccion y dejamos el resto como residual.
                    fallback_position, fallback_condition = range_conditions[0]

                    fallback_compatible = btree_indexes
                    fallback_range = self._build_range(
                        conditions=[fallback_condition],
                        table=table,
                        indexes=fallback_compatible,
                    )
                    fallback_plan = self._build_range_plan(
                        table=table,
                        indexes=fallback_compatible,
                        range_spec=fallback_range,
                    )

                    access_parts.append(fallback_plan)
                    used_indexes.update(index.name for index in fallback_compatible)
                    indexed_positions.add(fallback_position)

        # 2. Igualdades.
        unique_equalities = []
        seen_equalities: set[Condition] = set()

        for position, condition in equality_conditions:
            if condition not in seen_equalities:
                unique_equalities.append((position, condition))
                seen_equalities.add(condition)

        for position, equality in unique_equalities:
            column, literal = self._condition_column_literal(equality)
            assert column is not None
            assert literal is not None

            assigned = (assignment.get(equality) if assignment is not None else None)

            if assigned is not None:
                selected = [assigned]
            else:
                compatible = self._compatible_indexes(
                    indexes=indexes,
                    table=table,
                    column=column.name,
                    operator=ComparisonOperator.EQ,
                )

                if not compatible:
                    continue

                hash_indexes = [index for index in compatible if index.kind is IndexKind.HASH]

                if hash_indexes:
                    selected = hash_indexes
                else:
                    selected = [index for index in compatible if index.kind is IndexKind.BTREE]

                if not selected:
                    continue

            access_parts.append(
                self._build_equality_access(
                    table=table,
                    column=column.name,
                    value=literal.value,
                    indexes=selected,
                )
            )
            used_indexes.update(index.name for index in selected)
            indexed_positions.add(position)

        # 3. Todo lo que no se pudo resolver mediante indice queda
        #    como residual.
        residual_conditions = [condition for position, condition in conditions if position not in indexed_positions]

        if not access_parts:
            return AccessResult(
                plan=self._scan_plan(table, produce_rids),
                used_indexes=set(),
                residual=self._and_expression(residual_conditions),
            )

        if len(access_parts) == 1:
            plan = access_parts[0]
        else:
            plan = BitmapAndPlan(children=[self._as_bitmap(part) for part in access_parts])

        return AccessResult(
            plan=plan,
            used_indexes=used_indexes,
            residual=self._and_expression(residual_conditions),
        )

    def _build_or_access(
        self,
        table: str,
        expression: LogicalExpression,
        indexes: Sequence[IndexMetadata],
        produce_rids: bool,
        assignment: dict[Condition, IndexMetadata] | None,
    ) -> AccessResult:
        if len(expression.operands) != 2:
            raise PlannerError("OR requiere exactamente dos operandos")

        left = self._build_expression_access(
            table=table,
            expression=expression.operands[0],
            indexes=indexes,
            produce_rids=produce_rids,
            assignment=assignment,
        )

        right = self._build_expression_access(
            table=table,
            expression=expression.operands[1],
            indexes=indexes,
            produce_rids=produce_rids,
            assignment=assignment,
        )

        if isinstance(left.plan, EmptyResultPlan):
            return right

        if isinstance(right.plan, EmptyResultPlan):
            return left

        # Un OR indexado debe ser completo en ambas ramas.
        if (not left.used_indexes or left.residual is not None or not right.used_indexes or right.residual is not None):
            return AccessResult(
                plan=self._scan_plan(table, produce_rids),
                used_indexes=set(),
                residual=expression,
            )

        return AccessResult(
            plan=BitmapOrPlan(
                children=[
                    self._as_bitmap(left.plan),
                    self._as_bitmap(right.plan),
                ]
            ),
            used_indexes=left.used_indexes | right.used_indexes,
            residual=None,
        )

    # Asignacion global de indices forzados (FORCE INDEX)

    def _assign_forced_indexes(
        self,
        expression: Expression,
        indexes: Sequence[IndexMetadata],
    ) -> dict[Condition, IndexMetadata]:
        """
        Asigna indices forzados a condiciones del WHERE.

        Un mismo indice puede participar en varias condiciones (por ejemplo
        un BTREE en rangos de ramas distintas de un OR). La asignacion solo
        reparte igualdades cuando HASH y BTREE estan forzados sobre la misma
        columna y no existe un rango que desempate de forma natural.

        Detecta a nivel global:
        - indices forzados que no pueden participar en ninguna condicion;
        - el caso redundante en que HASH y BTREE solo pueden resolver la
          misma igualdad.
        """
        conditions: list[Condition] = []
        self._collect_conditions(expression, conditions)

        by_column: dict[str, list[Condition]] = {}
        for condition in conditions:
            column, literal = self._condition_column_literal(condition)
            if column is not None and literal is not None:
                by_column.setdefault(column.name, []).append(condition)

        # Cada indice forzado debe tener al menos una condicion elegible.
        for index in indexes:
            eligible = [condition for condition in by_column.get(index.column, []) if self._kind_supports(index.kind, condition.operator)]
            if not eligible:
                raise PlannerError(f"El indice '{index.name}' no puede participar en el " f"WHERE")

        # Redundancia global HASH/BTREE sobre la misma columna.
        by_column_indexes: dict[str, list[IndexMetadata]] = {}
        for index in indexes:
            by_column_indexes.setdefault(index.column, []).append(index)

        assignment: dict[Condition, IndexMetadata] = {}

        for column, column_indexes in by_column_indexes.items():
            kinds = {index.kind for index in column_indexes}
            if not {IndexKind.HASH, IndexKind.BTREE} <= kinds:
                continue

            conditions_of_column = by_column.get(column, [])
            equalities = [condition for condition in conditions_of_column if condition.operator is ComparisonOperator.EQ]
            ranges = [condition for condition in conditions_of_column if condition.operator in _RANGE_OPERATORS]

            # Con un rango, BTREE lo resuelve y HASH resuelve la igualdad:
            # no hay redundancia.
            if ranges:
                continue

            if len(equalities) < 2:
                btree = next(index for index in column_indexes if index.kind is IndexKind.BTREE)
                raise PlannerError(f"FORCE INDEX: el indice '{btree.name}' es redundante " f"para las condiciones del WHERE")

            hash_index = next(index for index in column_indexes if index.kind is IndexKind.HASH)
            btree_index = next(index for index in column_indexes if index.kind is IndexKind.BTREE)
            assignment[equalities[0]] = hash_index
            assignment[equalities[1]] = btree_index

        return assignment

    @classmethod
    def _collect_conditions(
        cls,
        expression: Expression,
        out: list[Condition],
    ) -> None:
        if isinstance(expression, Condition):
            out.append(expression)
            return

        if isinstance(expression, LogicalExpression):
            # NOT nunca se empuja al indice: sus condiciones no son asignables.
            if expression.operator is LogicalOperator.NOT:
                return
            for operand in expression.operands:
                cls._collect_conditions(operand, out)

    @staticmethod
    def _kind_supports(kind: IndexKind, operator: ComparisonOperator,) -> bool:
        if kind is IndexKind.HASH:
            return operator is ComparisonOperator.EQ

        return operator in {
            ComparisonOperator.EQ,
            ComparisonOperator.LT,
            ComparisonOperator.LE,
            ComparisonOperator.GT,
            ComparisonOperator.GE,
        }

    # Range / residual helpers

    def _build_range(
        self,
        conditions: Sequence[Condition],
        table: str,
        indexes: Sequence[IndexMetadata],
    ) -> RangeSpec:
        if not indexes:
            raise PlannerError(f"No hay indice BTREE para construir el rango en '{table}'")

        return self._range_spec(conditions, table, indexes[0].column)

    def _range_spec(
        self,
        conditions: Sequence[Condition],
        table: str,
        column: str,
    ) -> RangeSpec:
        lower: _Bound | None = None
        upper: _Bound | None = None
        index_column = column

        for condition in conditions:
            normalized = condition
            column, literal = self._condition_column_literal(normalized)

            if column is None or literal is None:
                continue

            value = self._normalize_access_key(
                table=table,
                column=index_column,
                value=literal.value,
            )
            op = normalized.operator

            if op is ComparisonOperator.GT:
                lower = self._stronger_lower(
                    lower,
                    _Bound(value, False),
                )
            elif op is ComparisonOperator.GE:
                lower = self._stronger_lower(
                    lower,
                    _Bound(value, True),
                )
            elif op is ComparisonOperator.LT:
                upper = self._stronger_upper(
                    upper,
                    _Bound(value, False),
                )
            elif op is ComparisonOperator.LE:
                upper = self._stronger_upper(
                    upper,
                    _Bound(value, True),
                )

        return RangeSpec(
            lower=lower.value if lower is not None else None,
            upper=upper.value if upper is not None else None,
            include_lower=lower.inclusive if lower is not None else True,
            include_upper=upper.inclusive if upper is not None else True,
        )

    @staticmethod
    def _range_is_valid(range_spec: RangeSpec) -> bool:
        if (range_spec.lower is None or range_spec.upper is None):
            return True

        if range_spec.lower < range_spec.upper:
            return True

        if range_spec.lower > range_spec.upper:
            return False

        return (range_spec.include_lower and range_spec.include_upper)

    @staticmethod
    def _stronger_lower(current: _Bound | None, candidate: _Bound,) -> _Bound:
        if current is None:
            return candidate

        if candidate.value > current.value:
            return candidate

        if candidate.value < current.value:
            return current

        return _Bound(value=current.value, inclusive=current.inclusive and candidate.inclusive,)

    @staticmethod
    def _stronger_upper(current: _Bound | None, candidate: _Bound,) -> _Bound:
        if current is None:
            return candidate

        if candidate.value < current.value:
            return candidate

        if candidate.value > current.value:
            return current

        return _Bound(value=current.value, inclusive=current.inclusive and candidate.inclusive,)

    @staticmethod
    def _and_expression(expressions: Iterable[Expression],) -> Expression | None:
        items: list[Expression] = []

        def collect(expression: Expression) -> None:
            if (isinstance(expression, LogicalExpression) and expression.operator is LogicalOperator.AND):
                if len(expression.operands) != 2:
                    raise PlannerError("AND requiere exactamente dos operandos")
                collect(expression.operands[0])
                collect(expression.operands[1])
            else:
                items.append(expression)

        for expression in expressions:
            collect(expression)

        if not items:
            return None

        result = items[0]

        for item in items[1:]:
            result = LogicalExpression(
                operator=LogicalOperator.AND,
                operands=(result, item),
            )

        return result

    @staticmethod
    def _flatten_and(expression: LogicalExpression) -> list[Expression]:
        result: list[Expression] = []

        def visit(item: Expression) -> None:
            if (isinstance(item, LogicalExpression) and item.operator is LogicalOperator.AND):
                if len(item.operands) != 2:
                    raise PlannerError("AND requiere exactamente dos operandos")
                visit(item.operands[0])
                visit(item.operands[1])
            else:
                result.append(item)

        for operand in expression.operands:
            visit(operand)

        return result

    # Index helpers

    def _explicit_indexes(
        self,
        table: str,
        names: Sequence[str] | None,
    ) -> list[IndexMetadata]:
        if names is None:
            return []

        seen: set[str] = set()
        result: list[IndexMetadata] = []

        for name in names:
            if name in seen:
                raise PlannerError(f"El indice '{name}' esta repetido en FORCE INDEX")
            seen.add(name)

            if not self.catalog.index_exists(name):
                raise PlannerError(f"El indice '{name}' no existe")

            index = self.catalog.get_index(name)

            if index.table != table:
                raise PlannerError(f"El indice '{name}' no pertenece a la tabla '{table}'")

            result.append(index)

        return result

    def _compatible_indexes(
        self,
        indexes: Sequence[IndexMetadata],
        table: str,
        column: str,
        operator: ComparisonOperator,
    ) -> list[IndexMetadata]:
        compatible: list[IndexMetadata] = []

        for index in indexes:
            if index.table != table or index.column != column:
                continue

            if index.kind is IndexKind.HASH:
                if operator is ComparisonOperator.EQ:
                    compatible.append(index)
                continue

            if index.kind is IndexKind.BTREE:
                if operator in {
                    ComparisonOperator.EQ,
                    ComparisonOperator.LT,
                    ComparisonOperator.LE,
                    ComparisonOperator.GT,
                    ComparisonOperator.GE,
                }:
                    compatible.append(index)

        return compatible

    def _normalize_access_key(
        self,
        table: str,
        column: str,
        value: Key,
    ) -> Key:
        metadata = self.catalog.get_table(table).metadata

        try:
            return metadata.normalize_value(column, value)
        except (TypeError, ValueError, IndexError) as exc:
            raise PlannerError(f"No se pudo normalizar el valor del indice '{column}' " f"en '{table}': {value!r}") from exc

    def _build_equality_access(
        self,
        table: str,
        column: str,
        value: Key,
        indexes: Sequence[IndexMetadata],
    ) -> PlanNode:
        if not indexes:
            raise PlannerError(f"No hay indice compatible para la igualdad de '{column}'")

        accesses = []

        for index in indexes:
            accesses.append(
                BitmapIndexScanPlan(
                    table=table,
                    index=index.name,
                    index_kind=index.kind,
                    key=self._normalize_access_key(
                        table=table,
                        column=column,
                        value=value,
                    ),
                )
            )

        if len(accesses) == 1:
            return IndexScanPlan(
                table=table,
                index=indexes[0].name,
                index_kind=indexes[0].kind,
                key=self._normalize_access_key(
                    table=table,
                    column=column,
                    value=value,
                ),
            )

        return BitmapAndPlan(children=accesses)

    def _build_range_plan(
        self,
        table: str,
        indexes: Sequence[IndexMetadata],
        range_spec: RangeSpec,
    ) -> PlanNode:
        if not indexes:
            raise PlannerError(f"No hay indice BTREE para construir el rango en '{table}'")

        if len(indexes) == 1:
            return IndexRangeScanPlan(
                table=table,
                index=indexes[0].name,
                range=range_spec,
            )

        return BitmapAndPlan(
            children=[
                BitmapIndexRangeScanPlan(
                    table=table,
                    index=index.name,
                    range=range_spec,
                )
                for index in indexes
            ]
        )

    # Access-tree typing / conversion

    @staticmethod
    def _is_bitmap_plan(plan: PlanNode) -> bool:
        return isinstance(
            plan,
            (
                BitmapIndexScanPlan,
                BitmapIndexRangeScanPlan,
                BitmapAndPlan,
                BitmapOrPlan,
            ),
        )

    @staticmethod
    def _as_bitmap(plan: PlanNode) -> PlanNode:
        if isinstance(plan, BitmapIndexScanPlan):
            return plan

        if isinstance(plan, BitmapIndexRangeScanPlan):
            return plan

        if isinstance(plan, BitmapAndPlan):
            return plan

        if isinstance(plan, BitmapOrPlan):
            return plan

        if isinstance(plan, IndexScanPlan):
            return BitmapIndexScanPlan(
                table=plan.table,
                index=plan.index,
                index_kind=plan.index_kind,
                key=plan.key,
            )

        if isinstance(plan, IndexRangeScanPlan):
            return BitmapIndexRangeScanPlan(
                table=plan.table,
                index=plan.index,
                range=plan.range,
            )

        raise PlannerError(f"No se puede convertir a bitmap el plan {type(plan).__name__}")

    def _records_from_access(
        self,
        table: str,
        access: PlanNode,
    ) -> PlanNode:
        if isinstance(access, EmptyResultPlan):
            return access

        if isinstance(access, SeqScanPlan):
            return access

        if isinstance(access, SequentialRangeScanPlan):
            return access

        if self._is_bitmap_plan(access):
            return BitmapHeapScanPlan(table=table, child=access,)

        if isinstance(access, IndexScanPlan):
            return HeapFetchPlan(table=table, child=access,)

        if isinstance(access, IndexRangeScanPlan):
            return HeapFetchPlan(table=table, child=access,)

        raise PlannerError(f"El access path {type(access).__name__} no produce un " "resultado valido para SELECT")

    def _scan_plan(
        self,
        table: str,
        produce_rids: bool,
    ) -> PlanNode:
        if produce_rids:
            return SeqScanRidsPlan(table=table)

        return SeqScanPlan(table=table)

    def _is_sequential_table(self, table: str) -> bool:
        return self.catalog.get_table(table).storage is StorageKind.SEQUENTIAL

    # AST / metadata helpers
    @staticmethod
    def _as_operand(value: PlannerExpression) -> Operand:
        if isinstance(value, (ColumnRef, LiteralValue)):
            return value

        raise PlannerError(f"Se esperaba un operando y se obtuvo {type(value)!r}")

    @staticmethod
    def _as_expression(value: PlannerExpression) -> Expression:
        if isinstance(value, Expression):
            return value

        raise PlannerError(f"Se esperaba una expresion y se obtuvo {type(value)!r}")

    @staticmethod
    def _comparison_operator(op: BinaryOp) -> ComparisonOperator:
        mapping = {
            BinaryOp.EQ: ComparisonOperator.EQ,
            BinaryOp.NEQ: ComparisonOperator.NE,
            BinaryOp.NEQ2: ComparisonOperator.NE,
            BinaryOp.LT: ComparisonOperator.LT,
            BinaryOp.LE: ComparisonOperator.LE,
            BinaryOp.GT: ComparisonOperator.GT,
            BinaryOp.GE: ComparisonOperator.GE,
        }

        try:
            return mapping[op]
        except KeyError as exc:
            raise PlannerError(f"Operador de comparacion no soportado: {op!r}") from exc

    @staticmethod
    def _normalize_condition(condition: Condition) -> Condition:
        left = condition.left
        right = condition.right
        operator = condition.operator

        if (isinstance(left, LiteralValue) and isinstance(right, ColumnRef)):
            operator = {
                ComparisonOperator.EQ: ComparisonOperator.EQ,
                ComparisonOperator.NE: ComparisonOperator.NE,
                ComparisonOperator.LT: ComparisonOperator.GT,
                ComparisonOperator.LE: ComparisonOperator.GE,
                ComparisonOperator.GT: ComparisonOperator.LT,
                ComparisonOperator.GE: ComparisonOperator.LE,
            }[operator]

            return Condition(
                left=right,
                operator=operator,
                right=left,
            )

        return condition

    @staticmethod
    def _condition_column_literal(condition: Condition,) -> tuple[ColumnRef | None, LiteralValue | None]:
        if (isinstance(condition.left, ColumnRef) and isinstance(condition.right, LiteralValue)):
            return condition.left, condition.right

        return None, None

    @staticmethod
    def _storage_kind(storage: StorageType) -> StorageKind:
        if storage is StorageType.SEQUENTIAL:
            return StorageKind.SEQUENTIAL

        return StorageKind.HEAP


    @staticmethod
    def _index_kind(index_type: IndexType) -> IndexKind:
        if index_type is IndexType.HASH:
            return IndexKind.HASH

        if index_type is IndexType.RTREE:
            return IndexKind.RTREE

        if index_type is IndexType.INVERTED:
            return IndexKind.INVERTED

        return IndexKind.BTREE


    @staticmethod
    def _table_metadata_from_columns(
        name: str,
        columns: Sequence[ColumnDef],
        primary_key: str | None,
    ) -> TableMetadata:
        built: list[dict] = []

        for column in columns:
            built.append(
                {
                    "name": column.name,
                    "type": column.data_type.name,
                    "size": column.size,
                    "unique": bool(column.unique),
                }
            )

        return TableMetadata(
            name,
            built,
            primary_key,
        )
