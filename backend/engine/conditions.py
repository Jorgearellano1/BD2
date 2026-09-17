from dataclasses import dataclass
from enum import Enum

from engine.catalog import Key, RawRecord, TableMetadata


class ComparisonOperator(Enum):
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class LogicalOperator(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass(frozen=True)
class Condition:
    """Condición basica de comparación sobre una columna."""
    column: str
    operator: ComparisonOperator
    value: Key


@dataclass(frozen=True)
class LogicalExpression:
    """
    Expresión logica que combina una o más expresiones.

    AND y OR requieren dos operandos.
    NOT requiere un solo operando.
    """
    operator: LogicalOperator
    operands: tuple["Expression", ...]


Expression = Condition | LogicalExpression


def evaluate_condition(
    record: RawRecord,
    condition: Condition,
    table_meta: TableMetadata
) -> bool:
    """
    Evalua una condicion basica sobre un registro.
    """
    col_idx = table_meta.col_index(condition.column)
    actual  = record[col_idx]
    target  = table_meta.normalize_value(condition.column, condition.value)

    op = condition.operator

    if op == ComparisonOperator.EQ:
        return actual == target
    if op == ComparisonOperator.NE:
        return actual != target
    if op == ComparisonOperator.LT:
        return actual < target
    if op == ComparisonOperator.LE:
        return actual <= target
    if op == ComparisonOperator.GT:
        return actual > target
    if op == ComparisonOperator.GE:
        return actual >= target

    raise ValueError(f"Operador de comparacion no soportado: {op!r}")


def evaluate_expression(
    record: RawRecord,
    expression: Expression,
    table_meta: TableMetadata
) -> bool:
    """
    Evalua una expresión logica o una condicion sobre un registro.
    """
    if isinstance(expression, Condition):
        return evaluate_condition(record, expression, table_meta)

    if isinstance(expression, LogicalExpression):
        op       = expression.operator
        operands = expression.operands

        if op == LogicalOperator.AND:
            if len(operands) != 2:
                raise ValueError("AND requiere exactamente dos operandos")
            return (
                evaluate_expression(record, operands[0], table_meta)
                and evaluate_expression(record, operands[1], table_meta)
            )

        if op == LogicalOperator.OR:
            if len(operands) != 2:
                raise ValueError("OR requiere exactamente dos operandos")
            return (
                evaluate_expression(record, operands[0], table_meta)
                or evaluate_expression(record, operands[1], table_meta)
            )

        if op == LogicalOperator.NOT:
            if len(operands) != 1:
                raise ValueError("NOT requiere exactamente un operando")
            return not evaluate_expression(record, operands[0], table_meta)

        raise ValueError(f"Operador logico no soportado: {op!r}")

    raise TypeError(f"Expresion no soportada: {type(expression)!r}")
