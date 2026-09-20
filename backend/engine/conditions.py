from abc import ABC
from dataclasses import dataclass
from enum import Enum

from engine.metadata import Key, RawRecord, TableMetadata


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


class Operand(ABC):
    """Operando de una condicion: columna o literal."""
    pass


@dataclass(frozen=True)
class ColumnRef(Operand):
    """Operando que referencia una columna del registro."""
    name: str


@dataclass(frozen=True)
class LiteralValue(Operand):
    """Operando que representa un valor literal."""
    value: Key


class Expression(ABC):
    """Expresion semantica/ejecutable."""
    pass


@dataclass(frozen=True)
class Condition(Expression):
    """Comparacion entre dos operandos (columna o literal)."""
    left: Operand
    operator: ComparisonOperator
    right: Operand


@dataclass(frozen=True)
class LogicalExpression(Expression):
    """
    Expresion logica que combina una o mas expresiones.

    AND y OR requieren dos operandos.
    NOT requiere un solo operando.
    """
    operator: LogicalOperator
    operands: tuple["Expression", ...]


def _resolve_operand(
    record: RawRecord,
    operand: Operand,
    table_meta: TableMetadata,
    other: Operand
) -> int | float | bytes | str:
    """
    Resuelve un operando contra el registro. Las columnas se leen del record;
    los literales se normalizan segun la columna del otro operando cuando esta
    disponible.
    """
    if isinstance(operand, ColumnRef):
        return record[table_meta.col_index(operand.name)]

    if isinstance(operand, LiteralValue):
        if isinstance(other, ColumnRef):
            return table_meta.normalize_value(other.name, operand.value)
        return operand.value

    raise TypeError(f"Operando no soportado: {type(operand)!r}")


def evaluate_condition(
    record: RawRecord,
    condition: Condition,
    table_meta: TableMetadata
) -> bool:
    """
    Evalua una condicion basica sobre un registro.
    """
    left  = _resolve_operand(record, condition.left, table_meta, condition.right)
    right = _resolve_operand(record, condition.right, table_meta, condition.left)

    op = condition.operator

    if op == ComparisonOperator.EQ:
        return left == right
    if op == ComparisonOperator.NE:
        return left != right
    if op == ComparisonOperator.LT:
        return left < right
    if op == ComparisonOperator.LE:
        return left <= right
    if op == ComparisonOperator.GT:
        return left > right
    if op == ComparisonOperator.GE:
        return left >= right

    raise ValueError(f"Operador de comparacion no soportado: {op!r}")


def evaluate_expression(
    record: RawRecord,
    expression: Expression,
    table_meta: TableMetadata
) -> bool:
    """
    Evalua una expresion logica o una condicion sobre un registro.
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
