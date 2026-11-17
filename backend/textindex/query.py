"""
Parser de la cadena de consulta usada en  columna MATCH '...'.

Gramatica (los operadores se escriben en MAYUSCULAS; en minusculas son
palabras normales y suelen ser stopwords):

    consulta  ::= conjuncion { [ OR ] conjuncion }
    conjuncion::= primario { AND [ NOT ] primario }
    primario  ::= "(" consulta ")" | PALABRA

    - Dos operandos seguidos sin operador se unen con OR. Asi una consulta de
      texto libre ("big data analisis") es una consulta vectorial clasica: se
      recuperan los documentos con al menos un termino y se ordenan por coseno.
    - AND tiene mayor precedencia que OR:  a OR b AND c  ==  a OR (b AND c).
    - NOT solo se admite como AND NOT (diferencia). Un NOT aislado requeriria
      el complemento sobre toda la coleccion y no se soporta.

Cada PALABRA pasa por el mismo TextPreprocessor del indice. Si la palabra
desaparece (stopword, signo), el operando es neutro: a AND <vacio> == a.
Si una palabra produce varios terminos (por ejemplo "e-mail"), se unen con OR,
igual que la adyacencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from textindex import postings as merge
from textindex.preprocessing import TextPreprocessor


class QueryError(ValueError):
    """Consulta textual mal formada."""


@dataclass(frozen=True)
class Term:
    term: str


@dataclass(frozen=True)
class And:
    left: 'QueryNode'
    right: 'QueryNode'


@dataclass(frozen=True)
class Or:
    left: 'QueryNode'
    right: 'QueryNode'


@dataclass(frozen=True)
class AndNot:
    left: 'QueryNode'
    right: 'QueryNode'


QueryNode = Union[Term, And, Or, AndNot]

_OPERATORS = {'AND', 'OR', 'NOT'}


def _lex(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char in '()':
            if current:
                tokens.append(''.join(current))
                current = []
            tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append(''.join(current))
    return tokens


def _combine(kind, left: QueryNode | None, right: QueryNode | None):
    """Construye un nodo tratando None (operando vacio) como neutro."""
    if left is None:
        # (vacio) AND NOT b no tiene conjunto base: queda vacio/neutro.
        return None if kind is AndNot else right
    if right is None:
        return left
    return kind(left, right)


class _Parser:
    def __init__(self, tokens: list[str], pre: TextPreprocessor) -> None:
        self.tokens = tokens
        self.pos = 0
        self.pre = pre

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> QueryNode | None:
        node = self.disjunction()
        if self.peek() is not None:
            raise QueryError(f"Token inesperado en la consulta: '{self.peek()}'")
        return node

    def disjunction(self) -> QueryNode | None:
        node = self.conjunction()
        while True:
            token = self.peek()
            if token is None or token == ')':
                return node
            if token == 'OR':
                self.take()
            elif token in ('AND', 'NOT'):
                raise QueryError(
                    "NOT solo puede usarse como 'AND NOT'"
                    if token == 'NOT' else "AND sin operando izquierdo"
                )
            node = _combine(Or, node, self.conjunction())

    def conjunction(self) -> QueryNode | None:
        node = self.primary()
        while self.peek() == 'AND':
            self.take()
            negated = False
            if self.peek() == 'NOT':
                self.take()
                negated = True
            right = self.primary()
            node = _combine(AndNot if negated else And, node, right)
        return node

    def primary(self) -> QueryNode | None:
        token = self.peek()
        if token is None:
            raise QueryError('La consulta termina antes de un operando')
        if token == '(':
            self.take()
            node = self.disjunction()
            if self.peek() != ')':
                raise QueryError("Falta ')' en la consulta")
            self.take()
            return node
        if token == ')':
            raise QueryError("')' inesperado en la consulta")
        if token == 'NOT':
            raise QueryError("NOT solo puede usarse como 'AND NOT'")
        if token in _OPERATORS:
            raise QueryError(f"Operador '{token}' sin operando")
        self.take()
        node: QueryNode | None = None
        for term in self.pre.terms(token):
            node = _combine(Or, node, Term(term))
        return node


def parse_query(text: str, pre: TextPreprocessor) -> QueryNode | None:
    """Devuelve el arbol de la consulta o None si no queda ningun termino."""
    tokens = _lex(text or '')
    if not tokens:
        return None
    return _Parser(tokens, pre).parse()


def positive_terms(node: QueryNode | None) -> list[str]:
    """
    Terminos que aportan al ranking, con repeticiones (tf en la consulta).
    Se excluye el lado derecho de AND NOT: un termino negado no puede
    aparecer en los documentos resultantes.
    """
    output: list[str] = []

    def visit(item: QueryNode | None) -> None:
        if item is None:
            return
        if isinstance(item, Term):
            output.append(item.term)
        elif isinstance(item, AndNot):
            visit(item.left)
        else:
            visit(item.left)
            visit(item.right)

    visit(node)
    return output


def all_terms(node: QueryNode | None) -> set[str]:
    """Todos los terminos (incluidos los negados)."""
    output: set[str] = set()

    def visit(item: QueryNode | None) -> None:
        if item is None:
            return
        if isinstance(item, Term):
            output.add(item.term)
        else:
            visit(item.left)
            visit(item.right)

    visit(node)
    return output


def evaluate(
    node: QueryNode | None,
    postings_of: Callable[[str], list[int]],
    df_of: Callable[[str], int],
) -> list[int]:
    """
    Evalua el arbol con mezclas lineales y devuelve doc_ids ordenados.

    Optimizacion clasica para cadenas de AND: se aplanan y se intersectan
    empezando por la lista de menor df, de modo que los resultados
    intermedios nunca crecen y el trabajo total se reduce.
    """
    if node is None:
        return []
    if isinstance(node, Term):
        return postings_of(node.term)
    if isinstance(node, Or):
        return merge.union(
            evaluate(node.left, postings_of, df_of),
            evaluate(node.right, postings_of, df_of),
        )
    if isinstance(node, AndNot):
        left = evaluate(node.left, postings_of, df_of)
        if not left:
            return []
        return merge.difference(left, evaluate(node.right, postings_of, df_of))

    operands: list[QueryNode] = []

    def flatten(item: QueryNode) -> None:
        if isinstance(item, And):
            flatten(item.left)
            flatten(item.right)
        else:
            operands.append(item)

    flatten(node)

    def estimate(item: QueryNode) -> int:
        if isinstance(item, Term):
            return df_of(item.term)
        return 1 << 62  # subexpresion compuesta: se evalua al final

    operands.sort(key=estimate)
    result = evaluate(operands[0], postings_of, df_of)
    for item in operands[1:]:
        if not result:
            return []
        result = merge.intersect(result, evaluate(item, postings_of, df_of))
    return result


def matches(node: QueryNode | None, terms: set[str] | dict) -> bool:
    """Evaluacion booleana sobre un solo documento (Full Text Scan)."""
    if node is None:
        return False
    if isinstance(node, Term):
        return node.term in terms
    if isinstance(node, And):
        return matches(node.left, terms) and matches(node.right, terms)
    if isinstance(node, Or):
        return matches(node.left, terms) or matches(node.right, terms)
    return matches(node.left, terms) and not matches(node.right, terms)
