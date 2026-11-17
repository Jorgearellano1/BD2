"""
Mezclas lineales sobre listas de doc_id ordenadas ascendentemente.

Cada funcion recorre las dos listas con dos punteros simultaneos, avanzando
siempre el que apunta al doc_id menor. Ningun elemento se visita dos veces,
por lo que el costo es O(n + m) comparaciones (PDF 09, p. 80-84; enunciado
P2, p. 9). La precondicion es que ambas listas esten ordenadas y sin
repetidos, que es exactamente como se guardan en postings.inv.
"""

from __future__ import annotations


def intersect(left: list[int], right: list[int]) -> list[int]:
    """AND: doc_ids presentes en ambas listas (PDF 09, p. 82)."""
    answer: list[int] = []
    i = j = 0
    n, m = len(left), len(right)
    while i < n and j < m:
        a, b = left[i], right[j]
        if a == b:
            answer.append(a)
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return answer


def union(left: list[int], right: list[int]) -> list[int]:
    """
    OR: doc_ids presentes en al menos una lista.

    A diferencia del pseudocodigo de la p. 83 del PDF 09, al terminar el bucle
    se copian las colas que quedan en la lista no agotada; sin ese paso se
    perderian documentos.
    """
    answer: list[int] = []
    i = j = 0
    n, m = len(left), len(right)
    while i < n and j < m:
        a, b = left[i], right[j]
        if a == b:
            answer.append(a)
            i += 1
            j += 1
        elif a < b:
            answer.append(a)
            i += 1
        else:
            answer.append(b)
            j += 1
    if i < n:
        answer.extend(left[i:])
    if j < m:
        answer.extend(right[j:])
    return answer


def difference(left: list[int], right: list[int]) -> list[int]:
    """
    AND NOT: doc_ids de 'left' que no aparecen en 'right' (PDF 09, p. 84).

    Cuando 'right' se agota, todo lo que queda en 'left' pertenece al
    resultado; el costo sigue siendo O(n + m).
    """
    answer: list[int] = []
    i = j = 0
    n, m = len(left), len(right)
    while i < n and j < m:
        a, b = left[i], right[j]
        if a == b:
            i += 1
            j += 1
        elif a < b:
            answer.append(a)
            i += 1
        else:
            j += 1
    if i < n:
        answer.extend(left[i:])
    return answer
