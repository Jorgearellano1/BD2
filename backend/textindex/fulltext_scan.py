"""
Full Text Scan: linea base sin indice para el Experimento 6.

Recorre todas las paginas del HeapFile, preprocesa cada documento con el
mismo TextPreprocessor y evalua la consulta registro a registro. Para el
ranking necesita estadisticas globales (N y df de todos los terminos, porque
la norma de cada documento depende de todos sus terminos); se obtienen en la
misma pasada: se acumula df de cada termino y se guardan los vectores tf solo
de los documentos que satisfacen la consulta. Al terminar se aplica la misma
formula TF-IDF / coseno que el indice, por lo que ambos devuelven el mismo
ranking sobre la misma coleccion (lo verifican las pruebas).

Costo: P lecturas de pagina del Heap y preprocesar los N documentos en cada
consulta, frente a leer solo las listas de los terminos con el indice.
"""

from __future__ import annotations

import math
from typing import Iterable

from textindex.inverted_index import SearchHit, top_k_heap
from textindex.preprocessing import TextPreprocessor
from textindex.query import matches, parse_query, positive_terms


def full_text_scan(
    documents: Iterable[tuple[tuple[int, int], str]],
    query_text: str,
    preprocessor: TextPreprocessor,
    top_k: int | None = None,
    rank: bool = True,
    descending: bool = True,
    accept=None,
) -> list[SearchHit]:
    """
    documents : pares (RID, texto) en orden fisico del Heap.
    accept    : funcion opcional RID -> bool (filtro relacional ya evaluado).
    """
    tree = parse_query(query_text, preprocessor)
    if tree is None:
        # Se recorre igual la tabla para que el costo sea comparable.
        for _ in documents:
            pass
        return []

    doc_freq: dict[str, int] = {}
    matched: list[tuple[int, tuple[int, int], dict[str, int]]] = []
    num_docs = 0
    for rid, text in documents:
        tf = preprocessor.term_frequencies(text)
        if rank:
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        if matches(tree, tf) and (accept is None or accept(rid)):
            matched.append((num_docs, rid, tf))
        num_docs += 1

    if not rank:
        hits = [SearchHit(doc_id, rid, None) for doc_id, rid, _ in matched]
        return hits[:top_k] if top_k is not None else hits

    def idf(term: str) -> float:
        df = doc_freq.get(term, 0)
        return math.log10(num_docs / df) if df else 0.0

    query_tf: dict[str, int] = {}
    for term in positive_terms(tree):
        query_tf[term] = query_tf.get(term, 0) + 1
    query_w = {
        term: (1.0 + math.log10(tf)) * idf(term)
        for term, tf in query_tf.items()
        if idf(term) > 0.0
    }
    query_norm = math.sqrt(sum(w * w for w in query_w.values()))

    scored: list[tuple[float, int]] = []
    rid_of: dict[int, tuple[int, int]] = {}
    for doc_id, rid, tf in matched:
        rid_of[doc_id] = rid
        norm_sq = 0.0
        dot = 0.0
        for term, count in tf.items():
            weight = (1.0 + math.log10(count)) * idf(term)
            norm_sq += weight * weight
            if term in query_w:
                dot += query_w[term] * weight
        norm = math.sqrt(norm_sq)
        score = dot / (norm * query_norm) if norm > 0 and query_norm > 0 else 0.0
        scored.append((score, doc_id))

    ordered = top_k_heap(scored, top_k, descending)
    return [SearchHit(doc_id, rid_of[doc_id], score) for score, doc_id in ordered]
