"""
Pruebas del nucleo textual (sin SQL): preprocesamiento, mezclas, parser de
consultas, persistencia del indice, booleano frente a referencia por
conjuntos y ranking frente a un calculo TF-IDF/coseno independiente.

Ejecutar desde la raiz:  .venv/bin/python -m pytest tests/test_textindex.py
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from textindex import postings as merge  # noqa: E402
from textindex.fulltext_scan import full_text_scan  # noqa: E402
from textindex.inverted_index import (  # noqa: E402
    DOCS_FILE,
    POSTINGS_FILE,
    VOCAB_FILE,
    InvertedIndex,
    InvertedIndexError,
    top_k_heap,
)
from textindex.preprocessing import TextPreprocessor, truncate_utf8  # noqa: E402
from textindex.query import (  # noqa: E402
    And,
    AndNot,
    Or,
    QueryError,
    Term,
    parse_query,
    positive_terms,
)

PRE = TextPreprocessor('spanish')


def rid_of(i: int) -> tuple[int, int]:
    return (i // 3, i % 3)


def reference_scores(docs: list[str], query: str, pre: TextPreprocessor) -> dict[int, float]:
    """
    Calculo independiente: vectores densos sobre todo el vocabulario,
    w = (1 + log10 tf) * log10(N/df), coseno = q.d / (|q||d|).
    Devuelve el score de cada documento con al menos un termino positivo.
    """
    bags = [pre.term_frequencies(text) for text in docs]
    n = len(docs)
    vocabulary = sorted({t for bag in bags for t in bag})
    df = {t: sum(1 for bag in bags if t in bag) for t in vocabulary}

    def weight(tf: int, term: str) -> float:
        return (1 + math.log10(tf)) * math.log10(n / df[term]) if tf > 0 else 0.0

    qtf: dict[str, int] = {}
    for term in positive_terms(parse_query(query, pre)):
        if term in df:
            qtf[term] = qtf.get(term, 0) + 1
    qvec = [weight(qtf.get(t, 0), t) for t in vocabulary]
    qnorm = math.sqrt(sum(x * x for x in qvec))
    out = {}
    for i, bag in enumerate(bags):
        dvec = [weight(bag.get(t, 0), t) for t in vocabulary]
        dnorm = math.sqrt(sum(x * x for x in dvec))
        dot = sum(a * b for a, b in zip(qvec, dvec))
        out[i] = dot / (dnorm * qnorm) if dnorm > 0 and qnorm > 0 else 0.0
    return out


def reference_boolean(docs: list[str], query: str, pre: TextPreprocessor) -> list[int]:
    """Evaluacion booleana por conjuntos de Python sobre cada documento."""
    tree = parse_query(query, pre)
    bags = [set(pre.terms(text)) for text in docs]

    def ev(node):
        if node is None:
            return set()
        if isinstance(node, Term):
            return {i for i, bag in enumerate(bags) if node.term in bag}
        if isinstance(node, And):
            return ev(node.left) & ev(node.right)
        if isinstance(node, Or):
            return ev(node.left) | ev(node.right)
        return ev(node.left) - ev(node.right)

    return sorted(ev(tree))


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix='bd2texto_')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, docs, name='idx', page_size=4096, block=500_000, pre=PRE):
        pairs = [(rid_of(i), text) for i, text in enumerate(docs)]
        return InvertedIndex.build(os.path.join(self.tmp, name), pairs, pre, page_size, block)


class PreprocessingTests(unittest.TestCase):
    def test_pdf09_example(self):
        # PDF 09, p. 24: "Mis amigas y amigos peruanos son estudiosos."
        self.assertEqual(
            PRE.terms('Mis amigas y amigos peruanos son estudiosos.'),
            ['amig', 'amig', 'peruan', 'estudi'],
        )

    def test_acronym_case_and_accents(self):
        self.assertEqual(PRE.terms('O.N.U.'), PRE.terms('onu'))
        self.assertEqual(PRE.terms('CANCIÓN'), PRE.terms('cancion'))
        self.assertEqual(PRE.terms('ﬁesta'), PRE.terms('fiesta'))  # ligadura NFKC

    def test_enie_is_preserved(self):
        self.assertNotEqual(PRE.terms('año'), PRE.terms('ano'))

    def test_stopwords_and_empty(self):
        self.assertEqual(PRE.terms('el de la y que'), [])
        self.assertEqual(PRE.terms(''), [])
        self.assertEqual(PRE.terms('!!! ... ---'), [])

    def test_truncation_keeps_valid_utf8(self):
        word = 'ñ' * 40  # 80 bytes
        cut = truncate_utf8(word, 64)
        self.assertEqual(cut, 'ñ' * 32)
        cut = truncate_utf8('a' + 'ñ' * 40, 64)
        self.assertLessEqual(len(cut.encode('utf-8')), 64)
        cut.encode('utf-8').decode('utf-8')

    def test_english_pipeline(self):
        eng = TextPreprocessor('english')
        self.assertEqual(eng.terms('The running databases'), ['run', 'databas'])

    def test_config_roundtrip(self):
        other = TextPreprocessor.from_config(PRE.config())
        self.assertEqual(other.terms('Bases de datos distribuidas'),
                         PRE.terms('Bases de datos distribuidas'))
        with self.assertRaises(ValueError):
            TextPreprocessor('klingon')


class MergeTests(unittest.TestCase):
    def test_against_sets(self):
        rng = random.Random(7)
        for _ in range(300):
            a = sorted(rng.sample(range(200), rng.randint(0, 60)))
            b = sorted(rng.sample(range(200), rng.randint(0, 60)))
            self.assertEqual(merge.intersect(a, b), sorted(set(a) & set(b)))
            self.assertEqual(merge.union(a, b), sorted(set(a) | set(b)))
            self.assertEqual(merge.difference(a, b), sorted(set(a) - set(b)))

    def test_union_keeps_tails(self):
        # El pseudocodigo OR de la p. 83 del PDF 09 omite las colas.
        self.assertEqual(merge.union([1, 2], [3, 4, 5]), [1, 2, 3, 4, 5])
        self.assertEqual(merge.difference([1, 5, 9], [5]), [1, 9])


class QueryParserTests(unittest.TestCase):
    def test_precedence_and_adjacency(self):
        tree = parse_query('casa OR perro AND gato', PRE)
        self.assertIsInstance(tree, Or)
        self.assertIsInstance(tree.right, And)
        self.assertEqual(parse_query('casa perro', PRE), parse_query('casa OR perro', PRE))

    def test_and_not(self):
        tree = parse_query('aprendizaje AND profundo AND NOT basico', PRE)
        self.assertIsInstance(tree, AndNot)
        self.assertEqual(positive_terms(tree), ['aprendizaj', 'profund'])

    def test_stopword_operand_is_neutral(self):
        self.assertEqual(parse_query('casa AND el', PRE), Term('cas'))
        self.assertIsNone(parse_query('el AND la', PRE))
        self.assertIsNone(parse_query('', PRE))

    def test_repeated_terms(self):
        self.assertEqual(positive_terms(parse_query('casa casa jardin', PRE)),
                         ['cas', 'cas', 'jardin'])

    def test_errors(self):
        for bad in ('NOT casa', 'casa OR NOT perro', '(casa', 'casa)', 'AND casa', 'casa AND'):
            with self.assertRaises(QueryError, msg=bad):
                parse_query(bad, PRE)


CORPUS = [
    'La casa es grande',
    'El gato está en la casa',
    'La casa es bonita y grande',
    'El sol brilla sobre la casa',
    'Gato negro, gato blanco y gato gris',
    'Bases de datos distribuidas y fragmentación horizontal',
    'Índice invertido en disco para recuperación de información',
    'Recuperación de información con similitud de coseno y TF-IDF',
    'El año pasado el niño compró una canción en España',
    'Datos, datos y más datos: la base de datos crece',
]


class IndexTests(TempDirCase):
    def test_files_are_paged_and_reopen(self):
        index, stats = self.build(CORPUS)
        self.assertEqual(stats.num_docs, len(CORPUS))
        for name in (VOCAB_FILE, POSTINGS_FILE, DOCS_FILE):
            size = os.path.getsize(os.path.join(index.directory, name))
            self.assertEqual(size % 4096, 0, name)
        before = index.search('casa AND grande', top_k=5)[0]
        terms = index.num_terms
        index.close()

        reopened = InvertedIndex(os.path.join(self.tmp, 'idx'))
        self.assertEqual(reopened.page_reads, 1)  # solo la cabecera al abrir
        self.assertEqual(reopened.num_terms, terms)
        self.assertEqual(reopened.config['language'], 'spanish')
        after = reopened.search('casa AND grande', top_k=5)[0]
        self.assertEqual(before, after)
        reopened.close()

    def test_spimi_runs_produce_identical_files(self):
        one, s1 = self.build(CORPUS, 'one', block=10**9)
        many, s2 = self.build(CORPUS, 'many', block=3)
        self.assertEqual(s1.runs, 1)
        self.assertGreater(s2.runs, 3)
        for name in (VOCAB_FILE, POSTINGS_FILE, DOCS_FILE):
            with open(os.path.join(one.directory, name), 'rb') as a, \
                    open(os.path.join(many.directory, name), 'rb') as b:
                self.assertEqual(a.read(), b.read(), name)
        one.close()
        many.close()

    def test_postings_sorted_and_tf(self):
        index, _ = self.build(CORPUS)
        entry = index.lookup('gat')
        ids, tfs = index.read_postings(entry)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(dict(zip(ids, tfs)), {1: 1, 4: 3})
        self.assertEqual(entry.df, 2)
        self.assertEqual(entry.nbytes, 16)
        index.close()

    def test_selective_reads(self):
        docs = [f'documento numero {i} tema{i % 50} comun' for i in range(3000)]
        index, _ = self.build(docs, page_size=1024)
        self.assertGreater(index.dir_pages, 0)
        buf = index._query_buffer()
        before = index.vocab.reads
        entry = index.lookup('tema7', buf)
        self.assertEqual(index.vocab.reads - before, index.dir_pages + 1)
        before = index.postings.reads
        index.read_postings(entry, buf)
        expected = (entry.offset + entry.nbytes - 1) // 1024 - entry.offset // 1024 + 1
        self.assertEqual(index.postings.reads - before, expected)
        total_pages = os.path.getsize(os.path.join(index.directory, POSTINGS_FILE)) // 1024
        self.assertLess(expected, total_pages)
        index.close()

    def test_empty_collection_and_page_mismatch(self):
        index, stats = self.build([], 'empty')
        self.assertEqual(stats.num_docs, 0)
        self.assertEqual(index.search('casa')[0], [])
        index.close()
        index, _ = self.build(CORPUS, 'b1024', page_size=1024)
        index.close()
        with self.assertRaises(InvertedIndexError):
            InvertedIndex(os.path.join(self.tmp, 'b1024'), page_size=4096)

    def test_unicode_terms(self):
        index, _ = self.build(CORPUS + ['数据库 索引', 'Ωmega ΣΟΦΙΑ'], 'uni')
        self.assertIsNotNone(index.lookup('数据库'))
        hits2, _ = index.search('σοφια')  # case folding griego
        self.assertEqual([h.doc_id for h in hits2], [11])
        self.assertEqual(index.search('niño')[0][0].doc_id, 8)
        index.close()


class BooleanAndRankingTests(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        rng = random.Random(2042)
        words = ['dato', 'indice', 'disco', 'pagina', 'arbol', 'hash', 'texto',
                 'consulta', 'vector', 'espacio', 'bloque', 'registro']
        self.words = words
        self.docs = [
            ' '.join(rng.choice(words) for _ in range(rng.randint(1, 12)))
            for _ in range(400)
        ]
        self.index, _ = self.build(self.docs, page_size=1024, block=700)
        self.rng = rng

    def tearDown(self) -> None:
        self.index.close()
        super().tearDown()

    def random_query(self) -> str:
        r = self.rng
        a, b, c = (r.choice(self.words) for _ in range(3))
        return r.choice([
            f'{a}', f'{a} AND {b}', f'{a} OR {b}', f'{a} AND NOT {b}',
            f'({a} OR {b}) AND NOT {c}', f'{a} AND {b} AND {c}', f'{a} {b} {c}',
            f'{a} AND ({b} OR {c})', f'{a} AND NOT {b} AND NOT {c}',
        ])

    def test_boolean_matches_reference(self):
        for _ in range(150):
            q = self.random_query()
            hits, _ = self.index.search(q, rank=False)
            self.assertEqual([h.doc_id for h in hits], reference_boolean(self.docs, q, PRE), q)

    def test_ranking_matches_independent_cosine(self):
        for _ in range(60):
            q = self.random_query()
            hits, _ = self.index.search(q, top_k=None)
            expected = reference_scores(self.docs, q, PRE)
            valid = set(reference_boolean(self.docs, q, PRE))
            self.assertEqual({h.doc_id for h in hits}, valid, q)
            for h in hits:
                self.assertAlmostEqual(h.score, expected[h.doc_id], places=9, msg=q)
            scores = [h.score for h in hits]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_is_prefix_and_full_scan_agrees(self):
        pairs = [(rid_of(i), t) for i, t in enumerate(self.docs)]
        for _ in range(40):
            q = self.random_query()
            full, _ = self.index.search(q)
            top, _ = self.index.search(q, top_k=10)
            self.assertEqual(top, full[:10], q)
            scan = full_text_scan(pairs, q, PRE, top_k=10)
            self.assertEqual([(h.doc_id, h.rid) for h in scan], [(h.doc_id, h.rid) for h in top], q)
            for a, b in zip(scan, top):
                self.assertAlmostEqual(a.score, b.score, places=12)

    def test_oov_repeated_and_prefilter(self):
        hits, stats = self.index.search('dato AND palabrainexistente')
        self.assertEqual(hits, [])
        self.assertEqual(stats.oov_terms, ['palabrainexistent'])
        hits, _ = self.index.search('dato OR palabrainexistente', top_k=None)
        self.assertEqual({h.doc_id for h in hits}, set(reference_boolean(self.docs, 'dato', PRE)))
        # tf en la consulta: 'dato dato' pondera 1 + log10(2) pero con un
        # solo termino el coseno es el mismo que 'dato'.
        a, _ = self.index.search('dato dato vector', top_k=None)
        expected = reference_scores(self.docs, 'dato dato vector', PRE)
        for h in a:
            self.assertAlmostEqual(h.score, expected[h.doc_id], places=9)
        allowed = {rid_of(i) for i in range(0, 400, 2)}
        pre, _ = self.index.search('dato', allowed_rids=allowed, top_k=None)
        self.assertTrue(all(h.rid in allowed for h in pre))
        self.assertEqual(len(pre), len([d for d in reference_boolean(self.docs, 'dato', PRE) if d % 2 == 0]))


class TopKTests(unittest.TestCase):
    def test_heap(self):
        data = [(0.5, 3), (0.9, 1), (0.5, 2), (0.1, 7), (0.9, 0)]
        self.assertEqual(top_k_heap(data, 3), [(0.9, 0), (0.9, 1), (0.5, 2)])
        self.assertEqual(top_k_heap(data, 10), top_k_heap(data, None))
        self.assertEqual(top_k_heap(data, 0), [])
        self.assertEqual(top_k_heap(data, 2, descending=False), [(0.1, 7), (0.5, 2)])


if __name__ == '__main__':
    unittest.main()
