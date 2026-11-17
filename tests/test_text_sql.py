"""
Pruebas integradas del modulo textual a traves de SQL: parser, semantica,
planner, executor, persistencia del catalogo, mantenimiento tras DML y la API
HTTP (FastAPI TestClient). Cada prueba usa un catalogo temporal propio.

Ejecutar desde la raiz:  .venv/bin/python -m pytest tests/test_text_sql.py
"""

from __future__ import annotations

import importlib
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from engine.catalog import Catalog  # noqa: E402
from engine.session import Session  # noqa: E402
from parser.ast import BinaryExp, CreateIndexStm, IdExp, IndexType, LiteralExp, SelectStm  # noqa: E402
from parser.parser import Parser  # noqa: E402
from parser.scanner import Scanner  # noqa: E402
from textindex.preprocessing import TextPreprocessor  # noqa: E402
from textindex.query import parse_query, positive_terms  # noqa: E402

DOCS = [
    'La casa es grande',
    'El gato está en la casa',
    'La casa es bonita y grande',
    'El sol brilla sobre la casa',
    'Gato negro, gato blanco y gato gris',
    'Aprendizaje profundo con redes neuronales',
    'Aprendizaje profundo básico para principiantes',
    'Aprendizaje automático profundo y redes',
    'Bases de datos distribuidas y fragmentación horizontal',
    'Índice invertido en disco para recuperación de información',
    'Recuperación de información con similitud de coseno',
    'El niño compró una canción el año pasado',
]


def setup_sql() -> str:
    sql = ('CREATE TABLE documentos (id INT PRIMARY KEY, titulo CHAR(30), '
           'cuerpo CHAR(160), anio INT);')
    for i, text in enumerate(DOCS, start=1):
        sql += f"INSERT INTO documentos VALUES ({i}, 'doc {i}', '{text}', {2014 + i});"
    return sql


def reference_ranking(docs: dict[int, str], query: str) -> dict[int, float]:
    """TF-IDF logaritmico y coseno con vectores densos (calculo independiente)."""
    pre = TextPreprocessor('spanish')
    bags = {k: pre.term_frequencies(v) for k, v in docs.items()}
    n = len(bags)
    vocab = sorted({t for b in bags.values() for t in b})
    df = {t: sum(1 for b in bags.values() if t in b) for t in vocab}

    def w(tf, t):
        return (1 + math.log10(tf)) * math.log10(n / df[t]) if tf else 0.0

    qtf: dict[str, int] = {}
    for t in positive_terms(parse_query(query, pre)):
        if t in df:
            qtf[t] = qtf.get(t, 0) + 1
    q = [w(qtf.get(t, 0), t) for t in vocab]
    qn = math.sqrt(sum(x * x for x in q))
    out = {}
    for key, bag in bags.items():
        d = [w(bag.get(t, 0), t) for t in vocab]
        dn = math.sqrt(sum(x * x for x in d))
        out[key] = sum(a * b for a, b in zip(q, d)) / (dn * qn) if dn and qn else 0.0
    return out


def walk(node):
    if node is None:
        return
    yield node
    for child in node.children:
        yield from walk(child)


class SqlCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix='bd2texto_sql_')
        self.catalog_path = os.path.join(self.tmp, 'catalog.json')
        self.data_dir = os.path.join(self.tmp, 'data')
        self.catalog = Catalog(self.catalog_path, self.data_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sql(self, sql: str, ok: bool = True, catalog: Catalog | None = None):
        program = Parser(Scanner(sql)).parse_program()
        output = Session(catalog or self.catalog, 4096).execute(program)
        if ok:
            self.assertTrue(output.success, output.error)
        return output

    def last(self, output):
        return output.results[-1], output.explains[-1]

    def node_types(self, explain) -> list[str]:
        return [node.node_type for node in walk(explain.root)]

    def strategy(self, explain) -> str | None:
        for node in walk(explain.root):
            if node.node_type in ('TextIndexScanPlan', 'FullTextScanPlan'):
                return node.detail.split(';')[0]
        return None


class ParserCompatibilityTests(unittest.TestCase):
    def parse(self, sql):
        return Parser(Scanner(sql)).parse_program().statements[0]

    def test_p1_statements_unchanged(self):
        stm = self.parse('SELECT id, nombre FROM t WHERE id = 5;')
        self.assertEqual(stm, SelectStm(['id', 'nombre'], 't', where=BinaryExp(IdExp('id'), LiteralExp(5), stm.where.op)))
        self.assertEqual(stm.output_columns(), ['id', 'nombre'])
        self.assertEqual((stm.score_alias, stm.order_by, stm.limit), (None, None, None))
        idx = self.parse('CREATE INDEX i ON t (id) USING BTREE;')
        self.assertEqual(idx, CreateIndexStm('i', 't', 'id', IndexType.BTREE))

    def test_enunciado_syntax(self):
        stm = self.parse(
            "SELECT id , titulo , COSINE_SCORE () AS relevancia FROM documentos "
            "WHERE cuerpo MATCH ' aprendizaje AND profundo AND NOT basico ' "
            "ORDER BY relevancia DESC LIMIT 10;")
        self.assertEqual(stm.output_columns(), ['id', 'titulo', 'relevancia'])
        self.assertEqual((stm.order_by, stm.order_desc, stm.limit), ('relevancia', True, 10))
        idx = self.parse("CREATE INDEX idx_texto ON documentos ( cuerpo ) USING INVERTED "
                         "WITH ( scoring = 'TFIDF_COSINE') ;")
        self.assertEqual(idx.options, {'scoring': 'TFIDF_COSINE'})

    def test_quote_escape(self):
        stm = self.parse("SELECT * FROM t WHERE c MATCH 'it''s';")
        self.assertEqual(stm.where.query, "it's")


class TextSqlTests(SqlCase):
    def setUp(self) -> None:
        super().setUp()
        self.run_sql(setup_sql())
        self.run_sql("CREATE INDEX idx_texto ON documentos (cuerpo) USING INVERTED "
                     "WITH (scoring = 'TFIDF_COSINE', language = 'spanish');")

    def test_index_files_and_catalog(self):
        info = self.catalog.get_index('idx_texto')
        self.assertEqual(info.kind.value, 'INVERTED')
        self.assertEqual(info.options['scoring'], 'TFIDF_COSINE')
        self.assertEqual(sorted(os.listdir(info.file_path)), ['docs.dat', 'postings.inv', 'vocab.dic'])

    def test_enunciado_query_and_full_scan_agree(self):
        query = ("SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos {force} "
                 "WHERE cuerpo MATCH 'aprendizaje AND profundo AND NOT basico' "
                 "ORDER BY relevancia DESC LIMIT 10;")
        indexed, explain = self.last(self.run_sql(query.format(force='')))
        scanned, explain2 = self.last(self.run_sql(query.format(force='FORCE SEQSCAN')))
        self.assertEqual(self.strategy(explain), 'INDEX_ONLY')
        self.assertEqual(self.strategy(explain2), 'FULL_TEXT_SCAN')
        self.assertEqual(indexed.columns, ['id', 'titulo', 'relevancia'])
        self.assertEqual({row[0] for row in indexed.rows}, {6, 8})
        self.assertEqual(indexed.rows, scanned.rows)

    def test_scores_match_reference(self):
        query = 'informacion recuperacion casa gato gato'
        result, _ = self.last(self.run_sql(
            f"SELECT id, COSINE_SCORE() AS s FROM documentos WHERE cuerpo MATCH '{query}' "
            "ORDER BY s DESC;"))
        expected = reference_ranking(dict(enumerate(DOCS, start=1)), query)
        self.assertTrue(result.rows)
        for doc_id, score in result.rows:
            self.assertAlmostEqual(score, round(expected[doc_id], 6), places=6)
        self.assertEqual([r[1] for r in result.rows], sorted((r[1] for r in result.rows), reverse=True))

    def test_boolean_without_score_and_limit(self):
        result, _ = self.last(self.run_sql(
            "SELECT id FROM documentos WHERE cuerpo MATCH 'casa OR gato';"))
        self.assertEqual([r[0] for r in result.rows], [1, 2, 3, 4, 5])
        result, _ = self.last(self.run_sql(
            "SELECT id FROM documentos WHERE cuerpo MATCH 'casa AND NOT grande' LIMIT 1;"))
        self.assertEqual(result.rows, [(2,)])

    def test_prefilter_postfilter_and_order_by_column(self):
        sql = ("SELECT id, anio, COSINE_SCORE() AS r FROM documentos WHERE cuerpo MATCH "
               "'casa gato' AND {cond} ORDER BY r DESC LIMIT 3;")
        pre, explain = self.last(self.run_sql(sql.format(cond='id <= 3')))
        self.assertEqual(self.strategy(explain), 'PRE_FILTER')
        self.assertIn('IndexRangeScanPlan', self.node_types(explain))
        self.assertTrue(all(row[0] <= 3 for row in pre.rows))
        post, explain = self.last(self.run_sql(sql.format(cond='anio >= 2018')))
        self.assertEqual(self.strategy(explain), 'POST_FILTER')
        self.assertTrue(all(row[1] >= 2018 for row in post.rows))
        scan, _ = self.last(self.run_sql(sql.format(cond='anio >= 2018').replace(
            'FROM documentos', 'FROM documentos FORCE SEQSCAN')))
        self.assertEqual(post.rows, scan.rows)
        ordered, explain = self.last(self.run_sql(
            "SELECT id FROM documentos WHERE cuerpo MATCH 'casa' ORDER BY anio DESC LIMIT 2;"))
        self.assertEqual(ordered.rows, [(4,), (3,)])
        self.assertIn('SortPlan', self.node_types(explain))

    def test_edge_queries(self):
        for query in ('palabrainexistente', 'el la de', 'casa AND palabrainexistente'):
            result, _ = self.last(self.run_sql(
                f"SELECT id, COSINE_SCORE() AS s FROM documentos WHERE cuerpo MATCH '{query}';"))
            self.assertEqual(result.rows, [], query)
        result, _ = self.last(self.run_sql(
            "SELECT id FROM documentos WHERE cuerpo MATCH 'niño canción año';"))
        self.assertEqual(result.rows, [(12,)])
        output = self.run_sql("SELECT id FROM documentos WHERE cuerpo MATCH 'NOT casa';", ok=False)
        self.assertIn('NOT', output.error)

    def test_reopen_after_restart_without_rebuild(self):
        before, _ = self.last(self.run_sql(
            "SELECT id, COSINE_SCORE() AS s FROM documentos WHERE cuerpo MATCH 'redes' ORDER BY s DESC;"))
        restarted = Catalog(self.catalog_path, self.data_dir)
        after, explain = self.last(self.run_sql(
            "SELECT id, COSINE_SCORE() AS s FROM documentos WHERE cuerpo MATCH 'redes' ORDER BY s DESC;",
            catalog=restarted))
        self.assertEqual(before.rows, after.rows)
        self.assertEqual(explain.root.pages_written, 0)

    def test_insert_and_delete_trigger_rebuild(self):
        self.run_sql("INSERT INTO documentos VALUES (50, 'nuevo', 'Un gato duerme en el tejado', 2030);")
        result, explain = self.last(self.run_sql(
            "SELECT id FROM documentos WHERE cuerpo MATCH 'tejado';"))
        self.assertEqual(result.rows, [(50,)])
        self.assertGreater(explain.root.pages_written, 0)  # reconstruccion
        self.run_sql('DELETE FROM documentos WHERE id = 50;')
        result, _ = self.last(self.run_sql("SELECT id FROM documentos WHERE cuerpo MATCH 'tejado';"))
        self.assertEqual(result.rows, [])
        docs = dict(enumerate(DOCS, start=1))
        scored, _ = self.last(self.run_sql(
            "SELECT id, COSINE_SCORE() AS s FROM documentos WHERE cuerpo MATCH 'gato' ORDER BY s DESC;"))
        expected = reference_ranking(docs, 'gato')
        for doc_id, score in scored.rows:
            self.assertAlmostEqual(score, round(expected[doc_id], 6), places=6)

    def test_semantic_errors(self):
        cases = {
            "SELECT id FROM documentos WHERE anio MATCH 'x';": 'CHAR',
            "SELECT id FROM documentos WHERE cuerpo MATCH 'x' OR id = 1;": 'AND',
            "SELECT id FROM documentos WHERE NOT cuerpo MATCH 'x';": 'AND',
            "SELECT id, COSINE_SCORE() AS s FROM documentos WHERE id = 1;": 'MATCH',
            "SELECT COSINE_SCORE() AS anio FROM documentos WHERE cuerpo MATCH 'x';": 'alias',
            "CREATE INDEX i2 ON documentos (anio) USING INVERTED;": 'CHAR',
            "CREATE INDEX i3 ON documentos (titulo) USING BTREE WITH (scoring = 'TFIDF_COSINE');": 'WITH',
            "CREATE INDEX i4 ON documentos (titulo) USING INVERTED WITH (scoring = 'BM25');": 'scoring',
            "DELETE FROM documentos WHERE cuerpo MATCH 'casa';": 'SELECT',
            "SELECT id FROM documentos WHERE cuerpo MATCH 'a' AND titulo MATCH 'b';": 'un MATCH',
            "SELECT id FROM documentos WHERE cuerpo MATCH 'casa' ORDER BY inexistente;": 'inexistente',
        }
        for sql, fragment in cases.items():
            output = self.run_sql(sql, ok=False)
            self.assertFalse(output.success, sql)
            self.assertIn(fragment, output.error, sql)
        output = self.run_sql("CREATE INDEX otro ON documentos (cuerpo) USING INVERTED;", ok=False)
        self.assertIn('Ya existe', output.error)

    def test_force_index_rules(self):
        _, explain = self.last(self.run_sql(
            "SELECT id FROM documentos FORCE INDEX(idx_texto) WHERE cuerpo MATCH 'casa';"))
        self.assertEqual(self.strategy(explain), 'INDEX_ONLY')
        _, explain = self.last(self.run_sql(
            "SELECT id FROM documentos FORCE INDEX(documentos_id_bplustree) "
            "WHERE cuerpo MATCH 'casa' AND id <= 2;"))
        self.assertEqual(self.strategy(explain), 'FULL_TEXT_SCAN')


class NonTextOrderLimitTests(SqlCase):
    def test_order_by_limit_heap_and_sequential(self):
        self.run_sql(setup_sql())
        result, _ = self.last(self.run_sql('SELECT id FROM documentos ORDER BY anio DESC LIMIT 3;'))
        self.assertEqual(result.rows, [(12,), (11,), (10,)])
        self.run_sql('CREATE TABLE s (id INT PRIMARY KEY, v INT) USING SEQUENTIAL;'
                     'INSERT INTO s VALUES (1, 30);INSERT INTO s VALUES (2, 10);INSERT INTO s VALUES (3, 20);')
        result, _ = self.last(self.run_sql('SELECT id FROM s WHERE id >= 1 ORDER BY v LIMIT 2;'))
        self.assertEqual(result.rows, [(2,), (3,)])
        result, _ = self.last(self.run_sql('SELECT id FROM documentos WHERE id = 1 AND id = 2 ORDER BY anio;'))
        self.assertEqual(result.rows, [])

    def test_full_text_scan_on_sequential_and_empty_index(self):
        self.run_sql("CREATE TABLE notas (id INT PRIMARY KEY, txt CHAR(60)) USING SEQUENTIAL;"
                     "INSERT INTO notas VALUES (1, 'gato en casa');INSERT INTO notas VALUES (2, 'perro');")
        result, explain = self.last(self.run_sql("SELECT id FROM notas WHERE txt MATCH 'gato';"))
        self.assertEqual(result.rows, [(1,)])
        self.assertEqual(self.strategy(explain), 'FULL_TEXT_SCAN')
        self.run_sql("CREATE TABLE vacia (id INT, txt CHAR(40));"
                     "CREATE INDEX iv ON vacia (txt) USING INVERTED;")
        result, _ = self.last(self.run_sql(
            "SELECT id, COSINE_SCORE() AS s FROM vacia WHERE txt MATCH 'gato';"))
        self.assertEqual(result.rows, [])


class ApiTests(unittest.TestCase):
    def test_http_query_reports_strategy_and_io(self):
        tmp = tempfile.mkdtemp(prefix='bd2texto_api_')
        try:
            os.environ['STORAGE_DIR'] = tmp
            from fastapi.testclient import TestClient
            import api.main as api_main
            api_main = importlib.reload(api_main)
            client = TestClient(api_main.app)
            response = client.post('/api/query', json={'sql': setup_sql() +
                "CREATE INDEX idx_texto ON documentos (cuerpo) USING INVERTED WITH (scoring = 'TFIDF_COSINE');"})
            self.assertEqual(response.status_code, 200, response.text)
            response = client.post('/api/query', json={'sql':
                "SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos "
                "WHERE cuerpo MATCH 'casa gato' AND id <= 4 ORDER BY relevancia DESC LIMIT 3;"})
            data = response.json()
            self.assertTrue(data['success'], data)
            self.assertEqual(data['columns'], ['id', 'titulo', 'relevancia'])
            self.assertEqual(data['strategy'], 'PRE_FILTER')
            self.assertGreater(data['io_breakdown']['index_reads'], 0)
            self.assertGreater(data['io_breakdown']['table_reads'], 0)
            self.assertEqual(data['metrics']['disk_reads'],
                             data['io_breakdown']['index_reads'] + data['io_breakdown']['table_reads'])
            indexes = client.get('/api/indexes').json()['indexes']
            inverted = [i for i in indexes if i['type'] == 'INVERTED']
            self.assertEqual(inverted[0]['options']['scoring'], 'TFIDF_COSINE')
        finally:
            os.environ.pop('STORAGE_DIR', None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
