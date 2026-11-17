"""Pruebas SQL del modulo vectorial: tipo VECTOR, IVF, SET y API."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from engine.catalog import Catalog  # noqa: E402
from engine.session import Session  # noqa: E402
from parser.parser import Parser  # noqa: E402
from parser.scanner import Scanner  # noqa: E402


SETUP = """
CREATE TABLE items (id INT PRIMARY KEY, categoria CHAR(10), embedding VECTOR(3)) USING HEAP;
INSERT INTO items VALUES (1, 'a', '[1.0, 0.0, 0.0]');
INSERT INTO items VALUES (2, 'a', '[0.9, 0.1, 0.0]');
INSERT INTO items VALUES (3, 'b', '[0.0, 1.0, 0.0]');
INSERT INTO items VALUES (4, 'b', '[0.0, 0.0, 1.0]');
INSERT INTO items VALUES (5, 'a', '[0.8, 0.2, 0.0]');
INSERT INTO items VALUES (6, 'c', '[0.7, 0.1, 0.1]');
"""

INDEX = """
CREATE INDEX idx_vec ON items(embedding) USING IVF
  WITH (lists = 2, metric = 'COSINE', iterations = 5, seed = 7);
"""


class VectorSqlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='bd2vector_sql_')
        self.catalog = Catalog(
            os.path.join(self.tmp, 'catalog.json'),
            os.path.join(self.tmp, 'storage'),
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sql(self, sql: str):
        program = Parser(Scanner(sql)).parse_program()
        return Session(self.catalog, 4096).execute(program)

    @staticmethod
    def last(output):
        return output.results[-1], output.explains[-1]

    @staticmethod
    def strategy(explain):
        def walk(node):
            if node is None:
                return None

            if node.node_type == 'VectorScanPlan':
                return node.detail.split(';', 1)[0]

            for child in node.children:
                found = walk(child)

                if found:
                    return found

            return None

        return walk(explain.root)

    def test_vector_type_and_projection(self):
        output = self.run_sql(SETUP + "SELECT * FROM items WHERE id = 1;")
        self.assertTrue(output.success, output.error)
        result, _ = self.last(output)
        self.assertEqual(result.columns, ['id', 'categoria', 'embedding'])
        self.assertEqual(result.rows[0][2], (1.0, 0.0, 0.0))

    def test_ivf_cosine_query(self):
        output = self.run_sql(
            SETUP + INDEX + "SET ivf.probes = 2;"
            "SELECT id, embedding <=> '[1, 0, 0]' AS dist "
            "FROM items ORDER BY dist ASC LIMIT 3;"
        )
        self.assertTrue(output.success, output.error)
        result, explain = self.last(output)
        self.assertEqual(result.columns, ['id', 'dist'])
        self.assertEqual([row[0] for row in result.rows], [1, 2, 6])
        self.assertAlmostEqual(result.rows[0][1], 0.0, places=5)
        self.assertEqual(self.strategy(explain), 'INDEX_ONLY')

    def test_l2_operator(self):
        output = self.run_sql(
            SETUP + "CREATE INDEX idx_l2 ON items(embedding) USING IVF "
            "WITH (lists = 2, metric = 'L2');"
            "SET ivf.probes = 2;"
            "SELECT id, embedding <-> '[0, 1, 0]' AS dist FROM items "
            "ORDER BY dist ASC LIMIT 1;"
        )
        self.assertTrue(output.success, output.error)
        result, _ = self.last(output)
        self.assertEqual(result.rows[0][0], 3)

    def test_hybrid_prefilter(self):
        output = self.run_sql(
            SETUP + INDEX + "CREATE INDEX idx_cat ON items(categoria) USING BTREE;"
            "SET ivf.probes = 2;"
            "SELECT id, embedding <=> '[1, 0, 0]' AS dist FROM items "
            "WHERE categoria = 'a' ORDER BY dist ASC LIMIT 2;"
        )
        self.assertTrue(output.success, output.error)
        result, explain = self.last(output)
        self.assertEqual([row[0] for row in result.rows], [1, 2])
        self.assertEqual(self.strategy(explain), 'PRE_FILTER')

    def test_force_seqscan_is_exact(self):
        output = self.run_sql(
            SETUP + INDEX + "SET ivf.probes = 2;"
            "SELECT id, embedding <=> '[1, 0, 0]' AS dist FROM items "
            "FORCE SEQSCAN ORDER BY dist ASC LIMIT 3;"
        )
        self.assertTrue(output.success, output.error)
        result, explain = self.last(output)
        self.assertEqual([row[0] for row in result.rows], [1, 2, 6])
        self.assertEqual(self.strategy(explain), 'SEQ_SCAN')

    def test_index_metric_must_match_operator(self):
        output = self.run_sql(
            SETUP + INDEX
            + "SELECT id, embedding <-> '[1, 0, 0]' AS dist FROM items "
            "ORDER BY dist ASC LIMIT 2;"
        )
        self.assertFalse(output.success)
        self.assertIn('metrica', output.error.lower())

    def test_dimension_validation(self):
        output = self.run_sql(
            "CREATE TABLE mal (id INT, v VECTOR(4));"
            "INSERT INTO mal VALUES (1, '[1, 2, 3]');"
        )
        self.assertFalse(output.success)
        self.assertIn('componentes', output.error)

    def test_distance_order_must_be_asc(self):
        output = self.run_sql(
            SETUP
            + "SELECT id, embedding <=> '[1, 0, 0]' AS dist FROM items "
            "ORDER BY dist DESC LIMIT 2;"
        )
        self.assertFalse(output.success)
        self.assertIn('ASC', output.error)

    def test_ivf_requires_vector_column(self):
        output = self.run_sql(
            "CREATE TABLE plano (id INT, texto CHAR(10));"
            "CREATE INDEX idx_malo ON plano(texto) USING IVF;"
        )
        self.assertFalse(output.success)
        self.assertIn('VECTOR', output.error)

    def test_negative_numeric_literals(self):
        output = self.run_sql(
            'CREATE TABLE neg (id INT PRIMARY KEY, v FLOAT) USING HEAP;'
            'INSERT INTO neg VALUES (1, -12.5);'
            'SELECT id FROM neg WHERE v >= -13.0;'
        )
        self.assertTrue(output.success, output.error)
        self.assertEqual(self.last(output)[0].rows, [(1,)])

    def test_csv_import_with_vector_column(self):
        handle = tempfile.NamedTemporaryFile(
            'w', suffix='.csv', delete=False, encoding='utf-8',
        )
        handle.write('1;a;[1,0,0]\n2;b;[0,1,0]\n3;c;[0,0,1]\n')
        handle.close()

        try:
            output = self.run_sql(
                'CREATE TABLE csv_vec (id INT PRIMARY KEY, c CHAR(5), '
                "v VECTOR(3)) USING HEAP "
                f"FROM CSV ('{handle.name}', DELIMITER = ';');"
                'CREATE INDEX idx_csv_vec ON csv_vec(v) USING IVF '
                "WITH (lists = 2, metric = 'L2');"
                'SET ivf.probes = 2;'
                "SELECT id, v <-> '[1,0,0]' AS d FROM csv_vec "
                'ORDER BY d ASC LIMIT 1;'
            )
            self.assertTrue(output.success, output.error)
            self.assertEqual(self.last(output)[0].rows[0][0], 1)
        finally:
            os.unlink(handle.name)

    def test_index_persists_and_rebuilds_after_insert(self):
        self.run_sql(SETUP + INDEX)
        output = self.run_sql(
            "SET ivf.probes = 2;"
            "SELECT id, embedding <=> '[0.81, 0.19, 0]' AS dist FROM items "
            "ORDER BY dist ASC LIMIT 1;"
        )
        self.assertTrue(output.success, output.error)
        self.assertEqual(self.last(output)[0].rows[0][0], 5)

        output = self.run_sql(
            "INSERT INTO items VALUES (7, 'a', '[0.81, 0.19, 0.0]');"
            "SET ivf.probes = 2;"
            "SELECT id, embedding <=> '[0.81, 0.19, 0]' AS dist FROM items "
            "ORDER BY dist ASC LIMIT 1;"
        )
        self.assertTrue(output.success, output.error)
        self.assertEqual(self.last(output)[0].rows[0][0], 7)

        reopened = Catalog(
            os.path.join(self.tmp, 'catalog.json'),
            os.path.join(self.tmp, 'storage'),
        )
        output = Session(reopened, 4096).execute(Parser(Scanner(
            "SET ivf.probes = 2;"
            "SELECT id FROM items;"
        )).parse_program())
        self.assertTrue(output.success, output.error)
        self.assertEqual(len(output.results[-1].rows), 7)

    def test_bovw_reuses_inverted_index(self):
        sql = """
CREATE TABLE imagenes (id INT PRIMARY KEY, categoria CHAR(10), bovw CHAR(200)) USING HEAP;
INSERT INTO imagenes VALUES (1, 'gato', 'v0001 v0001 v0002');
INSERT INTO imagenes VALUES (2, 'gato', 'v0001 v0002 v0002');
INSERT INTO imagenes VALUES (3, 'perro', 'v0003 v0003 v0004');
CREATE INDEX idx_bovw ON imagenes(bovw) USING INVERTED;
SELECT id, COSINE_SCORE() AS score FROM imagenes
WHERE bovw MATCH 'v0001 v0002' ORDER BY score DESC LIMIT 2;
"""
        output = self.run_sql(sql)
        self.assertTrue(output.success, output.error)
        result, _ = self.last(output)
        self.assertEqual(result.columns, ['id', 'score'])
        self.assertEqual([row[0] for row in result.rows], [1, 2])


class RecommenderApiTests(unittest.TestCase):
    def test_text_recommendation_with_filters(self):
        tmp = tempfile.mkdtemp(prefix='bd2rec_api_')

        try:
            os.environ['STORAGE_DIR'] = tmp
            from fastapi.testclient import TestClient
            import api.main as api_main
            api_main = importlib.reload(api_main)
            client = TestClient(api_main.app)

            setup = """
CREATE TABLE vehiculos (id INT PRIMARY KEY, titulo CHAR(150), descripcion CHAR(1000), precio FLOAT, categoria CHAR(30), tienda CHAR(80), ciudad CHAR(80), estado CHAR(10), latitud FLOAT, longitud FLOAT, imagen CHAR(30), palabras CHAR(1000), embedding VECTOR(4)) USING HEAP;
INSERT INTO vehiculos VALUES (1, 'Red sports car', 'Red sports car with low mileage and manual transmission', 25000, 'Coupe', 'Dealer A', 'San Francisco', 'CA', 37.77, -122.42, 'a.jpg', 'v0001 v0002', '[1,0,0,0]');
INSERT INTO vehiculos VALUES (2, 'Blue family SUV', 'Blue family SUV automatic with third row', 30000, 'SUV', 'Dealer B', 'Los Angeles', 'CA', 34.05, -118.24, 'b.jpg', 'v0003 v0004', '[0,1,0,0]');
INSERT INTO vehiculos VALUES (3, 'Red pickup truck', 'Red pickup truck diesel four wheel drive', 20000, 'Pickup', 'Dealer C', 'Houston', 'TX', 29.76, -95.37, 'c.jpg', 'v0001 v0005', '[0,0,1,0]');
CREATE INDEX idx_veh_desc ON vehiculos(descripcion) USING INVERTED WITH (language = 'english', stemming = 'true', stoplist = 'true');
CREATE INDEX idx_veh_vec ON vehiculos(embedding) USING IVF WITH (lists = 2, metric = 'COSINE');
"""
            response = client.post('/api/query', json={'sql': setup})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['success'], response.text)

            response = client.post('/api/recommend', json={
                'description': 'red sports car',
                'max_price': 26000,
                'lat': 37.77,
                'lon': -122.42,
                'radius_km': 100,
                'limit': 5,
            })
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertTrue(data['success'])
            self.assertEqual(data['mode'], 'text')
            self.assertEqual(
                [item['id'] for item in data['text']['results']], [1],
            )
            self.assertEqual(data['text']['results'][0]['imagen_url'], '/media/a.jpg')
            self.assertEqual(data['text']['results'][0]['distancia_km'], 0.0)
            self.assertEqual(
                data['text']['results'][0]['descripcion'],
                'Red sports car with low mileage and manual transmission',
            )
            self.assertEqual(data['limit'], 5)

            response = client.get('/api/recommend/samples?limit=3')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()['products']), 3)

            response = client.post('/api/recommend', json={})
            self.assertEqual(response.status_code, 400)
        finally:
            os.environ.pop('STORAGE_DIR', None)
            shutil.rmtree(tmp, ignore_errors=True)


class VectorApiTests(unittest.TestCase):
    def test_http_query_reports_vector_strategy_and_io(self):
        tmp = tempfile.mkdtemp(prefix='bd2vector_api_')

        try:
            os.environ['STORAGE_DIR'] = tmp
            from fastapi.testclient import TestClient
            import api.main as api_main
            api_main = importlib.reload(api_main)
            client = TestClient(api_main.app)

            response = client.post('/api/query', json={'sql': SETUP + INDEX + "CREATE INDEX idx_cat ON items(categoria) USING BTREE;"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['success'], response.text)

            response = client.post('/api/query', json={'sql':
                "SET ivf.probes = 2;"
                "SELECT id, embedding <=> '[1, 0, 0]' AS dist FROM items "
                "WHERE categoria = 'a' ORDER BY dist ASC LIMIT 2;"
            })
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertTrue(data['success'], data)
            self.assertEqual(data['columns'], ['id', 'dist'])
            self.assertEqual([row[0] for row in data['rows']], [1, 2])
            self.assertEqual(data['strategy'], 'PRE_FILTER')
            self.assertEqual(len(data['explains']), 2)
            self.assertEqual(
                [item['statement_type'] for item in data['explains']],
                ['SET', 'SELECT'],
            )
            self.assertTrue(data['explains'][-1]['explain'])
            self.assertEqual(
                data['explains'][-1]['strategy'], 'PRE_FILTER',
            )
            self.assertGreater(data['io_breakdown']['index_reads'], 0)
            self.assertGreater(data['io_breakdown']['table_reads'], 0)
            self.assertEqual(
                data['metrics']['disk_reads'],
                data['io_breakdown']['index_reads']
                + data['io_breakdown']['table_reads'],
            )

            indexes = client.get('/api/indexes').json()['indexes']
            ivf = [item for item in indexes if item['type'] == 'IVF']
            self.assertEqual(len(ivf), 1)
            self.assertEqual(ivf[0]['options']['lists'], 2)
            self.assertEqual(ivf[0]['options']['metric'], 'COSINE')
        finally:
            os.environ.pop('STORAGE_DIR', None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
