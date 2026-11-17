"""Pruebas de WITHIN_RADIUS y KNN sobre R-Tree."""

from __future__ import annotations

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
CREATE TABLE locales (
    id INT PRIMARY KEY,
    nombre CHAR(20),
    latitud FLOAT,
    longitud FLOAT
) USING HEAP;
INSERT INTO locales VALUES (1, 'centro', 0.0, 0.0);
INSERT INTO locales VALUES (2, 'norte', 0.0, 1.0);
INSERT INTO locales VALUES (3, 'este', 1.0, 0.0);
INSERT INTO locales VALUES (4, 'diagonal', 1.0, 1.0);
INSERT INTO locales VALUES (5, 'lejos', 10.0, 10.0);
CREATE INDEX idx_locales_geo ON locales(latitud, longitud) USING RTREE;
"""


class SpatialSqlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='bd2spatial_sql_')
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
    def uses_rtree(node) -> bool:
        if node is None:
            return False

        if any(operation.resource_type == 'RTree' for operation in node.operations):
            return True

        return any(SpatialSqlTests.uses_rtree(child) for child in node.children)

    def test_within_radius_uses_rtree_and_exact_distance(self):
        output = self.run_sql(
            SETUP
            + 'SELECT id, nombre FROM locales '
            'WHERE WITHIN_RADIUS(latitud, longitud, 0, 0, 140);'
        )
        self.assertTrue(output.success, output.error)
        result, explain = self.last(output)
        self.assertEqual({row[0] for row in result.rows}, {1, 2, 3})
        self.assertTrue(self.uses_rtree(explain.root))

    def test_knn_uses_rtree(self):
        output = self.run_sql(
            SETUP
            + 'SELECT id, nombre FROM locales '
            'WHERE KNN(latitud, longitud, 0, 0, 2);'
        )
        self.assertTrue(output.success, output.error)
        result, explain = self.last(output)
        self.assertEqual(result.rows[0][0], 1)
        self.assertEqual(len(result.rows), 2)
        self.assertTrue({row[0] for row in result.rows} <= {1, 2, 3})
        self.assertTrue(self.uses_rtree(explain.root))

    def test_knn_respects_k(self):
        output = self.run_sql(
            SETUP
            + 'SELECT id FROM locales '
            'WHERE KNN(latitud, longitud, 0, 0, 1);'
        )
        self.assertTrue(output.success, output.error)
        result, _ = self.last(output)
        self.assertEqual([row[0] for row in result.rows], [1])

    def test_within_radius_requires_two_numeric_columns(self):
        output = self.run_sql(
            'CREATE TABLE malo (id INT PRIMARY KEY, nombre CHAR(10)) USING HEAP;'
            'SELECT id FROM malo WHERE '
            'WITHIN_RADIUS(nombre, id, 0, 0, 10);'
        )
        self.assertFalse(output.success)
        self.assertIn('numerica', (output.error or '').lower())


if __name__ == '__main__':
    unittest.main()
