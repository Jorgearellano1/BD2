"""Pruebas del mapa SQL: metadatos espaciales y todos los puntos."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))


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


class SpatialApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='bd2spatial_api_')
        os.environ['STORAGE_DIR'] = self.tmp
        import api.main as api_main
        self.api = importlib.reload(api_main)
        self.client = TestClient(self.api.app)

    def tearDown(self):
        os.environ.pop('STORAGE_DIR', None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def query(self, sql: str, page_size: int = 50) -> dict:
        response = self.client.post(
            '/api/query',
            json={'sql': sql, 'page_size': page_size},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_radius_map_returns_all_rows(self):
        response = self.query(
            SETUP
            + 'SELECT id, nombre, latitud, longitud FROM locales '
            'WHERE WITHIN_RADIUS(latitud, longitud, 0, 0, 140);',
            page_size=2,
        )
        self.assertTrue(response['success'], response['error'])
        self.assertEqual(response['spatial']['kind'], 'radius')
        self.assertEqual(response['spatial']['center'], {'lat': 0.0, 'lon': 0.0})
        self.assertEqual(response['spatial']['radius_km'], 140.0)
        self.assertEqual(response['total_rows'], 3)
        self.assertEqual(len(response['rows']), 2)

        result_id = response['result_id']
        mapped = self.client.get(
            f'/api/results/{result_id}/map?limit=2000',
        ).json()
        self.assertEqual(mapped['shown'], 3)
        self.assertEqual(mapped['total'], 3)
        self.assertFalse(mapped['truncated'])
        self.assertEqual(mapped['spatial']['kind'], 'radius')
        self.assertEqual(
            {point[2] for point in mapped['points']},
            {'centro', 'norte', 'este'},
        )

    def test_rectangle_metadata(self):
        response = self.query(
            SETUP
            + 'SELECT id, nombre, latitud, longitud FROM locales '
            'WHERE latitud >= 0 AND latitud <= 1 '
            'AND longitud >= 0 AND longitud <= 1;'
        )
        self.assertEqual(response['spatial']['kind'], 'rect')
        bbox = response['spatial']['bbox']
        self.assertEqual(bbox['lat_min'], 0.0)
        self.assertEqual(bbox['lat_max'], 1.0)
        self.assertEqual(bbox['lon_min'], 0.0)
        self.assertEqual(bbox['lon_max'], 1.0)

    def test_knn_metadata(self):
        response = self.query(
            SETUP
            + 'SELECT id, nombre, latitud, longitud FROM locales '
            'WHERE KNN(latitud, longitud, 0, 0, 3);'
        )
        self.assertEqual(response['spatial']['kind'], 'knn')
        self.assertEqual(response['spatial']['k'], 3)
        self.assertEqual(response['spatial']['center'], {'lat': 0.0, 'lon': 0.0})
        self.assertEqual(response['total_rows'], 3)

    def test_map_without_location_returns_empty(self):
        response = self.query(
            SETUP + 'SELECT id, nombre FROM locales;'
        )
        self.assertIsNone(response['spatial'])
        mapped = self.client.get(
            f"/api/results/{response['result_id']}/map",
        ).json()
        self.assertEqual(mapped['points'], [])
        self.assertEqual(mapped['shown'], 0)


if __name__ == '__main__':
    unittest.main()
