"""Pruebas del recomendador multimodal: filtros y extraccion de features."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from api.recommender import FeatureExtractor, RecommenderService  # noqa: E402

CAR_DIR = ROOT / 'data' / 'datasets' / 'cars'
MODEL_PATH = ROOT / 'data' / 'models' / 'resnet18.pth'
VISION_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ('torch', 'torchvision', 'cv2', 'PIL', 'numpy')
)
DATASET_AVAILABLE = (
    (CAR_DIR / 'codebook.json').exists()
    and MODEL_PATH.exists()
    and (CAR_DIR / 'images').is_dir()
)


class RecommenderGeometryTests(unittest.TestCase):
    def test_filters_include_price_and_location(self):
        clauses, radius = RecommenderService._filters(
            25000, 37.77, -122.42, 100,
        )
        self.assertEqual(radius, 100)
        self.assertTrue(any('precio' in clause for clause in clauses))
        self.assertTrue(any('latitud >=' in clause for clause in clauses))
        self.assertTrue(any('longitud <=' in clause for clause in clauses))

    def test_distance_km(self):
        distance = RecommenderService._distance_km(
            37.77, -122.42, 34.05, -118.24,
        )
        self.assertGreater(distance, 500)
        self.assertLess(distance, 600)

    def test_base64_image_is_decoded(self):
        import base64

        data = base64.b64encode(b'abc').decode('ascii')
        self.assertEqual(
            RecommenderService.decode_image('data:image/png;base64,' + data),
            b'abc',
        )
        with self.assertRaises(Exception):
            RecommenderService.decode_image('no-es-base64')


@unittest.skipUnless(
    VISION_AVAILABLE and DATASET_AVAILABLE,
    'requiere torch, cv2 y el dataset de cars generado',
)
class FeatureExtractorTests(unittest.TestCase):
    def test_extract_histogram_and_embedding(self):
        images = sorted((CAR_DIR / 'images').glob('*.jpg'))

        if not images:
            self.skipTest('no hay imagenes de coches')

        extractor = FeatureExtractor()
        document, embedding = extractor.features(images[0].read_bytes())

        self.assertIsInstance(document, str)
        self.assertEqual(len(embedding), 512)
        norm = math.sqrt(sum(value * value for value in embedding))
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
