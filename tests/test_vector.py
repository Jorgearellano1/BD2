"""Pruebas del nucleo vectorial: metricas, K-Means, IVF, BoVW y datos."""

from __future__ import annotations

import math
import os
import random
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from vectorindex import bovw, features, scan  # noqa: E402
from vectorindex.ivf import IvfIndex  # noqa: E402
from vectorindex.kmeans import kmeans  # noqa: E402
from vectorindex.metrics import (  # noqa: E402
    Metric,
    cosine_distance,
    distance,
    dtw_distance,
    inner_product,
    l2_distance,
    lb_keogh,
    metric_by_name,
    metric_by_operator,
)
from vectorindex.options import normalize_vector_options  # noqa: E402
from vectorindex.vectors import (  # noqa: E402
    format_vector,
    parse_vector,
    vector_from_bytes,
    vector_to_bytes,
)


def clustered_vectors(count: int = 120) -> list[tuple[tuple[int, int], tuple[float, ...]]]:
    """Cuatro grupos separados, con ruido deterministico."""
    rng = random.Random(7)
    centers = [
        (0.0, 0.0, 0.0, 0.0),
        (10.0, 10.0, 0.0, 0.0),
        (-10.0, 10.0, 0.0, 0.0),
        (0.0, -10.0, 10.0, 0.0),
    ]
    vectors = []

    for index in range(count):
        center = centers[index % len(centers)]
        vector = tuple(value + rng.uniform(-1.0, 1.0) for value in center)
        vectors.append(((0, index), vector))

    return vectors


class VectorLiteralTests(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(parse_vector('[1, -2.5, 3]'), (1.0, -2.5, 3.0))
        self.assertEqual(parse_vector('[]'), ())
        self.assertEqual(format_vector((1.0, 2.5)), '[1.000000,2.500000]')

    def test_bytes_roundtrip(self):
        vector = (0.5, -0.25, 3.0)
        data = vector_to_bytes(vector)
        self.assertEqual(len(data), 12)
        self.assertEqual(vector_from_bytes(data, 3), vector)


class MetricTests(unittest.TestCase):
    def test_l2_and_cosine(self):
        self.assertAlmostEqual(l2_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(cosine_distance((1, 0), (0, 1)), 1.0)
        self.assertAlmostEqual(cosine_distance((1, 0), (2, 0)), 0.0)
        self.assertAlmostEqual(inner_product((1, 2), (3, 4)), 11.0)
        self.assertAlmostEqual(distance(Metric.IP, (1, 0), (2, 0)), -2.0)

    def test_operator_mapping(self):
        self.assertEqual(metric_by_operator('<->'), Metric.L2)
        self.assertEqual(metric_by_operator('<=>'), Metric.COSINE)
        self.assertEqual(metric_by_operator('<#>'), Metric.IP)
        self.assertEqual(metric_by_name('cosine'), Metric.COSINE)

    def test_dtw(self):
        self.assertEqual(dtw_distance((1, 2, 3), (1, 2, 3)), 0.0)
        self.assertAlmostEqual(dtw_distance((0, 0, 0), (1, 1, 1)), math.sqrt(3))

    def test_multivariate_dtw_and_lb_keogh(self):
        first = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        same = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        shifted = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        self.assertEqual(dtw_distance(first, same), 0.0)
        self.assertAlmostEqual(dtw_distance(first, shifted), 2.0)

        rng = random.Random(17)

        for _ in range(20):
            length = 8
            query = [tuple(rng.uniform(-3, 3) for _ in range(3)) for _ in range(length)]
            candidate = [tuple(rng.uniform(-3, 3) for _ in range(3)) for _ in range(length)]
            bound = lb_keogh(query, candidate, 2)
            actual = dtw_distance(query, candidate)
            self.assertLessEqual(bound, actual + 1e-9)

    def test_lb_keogh_is_lower_bound(self):
        rng = random.Random(11)

        for _ in range(30):
            length = 12
            query = tuple(rng.uniform(-5, 5) for _ in range(length))
            candidate = tuple(rng.uniform(-5, 5) for _ in range(length))
            bound = lb_keogh(query, candidate, 2)
            actual = dtw_distance(query, candidate)
            self.assertLessEqual(bound, actual + 1e-9)


class KMeansTests(unittest.TestCase):
    def test_two_clusters(self):
        points = [
            (0.0, 0.0), (0.1, 0.2), (-0.1, 0.1),
            (10.0, 10.0), (10.1, 9.9), (9.9, 10.1),
        ]
        centroids, assignments = kmeans(points, 2, iterations=20, seed=3)
        self.assertEqual(len(centroids), 2)
        self.assertEqual(assignments[:3], [assignments[0]] * 3)
        self.assertEqual(assignments[3:], [assignments[3]] * 3)
        self.assertNotEqual(assignments[0], assignments[3])

    def test_deterministic(self):
        points = [(float(i), float(i % 3)) for i in range(20)]
        first = kmeans(points, 3, iterations=5, seed=9)
        second = kmeans(points, 3, iterations=5, seed=9)
        self.assertEqual(first, second)


class IvfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bd2ivf_')
        self.vectors = clustered_vectors(120)
        self.index, self.stats = IvfIndex.build(
            os.path.join(self.tmp.name, 'idx'),
            self.vectors,
            lists=4,
            metric=Metric.L2,
            page_size=4096,
            iterations=5,
            seed=5,
        )

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_build_stats(self):
        self.assertEqual(self.stats.num_vectors, 120)
        self.assertEqual(self.stats.lists, 4)
        self.assertGreater(self.stats.pages_written, 0)

    def test_full_probe_matches_exact(self):
        query = (0.1, 0.0, 0.0, 0.0)
        exact = scan.exact_knn(
            iter(self.vectors), query, 5, Metric.L2,
        )
        approximate, stats = self.index.search(query, 5, probes=4)
        self.assertEqual(stats.cells_probed, 4)
        self.assertEqual(
            [rid for rid, _ in approximate],
            [rid for rid, _ in exact],
        )

    def test_more_probes_read_at_least_as_much(self):
        query = (0.1, 0.0, 0.0, 0.0)
        before_one = self.index.page_reads
        self.index.search(query, 5, probes=1)
        one = self.index.page_reads - before_one
        before_all = self.index.page_reads
        self.index.search(query, 5, probes=4)
        every = self.index.page_reads - before_all
        self.assertGreaterEqual(every, one)

    def test_persistence_and_dirty(self):
        path = self.index.directory
        self.index.close()
        reopened = IvfIndex(path, 4, Metric.L2, 4096)
        reopened.open()
        hits, _ = reopened.search((10.0, 10.0, 0.0, 0.0), 3, probes=4)
        self.assertEqual(len(hits), 3)

        reopened.mark_dirty()
        self.assertTrue(reopened.dirty)
        reopened.close()

        again = IvfIndex(path, 4, Metric.L2, 4096)
        again.open()
        self.assertTrue(again.dirty)
        again.close()

    def test_page_size_must_fit_the_vector(self):
        vectors = [((0, index), tuple(float(index) for _ in range(128)))
                   for index in range(8)]

        with tempfile.TemporaryDirectory(prefix='bd2ivf_page_') as tmp:
            with self.assertRaises(Exception) as context:
                IvfIndex.build(
                    os.path.join(tmp, 'idx'), vectors, lists=2,
                    metric=Metric.L2, page_size=512,
                )

            self.assertIn('pagina', str(context.exception).lower())


class BovwTests(unittest.TestCase):
    def test_codebook_histogram_and_document(self):
        descriptors = [
            (1.0, 0.0, 0.0), (0.9, 0.1, 0.0),
            (0.0, 1.0, 0.0), (0.1, 0.9, 0.0),
        ]
        codebook = bovw.build_codebook(descriptors, words=2, iterations=10, seed=1)
        first = bovw.histogram(descriptors[:2], codebook)
        second = bovw.histogram(descriptors[2:], codebook)
        self.assertEqual(sum(first.values()), 2)
        self.assertEqual(sum(second.values()), 2)
        self.assertNotEqual(set(first), set(second))

        document = bovw.histogram_document(first)
        self.assertEqual(bovw.parse_document(document), first)

    def test_tfidf_cosine(self):
        weights = bovw.tfidf_weights({0: 2, 1: 1}, {0: 3, 1: 3}, 6)
        other = bovw.tfidf_weights({0: 1, 1: 2}, {0: 3, 1: 3}, 6)
        self.assertAlmostEqual(bovw.cosine(weights, weights), 1.0)
        self.assertLess(bovw.cosine(weights, other), 1.0)


class FeatureTests(unittest.TestCase):
    def test_load_sift_and_embeddings(self):
        with tempfile.TemporaryDirectory(prefix='bd2features_') as tmp:
            sift_path = os.path.join(tmp, 'sift.bin')

            with open(sift_path, 'wb') as handle:
                handle.write(b'BD2SIFT1')
                handle.write(struct.pack('<II', 2, 128))
                handle.write(struct.pack('<I', 2))
                handle.write(bytes([0, 127, 255] + [0] * 125))
                handle.write(bytes([10, 20, 30] + [0] * 125))
                handle.write(struct.pack('<I', 1))
                handle.write(bytes([255] * 128))

            documents = features.load_sift_descriptors(sift_path)
            self.assertEqual(len(documents), 2)
            self.assertEqual(len(documents[0]), 2)
            self.assertAlmostEqual(documents[0][0][0], 0.0)
            self.assertAlmostEqual(documents[0][0][2], 1.0)

            embeddings_path = os.path.join(tmp, 'emb.bin')

            with open(embeddings_path, 'wb') as handle:
                handle.write(b'BD2EMB01')
                handle.write(struct.pack('<II', 2, 4))
                handle.write(struct.pack('<8f', 1, 0, 0, 0, 0, 1, 0, 0))

            vectors = features.load_embeddings(embeddings_path)
            self.assertEqual(vectors[0], (1.0, 0.0, 0.0, 0.0))

    def test_mfcc_frames_from_wav(self):
        with tempfile.TemporaryDirectory(prefix='bd2audio_') as tmp:
            path = os.path.join(tmp, 'chirp.wav')
            rate = 16000
            frames = rate // 2
            samples = []
            phase = 0.0

            for index in range(frames):
                frequency = 200.0 + 1800.0 * index / frames
                phase += 2 * math.pi * frequency / rate
                samples.append(int(12000 * math.sin(phase)))

            with wave.open(path, 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(struct.pack(f'<{frames}h', *samples))

            descriptors = features.mfcc_frames_from_wav(path)
            self.assertGreater(len(descriptors), 2)
            self.assertEqual(len(descriptors[0]), 13)
            self.assertNotEqual(descriptors[0], descriptors[-1])

            mean = features.mfcc_from_wav(path)
            expected = tuple(
                sum(frame[index] for frame in descriptors) / len(descriptors)
                for index in range(13)
            )

            for value, target in zip(mean, expected):
                self.assertAlmostEqual(value, target)

    def test_audio_bovw_pipeline(self):
        with tempfile.TemporaryDirectory(prefix='bd2audio2_') as tmp:
            documents = []

            for index, frequency in enumerate((220.0, 880.0)):
                path = os.path.join(tmp, f'tone{index}.wav')
                rate = 16000
                frames = rate // 4
                samples = [
                    int(9000 * math.sin(2 * math.pi * frequency * position / rate))
                    for position in range(frames)
                ]

                with wave.open(path, 'wb') as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(rate)
                    handle.writeframes(struct.pack(f'<{frames}h', *samples))

                documents.append(features.mfcc_frames_from_wav(path))

            codebook = bovw.build_codebook(
                documents[0] + documents[1], words=4, iterations=5, seed=2,
            )
            first = bovw.histogram(documents[0], codebook)
            second = bovw.histogram(documents[1], codebook)
            self.assertEqual(sum(first.values()), len(documents[0]))
            self.assertNotEqual(first, second)

            document = bovw.histogram_document(first)
            self.assertEqual(bovw.parse_document(document), first)


class ScanTests(unittest.TestCase):
    def test_range_and_dtw_refine(self):
        rng = random.Random(13)
        records = [
            ((0, index), tuple(rng.uniform(-1, 1) for _ in range(8)))
            for index in range(40)
        ]
        query = records[0][1]

        knn = scan.exact_knn(iter(records), query, 5, Metric.DTW)
        refined = scan.filter_and_refine(iter(records), query, 5, radius=2)
        self.assertEqual(knn, refined)

        ranged = scan.exact_range(
            iter(records), query, 0.5, Metric.L2,
        )
        self.assertTrue(all(distance <= 0.5 for _, distance in ranged))


class OptionTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        options = normalize_vector_options(None)
        self.assertEqual(options['lists'], 25)
        self.assertEqual(options['metric'], 'COSINE')
        options = normalize_vector_options(
            {'lists': 100, 'metric': 'l2', 'probes': 4},
        )
        self.assertEqual(options['metric'], 'L2')
        self.assertEqual(options['probes'], 4)

        with self.assertRaises(ValueError):
            normalize_vector_options({'metric': 'JACCARD'})

        with self.assertRaises(ValueError):
            normalize_vector_options({'lists': 0})


if __name__ == '__main__':
    unittest.main()
