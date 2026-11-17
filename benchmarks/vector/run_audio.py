"""
Demo del camino de audio: BoAW con MFCC y k-NN con DTW/LB_Keogh.

Genera un conjunto sintetico de clips WAV (tonos, chirp, bitono y ruido),
extrae MFCC por ventana y compara:

    1. BoAW: codebook K-Means sobre ventanas + indice invertido + coseno.
    2. k-NN exacto con DTW sobre la secuencia de MFCC.
    3. El mismo k-NN con poda LB_Keogh (Filter-and-Refine).

No requiere archivos de audio externos: los WAV se generan en un directorio
temporal con semilla fija. Salida en benchmarks/results/experiment_audio/.

Uso:
    python benchmarks/vector/run_audio.py
    python benchmarks/vector/run_audio.py --quick --no-plots
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import struct
import tempfile
import time
import wave
from pathlib import Path

from common import (
    Database, import_pyplot, mean, prepare_output, precision_at_k,
    reads_of, sql_string, stdev, write_csv, write_json,
)

from vectorindex import bovw, features
from vectorindex.metrics import Metric
from vectorindex.scan import exact_knn, filter_and_refine

RATE = 16000
DURATION = 0.4
CLASSES = ['tono220', 'tono440', 'tono880', 'chirp', 'bitono', 'ruido']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--clips-per-class', type=int, default=None)
    parser.add_argument('--words', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--queries-per-class', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def synthesize(kind: int, index: int, rng: random.Random) -> list[int]:
    """Genera un clip de 16 bits con variacion determinista por indice."""
    total = int(RATE * DURATION)
    phase = rng.uniform(0, 2 * math.pi)
    samples: list[int] = []

    for position in range(total):
        t = position / RATE
        progress = position / total

        if kind == 0:
            frequency = 220.0
        elif kind == 1:
            frequency = 440.0
        elif kind == 2:
            frequency = 880.0
        elif kind == 3:
            frequency = 200.0 + 1400.0 * progress
        elif kind == 4:
            frequency = 300.0 if int(progress * 8) % 2 == 0 else 900.0
        else:
            frequency = 0.0

        if kind == 5:
            value = rng.gauss(0.0, 5000.0)
        else:
            value = 9000.0 * math.sin(phase + 2 * math.pi * frequency * t)
            value += rng.uniform(-500.0, 500.0)

        samples.append(int(max(-32767, min(32767, value))))

    return samples


def write_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(struct.pack(f'<{len(samples)}h', *samples))


def build_audio_dataset(clips: int, work: Path) -> list[dict]:
    dataset = []
    clips_dir = work / 'clips'
    clips_dir.mkdir(parents=True, exist_ok=True)
    identifier = 0

    for kind, label in enumerate(CLASSES):
        for index in range(clips):
            rng = random.Random(kind * 1000 + index + 7)
            path = clips_dir / f'{identifier:04d}.wav'
            write_wav(path, synthesize(kind, index, rng))
            dataset.append({
                'id': identifier, 'label': label, 'path': path,
            })
            identifier += 1

    return dataset


def main() -> int:
    args = parse_args()
    quick = args.quick
    clips = args.clips_per_class if args.clips_per_class is not None else (4 if quick else 8)
    words = args.words if args.words is not None else (12 if quick else 24)
    queries_per_class = args.queries_per_class if args.queries_per_class is not None else 1
    profile = 'quick' if quick else 'full'

    output = prepare_output('experiment_audio', profile)
    work = Path(tempfile.mkdtemp(prefix='bd2audio_bench_'))

    try:
        dataset = build_audio_dataset(clips, work)
        started = time.perf_counter()
        descriptors = [
            features.mfcc_frames_from_wav(str(item['path']))
            for item in dataset
        ]
        extraction_ms = (time.perf_counter() - started) * 1000.0

        flat = [frame for frames in descriptors for frame in frames]
        sample = random.Random(args.seed).sample(
            flat, min(len(flat), 2000),
        )

        started = time.perf_counter()
        codebook = bovw.build_codebook(
            sample, words=min(words, len(sample)),
            iterations=args.iterations, seed=args.seed,
        )
        codebook_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        histograms = [bovw.histogram(frames, codebook) for frames in descriptors]
        documents = [bovw.histogram_document(histogram) for histogram in histograms]
        quantization_ms = (time.perf_counter() - started) * 1000.0

        csv_path = work / 'audio.csv'

        with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
            for item, document in zip(dataset, documents):
                handle.write(f"{item['id']};{item['label']};{document}\n")

        database = Database(os.path.join(work, 'db'))

        try:
            database.execute(
                'CREATE TABLE audio_bovw (id INT PRIMARY KEY, clase CHAR(20), '
                f"histograma CHAR(500)) USING HEAP FROM CSV ('{csv_path.as_posix()}', "
                "DELIMITER = ';');"
            )
            _, _, inverted_ms = database.execute(
                'CREATE INDEX idx_audio ON audio_bovw(histograma) USING INVERTED;'
            )

            queries = []

            for label in CLASSES:
                items = [item for item in dataset if item['label'] == label]
                queries.extend(items[:queries_per_class])

            per_query = []

            for item in queries:
                result, explain, latency = database.query(
                    'SELECT id, clase, COSINE_SCORE() AS score FROM audio_bovw '
                    f"WHERE histograma MATCH {sql_string(documents[item['id']])} "
                    'ORDER BY score DESC LIMIT 5;'
                )
                labels = [row[1] for row in result.rows]
                per_query.append({
                    'method': 'BoAW', 'query_id': item['id'],
                    'category': item['label'],
                    'precision@5': precision_at_k(labels, item['label'], 5),
                    'latency_ms': round(latency, 3),
                    'reads': reads_of(explain.root)[0],
                })

            records = [
                (item['id'], descriptors[item['id']]) for item in dataset
            ]

            for item in queries:
                query = descriptors[item['id']]

                started = time.perf_counter()
                exact = exact_knn(records, query, 5, Metric.DTW)
                exact_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                refined = filter_and_refine(records, query, 5, radius=2)
                refined_ms = (time.perf_counter() - started) * 1000.0

                exact_labels = [
                    dataset[rid]['label'] for rid, _ in exact
                ]
                refined_labels = [
                    dataset[rid]['label'] for rid, _ in refined
                ]
                per_query.append({
                    'method': 'DTW exacto', 'query_id': item['id'],
                    'category': item['label'],
                    'precision@5': precision_at_k(exact_labels, item['label'], 5),
                    'latency_ms': round(exact_ms, 3),
                    'reads': 0,
                })
                per_query.append({
                    'method': 'DTW LB_Keogh', 'query_id': item['id'],
                    'category': item['label'],
                    'precision@5': precision_at_k(refined_labels, item['label'], 5),
                    'latency_ms': round(refined_ms, 3),
                    'reads': 0,
                })

                if refined_labels != exact_labels:
                    raise RuntimeError(
                        'La poda LB_Keogh cambio el resultado del k-NN DTW.'
                    )

            summary = []

            for method in ('BoAW', 'DTW exacto', 'DTW LB_Keogh'):
                rows = [row for row in per_query if row['method'] == method]
                summary.append({
                    'method': method,
                    'precision@5': round(mean([row['precision@5'] for row in rows]), 4),
                    'precision@5_std': round(stdev([row['precision@5'] for row in rows]), 4),
                    'latency_ms': round(mean([row['latency_ms'] for row in rows]), 3),
                    'latency_ms_std': round(stdev([row['latency_ms'] for row in rows]), 3),
                    'reads': round(mean([row['reads'] for row in rows]), 2),
                })

            write_csv(
                output / 'queries.csv', per_query,
                ['method', 'query_id', 'category', 'precision@5',
                 'latency_ms', 'reads'],
            )
            write_csv(
                output / 'summary.csv', summary,
                ['method', 'precision@5', 'precision@5_std', 'latency_ms',
                 'latency_ms_std', 'reads'],
            )
            write_json(output / 'manifest.json', {
                'experiment': 'audio',
                'profile': profile,
                'dataset': {
                    'clips': len(dataset),
                    'classes': CLASSES,
                    'duration_s': DURATION,
                    'sample_rate': RATE,
                },
                'parameters': {
                    'words': words, 'iterations': args.iterations,
                    'clips_per_class': clips, 'seed': args.seed,
                },
                'times_ms': {
                    'mfcc': round(extraction_ms, 2),
                    'codebook': round(codebook_ms, 2),
                    'histograms': round(quantization_ms, 2),
                    'inverted_index': round(inverted_ms, 2),
                },
                'summary': summary,
            })

            print(
                f'Clips: {len(dataset)}  codebook={codebook.size}  '
                f'MFCC={extraction_ms:.0f} ms  INVERTED={inverted_ms:.0f} ms'
            )
            print(f'{"Metodo":<16}{"P@5":>8}{"ms":>10}')
            for row in summary:
                print(
                    f'{row["method"]:<16}{row["precision@5"]:>8.3f}'
                    f'{row["latency_ms"]:>10.2f}'
                )

            if not args.no_plots:
                make_plots(output, summary)

            print(f'\nResultados en {output}')
            return 0
        finally:
            database.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_plots(output: Path, summary: list[dict]) -> None:
    plt = import_pyplot()
    methods = [row['method'] for row in summary]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(methods, [row['precision@5'] for row in summary],
                color=['#2f6f8f', '#c58b2a', '#4c8a52'])
    axes[0].set_title('Precision@5 (audio)')
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(methods, [row['latency_ms'] for row in summary],
                color=['#2f6f8f', '#c58b2a', '#4c8a52'])
    axes[1].set_title('Latencia media (ms)')

    for axis in axes:
        axis.grid(axis='y', alpha=0.3)
        axis.tick_params(axis='x', rotation=15)

    figure.tight_layout()
    figure.savefig(output / 'audio_summary.png', dpi=140)
    plt.close(figure)


if __name__ == '__main__':
    raise SystemExit(main())
