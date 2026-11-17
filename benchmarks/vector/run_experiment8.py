"""
Experimento 8: compromiso eficiencia-calidad del indice IVF.

Toma como verdad base el k-NN lineal exacto y varia el numero de celdas
consultadas (probes) para medir Recall@10, latencia y bloques de disco.

Dataset: data/datasets/dry_beans.arff (13,687 vectores de 16 dimensiones y
7 clases). Se estandarizan las columnas antes de indexar.

Uso:
    python benchmarks/vector/run_experiment8.py
    python benchmarks/vector/run_experiment8.py --quick
    python benchmarks/vector/run_experiment8.py --sizes 1000,3000 --queries 15
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

from common import (
    DATA, Database, BenchmarkError, import_pyplot, mean, plan_name,
    prepare_output, reads_of, recall_at_k, sql_vector, stdev, write_csv,
    write_json,
)

from vectorindex.scan import exact_knn
from vectorindex.metrics import Metric

K = 10
PROBES = [1, 2, 5, 10, 20, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sizes', type=str, default=None)
    parser.add_argument('--queries', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def standardize(vectors: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    dimensions = len(vectors[0])
    means = [
        sum(vector[axis] for vector in vectors) / len(vectors)
        for axis in range(dimensions)
    ]
    deviations = []

    for axis in range(dimensions):
        variance = sum(
            (vector[axis] - means[axis]) ** 2 for vector in vectors
        ) / len(vectors)
        deviations.append(math.sqrt(variance) or 1.0)

    return [
        tuple(
            (vector[axis] - means[axis]) / deviations[axis]
            for axis in range(dimensions)
        )
        for vector in vectors
    ]


def load_dataset(sizes: list[int], seed: int):
    from vectorindex.features import load_arff_vectors

    path = DATA / 'dry_beans.arff'

    if not path.exists():
        raise BenchmarkError(f'No existe el dataset: {path}')

    rows = load_arff_vectors(str(path))
    labels = [label for label, _ in rows]
    vectors = standardize([vector for _, vector in rows])

    rng = random.Random(seed)
    order = list(range(len(vectors)))
    rng.shuffle(order)

    datasets = []

    for size in sizes:
        size = min(size, len(vectors))
        indices = order[:size]
        datasets.append((
            size,
            [vectors[index] for index in indices],
            [labels[index] for index in indices],
        ))

    return datasets


def write_vectors_csv(
    vectors: list[tuple[float, ...]],
    labels: list[str],
    path: Path,
) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        for identifier, (vector, label) in enumerate(zip(vectors, labels)):
            payload = ','.join(f'{value:.6f}' for value in vector)
            handle.write(f'{identifier};{label};[{payload}]\n')


def main() -> int:
    args = parse_args()
    quick = args.quick
    sizes = [500, 1500] if quick else [1000, 3000, 7000, 13687]
    queries = args.queries if args.queries is not None else (10 if quick else 20)
    iterations = args.iterations if args.iterations is not None else (3 if quick else 8)
    profile = 'quick' if quick else 'full'

    if args.sizes:
        sizes = [int(part) for part in args.sizes.split(',') if part.strip()]

    datasets = load_dataset(sizes, args.seed)
    output = prepare_output('experiment_8', profile)
    per_query: list[dict] = []
    summary: list[dict] = []
    manifest_runs: list[dict] = []

    for size, vectors, labels in datasets:
        work = tempfile.mkdtemp(prefix=f'bd2exp8_{size}_')
        database = Database(os.path.join(work, 'db'))
        lists = max(2, min(size, round(math.sqrt(size))))

        try:
            csv_path = Path(work) / 'vectors.csv'
            write_vectors_csv(vectors, labels, csv_path)

            database.execute(
                "CREATE TABLE granos (id INT PRIMARY KEY, clase CHAR(20), "
                f"v VECTOR(16)) USING HEAP FROM CSV ('{csv_path.as_posix()}', "
                "DELIMITER = ';');"
            )
            _, _, build_ms = database.execute(
                'CREATE INDEX idx_granos ON granos(v) USING IVF '
                f"WITH (lists = {lists}, metric = 'L2', "
                f'iterations = {iterations}, seed = {args.seed});'
            )

            rng = random.Random(args.seed + size)
            query_ids = rng.sample(range(size), min(queries, size))
            exact_results = {}

            for query_id in query_ids:
                truth = exact_knn(
                    ((index, vector) for index, vector in enumerate(vectors)),
                    vectors[query_id],
                    K,
                    Metric.L2,
                )
                exact_results[query_id] = [rid for rid, _ in truth]

                _, explain, latency = database.query(
                    'SELECT id, v <-> ' + sql_vector(vectors[query_id]) +
                    ' AS dist FROM granos FORCE SEQSCAN '
                    'ORDER BY dist ASC LIMIT 10;'
                )
                per_query.append({
                    'n': size, 'method': 'exact', 'probes': 0,
                    'query_id': query_id,
                    'recall@10': 1.0,
                    'latency_ms': round(latency, 3),
                    'reads': reads_of(explain.root)[0],
                    'plan': plan_name(explain.root),
                })

            active_probes = [value for value in PROBES if value <= lists] or [lists]

            for probes in active_probes:
                for query_id in query_ids:
                    output_, explain, latency = database.query(
                        f'SET ivf.probes = {probes};'
                        'SELECT id, v <-> ' + sql_vector(vectors[query_id]) +
                        ' AS dist FROM granos ORDER BY dist ASC LIMIT 10;'
                    )
                    retrieved = [row[0] for row in output_.rows]
                    per_query.append({
                        'n': size, 'method': 'ivf', 'probes': probes,
                        'query_id': query_id,
                        'recall@10': round(recall_at_k(
                            retrieved, exact_results[query_id], K,
                        ), 4),
                        'latency_ms': round(latency, 3),
                        'reads': reads_of(explain.root)[0],
                        'plan': plan_name(explain.root),
                    })

            exact_rows = [
                row for row in per_query
                if row['n'] == size and row['method'] == 'exact'
            ]
            summary.append({
                'n': size, 'lists': lists, 'method': 'exact', 'probes': 0,
                'recall@10': 1.0,
                'recall_std': 0.0,
                'latency_ms': round(mean([row['latency_ms'] for row in exact_rows]), 3),
                'latency_std': round(stdev([row['latency_ms'] for row in exact_rows]), 3),
                'reads': round(mean([row['reads'] for row in exact_rows]), 2),
            })

            for probes in active_probes:
                rows = [
                    row for row in per_query
                    if row['n'] == size and row['method'] == 'ivf'
                    and row['probes'] == probes
                ]
                summary.append({
                    'n': size, 'lists': lists, 'method': 'ivf', 'probes': probes,
                    'recall@10': round(mean([row['recall@10'] for row in rows]), 4),
                    'recall_std': round(stdev([row['recall@10'] for row in rows]), 4),
                    'latency_ms': round(mean([row['latency_ms'] for row in rows]), 3),
                    'latency_std': round(stdev([row['latency_ms'] for row in rows]), 3),
                    'reads': round(mean([row['reads'] for row in rows]), 2),
                })

            manifest_runs.append({
                'n': size, 'lists': lists, 'queries': len(query_ids),
                'build_ms': round(build_ms, 2),
                'probes': active_probes,
            })
            print(
                f'N={size:<6} lists={lists:<4} build={build_ms:8.1f} ms  '
                f'probes={active_probes}'
            )
        finally:
            database.close()
            shutil.rmtree(work, ignore_errors=True)

    fields = [
        'n', 'lists', 'method', 'probes', 'recall@10', 'recall_std',
        'latency_ms', 'latency_std', 'reads',
    ]
    write_csv(output / 'summary.csv', summary, fields)
    write_csv(
        output / 'queries.csv', per_query,
        ['n', 'method', 'probes', 'query_id', 'recall@10', 'latency_ms',
         'reads', 'plan'],
    )
    write_json(output / 'manifest.json', {
        'experiment': 8,
        'profile': profile,
        'dataset': 'data/datasets/dry_beans.arff',
        'parameters': {
            'k': K, 'probes': PROBES, 'sizes': sizes, 'queries': queries,
            'iterations': iterations, 'seed': args.seed,
        },
        'runs': manifest_runs,
        'summary': summary,
    })

    print()
    print(f'{"N":>6}{"lists":>7}{"probes":>8}{"R@10":>8}{"ms":>10}{"reads":>10}')
    for row in summary:
        print(
            f'{row["n"]:>6}{row["lists"]:>7}{row["probes"]:>8}'
            f'{row["recall@10"]:>8.3f}{row["latency_ms"]:>10.2f}'
            f'{row["reads"]:>10.1f}'
        )

    if not args.no_plots:
        make_plots(output, summary)

    print(f'\nResultados en {output}')
    return 0


def make_plots(output: Path, summary: list[dict]) -> None:
    plt = import_pyplot()
    sizes = sorted({row['n'] for row in summary})
    markers = ['o', 's', '^', 'D']

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    for index, size in enumerate(sizes):
        ivf = [
            row for row in summary if row['n'] == size and row['method'] == 'ivf'
        ]
        exact = [
            row for row in summary if row['n'] == size and row['method'] == 'exact'
        ]
        probes = [row['probes'] for row in ivf]
        marker = markers[index % len(markers)]

        axes[0][0].plot(probes, [row['recall@10'] for row in ivf],
                        marker=marker, label=f'N={size}')
        axes[0][1].plot(probes, [row['latency_ms'] for row in ivf],
                        marker=marker, label=f'N={size}')
        axes[1][0].plot(probes, [row['reads'] for row in ivf],
                        marker=marker, label=f'N={size}')
        axes[1][1].scatter(
            [row['latency_ms'] for row in ivf],
            [row['recall@10'] for row in ivf],
            marker=marker, label=f'N={size}',
        )

        if exact:
            axes[0][1].axhline(
                exact[0]['latency_ms'], linestyle='--', alpha=0.4,
            )
            axes[1][0].axhline(
                exact[0]['reads'], linestyle='--', alpha=0.4,
            )

    axes[0][0].set_title('Recall@10 vs probes')
    axes[0][0].set_xlabel('probes')
    axes[0][0].set_ylabel('Recall@10')
    axes[0][0].set_ylim(0, 1.05)

    axes[0][1].set_title('Latencia vs probes')
    axes[0][1].set_xlabel('probes')
    axes[0][1].set_ylabel('ms')

    axes[1][0].set_title('Lecturas de disco vs probes')
    axes[1][0].set_xlabel('probes')
    axes[1][0].set_ylabel('bloques leidos')

    axes[1][1].set_title('Compromiso recall-latencia')
    axes[1][1].set_xlabel('ms')
    axes[1][1].set_ylabel('Recall@10')
    axes[1][1].set_ylim(0, 1.05)

    for row in axes:
        for axis in row:
            axis.grid(alpha=0.3)
            axis.legend()

    figure.tight_layout()
    figure.savefig(output / 'exp8_ann.png', dpi=140)
    plt.close(figure)


if __name__ == '__main__':
    raise SystemExit(main())
