"""
Experimento 7: BoVW clasico contra embeddings densos.

Mide Precision@5 y Precision@10, latencia y lecturas de disco para las tres
rutas del motor:

    1. BoVW sobre el indice invertido (MATCH + COSINE_SCORE).
    2. Embeddings densos con k-NN lineal exacto (FORCE SEQSCAN).
    3. Embeddings densos con el indice IVF (ANN).

Uso:
    python benchmarks/vector/run_experiment7.py
    python benchmarks/vector/run_experiment7.py --quick
    python benchmarks/vector/run_experiment7.py --queries-per-class 3 --probes 5
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from common import (
    MULTIMEDIA, Database, BenchmarkError, check_dataset,
    import_pyplot, mean, plan_name, prepare_output, precision_at_k,
    reads_of, sql_string, sql_vector, stdev, strategy_of, write_csv,
    write_json,
)

from vectorindex import bovw, features
from vectorindex.vectors import format_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='dataset y modelo reducidos')
    parser.add_argument('--queries-per-class', type=int, default=None)
    parser.add_argument('--codebook-words', type=int, default=None)
    parser.add_argument('--codebook-sample', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--lists', type=int, default=None)
    parser.add_argument('--probes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def configure(args: argparse.Namespace) -> dict:
    quick = args.quick
    settings = {
        'queries_per_class': args.queries_per_class if args.queries_per_class is not None else (1 if quick else 3),
        'words': args.codebook_words if args.codebook_words is not None else (32 if quick else 64),
        'sample': args.codebook_sample if args.codebook_sample is not None else (1200 if quick else 3000),
        'iterations': args.iterations if args.iterations is not None else (3 if quick else 4),
        'lists': args.lists,
        'probes': args.probes if args.probes is not None else (4 if quick else 10),
        'seed': args.seed,
        'profile': 'quick' if quick else 'full',
    }
    return settings


def write_csv_data(rows: list[tuple[int, str, str]], path: Path) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        for identifier, category, payload in rows:
            handle.write(f'{identifier};{category};{payload}\n')


def build_codebook(
    descriptors: list[list],
    settings: dict,
) -> tuple[bovw.Codebook, float]:
    rng = random.Random(settings['seed'])
    flat = [descriptor for image in descriptors for descriptor in image]
    sample_size = min(settings['sample'], len(flat))
    sample = rng.sample(flat, sample_size)

    started = time.perf_counter()
    codebook = bovw.build_codebook(
        sample,
        words=min(settings['words'], sample_size),
        iterations=settings['iterations'],
        seed=settings['seed'],
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    return codebook, elapsed


def main() -> int:
    args = parse_args()
    settings = configure(args)
    check_dataset()

    manifest = __import__('json').loads(
        (MULTIMEDIA / 'manifest.json').read_text(encoding='utf-8')
    )
    descriptors = features.load_sift_descriptors(
        str(MULTIMEDIA / 'sift_descriptors.bin')
    )
    started = time.perf_counter()
    embeddings = features.load_embeddings(
        str(MULTIMEDIA / 'embeddings.bin')
    )
    embedding_ms = (time.perf_counter() - started) * 1000.0
    rows = features.load_embedding_manifest(
        str(MULTIMEDIA / 'embeddings_manifest.csv')
    )

    if not (len(descriptors) == len(embeddings) == len(rows)):
        raise BenchmarkError('El dataset multimedia esta inconsistente.')

    categories: dict[str, list[int]] = {}

    for row in rows:
        categories.setdefault(row.category, []).append(row.id)

    print(f'Imagenes: {len(rows)}  clases: {len(categories)}')
    print(
        f'Codebook: K={settings["words"]} sample={settings["sample"]} '
        f'iteraciones={settings["iterations"]}'
    )

    codebook, codebook_ms = build_codebook(descriptors, settings)

    started = time.perf_counter()
    histograms = [
        bovw.histogram(image, codebook) for image in descriptors
    ]
    histogram_ms = (time.perf_counter() - started) * 1000.0

    documents = [
        bovw.histogram_document(histogram) for histogram in histograms
    ]
    embedding_literals = [format_vector(vector, precision=7) for vector in embeddings]

    output = prepare_output('experiment_7', settings['profile'])
    work = tempfile.mkdtemp(prefix='bd2exp7_')
    database = Database(os.path.join(work, 'db'))

    try:
        bovw_csv = Path(work) / 'bovw.csv'
        vectors_csv = Path(work) / 'vectors.csv'
        write_csv_data(
            [
                (row.id, row.category, documents[row.id])
                for row in rows
            ],
            bovw_csv,
        )
        write_csv_data(
            [
                (row.id, row.category, embedding_literals[row.id])
                for row in rows
            ],
            vectors_csv,
        )

        print('Cargando tablas y construyendo indices...')

        _, _, ms_table_bovw = database.execute(
            "CREATE TABLE bovw (id INT PRIMARY KEY, categoria CHAR(20), "
            f"histograma CHAR(1000)) USING HEAP FROM CSV ('{bovw_csv.as_posix()}', "
            "DELIMITER = ';');"
        )
        _, _, ms_inverted = database.execute(
            'CREATE INDEX idx_bovw ON bovw(histograma) USING INVERTED;'
        )
        _, _, ms_table_vectors = database.execute(
            "CREATE TABLE vectores (id INT PRIMARY KEY, categoria CHAR(20), "
            f"embedding VECTOR(512)) USING HEAP FROM CSV ('{vectors_csv.as_posix()}', "
            "DELIMITER = ';');"
        )

        lists = settings['lists']
        if lists is None:
            lists = max(2, min(len(rows), round(math.sqrt(len(rows)))))
        lists = max(2, min(lists, len(rows)))

        _, _, ms_ivf = database.execute(
            'CREATE INDEX idx_vec ON vectores(embedding) USING IVF '
            f"WITH (lists = {lists}, metric = 'COSINE', "
            f"iterations = {settings['iterations']}, seed = {settings['seed']});"
        )
        database.execute(f'SET ivf.probes = {settings["probes"]};')

        print(
            f'Tablas: {ms_table_bovw:.1f} ms + {ms_table_vectors:.1f} ms  '
            f'INVERTED: {ms_inverted:.1f} ms  IVF: {ms_ivf:.1f} ms'
        )

        query_ids: list[int] = []

        for category in sorted(categories):
            query_ids.extend(categories[category][:settings['queries_per_class']])

        per_query: list[dict] = []

        for query_id in query_ids:
            row = rows[query_id]
            label = row.category

            # BoVW con indice invertido.
            result, explain, latency = database.query(
                'SELECT id, categoria, COSINE_SCORE() AS score FROM bovw '
                f"WHERE histograma MATCH {sql_string(documents[query_id])} "
                'ORDER BY score DESC LIMIT 10;'
            )
            per_query.append({
                'method': 'BoVW', 'query_id': query_id, 'category': label,
                'precision@5': precision_at_k([r[1] for r in result.rows], label, 5),
                'precision@10': precision_at_k([r[1] for r in result.rows], label, 10),
                'latency_ms': round(latency, 3),
                'reads': reads_of(explain.root)[0],
                'writes': reads_of(explain.root)[1],
                'plan': plan_name(explain.root),
                'strategy': strategy_of(explain.root),
            })

            # k-NN lineal exacto sobre embeddings.
            result, explain, latency = database.query(
                'SELECT id, categoria, embedding <=> ' + sql_vector(embeddings[query_id]) +
                ' AS dist FROM vectores FORCE SEQSCAN ORDER BY dist ASC LIMIT 10;'
            )
            per_query.append({
                'method': 'Dense exacto', 'query_id': query_id, 'category': label,
                'precision@5': precision_at_k([r[1] for r in result.rows], label, 5),
                'precision@10': precision_at_k([r[1] for r in result.rows], label, 10),
                'latency_ms': round(latency, 3),
                'reads': reads_of(explain.root)[0],
                'writes': reads_of(explain.root)[1],
                'plan': plan_name(explain.root),
                'strategy': strategy_of(explain.root),
            })

            # ANN con IVF.
            result, explain, latency = database.query(
                'SELECT id, categoria, embedding <=> ' + sql_vector(embeddings[query_id]) +
                ' AS dist FROM vectores ORDER BY dist ASC LIMIT 10;'
            )
            per_query.append({
                'method': 'Dense IVF', 'query_id': query_id, 'category': label,
                'precision@5': precision_at_k([r[1] for r in result.rows], label, 5),
                'precision@10': precision_at_k([r[1] for r in result.rows], label, 10),
                'latency_ms': round(latency, 3),
                'reads': reads_of(explain.root)[0],
                'writes': reads_of(explain.root)[1],
                'plan': plan_name(explain.root),
                'strategy': strategy_of(explain.root),
            })

        summary: list[dict] = []
        feature_ms = codebook_ms + histogram_ms

        for method in ('BoVW', 'Dense exacto', 'Dense IVF'):
            items = [row for row in per_query if row['method'] == method]
            summary.append({
                'method': method,
                'precision@5': round(mean([row['precision@5'] for row in items]), 4),
                'precision@10': round(mean([row['precision@10'] for row in items]), 4),
                'precision@5_std': round(stdev([row['precision@5'] for row in items]), 4),
                'precision@10_std': round(stdev([row['precision@10'] for row in items]), 4),
                'latency_ms': round(mean([row['latency_ms'] for row in items]), 3),
                'latency_ms_std': round(stdev([row['latency_ms'] for row in items]), 3),
                'reads': round(mean([row['reads'] for row in items]), 2),
                'writes': round(mean([row['writes'] for row in items]), 2),
                'feature_ms': round(
                    feature_ms if method == 'BoVW' else embedding_ms, 2
                ),
                'index_build_ms': round(
                    ms_inverted if method == 'BoVW'
                    else (ms_ivf if method == 'Dense IVF' else 0.0), 2
                ),
            })

        write_csv(
            output / 'summary.csv', summary,
            ['method', 'precision@5', 'precision@10', 'precision@5_std',
             'precision@10_std', 'latency_ms', 'latency_ms_std', 'reads',
             'writes', 'feature_ms', 'index_build_ms'],
        )
        write_csv(
            output / 'queries.csv', per_query,
            ['method', 'query_id', 'category', 'precision@5', 'precision@10',
             'latency_ms', 'reads', 'writes', 'plan', 'strategy'],
        )

        manifest_out = {
            'experiment': 7,
            'profile': settings['profile'],
            'dataset': {
                'images': len(rows),
                'classes': sorted(categories),
                'sift_descriptors': manifest['sift']['descriptors_total'],
                'embedding_dim': manifest['embeddings']['dim'],
            },
            'parameters': {
                **settings,
                'lists': lists,
            },
            'build_ms': {
                'codebook': round(codebook_ms, 2),
                'histograms': round(histogram_ms, 2),
                'embeddings_load': round(embedding_ms, 2),
                'inverted_index': round(ms_inverted, 2),
                'ivf_index': round(ms_ivf, 2),
                'tables': round(ms_table_bovw + ms_table_vectors, 2),
            },
            'summary': summary,
        }
        write_json(output / 'manifest.json', manifest_out)

        print()
        print(f'{"Metodo":<16}{"P@5":>8}{"P@10":>8}{"ms":>10}{"reads":>10}')
        for row in summary:
            print(
                f'{row["method"]:<16}{row["precision@5"]:>8.3f}'
                f'{row["precision@10"]:>8.3f}{row["latency_ms"]:>10.2f}'
                f'{row["reads"]:>10.1f}'
            )

        if not args.no_plots:
            make_plots(output, summary)

        print(f'\nResultados en {output}')
        return 0
    finally:
        database.close()
        shutil.rmtree(work, ignore_errors=True)


def make_plots(output: Path, summary: list[dict]) -> None:
    plt = import_pyplot()
    methods = [row['method'] for row in summary]
    colors = ['#2f6f8f', '#c58b2a', '#4c8a52']

    for metric, title, filename in (
        ('precision@10', 'Precision@10 por metodo', 'exp7_precision.png'),
        ('latency_ms', 'Latencia media por consulta (ms)', 'exp7_latencia.png'),
        ('reads', 'Bloques leidos por consulta', 'exp7_lecturas.png'),
    ):
        values = [row[metric] for row in summary]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(methods, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(metric)
        axis.grid(axis='y', alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=140)
        plt.close(figure)


if __name__ == '__main__':
    raise SystemExit(main())
