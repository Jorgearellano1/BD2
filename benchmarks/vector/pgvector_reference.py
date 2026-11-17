"""
Referencia opcional: IVF de pgvector contra el IVF del motor.

Requiere Docker en ejecucion. Levanta PostgreSQL con la extension pgvector,
carga el mismo conjunto de vectores y compara Recall@10 y latencia variando
pgvector.probes. El resultado se guarda junto al experimento 8.

Uso:
    python benchmarks/vector/pgvector_reference.py --start
    python benchmarks/vector/pgvector_reference.py --n 3000 --queries 20
    python benchmarks/vector/pgvector_reference.py --stop

Si Docker no esta disponible el script termina con un mensaje claro.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from common import DATA, ROOT, sql_vector, write_csv, write_json

from vectorindex.features import load_arff_vectors
from vectorindex.metrics import Metric
from vectorindex.scan import exact_knn

COMPOSE = ROOT / 'benchmarks' / 'vector' / 'docker-compose.yml'
PROBES = [1, 2, 5, 10, 20, 50]
K = 10


def ensure_docker() -> None:
    result = subprocess.run(
        ['docker', 'info'], text=True, capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            'Docker no esta disponible. Inicia Docker Desktop y vuelve a '
            'intentar; el resto de los benchmarks funciona sin Docker.'
        )


def docker(*args: str, check: bool = True, input_text: str | None = None,
           timeout: int = 120):
    result = subprocess.run(
        ['docker', 'compose', '-f', str(COMPOSE), *args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f'docker compose {" ".join(args)} fallo:\n{result.stderr.strip()}'
        )

    return result


def psql(sql: str | None = None, database: str = 'vectores',
         timeout: int = 120, stdin_text: str | None = None):
    args = [
        'exec', '-T', 'pgvector',
        'psql', '-U', 'postgres', '-d', database, '-v', 'ON_ERROR_STOP=1',
    ]

    if stdin_text is not None:
        args += ['-f', '-']

        return docker(*args, input_text=stdin_text, timeout=timeout)

    args += ['-t', '-A', '-c', sql]

    return docker(*args, timeout=timeout)


def wait_for_postgres(attempts: int = 40) -> None:
    for _ in range(attempts):
        result = docker('exec', '-T', 'pgvector', 'pg_isready', '-U', 'postgres', check=False)

        if result.returncode == 0:
            return

        time.sleep(2)

    raise RuntimeError('PostgreSQL no respondio a tiempo.')


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


def parse_execution_ms(text: str) -> float:
    match = re.search(r'Execution Time:\s*([0-9.]+)\s*ms', text)

    if not match:
        raise RuntimeError(f'No se encontro Execution Time en:\n{text}')

    return float(match.group(1))


def main() -> int:
    args = parse_args()
    ensure_docker()

    if args.start:
        docker('up', '-d')
        wait_for_postgres()
        print('pgvector listo en localhost:55432')

    if args.stop:
        docker('down', '-v')
        print('Contenedor detenido.')
        return 0

    rows = load_arff_vectors(str(DATA / 'dry_beans.arff'))
    vectors = standardize([vector for _, vector in rows])
    rng = random.Random(args.seed)
    order = list(range(len(vectors)))
    rng.shuffle(order)
    vectors = [vectors[index] for index in order[:args.n]]
    size = len(vectors)
    lists = max(2, round(math.sqrt(size)))

    work = tempfile.mkdtemp(prefix='bd2pgvector_')

    try:
        setup = Path(work) / 'setup.sql'
        with open(setup, 'w', encoding='utf-8') as handle:
            handle.write('CREATE EXTENSION IF NOT EXISTS vector;\n')
            handle.write('DROP TABLE IF EXISTS items;\n')
            handle.write(
                'CREATE TABLE items (id INT PRIMARY KEY, embedding vector(16));\n'
            )

            for start in range(0, size, 200):
                chunk = range(start, min(size, start + 200))
                values = ','.join(
                    f"({index},'{sql_vector(vectors[index])[1:-1]}')"
                    for index in chunk
                )
                handle.write(
                    f'INSERT INTO items (id, embedding) VALUES {values};\n'
                )

            handle.write(
                'CREATE INDEX idx_items_ivf ON items '
                f'USING ivfflat (embedding vector_l2_ops) WITH (lists = {lists});\n'
            )
            handle.write('ANALYZE items;\n')

        psql(stdin_text=setup.read_text(encoding='utf-8'), timeout=600)

        query_ids = rng.sample(range(size), min(args.queries, size))
        per_query = []

        for query_id in query_ids:
            truth = exact_knn(
                ((index, vector) for index, vector in enumerate(vectors)),
                vectors[query_id],
                K,
                Metric.L2,
            )
            truth_ids = [rid for rid, _ in truth]

            for probes in [value for value in PROBES if value <= lists]:
                explain = psql(
                    f'SET ivfflat.probes = {probes};'
                    'EXPLAIN (ANALYZE, TIMING OFF) '
                    "SELECT id FROM items ORDER BY embedding <-> " +
                    sql_vector(vectors[query_id]) + ' LIMIT 10;',
                )
                latency = parse_execution_ms(explain.stdout)

                result = psql(
                    f'SET ivfflat.probes = {probes};'
                    "SELECT id FROM items ORDER BY embedding <-> " +
                    sql_vector(vectors[query_id]) + ' LIMIT 10;',
                )
                retrieved = [int(line) for line in result.stdout.split() if line.strip()]
                recall = len(set(retrieved[:K]) & set(truth_ids)) / K

                per_query.append({
                    'engine': 'pgvector',
                    'n': size,
                    'lists': lists,
                    'probes': probes,
                    'query_id': query_id,
                    'recall@10': round(recall, 4),
                    'latency_ms': round(latency, 3),
                })

        summary = []

        for probes in sorted({row['probes'] for row in per_query}):
            values = [row for row in per_query if row['probes'] == probes]
            summary.append({
                'engine': 'pgvector',
                'n': size,
                'lists': lists,
                'probes': probes,
                'recall@10': round(statistics.mean(row['recall@10'] for row in values), 4),
                'latency_ms': round(statistics.mean(row['latency_ms'] for row in values), 3),
            })

        output = ROOT / 'benchmarks' / 'results' / 'experiment_8' / 'pgvector_reference'
        output.mkdir(parents=True, exist_ok=True)
        write_csv(
            output / 'summary.csv', summary,
            ['engine', 'n', 'lists', 'probes', 'recall@10', 'latency_ms'],
        )
        write_json(output / 'manifest.json', {
            'engine': 'pgvector',
            'dataset': 'data/datasets/dry_beans.arff',
            'parameters': {'n': size, 'lists': lists, 'probes': PROBES, 'k': K},
            'summary': summary,
        })
        print(f'Resultados en {output}')
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='levanta el contenedor')
    parser.add_argument('--stop', action='store_true', help='detiene el contenedor')
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--queries', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}')
        raise SystemExit(1)
