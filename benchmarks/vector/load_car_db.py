"""
Carga el dataset de vehiculos en el motor y crea sus indices.

Lee data/datasets/cars/ (products.csv, bovw.csv, embeddings.bin), genera un
CSV temporal con los vectores y crea la tabla `vehiculos` en el STORAGE_DIR
indicado, junto con los indices B+, R-Tree, invertidos e IVF.

Uso:
    python benchmarks/vector/load_car_db.py
    python benchmarks/vector/load_car_db.py --storage storage_demo
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from engine.catalog import Catalog  # noqa: E402
from engine.session import Session  # noqa: E402
from parser.parser import Parser  # noqa: E402
from parser.scanner import Scanner  # noqa: E402
from vectorindex.vectors import format_vector  # noqa: E402

DATA = ROOT / 'data' / 'datasets' / 'cars'
TABLE = 'vehiculos'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=str, default=str(ROOT / 'storage'))
    parser.add_argument('--lists', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=4)
    return parser.parse_args()


def read_embeddings(path: Path) -> list[tuple[float, ...]]:
    with open(path, 'rb') as handle:
        raw = handle.read()

    magic, count, dimension = struct.unpack_from('<8sII', raw, 0)

    if magic != b'BD2CAR01':
        raise SystemExit('embeddings.bin no tiene el formato esperado')

    vectors = []

    for index in range(count):
        start = 16 + index * dimension * 4
        vectors.append(struct.unpack_from(f'<{dimension}f', raw, start))

    return vectors


def read_bovw(path: Path) -> dict[int, str]:
    documents: dict[int, str] = {}

    with open(path, encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split(';', 2)

            if len(parts) == 3:
                documents[int(parts[0])] = parts[2]

    return documents


def run(session: Session, sql: str, label: str) -> float:
    started = time.perf_counter()
    output = session.execute(Parser(Scanner(sql)).parse_program())

    if not output.success:
        raise SystemExit(f'{label}: {output.error}')

    elapsed = (time.perf_counter() - started) * 1000.0
    print(f'  {label}: {elapsed:,.0f} ms')
    return elapsed


def main() -> int:
    args = parse_args()

    required = [
        DATA / 'products.csv', DATA / 'bovw.csv', DATA / 'embeddings.bin',
    ]

    for path in required:
        if not path.exists():
            raise SystemExit(
                f'Falta {path}. Genera el dataset con build_car_dataset.py.'
            )

    storage = Path(args.storage).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(str(storage / 'catalog.json'), str(storage))

    if catalog.table_exists(TABLE):
        raise SystemExit(
            f'La tabla {TABLE} ya existe en {storage}. Usa otro --storage.'
        )

    embeddings = read_embeddings(DATA / 'embeddings.bin')
    documents = read_bovw(DATA / 'bovw.csv')
    work = tempfile.mkdtemp(prefix='bd2cars_')
    csv_path = Path(work) / 'vehiculos.csv'

    try:
        with open(DATA / 'products.csv', encoding='utf-8', newline='') as source, \
                open(csv_path, 'w', encoding='utf-8', newline='') as target:
            reader = csv.reader(source, delimiter=';')
            header = next(reader)
            writer = csv.writer(target, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header + ['palabras', 'embedding'])

            for index, row in enumerate(reader):
                identifier = int(row[0])
                row[1] = row[1][:100]
                row[2] = row[2][:600]
                row[4] = row[4][:30]
                row[5] = row[5][:60]
                row[6] = row[6][:60]
                row[7] = row[7][:10]
                row[10] = row[10][:20]
                writer.writerow(
                    row
                    + [documents.get(identifier, '')[:400]]
                    + [format_vector(embeddings[identifier], precision=6)]
                )

                if (index + 1) % 2000 == 0:
                    print(f'  vectores preparados: {index + 1:,}')

        session = Session(catalog, 4096)
        print('Creando tabla...')
        run(
            session,
            f"CREATE TABLE {TABLE} ("
            'id INT PRIMARY KEY, titulo CHAR(100), descripcion CHAR(600), '
            'precio FLOAT, categoria CHAR(30), tienda CHAR(60), '
            'ciudad CHAR(60), estado CHAR(10), latitud FLOAT, longitud FLOAT, '
            'imagen CHAR(20), palabras CHAR(400), embedding VECTOR(512)'
            f") USING HEAP FROM CSV ('{csv_path.as_posix()}', DELIMITER = ';');",
            'tabla',
        )

        run(
            session,
            f'CREATE INDEX idx_veh_precio ON {TABLE}(precio) USING BTREE;',
            'B+ precio',
        )
        run(
            session,
            f'CREATE INDEX idx_veh_geo ON {TABLE}(latitud, longitud) USING RTREE;',
            'R-Tree ubicacion',
        )
        run(
            session,
            f"CREATE INDEX idx_veh_desc ON {TABLE}(descripcion) USING INVERTED "
            "WITH (language = 'english', stemming = 'true', stoplist = 'true');",
            'invertido descripcion',
        )
        run(
            session,
            f"CREATE INDEX idx_veh_bovw ON {TABLE}(palabras) USING INVERTED "
            "WITH (language = 'english', stemming = 'false', stoplist = 'false');",
            'invertido BoVW',
        )
        run(
            session,
            f'CREATE INDEX idx_veh_vec ON {TABLE}(embedding) USING IVF '
            f"WITH (lists = {args.lists}, metric = 'COSINE', "
            f'iterations = {args.iterations}, seed = 42);',
            'IVF embeddings',
        )

        print(f'\nTabla {TABLE}: {len(embeddings):,} vehiculos en {storage}')
        print('Indices: precio, ubicacion, descripcion, BoVW y embeddings')
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
