"""
Benchmark unico y reproducible de los experimentos 1-4.

Usa exclusivamente las implementaciones reales del proyecto:
HeapFile, SequentialFile, BPlusTree y ExtendibleHash.

Ejecutar desde la raiz del repositorio:

    python -u benchmarks/benchmark.py

Genera CSV, JSON y PNG en benchmarks/results/experiment_1..4/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / 'backend'
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'Falta matplotlib. Instala las dependencias con: '
        'pip install -r requirements.txt'
    ) from exc

from engine.conditions import (  # noqa: E402
    ColumnRef,
    ComparisonOperator,
    Condition,
    LiteralValue,
)
from engine.metadata import TableMetadata  # noqa: E402
from indexes.bplusTree import BPlusTree  # noqa: E402
from indexes.ExtendibleHash import ExtendibleHash  # noqa: E402
from storage.heapFile import HeapFile  # noqa: E402
from storage.sequentialFile import SequentialFile  # noqa: E402


SEED = 20240920
DATASET = ROOT / 'data' / 'datasets' / 'students_500k.csv'
RESULTS_DIR = ROOT / 'benchmarks' / 'results'
DATA_DIR = ROOT / 'benchmarks' / 'data_runs'
SIZES = [1000, 10000, 50000, 100000, 250000, 500000]
PAGE_SIZES = [1024, 2048, 4096, 8192]
DEFAULT_PAGE_SIZE = 4096
EQ_QUERIES = 1000
SELECTIVITIES = [0.001, 0.01, 0.05, 0.10, 0.25]
DELIMITER = ','
KEY_COLUMN = 'Student_ID'
TOTAL_ROWS = 500000

STRUCTURES = ['Heap', 'Sequential', 'Sequential (reorg)', 'B+ Tree', 'Hash']
COLORS = {
    'Heap': '#1f77b4',
    'Sequential': '#ff7f0e',
    'Sequential (reorg)': '#2ca02c',
    'B+ Tree': '#d62728',
    'Hash': '#9467bd',
}

COLUMNS = [
    {'name': 'Student_ID', 'type': 'INT', 'size': 4},
    {'name': 'Age', 'type': 'INT', 'size': 4},
    {'name': 'Gender', 'type': 'CHAR', 'size': 10},
    {'name': 'Region_Type', 'type': 'CHAR', 'size': 5},
    {'name': 'Major_Subject', 'type': 'CHAR', 'size': 23},
    {'name': 'University_Name', 'type': 'CHAR', 'size': 23},
    {'name': 'Home_City', 'type': 'CHAR', 'size': 16},
    {'name': 'Parent_Education', 'type': 'CHAR', 'size': 12},
    {'name': 'Previous_GPA', 'type': 'FLOAT', 'size': 4},
]


class BenchmarkError(RuntimeError):
    """Fallo de validacion del benchmark: no se generan resultados falsos."""


def require(condition, experiment, structure, operation, metric, cause):
    if not condition:
        raise BenchmarkError(
            'VALIDACION FALLIDA | '
            f'experimento={experiment} | estructura={structure} | '
            f'operacion={operation} | metrica={metric} | '
            f'causa probable={cause}'
        )


def print_header(title, subtitle):
    print()
    print('=' * 40)
    print(title)
    print(subtitle)
    print('=' * 40)


def print_table(title, headers, rows):
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    line = '  '.join(str(header).ljust(widths[index])
                     for index, header in enumerate(headers))
    print(title)
    print(line)
    print('-' * len(line))
    for row in rows:
        print('  '.join(str(cell).ljust(widths[index])
                        for index, cell in enumerate(row)))
    print()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, headers, rows):
    ensure_dir(path.parent)
    with open(path, 'w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows(rows)


def write_json(path: Path, payload):
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as target:
        json.dump(payload, target, indent=2, ensure_ascii=False)


def timed(function):
    start = time.perf_counter()
    value = function()
    return (time.perf_counter() - start) * 1000.0, value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_metadata() -> TableMetadata:
    return TableMetadata(
        'students_bench',
        [dict(column) for column in COLUMNS],
        primary_key=KEY_COLUMN,
    )


def count_dataset_rows() -> int:
    with open(DATASET, newline='', encoding='utf-8-sig') as source:
        reader = csv.reader(source)
        next(reader, None)
        return sum(1 for row in reader if row)


def write_prefix_csv(rows: int, path: Path) -> Path:
    ensure_dir(path.parent)
    with open(DATASET, newline='', encoding='utf-8-sig') as source, \
            open(path, 'w', newline='', encoding='utf-8') as target:
        reader = csv.reader(source)
        writer = csv.writer(target)
        header = next(reader)
        writer.writerow(header)
        written = 0
        for row in reader:
            if written >= rows:
                break
            if row:
                writer.writerow(row)
                written += 1

    require(
        written == rows,
        '1', '-', 'dataset', 'filas',
        f'el dataset no contiene {rows} filas de datos',
    )
    return path


def equals_condition(value) -> Condition:
    return Condition(
        ColumnRef(KEY_COLUMN),
        ComparisonOperator.EQ,
        LiteralValue(value),
    )


def make_ranges():
    ranges = []
    for selectivity in SELECTIVITIES:
        size = max(1, round(TOTAL_ROWS * selectivity))
        low = (TOTAL_ROWS - size) // 2 + 1
        high = low + size - 1
        ranges.append((selectivity, low, high))
    return ranges


# Experimentos
def experiment_1(meta: TableMetadata):
    print_header('EXPERIMENTO 1', 'COSTO DE INSERCION MASIVA')

    out_dir = RESULTS_DIR / 'experiment_1'
    ensure_dir(out_dir)

    time_rows = {name: [] for name in STRUCTURES}
    write_rows = {name: [] for name in STRUCTURES}
    detail_rows = []
    retained = None

    for size in SIZES:
        print(f'Registros: {size}')
        run_dir = DATA_DIR / f'exp1_{size}'
        if run_dir.exists():
            shutil.rmtree(run_dir)
        ensure_dir(run_dir)

        prefix_csv = write_prefix_csv(size, run_dir / 'prefix.csv')

        # Heap
        heap = HeapFile(meta, str(run_dir / 'students.heap'), DEFAULT_PAGE_SIZE)
        heap_ms, _ = timed(lambda: heap.rebulk_load(str(prefix_csv), DELIMITER))
        heap_writes = heap.page_writes
        require(heap.count() == size, '1', 'Heap', 'rebulk_load', 'registros',
                'el Heap no quedo con el tamano esperado')

        # Sequential sin reorganizacion
        seq_plain = SequentialFile(
            meta, str(run_dir / 'plain.seq'), DEFAULT_PAGE_SIZE,
            auto_reorganize=False,
        )
        seq_plain_ms, _ = timed(
            lambda: seq_plain.rebulk_load(str(prefix_csv), DELIMITER)
        )
        seq_plain_writes = seq_plain.page_writes
        require(seq_plain.count() == size, '1', 'Sequential', 'rebulk_load',
                'registros', 'el Sequential no quedo con el tamano esperado')

        # Sequential con reorganizacion automatica habilitada
        seq_reorg = SequentialFile(
            meta, str(run_dir / 'reorg.seq'), DEFAULT_PAGE_SIZE,
            auto_reorganize=True,
        )
        seq_reorg_ms, _ = timed(
            lambda: seq_reorg.rebulk_load(str(prefix_csv), DELIMITER)
        )
        seq_reorg_writes = seq_reorg.page_writes
        require(seq_reorg.count() == size, '1', 'Sequential (reorg)',
                'rebulk_load', 'registros',
                'el Sequential con reorg no quedo con el tamano esperado')

        # B+ Tree sobre el mismo Heap ya cargado
        heap_writes_after_load = heap.page_writes
        bplus_ms, tree = timed(
            lambda: BPlusTree.build_from_storage(
                meta, heap, KEY_COLUMN, DEFAULT_PAGE_SIZE,
                str(run_dir / 'students_bplus.dat'),
            )
        )
        bplus_stats = tree.stats()
        tree_writes_after_build = tree.page_writes
        tree.reset_stats()
        heap_writes_during_build = heap.page_writes - heap_writes_after_load
        index_writes = max(0, tree_writes_after_build - heap_writes_during_build)
        require(bplus_stats['total_entries'] == size, '1', 'B+ Tree',
                'build_from_storage', 'entradas',
                'el B+ Tree no quedo con el tamano esperado')

        # Hash sobre el mismo Heap ya cargado
        hash_ms, hash_index = timed(
            lambda: ExtendibleHash.build_from_storage(
                meta, heap, KEY_COLUMN, DEFAULT_PAGE_SIZE,
                str(run_dir / 'students_hash.hash'),
            )
        )
        require(hash_index.count() == size, '1', 'Hash', 'build_from_storage',
                'entradas', 'el Hash no quedo con el tamano esperado')

        heap_total_ms = heap_ms
        bplus_total_ms = heap_ms + bplus_ms
        hash_total_ms = heap_ms + hash_ms
        bplus_total_writes = heap_writes_after_load + index_writes
        hash_total_writes = heap_writes_after_load + hash_index.page_writes

        time_rows['Heap'].append(round(heap_total_ms, 4))
        time_rows['Sequential'].append(round(seq_plain_ms, 4))
        time_rows['Sequential (reorg)'].append(round(seq_reorg_ms, 4))
        time_rows['B+ Tree'].append(round(bplus_total_ms, 4))
        time_rows['Hash'].append(round(hash_total_ms, 4))

        write_rows['Heap'].append(heap_writes)
        write_rows['Sequential'].append(seq_plain_writes)
        write_rows['Sequential (reorg)'].append(seq_reorg_writes)
        write_rows['B+ Tree'].append(bplus_total_writes)
        write_rows['Hash'].append(hash_total_writes)

        detail_rows.extend([
            [size, 'Heap', f'{heap_total_ms:.4f}', heap_writes, 0],
            [size, 'Sequential', f'{seq_plain_ms:.4f}', seq_plain_writes,
             seq_plain.reorganizations],
            [size, 'Sequential (reorg)', f'{seq_reorg_ms:.4f}',
             seq_reorg_writes, seq_reorg.reorganizations],
            [size, 'B+ Tree', f'{bplus_total_ms:.4f}', bplus_total_writes,
             bplus_stats['total_nodes']],
            [size, 'Hash', f'{hash_total_ms:.4f}', hash_total_writes,
             hash_index.count()],
        ])

        print(f"  Heap {heap_total_ms:8.2f} ms / {heap_writes:6d} escrituras")
        print(f"  Sequential {seq_plain_ms:8.2f} ms / {seq_plain_writes:6d} escrituras")
        print(f"  Sequential (reorg) {seq_reorg_ms:8.2f} ms / "
              f"{seq_reorg_writes:6d} escrituras")
        print(f"  B+ Tree {bplus_total_ms:8.2f} ms / "
              f"{bplus_total_writes:6d} escrituras")
        print(f"  Hash {hash_total_ms:8.2f} ms / {hash_total_writes:6d} escrituras")

        if size == TOTAL_ROWS:
            require(heap.count() == TOTAL_ROWS, '1', 'Heap', 'validacion final',
                    'registros', 'se esperaban 500000 registros')
            seq_reorg.close()
            (run_dir / 'reorg.seq').unlink(missing_ok=True)
            retained = {
                'heap': heap,
                'seq': seq_plain,
                'tree': tree,
                'hash': hash_index,
                'dir': run_dir,
            }
        else:
            heap.close()
            seq_plain.close()
            seq_reorg.close()
            tree.close()
            hash_index.close()
            prefix_csv.unlink(missing_ok=True)
            shutil.rmtree(run_dir)

    headers = ['Registros'] + STRUCTURES
    write_csv(
        out_dir / 'insertion_time.csv', headers,
        [[size] + [time_rows[name][index] for name in STRUCTURES]
         for index, size in enumerate(SIZES)],
    )
    write_csv(
        out_dir / 'insertion_writes.csv', headers,
        [[size] + [write_rows[name][index] for name in STRUCTURES]
         for index, size in enumerate(SIZES)],
    )
    write_csv(
        out_dir / 'insertion_detail.csv',
        ['records', 'structure', 'milliseconds', 'writes', 'extra'],
        detail_rows,
    )
    write_json(out_dir / 'results.json', {
        'experiment': 1,
        'name': 'insercion masiva',
        'records': SIZES,
        'insertion_ms': time_rows,
        'writes': write_rows,
    })

    print_table(
        'Tabla 4 - Tiempo de insercion masiva (ms)', headers,
        [[size] + [f'{time_rows[name][index]:.2f}' for name in STRUCTURES]
         for index, size in enumerate(SIZES)],
    )
    print_table(
        'Tabla 5 - Escrituras de disco durante la insercion', headers,
        [[size] + [write_rows[name][index] for name in STRUCTURES]
         for index, size in enumerate(SIZES)],
    )

    plot_experiment_1(time_rows, write_rows, out_dir)
    return retained, {'time': time_rows, 'writes': write_rows}


def experiment_2(meta: TableMetadata, retained, keys):
    print_header('EXPERIMENTO 2', 'BUSQUEDAS PUNTUALES DE IGUALDAD (1000)')

    out_dir = RESULTS_DIR / 'experiment_2'
    ensure_dir(out_dir)

    heap = retained['heap']
    seq = retained['seq']
    tree = retained['tree']
    hash_index = retained['hash']

    per_query = []
    summary = {}

    def validate_record(record, key, name):
        require(record is not None, '2', name, 'busqueda', 'resultado',
                f'no se encontro la clave {key}')
        require(record[0] == key, '2', name, 'busqueda', 'clave',
                f'se esperaba {key} y se obtuvo {record[0]}')

    for name in ['Heap', 'Sequential', 'B+ Tree', 'Hash']:
        times = []
        reads = []
        for index, key in enumerate(keys):
            condition = equals_condition(key)

            if name == 'Heap':
                heap.reset_stats()
                milliseconds, rows = timed(lambda: heap.search_all(condition))
                read_count = heap.page_reads
                require(len(rows) == 1, '2', name, 'busqueda', 'filas',
                        f'la clave {key} devolvio {len(rows)} filas')
                validate_record(rows[0], key, name)
            elif name == 'Sequential':
                seq.reset_stats()
                milliseconds, rows = timed(lambda: seq.search_all(condition))
                read_count = seq.page_reads
                require(len(rows) == 1, '2', name, 'busqueda', 'filas',
                        f'la clave {key} devolvio {len(rows)} filas')
                validate_record(rows[0], key, name)
            elif name == 'B+ Tree':
                tree.reset_stats()
                milliseconds, record = timed(lambda: tree.search_record(key))
                read_count = tree.page_reads
                validate_record(record, key, name)
            else:
                hash_index.reset_stats()
                heap.reset_stats()

                def query():
                    rid = hash_index.search(key)
                    if rid is None:
                        return None
                    return meta.clean_tuple(heap.read(rid))

                milliseconds, record = timed(query)
                read_count = hash_index.page_reads + heap.page_reads
                validate_record(record, key, name)

            times.append(milliseconds)
            reads.append(read_count)
            per_query.append([name, index, key, f'{milliseconds:.6f}',
                              read_count])

            if (index + 1) % 200 == 0:
                print(f'  {name}: {index + 1}/{len(keys)} consultas')

        summary[name] = {
            'mean_reads': statistics.fmean(reads),
            'std_reads': statistics.pstdev(reads),
            'mean_ms': statistics.fmean(times),
            'std_ms': statistics.pstdev(times),
        }

    write_csv(
        out_dir / 'equality_per_query.csv',
        ['structure', 'query_index', 'key', 'milliseconds', 'reads'],
        per_query,
    )
    write_csv(
        out_dir / 'equality_summary.csv',
        ['structure', 'mean_reads', 'std_reads', 'mean_ms', 'std_ms'],
        [[
            name,
            f"{summary[name]['mean_reads']:.4f}",
            f"{summary[name]['std_reads']:.4f}",
            f"{summary[name]['mean_ms']:.6f}",
            f"{summary[name]['std_ms']:.6f}",
        ] for name in summary],
    )
    write_json(out_dir / 'results.json', {
        'experiment': 2,
        'name': 'busquedas puntuales de igualdad',
        'queries': EQ_QUERIES,
        'keys_sha256': hashlib.sha256(
            ','.join(str(key) for key in keys).encode()
        ).hexdigest(),
        'summary': summary,
    })

    print_table(
        'Tabla 6 - Busquedas de igualdad',
        ['Estructura', 'Lecturas Media', 'Lecturas Desv.',
         'Tiempo Media (ms)', 'Tiempo Desv.'],
        [[
            name,
            f"{summary[name]['mean_reads']:.2f}",
            f"{summary[name]['std_reads']:.2f}",
            f"{summary[name]['mean_ms']:.4f}",
            f"{summary[name]['std_ms']:.4f}",
        ] for name in summary],
    )

    plot_experiment_2(summary, out_dir)
    return {'summary': summary}


def experiment_3(meta: TableMetadata, retained):
    print_header('EXPERIMENTO 3', 'BUSQUEDAS POR RANGO')

    out_dir = RESULTS_DIR / 'experiment_3'
    ensure_dir(out_dir)

    heap = retained['heap']
    seq = retained['seq']
    tree = retained['tree']
    ranges = make_ranges()

    rows_out = []
    per_query = []

    for selectivity, low, high in ranges:
        results = {}
        for name in ['Heap', 'Sequential', 'B+ Tree']:
            if name == 'Heap':
                heap.reset_stats()
                milliseconds, values = timed(
                    lambda: heap.range_search(KEY_COLUMN, low, high)
                )
                read_count = heap.page_reads
            elif name == 'Sequential':
                seq.reset_stats()
                milliseconds, values = timed(
                    lambda: seq.range_search_raw(KEY_COLUMN, low, high)
                )
                read_count = seq.page_reads
            else:
                tree.reset_stats()
                milliseconds, values = timed(
                    lambda: tree.range_search_records(low, high)
                )
                read_count = tree.page_reads

            require(values, '3', name, 'range_search', 'filas',
                    f'el rango [{low}, {high}] no devolvio filas')

            key_values = [int(value[0]) for value in values]
            require(
                min(key_values) >= low and max(key_values) <= high,
                '3', name, 'range_search', 'limites',
                f'hay claves fuera de [{low}, {high}]',
            )
            results[name] = {
                'rows': len(values),
                'keys': sorted(key_values),
                'ms': milliseconds,
                'reads': read_count,
            }
            per_query.append([
                f'{selectivity}', low, high, name, len(values),
                read_count, f'{milliseconds:.6f}',
            ])

        require(
            len({results[name]['rows'] for name in results}) == 1,
            '3', '-', 'range_search', 'filas',
            'las estructuras devolvieron cantidades distintas',
        )
        require(
            results['Heap']['keys'] == results['Sequential']['keys']
            == results['B+ Tree']['keys'],
            '3', '-', 'range_search', 'contenido',
            'las estructuras devolvieron filas distintas',
        )

        rows_out.append({
            'selectivity': selectivity,
            'rows': results['Heap']['rows'],
            'Heap': results['Heap']['reads'],
            'Sequential': results['Sequential']['reads'],
            'B+ Tree': results['B+ Tree']['reads'],
            'Heap_ms': results['Heap']['ms'],
            'Sequential_ms': results['Sequential']['ms'],
            'B+ Tree_ms': results['B+ Tree']['ms'],
        })
        print(
            f"  Selectividad {selectivity * 100:>5.1f}% [{low}, {high}] "
            f"filas={results['Heap']['rows']} "
            f"(Heap {results['Heap']['reads']} lecturas, "
            f"Seq {results['Sequential']['reads']}, "
            f"B+ {results['B+ Tree']['reads']})"
        )

    write_csv(
        out_dir / 'range_per_query.csv',
        ['selectivity', 'low', 'high', 'structure', 'rows', 'reads',
         'milliseconds'],
        per_query,
    )
    write_csv(
        out_dir / 'range_reads.csv',
        ['selectivity', 'rows', 'Heap', 'Sequential', 'B+ Tree'],
        [[f"{row['selectivity']}", row['rows'], row['Heap'],
          row['Sequential'], row['B+ Tree']] for row in rows_out],
    )
    write_csv(
        out_dir / 'range_time.csv',
        ['selectivity', 'Heap', 'Sequential', 'B+ Tree'],
        [[f"{row['selectivity']}", f"{row['Heap_ms']:.6f}",
          f"{row['Sequential_ms']:.6f}", f"{row['B+ Tree_ms']:.6f}"]
         for row in rows_out],
    )
    write_json(out_dir / 'results.json', {
        'experiment': 3,
        'name': 'busquedas por rango',
        'ranges': [
            {'selectivity': selectivity, 'low': low, 'high': high}
            for selectivity, low, high in ranges
        ],
        'results': rows_out,
    })

    print_table(
        'Tabla 7 - Lecturas por consulta de rango',
        ['Selectividad', 'Filas', 'Heap', 'Sequential', 'B+ Tree'],
        [[f"{row['selectivity'] * 100:.1f}%", row['rows'], row['Heap'],
          row['Sequential'], row['B+ Tree']] for row in rows_out],
    )
    print_table(
        'Tabla 8 - Tiempo por consulta de rango (ms)',
        ['Selectividad', 'Heap', 'Sequential', 'B+ Tree'],
        [[f"{row['selectivity'] * 100:.1f}%", f"{row['Heap_ms']:.3f}",
          f"{row['Sequential_ms']:.3f}", f"{row['B+ Tree_ms']:.3f}"]
         for row in rows_out],
    )

    plot_experiment_3(rows_out, out_dir)
    return {'results': rows_out}


def experiment_4(meta: TableMetadata, keys):
    print_header('EXPERIMENTO 4', 'SENSIBILIDAD AL TAMANO DE BLOQUE')

    out_dir = RESULTS_DIR / 'experiment_4'
    ensure_dir(out_dir)

    ranges = make_ranges()
    rows_out = []
    per_query = []

    for page_size in PAGE_SIZES:
        print(f'Pagina: {page_size} bytes')
        run_dir = DATA_DIR / f'exp4_{page_size}'
        if run_dir.exists():
            shutil.rmtree(run_dir)
        ensure_dir(run_dir)

        heap = HeapFile(meta, str(run_dir / 'students.heap'), page_size)
        rebuild_ms, _ = timed(lambda: heap.rebulk_load(str(DATASET), DELIMITER))
        require(heap.count() == TOTAL_ROWS, '4', 'Heap', 'rebulk_load',
                'registros', 'el Heap no tiene 500000 registros')

        build_ms, tree = timed(
            lambda: BPlusTree.build_from_storage(
                meta, heap, KEY_COLUMN, page_size,
                str(run_dir / 'students_bplus.dat'),
            )
        )
        stats = tree.stats()
        require(stats['total_entries'] == TOTAL_ROWS, '4', 'B+ Tree',
                'build_from_storage', 'entradas',
                'el B+ Tree no tiene 500000 entradas')
        require(stats['page_size'] == page_size, '4', 'B+ Tree', 'stats',
                'page_size', 'el arbol no se construyo con el page_size pedido')

        rebuild_reads = heap.page_reads + tree.page_reads
        rebuild_writes = heap.page_writes + tree.page_writes
        tree.reset_stats()

        batch_ms = 0.0
        for index, key in enumerate(keys):
            before_reads = tree.page_reads
            milliseconds, record = timed(lambda: tree.search_record(key))
            batch_ms += milliseconds
            require(record is not None and record[0] == key, '4', 'B+ Tree',
                    'search_record', 'resultado',
                    f'clave {key} incorrecta o ausente')
            per_query.append([page_size, 'equality', index, key,
                              f'{milliseconds:.6f}',
                              tree.page_reads - before_reads])
            if (index + 1) % 200 == 0:
                print(f'  {page_size} bytes: {index + 1}/{len(keys)} igualdades')

        for selectivity, low, high in ranges:
            before_reads = tree.page_reads
            milliseconds, values = timed(
                lambda: tree.range_search_records(low, high)
            )
            batch_ms += milliseconds
            require(values, '4', 'B+ Tree', 'range_search_records', 'filas',
                    f'el rango [{low}, {high}] no devolvio filas')
            per_query.append([page_size, 'range', low, high,
                              f'{milliseconds:.6f}',
                              tree.page_reads - before_reads])

        batch_reads = tree.page_reads
        batch_writes = tree.page_writes

        rows_out.append({
            'page_size': page_size,
            'fanout': stats['fanout'],
            'height': stats['height'],
            'reads': batch_reads,
            'writes': batch_writes,
            'ms': batch_ms,
            'rebuild_ms': rebuild_ms + build_ms,
            'rebuild_reads': rebuild_reads,
            'rebuild_writes': rebuild_writes,
        })
        print(
            f"  fanout={stats['fanout']} altura={stats['height']} "
            f"lecturas={batch_reads} escrituras={batch_writes} "
            f"tiempo={batch_ms:.2f} ms"
        )

        tree.close()
        heap.close()
        shutil.rmtree(run_dir)

    write_csv(
        out_dir / 'page_size.csv',
        ['page_size', 'fanout', 'height', 'reads', 'writes', 'ms',
         'rebuild_ms', 'rebuild_reads', 'rebuild_writes'],
        [[
            row['page_size'], row['fanout'], row['height'], row['reads'],
            row['writes'], f"{row['ms']:.4f}", f"{row['rebuild_ms']:.4f}",
            row['rebuild_reads'], row['rebuild_writes'],
        ] for row in rows_out],
    )
    write_csv(
        out_dir / 'per_query.csv',
        ['page_size', 'kind', 'query_index', 'value', 'milliseconds', 'reads'],
        per_query,
    )
    write_json(out_dir / 'results.json', {
        'experiment': 4,
        'name': 'sensibilidad al tamano de bloque',
        'page_sizes': PAGE_SIZES,
        'query_batch': '1000 igualdades + 5 rangos',
        'results': rows_out,
    })

    print_table(
        'Tabla 9 - Sensibilidad al tamano de bloque',
        ['Pagina (bytes)', 'Fan-out', 'Altura', 'Lecturas', 'Escrituras',
         'Tiempo (ms)'],
        [[row['page_size'], row['fanout'], row['height'], row['reads'],
          row['writes'], f"{row['ms']:.2f}"] for row in rows_out],
    )

    plot_experiment_4(rows_out, out_dir)
    return {'results': rows_out}


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------


def save_figure(fig, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_experiment_1(time_rows, write_rows, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in STRUCTURES:
        ax.plot(SIZES, time_rows[name], marker='o', color=COLORS[name],
                label=name)
    ax.set_xscale('log')
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(size) for size in SIZES])
    ax.set_title('Experimento 1: tiempo de insercion masiva')
    ax.set_xlabel('Registros')
    ax.set_ylabel('Tiempo (ms)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / 'plot_time_vs_records.png')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in STRUCTURES:
        ax.plot(SIZES, write_rows[name], marker='o', color=COLORS[name],
                label=name)
    ax.set_xscale('log')
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(size) for size in SIZES])
    ax.set_title('Experimento 1: escrituras de disco en la insercion')
    ax.set_xlabel('Registros')
    ax.set_ylabel('Escrituras de disco')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / 'plot_writes_vs_records.png')


def plot_experiment_2(summary, out_dir: Path):
    names = list(summary)
    positions = range(len(names))

    for metric, title, ylabel, filename, error_key in (
        ('mean_reads', 'Experimento 2: media de lecturas por consulta',
         'Lecturas', 'plot_mean_reads.png', 'std_reads'),
        ('mean_ms', 'Experimento 2: media de tiempo por consulta',
         'Tiempo (ms)', 'plot_mean_time.png', 'std_ms'),
        ('std_reads', 'Experimento 2: desviacion estandar de lecturas',
         'Desviacion estandar', 'plot_std_reads.png', None),
        ('std_ms', 'Experimento 2: desviacion estandar de tiempo',
         'Desviacion estandar (ms)', 'plot_std_time.png', None),
    ):
        values = [summary[name][metric] for name in names]
        errors = ([summary[name][error_key] for name in names]
                  if error_key else None)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = ax.bar(positions, values,
                      color=[COLORS.get(name, '#333333') for name in names],
                      yerr=errors, capsize=4)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.3f}', (bar.get_x() + bar.get_width() / 2,
                                         bar.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        save_figure(fig, out_dir / filename)


def plot_experiment_3(rows_out, out_dir: Path):
    selectivities = [row['selectivity'] * 100 for row in rows_out]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ['Heap', 'Sequential', 'B+ Tree']:
        ax.plot(selectivities, [row[name] for row in rows_out], marker='o',
                color=COLORS[name], label=name)
    ax.set_xscale('log')
    ax.set_xticks(selectivities)
    ax.set_xticklabels([f'{value:g}%' for value in selectivities])
    ax.set_title('Experimento 3: lecturas de disco por rango')
    ax.set_xlabel('Selectividad')
    ax.set_ylabel('Lecturas de disco')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / 'plot_reads_vs_selectivity.png')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ['Heap', 'Sequential', 'B+ Tree']:
        ax.plot(selectivities, [row[f'{name}_ms'] for row in rows_out],
                marker='o', color=COLORS[name], label=name)
    ax.set_xscale('log')
    ax.set_xticks(selectivities)
    ax.set_xticklabels([f'{value:g}%' for value in selectivities])
    ax.set_title('Experimento 3: tiempo por consulta de rango')
    ax.set_xlabel('Selectividad')
    ax.set_ylabel('Tiempo (ms)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / 'plot_time_vs_selectivity.png')

    targets = [row['selectivity'] * TOTAL_ROWS for row in rows_out]
    real = [row['rows'] for row in rows_out]
    width = 0.38
    positions = range(len(selectivities))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([p - width / 2 for p in positions], targets, width,
           label='Objetivo', color='#7f7f7f')
    ax.bar([p + width / 2 for p in positions], real, width,
           label='Real', color='#1f77b4')
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f'{value:g}%' for value in selectivities])
    ax.set_title('Experimento 3: filas recuperadas vs selectividad objetivo')
    ax.set_xlabel('Selectividad')
    ax.set_ylabel('Filas')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / 'plot_rows_vs_selectivity.png')


def plot_experiment_4(rows_out, out_dir: Path):
    pages = [row['page_size'] for row in rows_out]
    positions = range(len(pages))

    for metric, title, ylabel, filename in (
        ('fanout', 'Experimento 4: fan-out del B+ Tree',
         'Fan-out (hijos por nodo interno)', 'plot_fanout.png'),
        ('height', 'Experimento 4: altura del B+ Tree',
         'Altura (niveles)', 'plot_height.png'),
        ('reads', 'Experimento 4: lecturas del lote de consultas',
         'Lecturas de disco', 'plot_reads.png'),
        ('writes', 'Experimento 4: escrituras del lote de consultas',
         'Escrituras de disco', 'plot_writes.png'),
        ('ms', 'Experimento 4: tiempo del lote de consultas',
         'Tiempo (ms)', 'plot_time.png'),
    ):
        values = [row[metric] for row in rows_out]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = ax.bar(positions, values, color='#d62728')
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.3f}' if isinstance(value, float)
                        else f'{value}',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(page) for page in pages])
        ax.set_title(title)
        ax.set_xlabel('Tamano de pagina (bytes)')
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        save_figure(fig, out_dir / filename)



# Orquestador
def main():
    started_at = datetime.now()
    print('=' * 40)
    print('BENCHMARK BD2')
    print('=' * 40)

    require(DATASET.exists(), '0', '-', 'dataset', 'archivo',
            f'no existe {DATASET}')

    rows_in_dataset = count_dataset_rows()
    require(rows_in_dataset == TOTAL_ROWS, '0', '-', 'dataset', 'filas',
            f'se esperaban {TOTAL_ROWS} y hay {rows_in_dataset}')
    dataset_hash = file_sha256(DATASET)

    meta = make_metadata()
    rng = random.Random(SEED)
    keys = sorted(rng.sample(range(1, TOTAL_ROWS + 1), EQ_QUERIES))
    require(len(set(keys)) == EQ_QUERIES, '0', '-', 'claves', 'unicidad',
            'las claves de igualdad no son unicas')

    print(f'Dataset      : {DATASET}')
    print(f'Registros    : {TOTAL_ROWS}')
    print(f'Semilla      : {SEED}')
    print(f'Claves       : {EQ_QUERIES}')
    print(f'Page sizes   : {PAGE_SIZES}')

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    ensure_dir(DATA_DIR)

    retained = None
    results = {}
    try:
        retained, results[1] = experiment_1(meta)
        results[2] = experiment_2(meta, retained, keys)
        results[3] = experiment_3(meta, retained)
    finally:
        if retained is not None:
            for key in ('heap', 'seq', 'tree', 'hash'):
                try:
                    retained[key].close()
                except Exception:
                    pass

    results[4] = experiment_4(meta, keys)

    finished_at = datetime.now()
    dataset_label = (str(DATASET.relative_to(ROOT))
                     if DATASET.is_relative_to(ROOT) else str(DATASET))
    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'finished_at': finished_at.isoformat(timespec='seconds'),
        'duration_seconds': (finished_at - started_at).total_seconds(),
        'python': sys.version,
        'seed': SEED,
        'dataset': dataset_label,
        'dataset_rows': rows_in_dataset,
        'dataset_sha256': dataset_hash,
        'sizes': SIZES,
        'page_sizes': PAGE_SIZES,
        'default_page_size': DEFAULT_PAGE_SIZE,
        'equality_queries': EQ_QUERIES,
        'selectivities': SELECTIVITIES,
        'keys_sha256': hashlib.sha256(
            ','.join(str(key) for key in keys).encode()
        ).hexdigest(),
        'experiments': {
            '1': 'benchmarks/results/experiment_1',
            '2': 'benchmarks/results/experiment_2',
            '3': 'benchmarks/results/experiment_3',
            '4': 'benchmarks/results/experiment_4',
        },
    }
    write_json(RESULTS_DIR / 'benchmark_summary.json', summary)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    print()
    print('=' * 40)
    print('BENCHMARK COMPLETADO')
    print('=' * 40)
    for path in sorted(RESULTS_DIR.rglob('*')):
        if path.is_file() and path.name != '.gitkeep':
            label = (str(path.relative_to(ROOT))
                     if path.is_relative_to(ROOT) else str(path))
            print(f'  {label}')
    print()
    print(f'Duracion total: {summary["duration_seconds"]:.1f} s')


if __name__ == '__main__':
    main()
