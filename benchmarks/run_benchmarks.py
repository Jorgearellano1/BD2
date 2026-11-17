#!/usr/bin/env python3
"""Local, reproducible benchmarks for the four BD2 experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from engine.catalog import Catalog
from engine.session import Session
from parser.parser import Parser
from parser.scanner import Scanner
from storage.heapFile import HeapFile
from storage.sequentialFile import SequentialFile
from indexes.bplusTree import BPlusTree
from indexes.ExtendibleHash import ExtendibleHash

SIZES = [1000, 10000, 50000, 100000, 250000, 500000]
PAGES = [1024, 2048, 4096, 8192]
RATES = [0.001, 0.01, 0.05, 0.10, 0.25]
KINDS = ['Heap', 'Sequential', 'B+', 'Hash']
COLS = ['Student_ID', 'Age', 'Gender', 'Region_Type', 'Major_Subject',
        'University_Name', 'Home_City', 'Parent_Education', 'Previous_GPA']
FIELDS = ['experiment', 'structure', 'n', 'page_size', 'repeat', 'query_id',
          'selectivity', 'lower', 'upper', 'rows', 'parse_ms', 'execution_ms',
          'disk_reads', 'disk_writes', 'plan', 'fanout', 'height',
          'overflow_before', 'overflow_after', 'sql']


def sql(catalog, size, text):
    start = time.perf_counter()
    program = Parser(Scanner(text)).parse_program()
    parsed = time.perf_counter()
    output = Session(catalog, size).execute(program)
    elapsed = (time.perf_counter() - parsed) * 1000
    if not output.success:
        raise RuntimeError(f'{text}\n{output.error_type}: {output.error}')
    def names(node):
        return [node.node_type] + [name for child in node.children for name in names(child)]
    roots = [item.root for item in output.explains if item.root]
    return output.results[-1].rows, {
        'parse_ms': (parsed - start) * 1000, 'execution_ms': elapsed,
        'disk_reads': sum(node.pages_read for node in roots),
        'disk_writes': sum(node.pages_written for node in roots),
        'plan': ' > '.join(name for node in roots for name in names(node)),
    }


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def schema(rows):
    parts = []
    for i, name in enumerate(COLS):
        kind = 'INT' if i < 2 else 'FLOAT' if i == 8 else f'CHAR({max(len(row[i].encode()) for row in rows)})'
        parts.append(f'{name} {kind}')
    return parts


def prepare(path, rows, parts, kind, size):
    path.mkdir(parents=True)
    source = path / 'input.csv'
    write_csv(source, COLS, [dict(zip(COLS, row)) for row in rows])
    catalog = Catalog(str(path / 'catalog.json'), str(path / 'data'))
    columns = list(parts)
    sequential = kind.startswith('Sequential')
    if sequential:
        columns[0] += ' PRIMARY KEY'
    source_sql = str(source).replace("'", "''")
    query = f"CREATE TABLE students ({', '.join(columns)}) USING {'SEQUENTIAL' if sequential else 'HEAP'} FROM CSV ('{source_sql}');"
    _, metrics = sql(catalog, size, query)
    if kind in ('B+', 'Hash'):
        _, added = sql(catalog, size, f"CREATE INDEX bench_idx ON students (Student_ID) USING {'BTREE' if kind == 'B+' else 'HASH'};")
        for key in ['parse_ms', 'execution_ms', 'disk_reads', 'disk_writes']:
            metrics[key] += added[key]
        metrics['plan'] += ' > ' + added['plan']
    return catalog, metrics


def force(kind):
    return ' FORCE SEQSCAN' if kind == 'Heap' else ' FORCE INDEX(bench_idx)' if kind in ('B+', 'Hash') else ''


def checked_query(catalog, size, kind, lower, upper, expected, equality=False):
    predicate = f'Student_ID = {lower}' if equality and kind != 'Sequential' else f'Student_ID >= {lower} AND Student_ID <= {upper}'
    statement = f'SELECT Student_ID FROM students{force(kind)} WHERE {predicate};'
    rows, metrics = sql(catalog, size, statement)
    found = [row[0] for row in rows]
    if len(found) != len(expected) or set(found) != expected:
        raise AssertionError(f'Wrong results: {kind} [{lower},{upper}]: {len(found)} instead of {len(expected)}')
    required = 'SeqScanPlan' if kind == 'Heap' else 'SequentialRangeScanPlan' if kind == 'Sequential' else 'IndexScanPlan' if equality else 'IndexRangeScanPlan'
    if required not in metrics['plan'].split(' > '):
        raise AssertionError(f'Unexpected plan {kind}: {metrics["plan"]}')
    return {'rows': len(found), 'lower': lower, 'upper': upper, 'sql': statement, **metrics}


def append_batch(catalog, size, kind, rows):
    entry = catalog.get_table('students')
    sequential = kind.startswith('Sequential')
    storage = SequentialFile(entry.metadata, entry.file_path, size, auto_reorganize=False) if sequential else HeapFile(entry.metadata, entry.file_path, size)
    index = None
    try:
        if kind in ('B+', 'Hash'):
            info = catalog.get_index('bench_idx')
            cls = BPlusTree if kind == 'B+' else ExtendibleHash
            index = cls(entry.metadata, 'Student_ID', storage, size, info.file_path)
            if kind == 'Hash':
                index.open()
        storage.reset_stats()
        if index:
            index.reset_stats()
        start = time.perf_counter()
        for raw in rows:
            values = entry.metadata.normalize_tuple(raw)
            rid = storage.insert(values)
            if index and not index.insert(values[0], rid):
                raise AssertionError('Unexpected duplicate')
        before = storage._pages.h.overflow_records if sequential else 0
        if kind == 'Sequential+reorg':
            storage.reorganize()
        after = storage._pages.h.overflow_records if sequential else 0
        if kind == 'Hash':
            index.flush()
        if index:
            index.close()
        storage.close()
        return {
            'execution_ms': (time.perf_counter() - start) * 1000,
            'disk_reads': storage.page_reads + (index.page_reads if index else 0),
            'disk_writes': storage.page_writes + (index.page_writes if index else 0),
            'overflow_before': before, 'overflow_after': after,
        }
    finally:
        if index:
            index.close()
        storage.close()


def tree_shape(catalog, size):
    entry = catalog.get_table('students')
    heap = HeapFile(entry.metadata, entry.file_path, size)
    index = BPlusTree(entry.metadata, 'Student_ID', heap, size, catalog.get_index('bench_idx').file_path)
    try:
        pointer = index._read_header(index.file)[0]
        height = 1
        node = index._read_node(index.file, pointer)
        while node.node_type == 0:
            height += 1
            node = index._read_node(index.file, node.pointers[0])
        return index.INTERNAL_ORDER + 1, height
    finally:
        index.close()
        heap.close()


def summarize(records, destination):
    groups = defaultdict(list)
    group_fields = ['experiment', 'structure', 'n', 'page_size', 'selectivity']
    for row in records:
        groups[tuple(row.get(key, '') for key in group_fields)].append(row)
    output = []
    for group, entries in groups.items():
        row = dict(zip(group_fields, group))
        row['samples'] = len(entries)
        for key in ['execution_ms', 'disk_reads', 'disk_writes', 'parse_ms']:
            values = [float(item[key]) for item in entries]
            row[key + '_mean'] = statistics.mean(values)
            row[key + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        row['fanout'] = entries[0].get('fanout', '')
        row['height'] = entries[0].get('height', '')
        output.append(row)
    write_csv(destination / 'summary.csv', list(output[0]), output)
    for experiment in range(1, 5):
        subset = [r for r in output if r['experiment'] == experiment]
        if subset:
            write_csv(destination / f'exp{experiment}_summary.csv', list(output[0]), subset)
    return output


def plots(summary, directory, quick):
    os.environ.setdefault('MPLCONFIGDIR', str(directory / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
    suffix = ' — PRUEBA RÁPIDA' if quick else ''
    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(directory / filename)
        plt.close(fig)
    def lines(exp, x, metric, filename, title, xlabel, ylabel):
        rows = [r for r in summary if r['experiment'] == exp]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for kind in dict.fromkeys(r['structure'] for r in rows):
            points = sorted((r for r in rows if r['structure'] == kind), key=lambda r: float(r[x]))
            ax.plot([float(r[x]) for r in points], [r[metric + '_mean'] for r in points], marker='o', label=kind)
        ax.set(title=title + suffix, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend()
        save(fig, filename)
    lines(1, 'n', 'execution_ms', 'exp1_tiempo.png', 'Carga masiva e inserción', 'Registros', 'Tiempo (ms)')
    lines(1, 'n', 'disk_writes', 'exp1_escrituras.png', 'Escrituras de carga', 'Registros', 'Páginas instrumentadas')
    for metric, filename, ylabel in [('disk_reads', 'exp2_lecturas.png', 'Páginas instrumentadas'), ('execution_ms', 'exp2_tiempo.png', 'Tiempo (ms)')]:
        rows = [r for r in summary if r['experiment'] == 2]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([r['structure'] for r in rows], [r[metric + '_mean'] for r in rows], yerr=[r[metric + '_std'] for r in rows], capsize=4)
        ax.set(title='Igualdad: media ± desviación estándar' + suffix, ylabel=ylabel)
        ax.grid(axis='y', alpha=.25)
        save(fig, filename)
    lines(3, 'selectivity', 'disk_reads', 'exp3_lecturas.png', 'Lecturas por rango', 'Selectividad (%)', 'Páginas instrumentadas')
    lines(3, 'selectivity', 'execution_ms', 'exp3_tiempo.png', 'Tiempo por rango', 'Selectividad (%)', 'Tiempo (ms)')
    rows = sorted((r for r in summary if r['experiment'] == 4), key=lambda r: r['page_size'])
    sizes = [r['page_size'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, label in zip(axes, ['fanout', 'height'], ['Fan-out máximo', 'Altura (niveles)']):
        ax.plot(sizes, [r[key] for r in rows], marker='o')
        ax.set(xlabel='Página (bytes)', ylabel=label)
        ax.set_xticks(sizes)
        ax.grid(alpha=.25)
    fig.suptitle('Estructura B+' + suffix)
    save(fig, 'exp4_estructura.png')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in [('disk_reads', 'Lecturas'), ('disk_writes', 'Escrituras')]:
        ax.plot(sizes, [r[key + '_mean'] for r in rows], marker='o', label=label)
    ax.set(title='B+: I/O medio por búsqueda de igualdad' + suffix, xlabel='Página (bytes)', ylabel='Páginas instrumentadas')
    ax.set_xticks(sizes)
    ax.legend()
    ax.grid(alpha=.25)
    save(fig, 'exp4_io.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Small validation run, not final report data')
    parser.add_argument('--seed', type=int, default=2042)
    parser.add_argument('--repeats', type=int, default=3, help='Independent runs for insertion and ranges')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/datasets/students_500k.csv')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    sizes = [100, 500, 1000] if args.quick else SIZES
    query_n = 1000 if args.quick else 100000
    range_n = 1000 if args.quick else 500000
    queries = 20 if args.quick else 1000
    repeats = 1 if args.quick else args.repeats
    dataset = args.dataset.resolve()
    with dataset.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        if next(reader) != COLS:
            raise ValueError('Dataset columns do not match students_500k.csv')
        rows = [row for row in reader]
    if len(rows) < max(sizes + [query_n, range_n]):
        raise ValueError('Dataset too small for requested profile')
    keys = [int(row[0]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Student_ID must be unique')
    columns = schema(rows)
    mode = 'quick' if args.quick else 'full'
    destination = ROOT / 'benchmarks/results' / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + mode)
    destination.mkdir(parents=True)
    manifest = {
        'status': 'running', 'mode': mode, 'seed': args.seed, 'sizes': sizes,
        'equality_n': query_n, 'range_n': range_n, 'queries': queries, 'repeats': repeats,
        'page_sizes': PAGES, 'base_page_size': 4096, 'dataset': str(dataset),
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'dataset_rows': len(rows),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'python': sys.version, 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_count': os.cpu_count(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sequential_equality': 'Uses >= key AND <= key because the current planner turns = into SeqScan; equivalent rows, verified ordered access',
        'cache_policy': 'OS cache not flushed; new Session per query; shuffled method order; one warm-up per structure excluded',
        'timing': 'SQL execution_ms includes semantic validation, planning, execution and close, excludes parse and result validation',
        'insertion': '50% CSV bulk load + index construction where applicable; 50% individual inserts; Sequential auto-reorganization disabled, +reorg adds final manual reorganization',
        'io_scope': 'Existing engine page counters, not measured physical device I/O; does not correct omitted header/open/close operations in engine instrumentation',
    }
    manifest_path = destination / 'manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    save_manifest()
    records = []
    print(f'Output: {destination}', flush=True)
    rawfile = (destination / 'raw.csv').open('w', newline='', encoding='utf-8')
    writer = csv.DictWriter(rawfile, fieldnames=FIELDS)
    writer.writeheader()
    def record(experiment, kind, n, size, repeat, metrics, **extra):
        row = dict(experiment=experiment, structure=kind, n=n, page_size=size, repeat=repeat, **metrics, **extra)
        writer.writerow(row)
        rawfile.flush()
        records.append(row)
    try:
        with tempfile.TemporaryDirectory(prefix='bd2-benchmark-') as tmp:
            scratch = Path(tmp)
            for n in sizes:
                for repeat in range(repeats):
                    batch = list(rows[:n])
                    random.Random(args.seed + repeat).shuffle(batch)
                    cut = max(1, int(n * .5))
                    order = ['Heap', 'Sequential', 'Sequential+reorg', 'B+', 'Hash']
                    random.Random(args.seed + n + repeat).shuffle(order)
                    for kind in order:
                        print(f'E1 n={n} run={repeat + 1} {kind}', flush=True)
                        with tempfile.TemporaryDirectory(dir=scratch) as case:
                            catalog, base = prepare(Path(case) / 'db', batch[:cut], columns, kind, 4096)
                            tail = append_batch(catalog, 4096, kind, batch[cut:])
                            for key in ['execution_ms', 'disk_reads', 'disk_writes']:
                                base[key] += tail[key]
                            base.update(overflow_before=tail['overflow_before'], overflow_after=tail['overflow_after'])
                            actual, _ = sql(catalog, 4096, 'SELECT Student_ID FROM students;')
                            if len(actual) != n or {r[0] for r in actual} != {int(r[0]) for r in batch}:
                                raise AssertionError(f'Insertion/persistence mismatch: {kind}')
                            if kind == 'Sequential+reorg' and tail['overflow_after'] != 0:
                                raise AssertionError('Reorganization left overflow')
                            record(1, kind, n, 4096, repeat, base, rows=n)
            selected_keys = random.Random(args.seed).choices(keys[:query_n], k=queries)
            write_csv(destination / 'equality_keys.csv', ['query_id', 'key'], [dict(query_id=i, key=k) for i,k in enumerate(selected_keys)])
            for experiment, n in [(2, query_n), (3, range_n)]:
                with tempfile.TemporaryDirectory(dir=scratch) as work:
                    catalogs = {}
                    kinds = KINDS if experiment == 2 else KINDS[:-1]
                    for kind in kinds:
                        print(f'E{experiment} prepare {kind} n={n}', flush=True)
                        catalogs[kind], _ = prepare(Path(work) / kind.replace('+', 'plus'), rows[:n], columns, kind, 4096)
                        checked_query(catalogs[kind], 4096, kind, keys[0], keys[0], {keys[0]}, True)
                    if experiment == 2:
                        cases = [(i, k, k, {k}, '') for i,k in enumerate(selected_keys)]
                    else:
                        sorted_keys = sorted(keys[:n])
                        cases = []
                        rng = random.Random(args.seed)
                        for repeat in range(repeats):
                            for rate in RATES:
                                count = max(1, round(n * rate))
                                start = rng.randrange(n - count + 1)
                                expected = set(sorted_keys[start:start + count])
                                cases.append((repeat, min(expected), max(expected), expected, rate * 100))
                    for i,(repeat,lo,hi,expected,rate) in enumerate(cases):
                        order = list(kinds)
                        random.Random(args.seed + i).shuffle(order)
                        for kind in order:
                            result = checked_query(catalogs[kind], 4096, kind, lo, hi, expected, experiment == 2)
                            record(experiment, kind, n, 4096, repeat if experiment == 3 else 0, result, query_id=i, selectivity=rate)
                        if i % 25 == 0:
                            print(f'E{experiment} queries {i + 1}/{len(cases)}', flush=True)
            for size in PAGES:
                print(f'E4 B+ page_size={size}', flush=True)
                with tempfile.TemporaryDirectory(dir=scratch) as work:
                    catalog, _ = prepare(Path(work) / 'db', rows[:query_n], columns, 'B+', size)
                    fanout, height = tree_shape(catalog, size)
                    checked_query(catalog, size, 'B+', selected_keys[0], selected_keys[0], {selected_keys[0]}, True)
                    for i,key in enumerate(selected_keys):
                        result = checked_query(catalog, size, 'B+', key, key, {key}, True)
                        record(4, 'B+', query_n, size, 0, result, query_id=i, fanout=fanout, height=height)
        summary = summarize(records, destination)
        plots(summary, destination, args.quick)
        manifest['status'] = 'complete'
        manifest['raw_measurements'] = len(records)
        print(f'PASS: {len(records)} measurements validated; CSV + 8 PNG in {destination}', flush=True)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        rawfile.close()
        save_manifest()


if __name__ == '__main__':
    main()
