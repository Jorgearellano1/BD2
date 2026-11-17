"""
Experimento 6: Indice invertido en disco vs Full Text Scan vs GIN (PostgreSQL).

Para cada tamano de corpus N (prefijos del corpus barajado):
  1. Crea una base temporal del motor (fuera de storage/), carga N documentos
     desde CSV en un HeapFile y construye el indice INVERTED por SQL.
     Registra tiempo de construccion, paginas leidas/escritas y tamano en disco
     de vocab.dic, postings.inv y docs.dat.
  2. Ejecuta el mismo conjunto de consultas (1, 3 y 5 terminos; termino unico,
     AND, OR, AND-NOT y ranking Top-10 por coseno) con:
        - indice propio  (SQL normal, plan TextIndexScan)
        - Full Text Scan (SQL con FORCE SEQSCAN, plan FullTextScan)
        - PostgreSQL GIN sobre to_tsvector('spanish', cuerpo), si se indica
          --pg-dsn (PostgreSQL solo como comparador externo).
     Latencia: mediana de --repeats ejecuciones (cache caliente).
     Lecturas: paginas logicas del motor (explain) y, en PostgreSQL, bloques
     de 8 KB 'shared hit + read' de EXPLAIN (ANALYZE, BUFFERS).
  3. Verifica que indice y Full Text Scan devuelven exactamente los mismos
     identificadores (y el mismo orden en ranking); la coincidencia con
     PostgreSQL se reporta como Jaccard porque su tokenizador y su ranking
     (ts_rank) son distintos.

Las consultas se generan con semilla a partir de los primeros 1000
documentos, de modo que existen en todos los prefijos.

Uso:
  .venv/bin/python benchmarks/text/run_experiment6.py --sizes 1000,2500,5000,10000,20000,30000 \\
      --pg-dsn postgresql://bench:bench@127.0.0.1:55432/bench
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))

from engine.catalog import Catalog  # noqa: E402
from engine.session import Session  # noqa: E402
from parser.parser import Parser  # noqa: E402
from parser.scanner import Scanner  # noqa: E402
from textindex.preprocessing import TextPreprocessor  # noqa: E402

CORPUS = ROOT / 'data' / 'datasets' / 'corpus_gutenberg_es.csv'
OUT_ROOT = ROOT / 'benchmarks' / 'results' / 'experiment_6'
PAGE_SIZE = 4096
CATEGORIES = [
    ('term', 1), ('and', 3), ('and', 5), ('or', 3), ('or', 5),
    ('andnot', 3), ('andnot', 5), ('rank', 1), ('rank', 3), ('rank', 5),
]
WORD = re.compile(r'[^\W\d_]{4,}')


# ---------------------------------------------------------------------------
# Corpus y consultas
# ---------------------------------------------------------------------------

def load_corpus(limit: int, path: Path = CORPUS) -> list[list[str]]:
    with open(path, encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def eligible_words(text: str, pre: TextPreprocessor) -> dict[str, str]:
    """termino -> una palabra de superficie que lo produce."""
    out: dict[str, str] = {}
    for word in WORD.findall(text):
        terms = pre.terms(word)
        if len(terms) == 1 and terms[0] not in out:
            out[terms[0]] = word.lower()
    return out


def generate_queries(rows: list[list[str]], per_category: int, seed: int) -> list[dict]:
    pre = TextPreprocessor('spanish')
    rng = random.Random(seed)
    base = [eligible_words(row[3], pre) for row in rows[:1000]]
    rich = {k: [b for b in base if len(b) >= k] for k in (1, 2, 3, 5)}
    queries = []

    def same_doc(k: int) -> list[str]:
        bag = rng.choice(rich[k])
        return rng.sample(sorted(bag.values()), k)

    def any_word(exclude: set[str]) -> str:
        while True:
            bag = rng.choice(base)
            if not bag:
                continue
            term, word = rng.choice(sorted(bag.items()))
            if term not in exclude:
                return word

    for kind, k in CATEGORIES:
        for number in range(per_category):
            if kind in ('term', 'rank') and k == 1:
                words = [any_word(set())]
                neg: list[str] = []
            elif kind in ('and', 'rank'):
                words, neg = same_doc(k), []
            elif kind == 'or':
                words, neg = [], []
                while len(words) < k:
                    word = any_word(set(pre.terms(' '.join(words))))
                    words.append(word)
            else:  # andnot: k=3 -> 2 positivos + 1 negado; k=5 -> 3 + 2
                positives = 2 if k == 3 else 3
                words = same_doc(positives)
                taken = set(pre.terms(' '.join(words)))
                neg = []
                while len(neg) < k - positives:
                    word = any_word(taken)
                    taken |= set(pre.terms(word))
                    neg.append(word)
            queries.append({
                'qid': f'{kind}{k}_{number:02d}', 'kind': kind, 'k': k,
                'words': words, 'neg': neg,
                'match': to_match(kind, words, neg), 'tsquery': to_tsquery(kind, words, neg),
            })
    return queries


def to_match(kind: str, words: list[str], neg: list[str]) -> str:
    if kind in ('term', 'rank'):
        return ' '.join(words)                       # texto libre (OR implicito)
    if kind == 'and':
        return ' AND '.join(words)
    if kind == 'or':
        return ' OR '.join(words)
    return ' AND '.join(words) + ''.join(f' AND NOT {w}' for w in neg)


def to_tsquery(kind: str, words: list[str], neg: list[str]) -> str:
    if kind in ('term', 'rank', 'or'):
        return ' | '.join(words)
    if kind == 'and':
        return ' & '.join(words)
    return ' & '.join(words) + ''.join(f' & !{w}' for w in neg)


def engine_sql(query: dict, force_scan: bool) -> str:
    force = ' FORCE SEQSCAN' if force_scan else ''
    text = query['match'].replace("'", "''")
    if query['kind'] == 'rank':
        return (f"SELECT id, COSINE_SCORE() AS s FROM docs{force} WHERE cuerpo MATCH "
                f"'{text}' ORDER BY s DESC LIMIT 10;")
    return f"SELECT id FROM docs{force} WHERE cuerpo MATCH '{text}';"


def pg_sql(query: dict) -> tuple[str, tuple]:
    if query['kind'] == 'rank':
        return ("SELECT id, ts_rank(to_tsvector('spanish', cuerpo), q) AS s "
                "FROM docs, to_tsquery('spanish', %s) q "
                "WHERE to_tsvector('spanish', cuerpo) @@ q ORDER BY s DESC LIMIT 10", (query['tsquery'],))
    return ("SELECT id FROM docs WHERE to_tsvector('spanish', cuerpo) @@ to_tsquery('spanish', %s)",
            (query['tsquery'],))


# ---------------------------------------------------------------------------
# Motor propio
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.catalog = Catalog(str(workdir / 'catalog.json'), str(workdir / 'data'))

    def execute(self, sql: str):
        start = perf_counter()
        program = Parser(Scanner(sql)).parse_program()
        output = Session(self.catalog, PAGE_SIZE).execute(program)
        wall = (perf_counter() - start) * 1000.0
        if not output.success:
            raise RuntimeError(f'{output.error_type}: {output.error} :: {sql}')
        return output, wall


def node_walk(node):
    yield node
    for child in node.children:
        yield from node_walk(child)


def split_reads(root) -> tuple[int, int]:
    index = table = 0
    for node in node_walk(root):
        for op in node.operations:
            if op.resource_type in ('HeapFile', 'SequentialFile'):
                table += op.pages_read
            else:
                index += op.pages_read
    return index, table


def plan_kind(root) -> str:
    for node in node_walk(root):
        if node.node_type in ('TextIndexScanPlan', 'FullTextScanPlan'):
            return node.node_type
    return root.node_type


def dir_size(path: str) -> dict[str, int]:
    return {name: os.path.getsize(os.path.join(path, name)) for name in sorted(os.listdir(path))}


# ---------------------------------------------------------------------------
# PostgreSQL (comparador externo)
# ---------------------------------------------------------------------------

class Postgres:
    def __init__(self, dsn: str) -> None:
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.version = self.conn.execute('SHOW server_version').fetchone()[0]

    def load(self, rows: list[list[str]]) -> dict:
        cur = self.conn.cursor()
        cur.execute('DROP TABLE IF EXISTS docs')
        cur.execute('CREATE TABLE docs (id INT PRIMARY KEY, libro INT, titulo TEXT, cuerpo TEXT)')
        with cur.copy('COPY docs (id, libro, titulo, cuerpo) FROM STDIN') as copy:
            for row in rows:
                copy.write_row((int(row[0]), int(row[1]), row[2], row[3]))
        cur.execute('VACUUM ANALYZE docs')
        start = perf_counter()
        cur.execute("CREATE INDEX docs_gin ON docs USING GIN (to_tsvector('spanish', cuerpo))")
        build_ms = (perf_counter() - start) * 1000.0
        cur.execute('VACUUM ANALYZE docs')
        size = cur.execute("SELECT pg_relation_size('docs_gin'), pg_total_relation_size('docs')").fetchone()
        return {'build_ms': build_ms, 'index_bytes': size[0], 'table_total_bytes': size[1]}

    def run(self, query: dict) -> tuple[list[int], float]:
        sql, params = pg_sql(query)
        start = perf_counter()
        rows = self.conn.execute(sql, params).fetchall()
        return [row[0] for row in rows], (perf_counter() - start) * 1000.0

    def buffers(self, query: dict) -> tuple[int, int, float, str]:
        sql, params = pg_sql(query)
        plan = self.conn.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ' + sql, params).fetchone()[0][0]
        node = plan['Plan']
        kinds = []

        def visit(item):
            kinds.append(item['Node Type'])
            for child in item.get('Plans', []):
                visit(child)

        visit(node)
        uses_gin = any('Bitmap Index Scan' in k or 'Index' in k for k in kinds)
        return (node.get('Shared Hit Blocks', 0), node.get('Shared Read Blocks', 0),
                plan.get('Execution Time', 0.0), 'GIN' if uses_gin else '+'.join(kinds))

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', default='1000,2500,5000,10000,20000,30000')
    parser.add_argument('--queries-per-category', type=int, default=20)
    parser.add_argument('--scan-queries-per-category', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2042)
    parser.add_argument('--pg-dsn', default=None)
    parser.add_argument('--tag', default='full')
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(',')]
    corpus = load_corpus(max(sizes), args.corpus)
    if len(corpus) < max(sizes):
        raise SystemExit(f'El corpus solo tiene {len(corpus)} documentos')
    queries = generate_queries(corpus, args.queries_per_category, args.seed)
    scan_ids = set()
    for kind, k in CATEGORIES:
        ids = [q['qid'] for q in queries if q['kind'] == kind and q['k'] == k]
        scan_ids.update(ids[:args.scan_queries_per_category])

    out = args.out_root / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + args.tag)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / 'queries.json', 'w', encoding='utf-8') as handle:
        json.dump(queries, handle, ensure_ascii=False, indent=1)

    pg = Postgres(args.pg_dsn) if args.pg_dsn else None
    manifest = {
        'status': 'running', 'started': datetime.now().isoformat(timespec='seconds'),
        'sizes': sizes, 'page_size': PAGE_SIZE, 'repeats': args.repeats, 'seed': args.seed,
        'queries_per_category': args.queries_per_category,
        'scan_queries_per_category': args.scan_queries_per_category,
        'corpus': str(args.corpus.resolve().relative_to(ROOT)) if args.corpus.resolve().is_relative_to(ROOT) else str(args.corpus),
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'postgres': pg.version if pg else None,
        'commit_base': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                      capture_output=True).stdout.strip(),
        'note': 'Cache caliente. Lecturas del motor = paginas logicas de 4 KB; '
                'PostgreSQL = bloques de 8 KB (shared hit + read).',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    build_rows: list[dict] = []
    query_rows: list[dict] = []
    scratch = Path(tempfile.mkdtemp(prefix='bd2-exp6-'))
    try:
        for n in sizes:
            print(f'N={n}: carga', flush=True)
            workdir = scratch / f'n{n}'
            workdir.mkdir()
            csv_path = workdir / 'docs.csv'
            with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'libro', 'titulo', 'cuerpo'])
                writer.writerows(corpus[:n])
            engine = Engine(workdir)
            output, load_ms = engine.execute(
                "CREATE TABLE docs (id INT PRIMARY KEY, libro INT, titulo CHAR(80), cuerpo CHAR(1000)) "
                f"FROM CSV('{csv_path}');")
            heap_bytes = os.path.getsize(workdir / 'data' / 'docs.heap')

            print(f'N={n}: construccion del indice invertido', flush=True)
            output, build_wall = engine.execute(
                "CREATE INDEX idx_cuerpo ON docs (cuerpo) USING INVERTED "
                "WITH (scoring = 'TFIDF_COSINE', language = 'spanish');")
            root = output.explains[-1].root
            spimi = next(op for node in node_walk(root) for op in node.operations if op.operation == 'spimi_build')
            index_dir = engine.catalog.get_index('idx_cuerpo').file_path
            sizes_on_disk = dir_size(index_dir)
            build = {
                'n': n, 'load_ms': round(load_ms, 3), 'heap_bytes': heap_bytes,
                'build_ms': round(root.elapsed_ms, 3), 'build_wall_ms': round(build_wall, 3),
                'build_pages_read': root.pages_read, 'build_pages_written': root.pages_written,
                'postings': spimi.entries_affected,
                'vocab_bytes': sizes_on_disk['vocab.dic'], 'postings_bytes': sizes_on_disk['postings.inv'],
                'docs_bytes': sizes_on_disk['docs.dat'],
                'index_bytes': sum(sizes_on_disk.values()),
            }
            from textindex.inverted_index import InvertedIndex
            opened = InvertedIndex(index_dir)
            build['terms'] = opened.num_terms
            build['vocab_pages'] = opened.vocab_pages
            build['dir_pages'] = opened.dir_pages
            opened.close()
            if pg:
                print(f'N={n}: PostgreSQL carga y GIN', flush=True)
                info = pg.load(corpus[:n])
                build.update({'pg_build_ms': round(info['build_ms'], 3),
                              'pg_index_bytes': info['index_bytes'],
                              'pg_table_total_bytes': info['table_total_bytes']})
            build_rows.append(build)
            print(f'  {build}', flush=True)

            # Calentamiento
            engine.execute(engine_sql(queries[0], False))
            if pg:
                pg.run(queries[0])

            for position, query in enumerate(queries, start=1):
                ranked = query['kind'] == 'rank'
                systems = [('indice', False)]
                if query['qid'] in scan_ids:
                    systems.append(('full_scan', True))
                ids_by_system: dict[str, list[int]] = {}
                for system, force in systems:
                    repeats = args.repeats if system == 'indice' else 1
                    walls, execs = [], []
                    for _ in range(repeats):
                        output, wall = engine.execute(engine_sql(query, force))
                        walls.append(wall)
                        execs.append(output.explains[-1].root.elapsed_ms)
                    root = output.explains[-1].root
                    result = output.results[-1]
                    ids = [row[0] for row in result.rows]
                    ids_by_system[system] = ids
                    index_reads, table_reads = split_reads(root)
                    query_rows.append({
                        'n': n, 'system': system, 'qid': query['qid'], 'kind': query['kind'], 'k': query['k'],
                        'plan': plan_kind(root), 'latency_ms': round(statistics.median(walls), 4),
                        'exec_ms': round(statistics.median(execs), 4), 'repeats': repeats,
                        'reads': root.pages_read, 'index_reads': index_reads, 'table_reads': table_reads,
                        'rows': len(ids), 'agree_with_index': '',
                    })
                if 'full_scan' in ids_by_system:
                    same = (ids_by_system['full_scan'] == ids_by_system['indice'] if ranked
                            else sorted(ids_by_system['full_scan']) == sorted(ids_by_system['indice']))
                    query_rows[-1]['agree_with_index'] = int(same)
                    if not same:
                        raise RuntimeError(f"Indice y Full Text Scan difieren en {query['qid']} (N={n})")
                if pg:
                    walls = []
                    for _ in range(args.repeats):
                        ids, wall = pg.run(query)
                        walls.append(wall)
                    hit, read, exec_ms, pg_plan = pg.buffers(query)
                    mine = set(ids_by_system['indice'])
                    theirs = set(ids)
                    union = mine | theirs
                    query_rows.append({
                        'n': n, 'system': 'postgres_gin', 'qid': query['qid'], 'kind': query['kind'],
                        'k': query['k'], 'plan': pg_plan, 'latency_ms': round(statistics.median(walls), 4),
                        'exec_ms': round(exec_ms, 4), 'repeats': args.repeats, 'reads': hit + read,
                        'index_reads': '', 'table_reads': '', 'rows': len(ids),
                        'agree_with_index': round(len(mine & theirs) / len(union), 4) if union else 1.0,
                    })
                if position % 50 == 0:
                    print(f'  N={n}: {position}/{len(queries)} consultas', flush=True)

            write_csv(out / 'build.csv', build_rows)
            write_csv(out / 'queries_raw.csv', query_rows)
            shutil.rmtree(workdir)

        write_summary(out, query_rows)
        manifest['status'] = 'complete'
        manifest['finished'] = datetime.now().isoformat(timespec='seconds')
    except BaseException as exc:
        manifest['status'] = f'failed: {type(exc).__name__}: {exc}'
        raise
    finally:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        shutil.rmtree(scratch, ignore_errors=True)
        if pg:
            pg.close()
    print(f'Resultados en {out}')
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out: Path, rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row['n'], row['system'], row['kind'], row['k']), []).append(row)
    summary = []
    for (n, system, kind, k), items in sorted(groups.items()):
        lat = [r['latency_ms'] for r in items]
        reads = [r['reads'] for r in items]
        agree = [r['agree_with_index'] for r in items if r['agree_with_index'] != '']
        summary.append({
            'n': n, 'system': system, 'kind': kind, 'k': k, 'queries': len(items),
            'latency_mean_ms': round(statistics.mean(lat), 4),
            'latency_median_ms': round(statistics.median(lat), 4),
            'latency_std_ms': round(statistics.stdev(lat), 4) if len(lat) > 1 else 0.0,
            'reads_mean': round(statistics.mean(reads), 2),
            'reads_std': round(statistics.stdev(reads), 2) if len(reads) > 1 else 0.0,
            'rows_mean': round(statistics.mean(r['rows'] for r in items), 2),
            'agreement_mean': round(statistics.mean(agree), 4) if agree else '',
        })
    write_csv(out / 'summary.csv', summary)


if __name__ == '__main__':
    sys.exit(main())
