"""
Utilidades compartidas por los experimentos del modulo vectorial.

Los scripts de este directorio usan el motor real (parser + Session) sobre
directorios temporales y escriben sus resultados en benchmarks/results/.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / 'backend'
DATA = ROOT / 'data' / 'datasets'
MULTIMEDIA = DATA / 'multimedia'
RESULTS = ROOT / 'benchmarks' / 'results'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from engine.catalog import Catalog  # noqa: E402
from engine.session import Session  # noqa: E402
from parser.parser import Parser  # noqa: E402
from parser.scanner import Scanner  # noqa: E402
from vectorindex.vectors import format_vector  # noqa: E402


class BenchmarkError(RuntimeError):
    """Error de configuracion o de ejecucion del benchmark."""


def check_dataset() -> None:
    required = [
        MULTIMEDIA / 'sift_descriptors.bin',
        MULTIMEDIA / 'embeddings.bin',
        MULTIMEDIA / 'embeddings_manifest.csv',
        MULTIMEDIA / 'manifest.json',
    ]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise BenchmarkError(
            'Falta el dataset multimedia. Regeneralo con '
            'python benchmarks/vector/build_dataset.py. Faltan: '
            + ', '.join(missing)
        )


def sql_vector(vector) -> str:
    """Literal SQL de un vector, listo para incluir entre comillas."""
    return "'" + format_vector(vector, precision=7) + "'"


def sql_string(value: str) -> str:
    """Literal SQL de texto escapando comillas simples."""
    return "'" + value.replace("'", "''") + "'"


class Database:
    """Catalogo temporal sobre el que corren todas las sentencias del run."""

    def __init__(self, directory: str | None = None) -> None:
        self._temporary = directory is None

        if directory is None:
            directory = tempfile.mkdtemp(prefix='bd2vector_bench_')

        self.directory = directory
        self.catalog = Catalog(
            os.path.join(directory, 'catalog.json'),
            os.path.join(directory, 'storage'),
        )

    def execute(self, sql: str):
        started = time.perf_counter()
        program = Parser(Scanner(sql)).parse_program()
        parsed = time.perf_counter()
        output = Session(self.catalog, 4096).execute(program)
        finished = time.perf_counter()

        if not output.success:
            raise BenchmarkError(
                f'{output.error_type}: {output.error}\nSQL: {sql[:400]}'
            )

        return output, (parsed - started) * 1000.0, (finished - parsed) * 1000.0

    def query(self, sql: str):
        """Ejecuta una sentencia y devuelve (resultado, explain, latencia_ms)."""
        output, parse_ms, execution_ms = self.execute(sql)
        explain = output.explains[-1]
        return output.results[-1], explain, execution_ms + parse_ms

    def close(self) -> None:
        self.catalog = None


def find_node(root, node_type: str):
    """Primer nodo del arbol de metricas con el tipo indicado."""
    if root is None:
        return None

    if root.node_type == node_type:
        return root

    for child in root.children:
        found = find_node(child, node_type)

        if found is not None:
            return found

    return None


def plan_name(root) -> str:
    """Nombre corto del acceso principal del plan."""
    preferred = [
        'VectorScanPlan', 'TextIndexScanPlan', 'FullTextScanPlan',
        'IndexScanPlan', 'IndexRangeScanPlan', 'SeqScanPlan',
        'BitmapHeapScanPlan',
    ]

    for name in preferred:
        if find_node(root, name) is not None:
            return name.removesuffix('Plan')

    return root.node_type.removesuffix('Plan') if root is not None else 'None'


def strategy_of(root) -> str | None:
    node = find_node(root, 'VectorScanPlan') or find_node(
        root, 'TextIndexScanPlan'
    )

    if node is None or not node.detail:
        return None

    return node.detail.split(';', 1)[0]


def reads_of(root) -> tuple[int, int]:
    if root is None:
        return 0, 0

    return root.pages_read, root.pages_written


def precision_at_k(retrieved: list[str], label: str, k: int) -> float:
    if k <= 0:
        return 0.0

    return sum(1 for item in retrieved[:k] if item == label) / k


def recall_at_k(retrieved: list[int], truth: list[int], k: int) -> float:
    if not truth:
        return 0.0

    return len(set(retrieved[:k]) & set(truth[:k])) / min(k, len(truth))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def timestamp() -> str:
    return time.strftime('%Y%m%d-%H%M%S')


def prepare_output(experiment: str, profile: str) -> Path:
    directory = RESULTS / experiment / f'{timestamp()}-{profile}'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path: Path, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def plots_enabled(args) -> bool:
    return not getattr(args, 'no_plots', False)


def import_pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise BenchmarkError(
            'matplotlib no esta instalado; usa --no-plots para omitir las '
            f'graficas (error original: {exc})'
        ) from exc
