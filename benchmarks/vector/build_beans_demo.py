"""
Genera un CSV pequeno del dataset de granos para el ejemplo del frontend.

Toma 30 filas por clase del ARFF original, las numera y guarda cada vector
como un literal entre corchetes. El archivo se usa con:

    CREATE TABLE beans (id INT PRIMARY KEY, clase CHAR(20), v VECTOR(16))
      USING HEAP FROM CSV ('data/datasets/dry_beans_demo.csv', DELIMITER = ';');

Uso:
    python benchmarks/vector/build_beans_demo.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from vectorindex.features import load_arff_vectors  # noqa: E402

SOURCE = ROOT / 'data' / 'datasets' / 'dry_beans.arff'
TARGET = ROOT / 'data' / 'datasets' / 'dry_beans_demo.csv'
PER_CLASS = 30
SEED = 42


def main() -> int:
    rows = load_arff_vectors(str(SOURCE))
    groups: dict[str, list[tuple[float, ...]]] = {}

    for label, vector in rows:
        groups.setdefault(label, []).append(vector)

    rng = random.Random(SEED)
    selected: list[tuple[str, tuple[float, ...]]] = []

    for label in sorted(groups):
        vectors = groups[label]
        chosen = rng.sample(vectors, min(PER_CLASS, len(vectors)))
        selected.extend((label, vector) for vector in chosen)

    with open(TARGET, 'w', encoding='utf-8', newline='') as handle:
        for identifier, (label, vector) in enumerate(selected):
            payload = ','.join(f'{value:.6f}' for value in vector)
            handle.write(f'{identifier};{label};[{payload}]\n')

    print(f'{TARGET}: {len(selected)} filas')
    print(f'primera fila: {selected[0][0]}')
    print('[' + ','.join(f'{value:.6f}' for value in selected[0][1]) + ']')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
