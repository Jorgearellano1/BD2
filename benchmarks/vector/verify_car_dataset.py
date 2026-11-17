"""Verifica que el dataset de vehiculos este completo."""

from __future__ import annotations

import csv
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'datasets' / 'cars'


def main() -> int:
    products = list(csv.DictReader(open(DATA / 'products.csv', encoding='utf-8'), delimiter=';'))
    images = list((DATA / 'images').glob('*.jpg'))
    raw = (DATA / 'embeddings.bin').read_bytes()
    magic, count, dimension = struct.unpack_from('<8sII', raw, 0)
    codebook = json.loads((DATA / 'codebook.json').read_text(encoding='utf-8'))
    bovw = sum(1 for _ in open(DATA / 'bovw.csv', encoding='utf-8'))

    checks = {
        'productos': (len(products), 12000),
        'imagenes': (len(images), len(products)),
        'embeddings': (count, len(products)),
        'dim_embedding': (dimension, 512),
        'codebook': (len(codebook['centroids']), 64),
        'filas_bovw': (bovw, len(products)),
    }
    ok = True

    for name, (value, expected) in checks.items():
        status = 'OK' if value == expected else 'ERROR'
        print(f'{name}: {value:,} (esperado {expected:,}) [{status}]')
        ok = ok and value == expected

    print('magic:', magic)
    print('RESULTADO:', 'OK' if ok else 'ERROR')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
