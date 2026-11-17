r"""
Verifica los artefactos del dataset multimedia (data/datasets/multimedia).

Uso:
    python benchmarks/vector/verify_dataset.py
    python benchmarks/vector/verify_dataset.py --out-dir data/datasets/multimedia

Solo usa la biblioteca estandar: el motor no necesita OpenCV ni PyTorch para
leer sift_descriptors.bin ni embeddings.bin.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / 'data' / 'datasets' / 'multimedia'

SIFT_MAGIC = b'BD2SIFT1'
EMB_MAGIC = b'BD2EMB01'


def check(condition: bool, message: str) -> bool:
    print(f'  [{"OK" if condition else "FAIL"}] {message}')
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description='Verifica el dataset multimedia de BD2.')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    ok = True

    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))
    sift_path = out_dir / manifest['sift']['file']
    emb_path = out_dir / manifest['embeddings']['file']

    print('sift_descriptors.bin')
    data = sift_path.read_bytes()
    magic, num_images, dim = struct.unpack_from('<8sII', data, 0)
    offset = 16
    counts: list[int] = []
    for _ in range(num_images):
        (count,) = struct.unpack_from('<I', data, offset)
        offset += 4 + count * dim
        counts.append(count)
    ok &= check(magic == SIFT_MAGIC, f'magic == {SIFT_MAGIC!r}')
    ok &= check(num_images == 600, f'num_images == 600 (obtenido {num_images})')
    ok &= check(dim == 128, f'dim == 128 (obtenido {dim})')
    ok &= check(offset == len(data), f'el archivo se consume completo ({offset}/{len(data)})')
    ok &= check(sum(counts) == manifest['sift']['descriptors_total'],
                f"sum(count) == descriptors_total ({sum(counts)})")
    ok &= check(max(counts) <= manifest['sift']['max_descriptors'],
                f"count <= max_descriptors ({max(counts)})")
    ok &= check(len(data) == manifest['sift']['bytes'],
                f"bytes == manifest ({len(data)})")

    print('embeddings.bin')
    edata = emb_path.read_bytes()
    emagic, enum, edim = struct.unpack_from('<8sII', edata, 0)
    vectors = struct.unpack_from(f'<{enum * edim}f', edata, 16)
    norms = [sum(v * v for v in vectors[i * edim:(i + 1) * edim]) ** 0.5 for i in range(enum)]
    ok &= check(emagic == EMB_MAGIC, f'magic == {EMB_MAGIC!r}')
    ok &= check(enum == 600, f'num_images == 600 (obtenido {enum})')
    ok &= check(edim == 512, f'dim == 512 (obtenido {edim})')
    ok &= check(len(edata) == 16 + enum * edim * 4,
                f'tamano == 16 + {enum}*{edim}*4 ({len(edata)})')
    ok &= check(max(abs(n - 1.0) for n in norms) < 1e-5,
                f'vectores L2 normalizados (max |norm-1| = {max(abs(n - 1.0) for n in norms):.2e})')
    ok &= check(len(edata) == manifest['embeddings']['bytes'],
                f"bytes == manifest ({len(edata)})")

    print('embeddings_manifest.csv')
    with open(out_dir / 'embeddings_manifest.csv', encoding='utf-8', newline='') as handle:
        rows = list(csv.reader(handle))
    ok &= check(rows[0] == ['id', 'category', 'filename'], f'cabecera exacta ({rows[0]})')
    ok &= check(len(rows) - 1 == 600, f'600 filas de datos (obtenido {len(rows) - 1})')
    ids = [int(row[0]) for row in rows[1:]]
    ok &= check(ids == list(range(600)), 'ids correlativos 0..599')
    pairs = [(row[1], row[2]) for row in rows[1:]]
    ok &= check(pairs == sorted(pairs), 'orden por clase (alfabetico) y luego por archivo')

    print('imagenes')
    images_dir = out_dir / 'images'
    files = sorted(p for p in images_dir.rglob('*') if p.is_file())
    ok &= check(len(files) == 600, f'600 archivos jpg (obtenido {len(files)})')
    ok &= check(sorted(d.name for d in images_dir.iterdir() if d.is_dir()) == manifest['classes'],
                f"clases == {manifest['classes']}")

    print('manifest.json')
    ok &= check(manifest['images'] == 600, 'images == 600')
    ok &= check(
        manifest['source'] == 'data/datasets/multimedia/images',
        'source correcto',
    )
    ok &= check(manifest['extracted_with'] == {'sift': 'opencv SIFT',
                                               'embeddings': 'torchvision resnet18'},
                'extracted_with correcto')

    total = sum(p.stat().st_size for p in out_dir.rglob('*') if p.is_file())
    print(f'total del dataset (incluye imagenes): {total} bytes')
    print('RESULTADO:', 'OK' if ok else 'FALLO')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
