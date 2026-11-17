"""
Genera el CSV BoVW de las 600 imagenes para el ejemplo del frontend.

Lee los descriptores SIFT del dataset, construye un codebook con K-Means y
escribe una fila por imagen con su histograma de palabras visuales.

Salidas en data/datasets/multimedia/:
    bovw.csv            id;categoria;histograma
    bovw_codebook.json  centroides y parametros del codebook
    bovw_manifest.json  resumen del artefacto

Uso:
    python benchmarks/vector/build_bovw_csv.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from vectorindex import bovw, features  # noqa: E402

DATA = ROOT / 'data' / 'datasets' / 'multimedia'
SIFT = DATA / 'sift_descriptors.bin'
MANIFEST = DATA / 'embeddings_manifest.csv'
TARGET = DATA / 'bovw.csv'
CODEBOOK = DATA / 'bovw_codebook.json'
SUMMARY = DATA / 'bovw_manifest.json'

WORDS = 64
SAMPLE = 4000
ITERATIONS = 5
SEED = 42


def main() -> int:
    started = time.perf_counter()
    descriptors = features.load_sift_descriptors(str(SIFT))
    rows = features.load_embedding_manifest(str(MANIFEST))

    if len(descriptors) != len(rows):
        raise SystemExit('El numero de imagenes y descriptores no coincide.')

    flat = [descriptor for image in descriptors for descriptor in image]
    rng = random.Random(SEED)
    sample = rng.sample(flat, min(SAMPLE, len(flat)))

    codebook = bovw.build_codebook(
        sample,
        words=min(WORDS, len(sample)),
        iterations=ITERATIONS,
        seed=SEED,
    )

    histograms = [bovw.histogram(image, codebook) for image in descriptors]

    with open(TARGET, 'w', encoding='utf-8', newline='') as handle:
        for row, histogram in zip(rows, histograms):
            handle.write(
                f'{row.id};{row.category};{bovw.histogram_document(histogram)}\n'
            )

    with open(CODEBOOK, 'w', encoding='utf-8') as handle:
        json.dump({
            'words': codebook.size,
            'dimension': codebook.dimension,
            'iterations': ITERATIONS,
            'sample': len(sample),
            'seed': SEED,
            'centroids': [list(centroid) for centroid in codebook.centroids],
        }, handle, ensure_ascii=False)

    query_words = sorted(
        histograms[0].items(), key=lambda item: (-item[1], item[0]),
    )[:20]
    query_tokens = []

    for word, count in query_words:
        query_tokens.extend([bovw.word_token(word)] * min(count, 3))

    query = ' '.join(query_tokens)

    with open(SUMMARY, 'w', encoding='utf-8') as handle:
        json.dump({
            'images': len(rows),
            'descriptors': len(flat),
            'codebook_words': codebook.size,
            'codebook_dimension': codebook.dimension,
            'csv': TARGET.name,
            'codebook': CODEBOOK.name,
            'query_example': query,
        }, handle, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - started
    print(f'{TARGET}: {len(rows)} filas')
    print(f'{CODEBOOK}: {codebook.size} palabras de {codebook.dimension}D')
    print(f'Tiempo: {elapsed:.1f} s')
    print(f'Consulta de ejemplo (imagen 0): {query}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
