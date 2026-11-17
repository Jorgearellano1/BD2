"""
Construye el corpus real del Experimento 6 a partir de libros en espanol de
dominio publico de Project Gutenberg.

Pasos (reproducibles con la misma semilla):
  1. Descarga el catalogo pg_catalog.csv del mirror gutenberg.pglaf.org.
  2. Selecciona libros con Language == 'es' y Type == 'Text', en orden de
     numero de libro, hasta reunir suficientes parrafos.
  3. Descarga cada libro (texto plano UTF-8), elimina la cabecera y el pie de
     licencia de Gutenberg y divide en parrafos (separados por linea vacia).
  4. Filtra parrafos de 200 a 1000 caracteres con mayoria de letras y corta
     el cuerpo a 1000 bytes UTF-8 en limite de palabra (CHAR(1000) del motor).
  5. Elimina duplicados, baraja con semilla fija (asi los prefijos de 1.000,
     2.500, ... documentos son muestras aleatorias) y escribe el CSV.

Los archivos descargados se guardan en data/raw/gutenberg_es/ (cache, no se
versiona). Las descargas se hacen con curl, de una en una y con pausa.

Uso:
    .venv/bin/python benchmarks/text/build_corpus.py --target 32000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw' / 'gutenberg_es'
OUT_CSV = ROOT / 'data' / 'datasets' / 'corpus_gutenberg_es.csv'
OUT_MANIFEST = ROOT / 'data' / 'datasets' / 'corpus_gutenberg_es.manifest.json'
MIRROR = 'https://gutenberg.pglaf.org'
CATALOG_URL = f'{MIRROR}/cache/epub/feeds/pg_catalog.csv'

MIN_CHARS = 200
MAX_CHARS = 1000
MAX_BYTES = 1000
TITLE_BYTES = 80

_START = re.compile(r'\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*', re.I)
_END = re.compile(r'\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG', re.I)


def fetch(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    subprocess.run(['curl', '-sSfL', '--retry', '2', '-o', str(tmp), url], check=True)
    tmp.replace(path)
    time.sleep(1.0)  # pausa entre descargas por cortesia con el mirror
    return path


def cut_bytes(text: str, limit: int) -> str:
    data = text.encode('utf-8')
    if len(data) <= limit:
        return text
    cut = data[:limit].decode('utf-8', errors='ignore')
    space = cut.rfind(' ')
    return cut[:space] if space > limit // 2 else cut


def paragraphs(raw: str) -> list[str]:
    start = _START.search(raw)
    end = _END.search(raw)
    body = raw[start.end() if start else 0:end.start() if end else len(raw)]
    body = body.replace('\r\n', '\n').replace('\r', '\n')
    output = []
    for block in re.split(r'\n\s*\n', body):
        text = ' '.join(block.split())
        text = text.replace('_', '')  # enfasis de Gutenberg: _palabra_
        if not (MIN_CHARS <= len(text) <= MAX_CHARS * 2):
            continue
        letters = sum(ch.isalpha() for ch in text)
        if letters < 0.6 * len(text):
            continue
        if text.upper() == text:
            continue
        output.append(cut_bytes(text, MAX_BYTES))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=32000)
    parser.add_argument('--seed', type=int, default=2042)
    parser.add_argument('--max-books', type=int, default=120)
    args = parser.parse_args()

    catalog_path = fetch(CATALOG_URL, RAW_DIR / 'pg_catalog.csv')
    with open(catalog_path, encoding='utf-8', newline='') as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get('Language', '').strip() == 'es' and row.get('Type') == 'Text'
        ]
    rows.sort(key=lambda row: int(row['Text#']))

    seen: set[str] = set()
    documents: list[tuple[int, str, str]] = []
    books = []
    for row in rows:
        if len(documents) >= args.target or len(books) >= args.max_books:
            break
        book_id = int(row['Text#'])
        url = f'{MIRROR}/cache/epub/{book_id}/pg{book_id}.txt'
        try:
            path = fetch(url, RAW_DIR / f'pg{book_id}.txt')
        except subprocess.CalledProcessError:
            print(f'omitido {book_id}: descarga fallida', file=sys.stderr)
            continue
        raw = path.read_text(encoding='utf-8', errors='replace')
        title = ' '.join(row['Title'].split())
        added = 0
        for text in paragraphs(raw):
            digest = hashlib.sha1(text.encode('utf-8')).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            documents.append((book_id, title, text))
            added += 1
        books.append({'id': book_id, 'title': title, 'authors': row.get('Authors', ''),
                      'issued': row.get('Issued', ''), 'paragraphs': added})
        print(f'libro {book_id}: {added} parrafos (total {len(documents)})', flush=True)

    random.Random(args.seed).shuffle(documents)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['id', 'libro', 'titulo', 'cuerpo'])
    for doc_id, (book_id, title, text) in enumerate(documents, start=1):
        writer.writerow([doc_id, book_id, cut_bytes(title, TITLE_BYTES), text])
    data = buffer.getvalue().encode('utf-8')
    OUT_CSV.write_bytes(data)

    manifest = {
        'source': 'Project Gutenberg (dominio publico), mirror ' + MIRROR,
        'catalog': CATALOG_URL,
        'filters': {'language': 'es', 'type': 'Text', 'min_chars': MIN_CHARS,
                    'max_bytes': MAX_BYTES, 'dedupe': 'sha1 del parrafo'},
        'seed': args.seed,
        'documents': len(documents),
        'books': books,
        'csv_sha256': hashlib.sha256(data).hexdigest(),
        'csv_bytes': len(data),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(documents)} documentos de {len(books)} libros -> {OUT_CSV}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
