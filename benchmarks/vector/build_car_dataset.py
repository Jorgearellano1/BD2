"""
Genera el dataset de vehiculos para el recomendador multimodal.

Lee el CSV de Craigslist (Kaggle: austinreese/craigslist-carstrucks-data),
selecciona un subconjunto balanceado por tipo, descarga las imagenes, extrae
SIFT/BoVW y embeddings con ResNet-18, y escribe todos los artefactos en
data/datasets/cars/.

Uso:
    python benchmarks/vector/build_car_dataset.py --target 12000
    python benchmarks/vector/build_car_dataset.py --zip <ruta>/cars.zip --target 12000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from vectorindex import bovw  # noqa: E402

OUT_DIR = ROOT / 'data' / 'datasets' / 'cars'
IMAGES_DIR = OUT_DIR / 'images'
MODELS_DIR = ROOT / 'data' / 'models'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 192
FEATURE_SIZE = 224
WORDS = 64
CODEBOOK_SAMPLE = 20000
SEED = 42
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=str, default=None)
    parser.add_argument('--target', type=int, default=12000)
    parser.add_argument('--per-type', type=int, default=1400)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--width', type=int, default=IMAGE_WIDTH)
    parser.add_argument('--height', type=int, default=IMAGE_HEIGHT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--max-candidates', type=int, default=24000)
    return parser.parse_args()


def find_zip(explicit: str | None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        ROOT / 'data' / 'raw' / 'craigslist' / 'cars.zip',
        Path(os.environ.get('CRAIGSLIST_ZIP', '')),
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise SystemExit(
        'Falta el zip de Craigslist. Descargalo con:\n'
        '  curl -L -o data/raw/craigslist/cars.zip '
        'https://www.kaggle.com/api/v1/datasets/download/'
        'austinreese/craigslist-carstrucks-data'
    )


def clean(value: str, limit: int) -> str:
    value = ' '.join((value or '').split())
    return value.replace(';', ',').replace('\t', ' ')[:limit]


def row_valid(row: dict) -> bool:
    try:
        price = float(row.get('price') or 0)
        lat = float(row.get('lat') or 0)
        lon = float(row.get('long') or 0)
    except ValueError:
        return False

    return (
        500 <= price <= 150000
        and 24 <= lat <= 50
        and -125 <= lon <= -66
        and len((row.get('title') or '').strip()) >= 5
        and len((row.get('description') or '').strip()) >= 20
        and bool((row.get('image_url') or '').strip())
        and bool((row.get('type') or '').strip())
    )


def candidate_from_row(row: dict) -> dict:
    return {
        'title': clean(row.get('title'), 120),
        'description': clean(row.get('description'), 900),
        'price': round(float(row.get('price')), 2),
        'category': clean(row.get('type'), 30).title(),
        'region': clean(row.get('region'), 60),
        'state': clean(row.get('state'), 10).upper(),
        'city': clean(row.get('address_city') or row.get('region'), 60),
        'lat': float(row.get('lat')),
        'lon': float(row.get('long')),
        'seller': clean(row.get('seller_type') or 'owner', 20).capitalize(),
        'url': (row.get('image_url') or '').strip(),
        'year': clean(row.get('year'), 6),
        'manufacturer': clean(row.get('manufacturer'), 30),
        'model': clean(row.get('model'), 50),
    }


def collect_candidates(zip_path: Path, per_type: int, max_rows: int = 0, max_candidates: int = 0) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open('vehicles.csv') as raw:
            text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace')
            reader = csv.DictReader(text)

            for index, row in enumerate(reader):
                if max_rows and index >= max_rows:
                    break

                if not row_valid(row):
                    continue

                category = clean(row.get('type'), 30).title()
                bucket = by_type[category]

                if len(bucket) < per_type:
                    bucket.append(candidate_from_row(row))

                total = sum(len(items) for items in by_type.values())

                if index % 50000 == 0:
                    print(f'  filas leidas={index:,} candidatos={total:,}')

                if max_candidates and total >= max_candidates:
                    print(f'  limite de candidatos alcanzado en fila {index:,}')
                    break

    rng = random.Random(SEED)
    selected: list[dict] = []

    for category in sorted(by_type):
        bucket = by_type[category]
        rng.shuffle(bucket)
        selected.extend(bucket)

    print(
        'Candidatos por tipo: '
        + ', '.join(f'{name}={len(items):,}' for name, items in sorted(by_type.items()))
    )
    return selected


def download_one(index: int, candidate: dict, width: int, height: int) -> dict | None:
    request = urllib.request.Request(
        candidate['url'], headers={'User-Agent': USER_AGENT},
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            content_type = response.headers.get('Content-Type', '')
            data = response.read()

        if 'image' not in content_type.lower() and len(data) < 2000:
            return None

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image = image.convert('RGB')
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            filename = f'{index:05d}.jpg'
            image.save(IMAGES_DIR / filename, 'JPEG', quality=82)

        candidate = dict(candidate)
        candidate['image'] = filename
        return candidate
    except Exception:
        return None


def download_images(
    candidates: list[dict], target: int, workers: int,
    width: int, height: int,
) -> list[dict]:
    selected: list[dict] = []
    pending = list(candidates)
    position = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        next_id = 0

        while len(selected) < target and (pending or futures):
            while pending and len(futures) < workers * 2:
                candidate = pending.pop(0)
                futures[executor.submit(
                    download_one, next_id, candidate, width, height,
                )] = next_id
                next_id += 1

            if not futures:
                break

            for future in as_completed(list(futures), timeout=60):
                futures.pop(future, None)
                result = future.result()
                position += 1

                if result is not None and len(selected) < target:
                    selected.append(result)

                if position % 500 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f'  imagenes={len(selected):,}/{target:,} '
                        f'intentos={position:,} {elapsed:.0f}s'
                    )

                break

    return selected


def numpy_kmeans(points: np.ndarray, k: int, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    centroids = [points[rng.integers(len(points))]]

    for _ in range(1, k):
        distances = np.min(
            ((points[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        total = distances.sum()

        if total <= 0:
            centroids.append(points[rng.integers(len(points))])
            continue

        centroids.append(points[rng.choice(len(points), p=distances / total)])

    centroids = np.array(centroids, dtype=np.float32)

    for _ in range(iterations):
        distances = (
            (points ** 2).sum(axis=1)[:, None]
            - 2 * points @ centroids.T
            + (centroids ** 2).sum(axis=1)[None, :]
        )
        labels = distances.argmin(axis=1)

        for group in range(k):
            mask = labels == group

            if mask.any():
                centroids[group] = points[mask].mean(axis=0)

    return centroids


def extract_features(selected: list[dict]) -> None:
    import cv2
    import torch
    from PIL import Image
    from torchvision import models, transforms

    print('Extrayendo SIFT...')
    descriptors_by_image: list[list] = []
    sift = cv2.SIFT_create(nfeatures=40)

    for position, item in enumerate(selected):
        path = IMAGES_DIR / item['image']
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (FEATURE_SIZE, FEATURE_SIZE))
        _, descriptors = sift.detectAndCompute(image, None)

        if descriptors is None:
            descriptors_by_image.append([])
        else:
            descriptors_by_image.append([
                tuple(float(value) / 255.0 for value in row)
                for row in descriptors
            ])

        if (position + 1) % 2000 == 0:
            print(f'  SIFT {position + 1:,}/{len(selected):,}')

    flat = [descriptor for image in descriptors_by_image for descriptor in image]
    rng = random.Random(SEED)
    sample = rng.sample(flat, min(CODEBOOK_SAMPLE, len(flat)))
    points = np.array(sample, dtype=np.float32)

    print(f'Entrenando codebook K={WORDS} con {len(sample):,} descriptores...')
    centroids = numpy_kmeans(points, min(WORDS, len(sample)), 5, SEED)

    print('Cuantizando histogramas...')
    histograms = []

    for image in descriptors_by_image:
        if not image:
            histograms.append({})
            continue

        matrix = np.array(image, dtype=np.float32)
        distances = (
            (matrix ** 2).sum(axis=1)[:, None]
            - 2 * matrix @ centroids.T
            + (centroids ** 2).sum(axis=1)[None, :]
        )
        labels = distances.argmin(axis=1)
        counts: dict[int, int] = {}

        for label in labels:
            word = int(label)
            counts[word] = counts.get(word, 0) + 1

        histograms.append(counts)

    with open(OUT_DIR / 'bovw.csv', 'w', encoding='utf-8', newline='') as handle:
        for item, histogram in zip(selected, histograms):
            handle.write(
                f"{item['id']};{item['category']};{bovw.histogram_document(histogram)}\n"
            )

    with open(OUT_DIR / 'codebook.json', 'w', encoding='utf-8') as handle:
        json.dump({
            'words': len(centroids),
            'dimension': int(centroids.shape[1]),
            'iterations': 5,
            'seed': SEED,
            'centroids': centroids.tolist(),
        }, handle)

    print('Extrayendo embeddings con ResNet-18...')
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / 'resnet18.pth')
    model.fc = torch.nn.Identity()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((FEATURE_SIZE, FEATURE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(selected), 64):
            batch = selected[start:start + 64]
            tensors = []

            for item in batch:
                with Image.open(IMAGES_DIR / item['image']) as image:
                    tensors.append(transform(image.convert('RGB')))

            output = model(torch.stack(tensors)).cpu().numpy()

            for vector in output:
                norm = np.linalg.norm(vector)
                embeddings.append(vector / norm if norm else vector)

            if (start + 64) % 1280 == 0:
                print(f'  embeddings {min(start + 64, len(selected)):,}/{len(selected):,}')

    embeddings = np.array(embeddings, dtype=np.float32)

    with open(OUT_DIR / 'embeddings.bin', 'wb') as handle:
        handle.write(b'BD2CAR01')
        handle.write(struct.pack('<II', len(embeddings), embeddings.shape[1]))
        handle.write(embeddings.tobytes())

    print(f'Codebook: {len(centroids)} palabras; embeddings: {embeddings.shape}')


def write_products(selected: list[dict]) -> None:
    with open(OUT_DIR / 'products.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            'id', 'titulo', 'descripcion', 'precio', 'categoria', 'tienda',
            'ciudad', 'estado', 'latitud', 'longitud', 'imagen',
        ])

        for item in selected:
            writer.writerow([
                item['id'], item['title'], item['description'], item['price'],
                item['category'], item['region'], item['city'], item['state'],
                item['lat'], item['lon'], item['image'],
            ])


def main() -> int:
    args = parse_args()
    zip_path = find_zip(args.zip)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    IMAGES_DIR.mkdir(parents=True)

    print(f'Leyendo {zip_path} ...')
    candidates = collect_candidates(
        zip_path, args.per_type, args.max_rows, args.max_candidates,
    )
    print(f'Candidatos totales: {len(candidates):,}')

    selected = download_images(
        candidates, args.target, args.workers, args.width, args.height,
    )

    if len(selected) < args.target:
        raise SystemExit(
            f'Solo se descargaron {len(selected)} imagenes de {args.target}.'
        )

    for identifier, item in enumerate(selected):
        item['id'] = identifier

    write_products(selected)
    extract_features(selected)

    categories: dict[str, int] = defaultdict(int)

    for item in selected:
        categories[item['category']] += 1

    with open(OUT_DIR / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump({
            'source': 'austinreese/craigslist-carstrucks-data (CC0)',
            'products': len(selected),
            'image_width': args.width,
            'image_height': args.height,
            'feature_size': FEATURE_SIZE,
            'categories': dict(sorted(categories.items())),
            'words': WORDS,
            'embedding_dim': 512,
            'files': [
                'products.csv', 'bovw.csv', 'codebook.json',
                'embeddings.bin', 'images/',
            ],
        }, handle, indent=2, ensure_ascii=False)

    print(f'\nDataset en {OUT_DIR}: {len(selected):,} vehiculos')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
