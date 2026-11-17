r"""
Regenera el dataset del modulo multimedia y de busqueda vectorial.

Artefactos en data/datasets/multimedia/:

  1. images/<clase>/<archivo>.jpg   copia de las imagenes originales.
  2. sift_descriptors.bin            descriptores locales SIFT (128D, uint8).
  3. embeddings.bin                  embeddings densos (512D, float32).
  4. embeddings_manifest.csv         id,category,filename en el orden de los vectores.
  5. manifest.json                   metadatos del dataset.

Formato de sift_descriptors.bin (little-endian):
    magic  : 8 bytes  b'BD2SIFT1'
    uint32 : num_images
    uint32 : dim (128)
    por cada imagen, en orden:
        uint32 : count
        count*dim valores uint8 (un byte por componente del descriptor)

Formato de embeddings.bin (little-endian):
    magic  : 8 bytes  b'BD2EMB01'
    uint32 : num_images
    uint32 : dim (512)
    num_images*dim float32 (fila = orden del manifiesto, vector normalizado L2)

SIFT se extrae con OpenCV sobre grises reescalados a 224x224 y los embeddings
con ResNet-18 (fc = Identity) sobre RGB reescalado a 224x224 con la
normalizacion de ImageNet. OpenCV y PyTorch solo hacen falta aqui: el motor
lee los archivos con struct y no depende de ellas.

Uso:
    python benchmarks/vector/build_dataset.py
    python benchmarks/vector/build_dataset.py --images-dir <dir> --out-dir <dir>
    python benchmarks/vector/build_dataset.py --device cpu
    python benchmarks/vector/build_dataset.py --copy-from <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = ROOT / 'data' / 'datasets' / 'multimedia' / 'images'
DEFAULT_OUT_DIR = ROOT / 'data' / 'datasets' / 'multimedia'

SIFT_MAGIC = b'BD2SIFT1'
EMB_MAGIC = b'BD2EMB01'
SIFT_DIM = 128
EMB_DIM = 512
MAX_DESCRIPTORS = 150
IMAGE_SIZE = 224

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _die(message: str) -> None:
    print(f'[build_dataset] ERROR: {message}', file=sys.stderr)
    raise SystemExit(2)


def _import_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depende del entorno
        _die(
            'falta OpenCV (cv2). Instala opencv-python para regenerar el '
            f'dataset; el motor no lo necesita (error original: {exc})'
        )
    return cv2


def _import_torch():
    try:
        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depende del entorno
        _die(
            'falta PyTorch/torchvision. Instala torch y torchvision para '
            f'regenerar el dataset; el motor no los necesita (error original: {exc})'
        )
    return torch, torchvision


def list_images(images_dir: Path) -> list[tuple[str, str, Path]]:
    """Devuelve (clase, archivo, ruta) ordenados por clase y luego por nombre."""
    if not images_dir.is_dir():
        _die(f'no existe el directorio de imagenes: {images_dir}')
    entries: list[tuple[str, str, Path]] = []
    classes = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    for category in classes:
        class_dir = images_dir / category
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        for path in files:
            entries.append((category, path.name, path))
    if not entries:
        _die(f'no se encontraron imagenes bajo {images_dir}')
    return entries


def copy_images(source_dir: Path, images_dir: Path) -> int:
    """Copia literal (byte a byte) las imagenes de origen a images/<clase>/."""
    copied = 0
    for category, filename, path in list_images(source_dir):
        target_dir = images_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        with open(path, 'rb') as src, open(target, 'wb') as dst:
            dst.write(src.read())
        copied += 1
    return copied


def build_sift(entries: list[tuple[str, str, Path]], out_path: Path) -> tuple[int, int]:
    """Extrae SIFT (128D) y lo serializa como uint8. Devuelve (total, bytes)."""
    cv2 = _import_cv2()
    import numpy as np  # noqa: PLC0415

    sift = cv2.SIFT_create(nfeatures=MAX_DESCRIPTORS)
    payload = bytearray()
    total = 0
    for index, (_category, _filename, path) in enumerate(entries):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'[build_dataset] aviso: no se pudo leer {path.name}, count=0')
            payload += struct.pack('<I', 0)
            continue
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None or len(descriptors) == 0:
            payload += struct.pack('<I', 0)
            continue
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.shape[1] != SIFT_DIM:
            _die(f'SIFT devolvio dim={descriptors.shape[1]}, se esperaba {SIFT_DIM}')
        if len(descriptors) > MAX_DESCRIPTORS:
            # OpenCV usa nfeatures como cota superior aproximada y puede devolver
            # 1-2 keypoints de mas: nos quedamos con los MAX_DESCRIPTORS mas
            # fuertes (por respuesta) respetando el orden de deteccion original.
            responses = np.array([kp.response for kp in keypoints], dtype=np.float64)
            keep = np.sort(np.argsort(-responses, kind='stable')[:MAX_DESCRIPTORS])
            descriptors = descriptors[keep]
        quantized = np.clip(np.round(descriptors), 0, 255).astype(np.uint8)
        count = int(quantized.shape[0])
        payload += struct.pack('<I', count)
        payload += quantized.tobytes(order='C')
        total += count
        if (index + 1) % 100 == 0:
            print(f'[build_dataset]   SIFT {index + 1}/{len(entries)} imagenes, '
                  f'{total} descriptores', flush=True)

    header = SIFT_MAGIC + struct.pack('<II', len(entries), SIFT_DIM)
    out_path.write_bytes(header + bytes(payload))
    return total, os.path.getsize(out_path)


def build_embeddings(entries: list[tuple[str, str, Path]], out_path: Path,
                     device: str) -> tuple[int, int]:
    """Extrae embeddings densos ResNet-18 (512D, L2-normalizados)."""
    torch, torchvision = _import_torch()
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from torchvision import models, transforms  # noqa: PLC0415

    if device == 'cuda' and not torch.cuda.is_available():
        print('[build_dataset] aviso: CUDA no disponible, se usa CPU')
        device = 'cpu'
    dev = torch.device(device)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(dev).eval()

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vectors = np.zeros((len(entries), EMB_DIM), dtype=np.float32)
    with torch.no_grad():
        for index, (_category, _filename, path) in enumerate(entries):
            with Image.open(path) as handle:
                pil_image = handle.convert('RGB')
                tensor = preprocess(pil_image).unsqueeze(0).to(dev)
            feature = resnet(tensor).cpu().numpy().flatten().astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm > 0:
                feature = feature / norm
            vectors[index] = feature
            if (index + 1) % 100 == 0:
                print(f'[build_dataset]   embeddings {index + 1}/{len(entries)} imagenes',
                      flush=True)

    header = EMB_MAGIC + struct.pack('<II', len(entries), EMB_DIM)
    out_path.write_bytes(header + vectors.astype('<f4').tobytes(order='C'))
    return len(entries), os.path.getsize(out_path)


def write_manifest_csv(entries: list[tuple[str, str, Path]], out_path: Path) -> None:
    with open(out_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['id', 'category', 'filename'])
        for index, (category, filename, _path) in enumerate(entries):
            writer.writerow([index, category, filename])


def write_manifest_json(classes: list[str], images: int, descriptors_total: int,
                        sift_bytes: int, emb_bytes: int, out_path: Path) -> None:
    manifest = {
        'source': 'data/datasets/multimedia/images',
        'images': images,
        'classes': classes,
        'sift': {
            'file': 'sift_descriptors.bin',
            'dim': SIFT_DIM,
            'max_descriptors': MAX_DESCRIPTORS,
            'descriptors_total': descriptors_total,
            'bytes': sift_bytes,
        },
        'embeddings': {
            'file': 'embeddings.bin',
            'dim': EMB_DIM,
            'bytes': emb_bytes,
        },
        'extracted_with': {
            'sift': 'opencv SIFT',
            'embeddings': 'torchvision resnet18',
        },
        'created': datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Genera el dataset multimedia (SIFT + embeddings) del proyecto BD2.')
    parser.add_argument('--images-dir', type=Path, default=DEFAULT_IMAGES_DIR,
                        help='directorio con las imagenes en <clase>/<archivo> (por defecto: '
                             'data/datasets/multimedia/images)')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                        help='directorio de salida de los artefactos (por defecto: '
                             'data/datasets/multimedia)')
    parser.add_argument('--device', default='cpu',
                        help='dispositivo para ResNet-18: cpu o cuda (por defecto: cpu)')
    parser.add_argument('--copy-from', type=Path, default=None,
                        help='directorio de origen a copiar literalmente en <out-dir>/images')
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_from is not None:
        copied = copy_images(args.copy_from.resolve(), out_dir / 'images')
        print(f'[build_dataset] {copied} imagenes copiadas a {out_dir / "images"}', flush=True)

    entries = list_images(images_dir)
    classes = sorted({category for category, _filename, _path in entries})
    print(f'[build_dataset] {len(entries)} imagenes en {len(classes)} clases: '
          f'{", ".join(classes)}', flush=True)

    sift_path = out_dir / 'sift_descriptors.bin'
    emb_path = out_dir / 'embeddings.bin'
    csv_path = out_dir / 'embeddings_manifest.csv'
    json_path = out_dir / 'manifest.json'

    descriptors_total, sift_bytes = build_sift(entries, sift_path)
    print(f'[build_dataset] SIFT: {descriptors_total} descriptores -> {sift_path}', flush=True)

    num_images, emb_bytes = build_embeddings(entries, emb_path, args.device)
    print(f'[build_dataset] embeddings: {num_images} vectores de {EMB_DIM}D -> {emb_path}',
          flush=True)

    write_manifest_csv(entries, csv_path)
    write_manifest_json(classes, num_images, descriptors_total, sift_bytes, emb_bytes,
                        json_path)

    print('[build_dataset] resumen:')
    print(f'  imagenes          : {num_images}')
    print(f'  descriptores SIFT : {descriptors_total} (dim {SIFT_DIM}, '
          f'max {MAX_DESCRIPTORS}/imagen)')
    print(f'  sift_descriptors.bin    : {sift_bytes} bytes')
    print(f'  embeddings.bin           : {emb_bytes} bytes')
    print(f'  embeddings_manifest.csv  : {os.path.getsize(csv_path)} bytes')
    print(f'  manifest.json            : {os.path.getsize(json_path)} bytes')
    print(f'  total artefactos         : '
          f'{sift_bytes + emb_bytes + os.path.getsize(csv_path) + os.path.getsize(json_path)} bytes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
