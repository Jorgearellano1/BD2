"""
Servicio del recomendador multimodal de vehiculos.

Combina los tres accesos del motor:
    - BoVW sobre el indice invertido.
    - Embeddings densos sobre el indice IVF.
    - Descripcion textual sobre el indice invertido.

La extraccion de caracteristicas de una imagen subida usa el mismo codebook y
el mismo ResNet-18 con el que se genero el dataset.
"""

from __future__ import annotations

import base64
import io
import json
import math
from collections import Counter
from pathlib import Path

from fastapi import HTTPException

from vectorindex import bovw

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'datasets' / 'cars'
MODELS_DIR = ROOT / 'data' / 'models'
CODEBOOK_PATH = DATA_DIR / 'codebook.json'
MODEL_PATH = MODELS_DIR / 'resnet18.pth'
TABLE = 'vehiculos'
IMAGE_FEATURE_SIZE = 224


class FeatureExtractor:
    """Extrae histograma BoVW y embedding ResNet-18 de una imagen."""

    def __init__(self) -> None:
        self._centroids = None
        self._model = None
        self._transform = None
        self._sift = None

    def _codebook(self):
        if self._centroids is None:
            if not CODEBOOK_PATH.exists():
                raise HTTPException(
                    503,
                    'Falta el codebook. Genera el dataset con '
                    'build_car_dataset.py.',
                )

            import numpy as np

            with open(CODEBOOK_PATH, encoding='utf-8') as handle:
                data = json.load(handle)

            self._centroids = np.array(data['centroids'], dtype=np.float32)

        return self._centroids

    def _load_model(self):
        if self._model is not None:
            return

        if not MODEL_PATH.exists():
            raise HTTPException(
                503,
                'Falta data/models/resnet18.pth. Genera el dataset con '
                'build_car_dataset.py.',
            )

        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise HTTPException(
                503,
                'Instala las dependencias de vision '
                '(pip install -r requirements-vision.txt).',
            ) from exc

        model = models.resnet18(weights=None)
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=False)
        model.fc = torch.nn.Identity()
        model.eval()

        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_FEATURE_SIZE, IMAGE_FEATURE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def _open(image_bytes: bytes):
        try:
            from PIL import Image
        except ImportError as exc:
            raise HTTPException(
                503,
                'Instala Pillow para procesar imagenes.',
            ) from exc

        try:
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as exc:
            raise HTTPException(400, 'La imagen no es valida.') from exc

    def histogram(self, image_bytes: bytes) -> str:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise HTTPException(
                503,
                'Instala opencv-python-headless para extraer SIFT.',
            ) from exc

        image = self._open(image_bytes).convert('L').resize(
            (IMAGE_FEATURE_SIZE, IMAGE_FEATURE_SIZE),
        )
        matrix = np.array(image)

        if self._sift is None:
            self._sift = cv2.SIFT_create(nfeatures=40)

        _, descriptors = self._sift.detectAndCompute(matrix, None)

        if descriptors is None or len(descriptors) == 0:
            return ''

        descriptors = descriptors.astype(np.float32) / 255.0
        centroids = self._codebook()
        distances = (
            (descriptors ** 2).sum(axis=1)[:, None]
            - 2 * descriptors @ centroids.T
            + (centroids ** 2).sum(axis=1)[None, :]
        )
        labels = distances.argmin(axis=1)
        counts = Counter(int(label) for label in labels)
        return bovw.histogram_document(dict(counts))

    def embedding(self, image_bytes: bytes) -> list[float]:
        self._load_model()

        import torch

        image = self._open(image_bytes)
        tensor = self._transform(image).unsqueeze(0)

        with torch.no_grad():
            vector = self._model(tensor).cpu().numpy()[0]

        norm = float((vector ** 2).sum() ** 0.5)
        if norm:
            vector = vector / norm

        return [float(value) for value in vector]

    def features(self, image_bytes: bytes) -> tuple[str, list[float]]:
        return self.histogram(image_bytes), self.embedding(image_bytes)


class RecommenderService:
    """Consultas hibridas sobre la tabla de vehiculos."""

    def __init__(self, database) -> None:
        self.database = database
        self.extractor = FeatureExtractor()

    # Utilidades

    @staticmethod
    def decode_image(value: str) -> bytes:
        if ',' in value and value.strip().lower().startswith('data:'):
            value = value.split(',', 1)[1]

        try:
            return base64.b64decode(value, validate=True)
        except Exception as exc:
            raise HTTPException(400, 'La imagen base64 no es valida.') from exc

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _filters(max_price, lat, lon, radius_km) -> tuple[list[str], float | None]:
        clauses: list[str] = []
        radius = radius_km

        if max_price is not None:
            clauses.append(f'precio <= {float(max_price):.2f}')

        if lat is not None and lon is not None:
            radius = float(radius_km or 50.0)
            delta_lat = radius / 111.0
            delta_lon = radius / (111.0 * max(0.1, math.cos(math.radians(lat))))
            clauses.append(f'latitud >= {lat - delta_lat:.6f}')
            clauses.append(f'latitud <= {lat + delta_lat:.6f}')
            clauses.append(f'longitud >= {lon - delta_lon:.6f}')
            clauses.append(f'longitud <= {lon + delta_lon:.6f}')

        return clauses, radius

    def _run(self, sql: str) -> dict:
        payload = self.database.execute_sql(sql, page_size=50)

        if not payload.get('success', False):
            raise HTTPException(400, payload.get('error') or 'Consulta fallida.')

        return payload

    def _rows(self, payload: dict, lat, lon, radius=None) -> list[dict]:
        columns = payload.get('columns', [])
        index = {name: position for position, name in enumerate(columns)}
        results = []

        for row in payload.get('rows', []):
            item = {
                name: row[position]
                for name, position in index.items()
                if name in {
                    'id', 'titulo', 'descripcion', 'precio', 'categoria',
                    'tienda', 'ciudad', 'estado', 'latitud', 'longitud',
                    'imagen', 'score',
                }
            }
            item['imagen_url'] = f"/media/{item.get('imagen', '')}"

            if lat is not None and lon is not None:
                try:
                    distance = round(self._distance_km(
                        float(lat), float(lon),
                        float(item['latitud']), float(item['longitud']),
                    ), 1)
                except (KeyError, TypeError, ValueError):
                    distance = None

                if distance is not None:
                    if radius is not None and distance > radius:
                        continue
                    item['distancia_km'] = distance

            results.append(item)

        return results

    @staticmethod
    def _summary(payload: dict) -> dict:
        return {
            'plan': payload.get('plan'),
            'strategy': payload.get('strategy'),
            'metrics': payload.get('metrics'),
            'io_breakdown': payload.get('io_breakdown'),
        }

    # API

    def recommend(self, request) -> dict:
        has_image = bool(request.image_base64)
        has_text = bool((request.description or '').strip())

        if not has_image and not has_text:
            raise HTTPException(
                400, 'Sube una imagen o escribe una descripcion.',
            )

        clauses, radius = self._filters(
            request.max_price, request.lat, request.lon, request.radius_km,
        )
        where = (' AND ' + ' AND '.join(clauses)) if clauses else ''
        where_clause = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        columns = (
            'id, titulo, descripcion, precio, categoria, tienda, ciudad, '
            'estado, latitud, longitud, imagen'
        )
        limit = max(1, min(int(request.limit), 20))
        fetch_limit = (
            min(limit * 4, 50)
            if request.lat is not None and request.lon is not None
            else limit
        )
        response = {
            'success': True,
            'mode': 'image' if has_image else 'text',
            'limit': limit,
            'filters': {
                'max_price': request.max_price,
                'lat': request.lat,
                'lon': request.lon,
                'radius_km': radius,
            },
            'bovw': None,
            'embeddings': None,
            'text': None,
        }

        if has_image:
            image_bytes = self.decode_image(request.image_base64)
            document, embedding = self.extractor.features(image_bytes)

            embeddings_sql = (
                'SET ivf.probes = 10;'
                f'SELECT {columns}, embedding <=> '
                f"'[{','.join(f'{value:.7f}' for value in embedding)}]' AS score "
                f'FROM {TABLE} {where_clause} ORDER BY score ASC LIMIT {fetch_limit};'
            )
            embeddings_payload = self._run(embeddings_sql)
            response['embeddings'] = {
                'results': self._rows(
                    embeddings_payload, request.lat, request.lon, radius,
                )[:limit],
                **self._summary(embeddings_payload),
            }

            if document:
                bovw_sql = (
                    f'SELECT {columns}, COSINE_SCORE() AS score FROM {TABLE} '
                    f"WHERE palabras MATCH '{self._escape(document)}'{where} "
                    f'ORDER BY score DESC LIMIT {fetch_limit};'
                )
                bovw_payload = self._run(bovw_sql)
                response['bovw'] = {
                    'results': self._rows(
                        bovw_payload, request.lat, request.lon, radius,
                    )[:limit],
                    **self._summary(bovw_payload),
                }

            response['query'] = {
                'histogram_terms': len(document.split()),
                'embedding_dim': len(embedding),
            }

            bovw_ids = {
                item['id'] for item in (response['bovw'] or {}).get('results', [])
            }
            embedding_ids = {
                item['id'] for item in response['embeddings']['results']
            }
            response['comparison'] = {
                'overlap': len(bovw_ids & embedding_ids),
                'limit': limit,
            }
        else:
            text_sql = (
                f'SELECT {columns}, COSINE_SCORE() AS score FROM {TABLE} '
                f"WHERE descripcion MATCH '{self._escape(request.description.strip())}'"
                f'{where} ORDER BY score DESC LIMIT {fetch_limit};'
            )
            text_payload = self._run(text_sql)
            response['text'] = {
                'results': self._rows(
                    text_payload, request.lat, request.lon, radius,
                )[:limit],
                **self._summary(text_payload),
            }

        return response

    def samples(self, limit: int = 12) -> dict:
        payload = self._run(
            f'SELECT id, titulo, precio, categoria, tienda, ciudad, estado, '
            f'imagen FROM {TABLE} ORDER BY id LIMIT {max(1, min(limit, 30))};'
        )
        return {'products': self._rows(payload, None, None)}
