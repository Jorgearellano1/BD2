from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles

from engine.catalog import Catalog

from api import schemas
from api.recommender import RecommenderService
from api.service import DatabaseService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = Path(os.environ.get('STORAGE_DIR', str(PROJECT_ROOT / 'storage')))
CATALOG_PATH = STORAGE_DIR / 'catalog.json'
TMP_DIR = STORAGE_DIR / 'tmp'
PAGE_SIZE = int(os.environ.get('PAGE_SIZE', '4096'))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

catalog = Catalog(str(CATALOG_PATH), str(STORAGE_DIR))
service = DatabaseService(catalog, PAGE_SIZE)
recommender = RecommenderService(service)

MEDIA_DIR = PROJECT_ROOT / 'data' / 'datasets' / 'cars' / 'images'

app = FastAPI(title='MiniDB API', version='1.0')


@app.get('/api/health')
def health() -> schemas.HealthResponse:
    return schemas.HealthResponse(status='ok')


@app.get('/api/tables')
def list_tables() -> schemas.TablesResponse:
    return schemas.TablesResponse(**service.tables())


@app.get('/api/indexes')
def list_indexes() -> schemas.IndexesResponse:
    return schemas.IndexesResponse(**service.indexes())


@app.post('/api/query')
def execute_query(request: schemas.QueryRequest) -> schemas.QueryResponse:
    return schemas.QueryResponse(**service.execute_sql(request.sql, request.page_size))


@app.get('/api/results/{result_id}')
def result_page(result_id: str, page: int = Query(default=1, ge=1)) -> schemas.QueryResponse:
    return schemas.QueryResponse(**service.result_page(result_id, page))


@app.post('/api/tables/reorganize')
def reorganize(request: schemas.ReorganizeRequest) -> schemas.QueryResponse:
    return schemas.QueryResponse(**service.reorganize(request.table))


@app.post('/api/recommend')
def recommend(request: schemas.RecommendRequest) -> dict:
    return recommender.recommend(request)


@app.get('/api/recommend/samples')
def recommend_samples(limit: int = Query(default=12, ge=1, le=30)) -> dict:
    return recommender.samples(limit)


if MEDIA_DIR.exists():
    app.mount('/media', StaticFiles(directory=str(MEDIA_DIR)), name='media')

app.mount('/', StaticFiles(directory=PROJECT_ROOT / 'frontend', html=True), name='frontend')
