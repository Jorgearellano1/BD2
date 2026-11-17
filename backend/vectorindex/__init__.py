"""
Modulo multimedia y busqueda vectorial.

Contenido:
    vectors.py   literales y utilidades sobre vectores densos.
    metrics.py   L2, coseno, producto interno, DTW y LB_Keogh.
    kmeans.py    K-Means determinista (codebook BoVW y celdas de IVF).
    ivf.py       indice vectorial aproximado IVF en disco.
    bovw.py      Bag of Visual/Audio Words sobre el indice invertido.
    features.py  descriptores SIFT, embeddings, descriptores de imagen y MFCC.
    options.py   opciones de CREATE INDEX ... USING IVF WITH (...).
    scan.py      busqueda exacta (linea base) y busqueda por rango.
"""

from __future__ import annotations

__all__ = [
    'bovw',
    'features',
    'ivf',
    'kmeans',
    'metrics',
    'options',
    'pagedfile',
    'scan',
    'vectors',
]
