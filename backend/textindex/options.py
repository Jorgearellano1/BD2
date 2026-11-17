"""
Opciones de  CREATE INDEX ... USING INVERTED WITH (...).

    scoring        'TFIDF_COSINE' (unico esquema implementado; BM25 es una
                   alternativa no implementada)
    language       'spanish' (defecto) o 'english'
    stemming       'true' / 'false'   (Snowball de NLTK)
    stoplist       'true' / 'false'
    block_postings entero > 0: tamano de bloque SPIMI en postings
"""

from __future__ import annotations

from textindex.inverted_index import DEFAULT_BLOCK_POSTINGS, SCORING
from textindex.preprocessing import LANGUAGES

DEFAULT_OPTIONS = {
    'scoring': SCORING,
    'language': 'spanish',
    'stemming': True,
    'stoplist': True,
    'block_postings': DEFAULT_BLOCK_POSTINGS,
}


def _as_bool(key: str, value) -> bool:
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes', 'si'):
        return True
    if text in ('false', '0', 'no'):
        return False
    raise ValueError(f"La opcion '{key}' debe ser 'true' o 'false'")


def normalize_text_options(options: dict | None) -> dict:
    """Valida y completa las opciones. Lanza ValueError si son invalidas."""
    result = dict(DEFAULT_OPTIONS)
    for key, value in (options or {}).items():
        key = key.lower()
        if key == 'scoring':
            scoring = str(value).upper()
            if scoring != SCORING:
                raise ValueError(
                    f"scoring '{value}' no soportado; use '{SCORING}'"
                )
            result['scoring'] = scoring
        elif key == 'language':
            language = str(value).lower()
            if language not in LANGUAGES:
                raise ValueError(
                    f"language '{value}' no soportado; opciones: "
                    f"{', '.join(sorted(LANGUAGES))}"
                )
            result['language'] = language
        elif key in ('stemming', 'stoplist'):
            result[key] = _as_bool(key, value)
        elif key == 'block_postings':
            if not isinstance(value, int) or value <= 0:
                raise ValueError("block_postings debe ser un entero positivo")
            result['block_postings'] = value
        else:
            raise ValueError(f"Opcion de indice invertido desconocida: '{key}'")
    return result
