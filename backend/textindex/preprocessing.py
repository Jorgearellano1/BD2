"""
Pipeline de preprocesamiento lexico (modelo Bag of Words).

Etapas, en el orden aplicado (PDF 09, p. 23-25; enunciado P2, p. 9):

    1. Normalizacion Unicode NFKC: unifica formas de compatibilidad
       (ligaduras, digitos de ancho completo) para que el mismo texto
       siempre produzca los mismos bytes.
    2. Siglas con puntos: "O.N.U." se convierte en "ONU" (p. 23 del PDF 09).
    3. Case folding (casefold), mas agresivo que lower() para Unicode.
    4. Tokenizacion: secuencias maximas de letras o digitos Unicode; todo lo
       demas (espacios, signos, guiones, apostrofos) separa tokens.
    5. Descarte de tokens de un solo caracter y de stopwords (stoplist del
       idioma configurado; se compara con y sin tildes).
    6. Stemming Snowball de NLTK para el idioma configurado.
    7. Eliminacion de tildes en la raiz resultante (se conserva la enie), para
       que "tambien" y "tambien" con tilde produzcan el mismo termino.
    8. Corte a MAX_TERM_BYTES bytes UTF-8 (limite fisico del diccionario).

El indice y la consulta usan exactamente el mismo objeto configurado; la
configuracion se guarda en el catalogo y en la cabecera de vocab.dic, de modo
que al reabrir el indice se reconstruye el mismo pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent / 'resources'

# Idiomas soportados: nombre Snowball -> sufijo del archivo de stoplist.
LANGUAGES = {
    'spanish': 'es',
    'english': 'en',
}

MAX_TERM_BYTES = 64
MIN_TOKEN_CHARS = 2

_ACRONYM = re.compile(r'(?<![^\W\d_])((?:[^\W\d_]\.){2,})')
_TOKEN = re.compile(r'[^\W_]+')


def strip_accents(text: str) -> str:
    """Elimina marcas diacriticas conservando la enie."""
    text = text.replace('ñ', '\x00')
    decomposed = unicodedata.normalize('NFD', text)
    stripped = ''.join(
        char for char in decomposed if unicodedata.category(char) != 'Mn'
    )
    return unicodedata.normalize('NFC', stripped).replace('\x00', 'ñ')


def truncate_utf8(term: str, limit: int = MAX_TERM_BYTES) -> str:
    """Corta un termino a 'limit' bytes UTF-8 sin partir un caracter."""
    data = term.encode('utf-8')
    if len(data) <= limit:
        return term
    return data[:limit].decode('utf-8', errors='ignore')


def load_stoplist(language: str) -> frozenset[str]:
    """Lee resources/stoplist_<xx>.txt. Devuelve formas con y sin tilde."""
    suffix = LANGUAGES[language]
    path = RESOURCES_DIR / f'stoplist_{suffix}.txt'
    words: set[str] = set()
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            word = line.split('#', 1)[0].strip()
            if not word:
                continue
            word = unicodedata.normalize('NFKC', word).casefold()
            words.add(word)
            words.add(strip_accents(word))
    return frozenset(words)


class TextPreprocessor:
    """
    Convierte texto libre en la lista de terminos del indice.

    language     : 'spanish' o 'english' (stoplist y stemmer Snowball)
    stemming     : aplica SnowballStemmer de NLTK
    use_stoplist : elimina stopwords
    """

    def __init__(
        self,
        language: str = 'spanish',
        stemming: bool = True,
        use_stoplist: bool = True,
    ) -> None:
        language = language.lower()
        if language not in LANGUAGES:
            raise ValueError(
                f"Idioma no soportado: '{language}'. "
                f"Opciones: {', '.join(sorted(LANGUAGES))}"
            )
        self.language = language
        self.stemming = bool(stemming)
        self.use_stoplist = bool(use_stoplist)
        self.stoplist = load_stoplist(language) if use_stoplist else frozenset()
        self._stemmer = None
        if self.stemming:
            try:
                from nltk.stem.snowball import SnowballStemmer
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise RuntimeError(
                    'El modulo textual requiere nltk (pip install nltk) '
                    'para el stemming Snowball.'
                ) from exc
            self._stemmer = SnowballStemmer(language)
        # Cache palabra -> termino final; el vocabulario de un idioma se
        # repite mucho, por lo que evita volver a ejecutar el stemmer.
        self._cache: dict[str, str | None] = {}

    # Configuracion persistible

    def config(self) -> dict:
        return {
            'language': self.language,
            'stemming': self.stemming,
            'stoplist': self.use_stoplist,
        }

    @classmethod
    def from_config(cls, config: dict) -> 'TextPreprocessor':
        return cls(
            language=config.get('language', 'spanish'),
            stemming=bool(config.get('stemming', True)),
            use_stoplist=bool(config.get('stoplist', True)),
        )

    # Pipeline

    @staticmethod
    def normalize(text: str) -> str:
        """Etapas 1 a 3: NFKC, siglas con puntos y case folding."""
        text = unicodedata.normalize('NFKC', text)
        text = _ACRONYM.sub(lambda match: match.group(1).replace('.', ''), text)
        return text.casefold()

    @staticmethod
    def tokenize(normalized: str) -> list[str]:
        """Etapa 4: tokens alfanumericos Unicode."""
        return _TOKEN.findall(normalized)

    def term_of(self, token: str) -> str | None:
        """Etapas 5 a 8 para un token ya normalizado. None si se descarta."""
        cached = self._cache.get(token, False)
        if cached is not False:
            return cached

        result: str | None
        if len(token) < MIN_TOKEN_CHARS:
            result = None
        elif self.use_stoplist and (
            token in self.stoplist or strip_accents(token) in self.stoplist
        ):
            result = None
        else:
            stem = self._stemmer.stem(token) if self._stemmer else token
            result = truncate_utf8(strip_accents(stem)) or None

        if len(self._cache) < 500_000:
            self._cache[token] = result
        return result

    def terms(self, text: str) -> list[str]:
        """Texto libre -> lista de terminos (con repeticiones, en orden)."""
        if not text:
            return []
        output: list[str] = []
        for token in self.tokenize(self.normalize(text)):
            term = self.term_of(token)
            if term is not None:
                output.append(term)
        return output

    def term_frequencies(self, text: str) -> dict[str, int]:
        """Bolsa de palabras: termino -> tf dentro del texto."""
        counts: dict[str, int] = {}
        for term in self.terms(text):
            counts[term] = counts.get(term, 0) + 1
        return counts
