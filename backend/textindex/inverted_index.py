"""
Indice invertido persistido en memoria secundaria.

Archivos dentro del directorio del indice (todos en paginas de B bytes,
leidas y escritas con seek + read/write + struct; little-endian):

vocab.dic  Diccionario paginado ordenado por termino.
    pagina 0          cabecera: magic, B, num_terms, paginas de vocabulario,
                      inicio y numero de paginas del directorio, N, dirty,
                      idioma, stemming, stoplist y esquema de scoring.
    paginas 1..P      entradas ordenadas  <len u8><termino utf-8><df u32>
                      <nbytes u32><offset u64>; cabecera de pagina u16 con
                      la cantidad de entradas.
    paginas P+1..P+D  directorio disperso: primer termino de cada pagina de
                      vocabulario y su numero de pagina (como un indice ISAM
                      de un nivel). Buscar un termino cuesta D + 1 lecturas.

postings.inv  Listas de posteo contiguas <doc_id u32><tf u32> ordenadas por
    doc_id. Cada lista empieza en 'offset' (bytes) y ocupa 'nbytes'. Como B
    es multiplo de 8, una entrada nunca cruza el limite de una pagina. Solo
    se leen las paginas que cubren las listas de los terminos de la consulta.

docs.dat  Tabla de documentos: pagina 0 cabecera; luego registros fijos
    <page_id i64><slot i64><norma f64> indexados por doc_id. Guarda el RID
    del registro en el HeapFile y la norma euclidiana precalculada del
    vector TF-IDF del documento.

Pesos TF-IDF:
    w(t,d) = (1 + log10 tf(t,d)) * log10(N / df(t))
    |d|    = sqrt( sum_t w(t,d)^2 )                   (precalculado)
    score(q,d) = sum_{t en q} w(t,q) w(t,d) / (|q| |d|)   (CosineScore)

Construccion SPIMI por bloques: los documentos se
procesan en bloques; cada bloque acumula un diccionario termino -> postings
en RAM hasta 'block_postings' entradas, se ordena por termino y se escribe
como una corrida en disco. Al final las corridas se mezclan con un heap
(k-way merge) y se escriben postings.inv y vocab.dic de forma secuencial.
Como los doc_id se asignan en orden, concatenar las listas de un termino en el
orden de las corridas mantiene la lista ordenada.
"""

from __future__ import annotations

import bisect
import heapq
import math
import os
import shutil
import struct
from array import array
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Iterator

from textindex.preprocessing import MAX_TERM_BYTES, TextPreprocessor
from textindex.query import QueryNode, evaluate, parse_query, positive_terms

VOCAB_FILE = 'vocab.dic'
POSTINGS_FILE = 'postings.inv'
DOCS_FILE = 'docs.dat'

VOCAB_MAGIC = b'BD2VOC01'
DOCS_MAGIC = b'BD2DOC01'

# magic, page_size, num_terms, vocab_pages, dir_start, dir_pages, num_docs,
# dirty, language, stemming, stoplist, scoring
VOCAB_HEADER = struct.Struct('<8sIIIIIIB16sBB16s')
DOCS_HEADER = struct.Struct('<8sIII')       # magic, page_size, num_docs, rpp
PAGE_COUNT = struct.Struct('<H')            # entradas en la pagina
ENTRY_TAIL = struct.Struct('<IIQ')          # df, nbytes, offset
DIR_TAIL = struct.Struct('<I')              # numero de pagina de vocabulario
POSTING = struct.Struct('<II')              # doc_id, tf
DOC_RECORD = struct.Struct('<qqd')          # page_id, slot, norma
RUN_HEAD = struct.Struct('<BI')             # len termino, cantidad postings

DEFAULT_BLOCK_POSTINGS = 500_000
SCORING = 'TFIDF_COSINE'


class InvertedIndexError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# E/S por paginas
# --------------------------------------------------------------------------

class PagedFile:
    """Archivo binario accedido exclusivamente por paginas de B bytes."""

    def __init__(self, path: str, page_size: int, mode: str = 'rb') -> None:
        self.path = path
        self.page_size = page_size
        self.file = open(path, mode)
        self.reads = 0
        self.writes = 0

    def read_page(self, page_no: int) -> bytes:
        self.file.seek(page_no * self.page_size)
        data = self.file.read(self.page_size)
        self.reads += 1
        if len(data) < self.page_size:
            data = data + b'\x00' * (self.page_size - len(data))
        return data

    def write_page(self, page_no: int, data: bytes) -> None:
        if len(data) > self.page_size:
            raise InvertedIndexError('Pagina excede el tamano de bloque')
        self.file.seek(page_no * self.page_size)
        self.file.write(data.ljust(self.page_size, b'\x00'))
        self.writes += 1

    def num_pages(self) -> int:
        self.file.seek(0, os.SEEK_END)
        return self.file.tell() // self.page_size

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


class PageWriter:
    """Escritura secuencial de un flujo de bytes en paginas completas."""

    def __init__(self, paged: PagedFile, first_page: int = 0) -> None:
        self.paged = paged
        self.next_page = first_page
        self.buffer = bytearray()
        self.position = first_page * paged.page_size  # bytes logicos escritos

    def write(self, data: bytes) -> None:
        self.buffer += data
        self.position += len(data)
        size = self.paged.page_size
        while len(self.buffer) >= size:
            self.paged.write_page(self.next_page, bytes(self.buffer[:size]))
            del self.buffer[:size]
            self.next_page += 1

    def finish(self) -> int:
        """Escribe la pagina parcial final. Devuelve paginas usadas."""
        if self.buffer:
            self.paged.write_page(self.next_page, bytes(self.buffer))
            self.buffer = bytearray()
            self.next_page += 1
        return self.next_page


class PageStreamReader:
    """Lectura secuencial de un flujo de bytes almacenado en paginas."""

    def __init__(self, paged: PagedFile, total_bytes: int) -> None:
        self.paged = paged
        self.total = total_bytes
        self.page_no = 0
        self.buffer = b''
        self.consumed = 0

    def read(self, n: int) -> bytes:
        while len(self.buffer) < n and self.page_no * self.paged.page_size < self.total:
            self.buffer += self.paged.read_page(self.page_no)
            self.page_no += 1
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        self.consumed += len(data)
        return data

    def exhausted(self) -> bool:
        return self.consumed >= self.total


# --------------------------------------------------------------------------
# Estructuras auxiliares
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TermEntry:
    term: str
    df: int
    nbytes: int
    offset: int


@dataclass
class SearchStats:
    """Telemetria de una busqueda, desglosada por archivo del indice."""
    query_terms: list[str] = field(default_factory=list)
    oov_terms: list[str] = field(default_factory=list)
    candidates: int = 0
    vocab_reads: int = 0
    postings_reads: int = 0
    docs_reads: int = 0
    vocab_ms: float = 0.0
    postings_ms: float = 0.0
    docs_ms: float = 0.0
    scoring_ms: float = 0.0

    @property
    def total_reads(self) -> int:
        return self.vocab_reads + self.postings_reads + self.docs_reads


@dataclass(frozen=True)
class SearchHit:
    doc_id: int
    rid: tuple[int, int]
    score: float | None


@dataclass
class BuildStats:
    num_docs: int = 0
    num_terms: int = 0
    num_postings: int = 0
    runs: int = 0
    pages_read: int = 0
    pages_written: int = 0
    elapsed_ms: float = 0.0
    bytes_vocab: int = 0
    bytes_postings: int = 0
    bytes_docs: int = 0


def _encode_term(term: str) -> bytes:
    data = term.encode('utf-8')
    if len(data) > MAX_TERM_BYTES:
        raise InvertedIndexError(f'Termino demasiado largo: {term!r}')
    return data


def _fixed(text: str, size: int) -> bytes:
    return text.encode('ascii')[:size].ljust(size, b'\x00')


def _unfixed(data: bytes) -> str:
    return data.rstrip(b'\x00').decode('ascii')


# --------------------------------------------------------------------------
# Indice
# --------------------------------------------------------------------------

class InvertedIndex:
    """
    Indice invertido en disco. Se abre leyendo solo la pagina 0 de vocab.dic;
    ninguna consulta carga el diccionario ni las listas completas en RAM.
    """

    def __init__(self, directory: str, page_size: int | None = None) -> None:
        self.directory = directory
        vocab_path = os.path.join(directory, VOCAB_FILE)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f'No existe el indice invertido: {directory}')

        # La pagina 0 se lee primero con un tamano minimo para descubrir B.
        with open(vocab_path, 'rb') as handle:
            raw = handle.read(VOCAB_HEADER.size)
        fields = VOCAB_HEADER.unpack(raw)
        if fields[0] != VOCAB_MAGIC:
            raise InvertedIndexError('vocab.dic no tiene el formato esperado')
        stored_page = fields[1]
        if page_size is not None and page_size != stored_page:
            raise InvertedIndexError(
                f'El indice fue construido con B={stored_page} y se abrio con B={page_size}'
            )
        self.page_size = stored_page

        self.vocab = PagedFile(vocab_path, self.page_size, 'r+b')
        self.postings = PagedFile(os.path.join(directory, POSTINGS_FILE), self.page_size, 'rb')
        self.docs = PagedFile(os.path.join(directory, DOCS_FILE), self.page_size, 'rb')
        self._read_header()

    # Contadores agregados (el Executor mide por diferencia de estos valores)

    @property
    def page_reads(self) -> int:
        return self.vocab.reads + self.postings.reads + self.docs.reads

    @property
    def page_writes(self) -> int:
        return self.vocab.writes + self.postings.writes + self.docs.writes

    def _read_header(self) -> None:
        page = self.vocab.read_page(0)
        (_, _, self.num_terms, self.vocab_pages, self.dir_start, self.dir_pages,
         self.num_docs, dirty, language, stemming, stoplist,
         scoring) = VOCAB_HEADER.unpack_from(page, 0)
        self.dirty = bool(dirty)
        self.config = {
            'language': _unfixed(language),
            'stemming': bool(stemming),
            'stoplist': bool(stoplist),
            'scoring': _unfixed(scoring),
        }
        self.preprocessor = TextPreprocessor.from_config(self.config)
        self.records_per_page = self.page_size // DOC_RECORD.size

    def _write_header(self) -> None:
        header = VOCAB_HEADER.pack(
            VOCAB_MAGIC, self.page_size, self.num_terms, self.vocab_pages,
            self.dir_start, self.dir_pages, self.num_docs, int(self.dirty),
            _fixed(self.config['language'], 16), int(self.config['stemming']),
            int(self.config['stoplist']), _fixed(self.config['scoring'], 16),
        )
        self.vocab.write_page(0, header)
        self.vocab.file.flush()

    def mark_dirty(self) -> None:
        """
        Marca el indice como desactualizado (1 escritura de pagina). Lo usa el
        Executor tras INSERT o DELETE sobre la tabla: N, df y las normas
        dependen de toda la coleccion, de modo que se reconstruye antes de la
        siguiente consulta textual.
        """
        if not self.dirty:
            self.dirty = True
            self._write_header()

    def close(self) -> None:
        for paged in (self.vocab, self.postings, self.docs):
            paged.close()

    def disk_size(self) -> dict[str, int]:
        return {
            name: os.path.getsize(os.path.join(self.directory, name))
            for name in (VOCAB_FILE, POSTINGS_FILE, DOCS_FILE)
        }

    # ------------------------------------------------------------------
    # Construccion (SPIMI)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        directory: str,
        documents: Iterable[tuple[tuple[int, int], str]],
        preprocessor: TextPreprocessor,
        page_size: int,
        block_postings: int = DEFAULT_BLOCK_POSTINGS,
    ) -> tuple['InvertedIndex', BuildStats]:
        """
        Construye el indice a partir de pares (RID, texto) y lo deja abierto.

        Se escribe en '<directory>.building' y al final se reemplaza el
        directorio definitivo, de modo que un fallo nunca deja un indice a
        medio escribir con el nombre valido.
        """
        if page_size % POSTING.size != 0:
            raise InvertedIndexError('B debe ser multiplo de 8')
        start = perf_counter()
        stats = BuildStats()
        work = directory + '.building'
        if os.path.exists(work):
            shutil.rmtree(work)
        os.makedirs(work)

        rids = array('q')
        run_paths: list[str] = []
        block: dict[str, array] = {}
        block_size = 0

        def flush() -> None:
            nonlocal block, block_size
            if not block:
                return
            path = os.path.join(work, f'run_{len(run_paths):04d}.tmp')
            paged = PagedFile(path, page_size, 'w+b')
            writer = PageWriter(paged)
            for term in sorted(block):
                data = block[term]
                writer.write(RUN_HEAD.pack(len(term.encode('utf-8')), len(data) // 2))
                writer.write(term.encode('utf-8'))
                writer.write(data.tobytes())
            writer.finish()
            stats.pages_written += paged.writes
            total = writer.position
            paged.close()
            run_paths.append((path, total))
            block = {}
            block_size = 0

        for rid, text in documents:
            doc_id = len(rids) // 2
            rids.append(int(rid[0]))
            rids.append(int(rid[1]))
            for term, tf in preprocessor.term_frequencies(text).items():
                postings = block.get(term)
                if postings is None:
                    postings = block[term] = array('I')
                postings.append(doc_id)
                postings.append(tf)
                block_size += 1
            if block_size >= block_postings:
                flush()
        flush()

        num_docs = len(rids) // 2
        stats.num_docs = num_docs
        stats.runs = len(run_paths)
        norms_sq = array('d', bytes(8 * num_docs))

        vocab = PagedFile(os.path.join(work, VOCAB_FILE), page_size, 'w+b')
        postings_file = PagedFile(os.path.join(work, POSTINGS_FILE), page_size, 'w+b')
        post_writer = PageWriter(postings_file)

        vocab_page_no = 1
        page_entries: list[bytes] = []
        page_used = PAGE_COUNT.size
        directory_entries: list[tuple[str, int]] = []
        num_terms = 0

        def emit_vocab_page() -> None:
            nonlocal vocab_page_no, page_entries, page_used
            if not page_entries:
                return
            vocab.write_page(
                vocab_page_no,
                PAGE_COUNT.pack(len(page_entries)) + b''.join(page_entries),
            )
            vocab_page_no += 1
            page_entries = []
            page_used = PAGE_COUNT.size

        # k-way merge de corridas ordenadas por termino.
        readers = []
        for path, total in run_paths:
            paged = PagedFile(path, page_size, 'rb')
            readers.append((paged, PageStreamReader(paged, total)))

        def next_term(reader: PageStreamReader):
            if reader.exhausted():
                return None
            length, count = RUN_HEAD.unpack(reader.read(RUN_HEAD.size))
            term = reader.read(length).decode('utf-8')
            return term, reader.read(count * POSTING.size)

        heap: list[tuple[str, int, bytes]] = []
        for index, (_, reader) in enumerate(readers):
            item = next_term(reader)
            if item is not None:
                heapq.heappush(heap, (item[0], index, item[1]))

        while heap:
            term, index, data = heapq.heappop(heap)
            chunks = [data]
            item = next_term(readers[index][1])
            if item is not None:
                heapq.heappush(heap, (item[0], index, item[1]))
            while heap and heap[0][0] == term:
                _, other, more = heapq.heappop(heap)
                chunks.append(more)
                item = next_term(readers[other][1])
                if item is not None:
                    heapq.heappush(heap, (item[0], other, item[1]))

            blob = b''.join(chunks)
            df = len(blob) // POSTING.size
            offset = post_writer.position
            post_writer.write(blob)

            idf = math.log10(num_docs / df)
            if idf > 0.0:
                values = array('I')
                values.frombytes(blob)
                for k in range(0, len(values), 2):
                    weight = (1.0 + math.log10(values[k + 1])) * idf
                    norms_sq[values[k]] += weight * weight

            encoded = _encode_term(term)
            entry = (bytes([len(encoded)]) + encoded
                     + ENTRY_TAIL.pack(df, len(blob), offset))
            if page_used + len(entry) > page_size:
                emit_vocab_page()
            if not page_entries:
                directory_entries.append((term, vocab_page_no))
            page_entries.append(entry)
            page_used += len(entry)
            num_terms += 1
            stats.num_postings += df
        emit_vocab_page()
        post_writer.finish()

        for paged, _ in readers:
            stats.pages_read += paged.reads
            paged.close()
        for path, _ in run_paths:
            os.remove(path)

        # Directorio disperso al final de vocab.dic.
        dir_start = vocab_page_no
        dir_writer_page = dir_start
        entries: list[bytes] = []
        used = PAGE_COUNT.size
        for term, page_no in directory_entries:
            encoded = term.encode('utf-8')
            entry = bytes([len(encoded)]) + encoded + DIR_TAIL.pack(page_no)
            if used + len(entry) > page_size:
                vocab.write_page(dir_writer_page, PAGE_COUNT.pack(len(entries)) + b''.join(entries))
                dir_writer_page += 1
                entries, used = [], PAGE_COUNT.size
            entries.append(entry)
            used += len(entry)
        if entries:
            vocab.write_page(dir_writer_page, PAGE_COUNT.pack(len(entries)) + b''.join(entries))
            dir_writer_page += 1

        config = dict(preprocessor.config(), scoring=SCORING)
        header = VOCAB_HEADER.pack(
            VOCAB_MAGIC, page_size, num_terms, dir_start - 1, dir_start,
            dir_writer_page - dir_start, num_docs, 0,
            _fixed(config['language'], 16), int(config['stemming']),
            int(config['stoplist']), _fixed(config['scoring'], 16),
        )
        vocab.write_page(0, header)

        # docs.dat: RID y norma por doc_id.
        docs = PagedFile(os.path.join(work, DOCS_FILE), page_size, 'w+b')
        rpp = page_size // DOC_RECORD.size
        docs.write_page(0, DOCS_HEADER.pack(DOCS_MAGIC, page_size, num_docs, rpp))
        page = bytearray()
        page_no = 1
        for doc_id in range(num_docs):
            page += DOC_RECORD.pack(rids[2 * doc_id], rids[2 * doc_id + 1],
                                    math.sqrt(norms_sq[doc_id]))
            if (doc_id + 1) % rpp == 0:
                docs.write_page(page_no, bytes(page))
                page_no += 1
                page = bytearray()
        if page:
            docs.write_page(page_no, bytes(page))

        stats.pages_written += vocab.writes + postings_file.writes + docs.writes
        stats.num_terms = num_terms
        for paged in (vocab, postings_file, docs):
            paged.file.flush()
            os.fsync(paged.file.fileno())
            paged.close()

        if os.path.exists(directory):
            old = directory + '.old'
            if os.path.exists(old):
                shutil.rmtree(old)
            os.replace(directory, old)
            os.replace(work, directory)
            shutil.rmtree(old)
        else:
            os.replace(work, directory)

        index = cls(directory, page_size)
        index.vocab.reads = 0  # la lectura de cabecera al abrir no es de la construccion
        sizes = index.disk_size()
        stats.bytes_vocab = sizes[VOCAB_FILE]
        stats.bytes_postings = sizes[POSTINGS_FILE]
        stats.bytes_docs = sizes[DOCS_FILE]
        stats.elapsed_ms = (perf_counter() - start) * 1000.0
        return index, stats

    # ------------------------------------------------------------------
    # Lectura (cada busqueda usa un buffer propio: una pagina se lee a lo
    # sumo una vez por consulta)
    # ------------------------------------------------------------------

    def _query_buffer(self) -> dict:
        return {'directory': None, 'vocab': {}, 'postings': {}, 'docs': {}, 'terms': {}}

    def _directory(self, buf: dict) -> tuple[list[str], list[int]]:
        if buf['directory'] is None:
            firsts: list[str] = []
            pages: list[int] = []
            for page_no in range(self.dir_start, self.dir_start + self.dir_pages):
                data = self.vocab.read_page(page_no)
                (count,) = PAGE_COUNT.unpack_from(data, 0)
                pos = PAGE_COUNT.size
                for _ in range(count):
                    length = data[pos]
                    term = data[pos + 1:pos + 1 + length].decode('utf-8')
                    pos += 1 + length
                    (vocab_page,) = DIR_TAIL.unpack_from(data, pos)
                    pos += DIR_TAIL.size
                    firsts.append(term)
                    pages.append(vocab_page)
            buf['directory'] = (firsts, pages)
        return buf['directory']

    def _vocab_page(self, buf: dict, page_no: int) -> tuple[list[str], list[TermEntry]]:
        cached = buf['vocab'].get(page_no)
        if cached is None:
            data = self.vocab.read_page(page_no)
            (count,) = PAGE_COUNT.unpack_from(data, 0)
            pos = PAGE_COUNT.size
            terms: list[str] = []
            entries: list[TermEntry] = []
            for _ in range(count):
                length = data[pos]
                term = data[pos + 1:pos + 1 + length].decode('utf-8')
                pos += 1 + length
                df, nbytes, offset = ENTRY_TAIL.unpack_from(data, pos)
                pos += ENTRY_TAIL.size
                terms.append(term)
                entries.append(TermEntry(term, df, nbytes, offset))
            cached = buf['vocab'][page_no] = (terms, entries)
        return cached

    def lookup(self, term: str, buf: dict | None = None) -> TermEntry | None:
        """Busca un termino: directorio (cacheado) + 1 pagina de vocabulario."""
        buf = buf if buf is not None else self._query_buffer()
        if term in buf['terms']:
            return buf['terms'][term]
        firsts, pages = self._directory(buf)
        position = bisect.bisect_right(firsts, term) - 1
        entry = None
        if position >= 0:
            terms, entries = self._vocab_page(buf, pages[position])
            k = bisect.bisect_left(terms, term)
            if k < len(terms) and terms[k] == term:
                entry = entries[k]
        buf['terms'][term] = entry
        return entry

    def read_postings(self, entry: TermEntry, buf: dict | None = None) -> tuple[list[int], list[int]]:
        """Lee solo las paginas de postings.inv que cubren la lista."""
        buf = buf if buf is not None else self._query_buffer()
        if entry.nbytes == 0:
            return [], []
        size = self.page_size
        first = entry.offset // size
        last = (entry.offset + entry.nbytes - 1) // size
        chunks = []
        for page_no in range(first, last + 1):
            page = buf['postings'].get(page_no)
            if page is None:
                page = buf['postings'][page_no] = self.postings.read_page(page_no)
            chunks.append(page)
        blob = b''.join(chunks)
        start = entry.offset - first * size
        values = array('I')
        values.frombytes(blob[start:start + entry.nbytes])
        return list(values[0::2]), list(values[1::2])

    def read_doc_records(self, doc_ids: Iterable[int], buf: dict | None = None) -> dict[int, tuple[tuple[int, int], float]]:
        """RID y norma de cada doc_id; agrupa por pagina de docs.dat."""
        buf = buf if buf is not None else self._query_buffer()
        rpp = self.records_per_page
        output: dict[int, tuple[tuple[int, int], float]] = {}
        for doc_id in doc_ids:
            page_no = 1 + doc_id // rpp
            page = buf['docs'].get(page_no)
            if page is None:
                page = buf['docs'][page_no] = self.docs.read_page(page_no)
            page_id, slot, norm = DOC_RECORD.unpack_from(page, (doc_id % rpp) * DOC_RECORD.size)
            output[doc_id] = ((page_id, slot), norm)
        return output

    # ------------------------------------------------------------------
    # Busqueda
    # ------------------------------------------------------------------

    def idf(self, df: int) -> float:
        if df <= 0 or self.num_docs <= 0:
            return 0.0
        return math.log10(self.num_docs / df)

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        rank: bool = True,
        allowed_rids: set[tuple[int, int]] | None = None,
        descending: bool = True,
    ) -> tuple[list[SearchHit], SearchStats]:
        """
        1. Parsea la consulta con el preprocesador del indice.
        2. Evalua el arbol booleano con mezclas lineales -> candidatos.
        3. Lee RID y norma de los candidatos en docs.dat; si hay prefiltro
           relacional (allowed_rids) descarta los que no esten en el.
        4. Si rank: CosineScore termino a termino solo sobre los candidatos y
           Top-K con un min-heap de tamano K.
        """
        stats = SearchStats()
        buf = self._query_buffer()
        tree = parse_query(query_text, self.preprocessor)
        stats.query_terms = positive_terms(tree)
        postings_cache: dict[str, tuple[list[int], list[int]]] = {}

        def entry_of(term: str) -> TermEntry | None:
            before = self.vocab.reads
            started = perf_counter()
            entry = self.lookup(term, buf)
            stats.vocab_reads += self.vocab.reads - before
            stats.vocab_ms += (perf_counter() - started) * 1000.0
            if entry is None and term not in stats.oov_terms:
                stats.oov_terms.append(term)
            return entry

        def postings_of(term: str) -> list[int]:
            if term not in postings_cache:
                entry = entry_of(term)
                if entry is None:
                    postings_cache[term] = ([], [])
                else:
                    before = self.postings.reads
                    started = perf_counter()
                    postings_cache[term] = self.read_postings(entry, buf)
                    stats.postings_reads += self.postings.reads - before
                    stats.postings_ms += (perf_counter() - started) * 1000.0
            return postings_cache[term][0]

        def df_of(term: str) -> int:
            entry = entry_of(term)
            return entry.df if entry is not None else 0

        candidates = evaluate(tree, postings_of, df_of)

        before = self.docs.reads
        started = perf_counter()
        records = self.read_doc_records(candidates, buf)
        stats.docs_reads += self.docs.reads - before
        stats.docs_ms += (perf_counter() - started) * 1000.0

        if allowed_rids is not None:
            candidates = [d for d in candidates if records[d][0] in allowed_rids]
        stats.candidates = len(candidates)

        if not rank:
            hits = [SearchHit(d, records[d][0], None) for d in candidates]
            if top_k is not None:
                hits = hits[:top_k]
            return hits, stats

        started = perf_counter()
        query_tf: dict[str, int] = {}
        for term in stats.query_terms:
            query_tf[term] = query_tf.get(term, 0) + 1

        candidate_set = set(candidates)
        scores = dict.fromkeys(candidates, 0.0)
        query_norm_sq = 0.0
        for term, tf_q in query_tf.items():
            entry = entry_of(term)
            if entry is None:
                continue
            idf = self.idf(entry.df)
            if idf == 0.0:
                continue
            w_q = (1.0 + math.log10(tf_q)) * idf
            query_norm_sq += w_q * w_q
            postings_of(term)
            doc_ids, tfs = postings_cache[term]
            for doc_id, tf in zip(doc_ids, tfs):
                if doc_id in candidate_set:
                    scores[doc_id] += w_q * (1.0 + math.log10(tf)) * idf

        query_norm = math.sqrt(query_norm_sq)
        final: list[tuple[float, int]] = []
        for doc_id in candidates:
            norm = records[doc_id][1]
            score = scores[doc_id] / (norm * query_norm) if norm > 0 and query_norm > 0 else 0.0
            final.append((score, doc_id))

        ordered = top_k_heap(final, top_k, descending)
        stats.scoring_ms = (perf_counter() - started) * 1000.0
        hits = [SearchHit(doc_id, records[doc_id][0], score) for score, doc_id in ordered]
        return hits, stats


def top_k_heap(
    scored: Iterable[tuple[float, int]],
    k: int | None,
    descending: bool = True,
) -> list[tuple[float, int]]:
    """
    Top-K con una cola de prioridad de tamano K.

    Para DESC se mantiene un min-heap con los K mejores vistos: la raiz es el
    peor de ellos y se reemplaza cuando llega uno mejor. Costo O(n log K) y
    memoria O(K). El empate se resuelve por doc_id ascendente.
    Sin K se ordena toda la lista (equivale a K = n).
    """
    if descending:
        def key(item):
            return (item[0], -item[1])
    else:
        def key(item):
            return (-item[0], -item[1])

    if k is None:
        return sorted(scored, key=key, reverse=True)
    if k <= 0:
        return []
    heap: list[tuple[tuple[float, int], tuple[float, int]]] = []
    for item in scored:
        entry = (key(item), item)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [item for _, item in sorted(heap, key=lambda e: e[0], reverse=True)]
