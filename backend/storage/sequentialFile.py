from __future__ import annotations

import csv
import heapq
import hashlib
import json
import math
import os
import struct
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

from engine.metadata import TableMetadata, RawRecord, CleanRecord, RID, Key
from engine.conditions import Expression, Condition, LogicalExpression, ComparisonOperator, ColumnRef, LiteralValue, evaluate_expression
from storage.storageFile import StorageFile


@dataclass(frozen=True, order=True)
class _RID:
    page_id: int
    slot_number: int

    def encode(self) -> int:
        return (self.page_id << 16) | self.slot_number

    @classmethod
    def decode(cls, value: int) -> Optional[_RID]:
        return None if value == -1 else cls(value >> 16, value & 65535)


@dataclass
class DiskCounter:
    disk_reads: int = 0
    disk_writes: int = 0

    def reset(self) -> None:
        self.disk_reads = 0
        self.disk_writes = 0

    def snapshot(self) -> dict:
        return asdict(self)


@dataclass
class _Record:
    key: tuple
    values: RawRecord
    active: bool = True
    prev: int = -1
    next: int = -1


@dataclass
class Header:
    page_size: int
    main_pages: int
    total_pages: int
    active_records: int
    overflow_records: int
    overflow_tail: int
    k_threshold: int
    fill_factor: float
    min_fill_factor: float
    next_id: int
    schema_hash: bytes


class _SequentialPages:
    MAGIC = b'SEQGEN03'
    VERSION = 3
    META = struct.Struct('<8sIIQQQQqQddQ32s')
    PAGE = struct.Struct('<qIIqqB7xqqI')
    HEADER_SIZE = META.size
    MAIN = 1
    OVERFLOW = 2
    RUN = 3

    def __init__(self, table_meta: TableMetadata, filename: str, key_column: str, aux_capacity: int = 128,
                 k_threshold: Optional[int] = None, min_fill_factor: float = 0.5,
                 page_size: int = 4096, fill_factor: float = 0.75,
                 counter: Optional[DiskCounter] = None, auto_reorganize: bool = True):
        if page_size not in (1024, 2048, 4096, 8192):
            raise ValueError('page_size debe ser 1024, 2048, 4096 u 8192')
        if not 0.70 <= fill_factor <= 0.80:
            raise ValueError('fill_factor debe estar entre 0.70 y 0.80')
        if not 0 <= min_fill_factor < fill_factor:
            raise ValueError('min_fill_factor debe ser menor que fill_factor')
        if not isinstance(aux_capacity, int) or aux_capacity < 1:
            raise ValueError('aux_capacity debe ser un entero positivo')
        if k_threshold is not None and (not isinstance(k_threshold, int) or k_threshold < 1):
            raise ValueError('k_threshold debe ser positivo')
        self.table_meta = table_meta
        self.key_index = table_meta.col_index(key_column)
        self.RECORD = struct.Struct(table_meta.struct_format + 'q?qq')
        self.RECORD_SIZE = self.RECORD.size
        self.capacity = (page_size - self.PAGE.size) // (self.RECORD_SIZE + 2)
        if self.capacity < 1:
            raise ValueError('El registro no cabe en una pagina')
        schema = json.dumps([table_meta.columns, table_meta.primary_key, key_column], sort_keys=True)
        schema_hash = hashlib.sha256(schema.encode()).digest()
        self.filename = os.fspath(filename)
        self.counter = counter if counter is not None else DiskCounter()
        self.auto_reorganize = auto_reorganize
        self.reorganizations = 0
        self.page_size = page_size
        self.last_metrics = {}
        self.file = None
        if os.path.exists(self.filename) and os.path.getsize(self.filename):
            self.file = open(self.filename, 'r+b', buffering=0)
            try:
                prefix = self._read_block(self.file, 0)
                values = self.META.unpack_from(prefix)
                if values[0] != self.MAGIC or values[1] != self.VERSION:
                    raise ValueError('Formato incompatible; importe los datos mediante CSV')
                if values[2] != self.page_size:
                    raise ValueError(f'Abra el archivo con page_size={values[2]}')
                self.h = Header(*values[2:])
                if self.h.schema_hash != schema_hash:
                    raise ValueError('El esquema no coincide con el archivo')
                if os.fstat(self.file.fileno()).st_size != self.h.total_pages * self.page_size:
                    raise ValueError('Longitud de archivo invalida')
                if not 1 <= self.h.main_pages < self.h.total_pages:
                    raise ValueError('Cabecera invalida')
            except BaseException:
                self.file.close()
                raise
        else:
            self.file = open(self.filename, 'w+b', buffering=0)
            self.h = Header(page_size, 1, 2, 0, 0, -1,
                            k_threshold or aux_capacity, fill_factor, min_fill_factor, 0, schema_hash)
            self._write_header()
            self._write_page(self._new_page(1, self.MAIN))
        self.capacity = (self.page_size - self.PAGE.size) // (self.RECORD_SIZE + 2)

    def _read_block(self, file, page_id: int, size: Optional[int] = None) -> bytes:
        size = size or self.page_size
        file.seek(page_id * size)
        data = file.read(size)
        self.counter.disk_reads += 1
        if len(data) != size:
            raise ValueError('Pagina incompleta')
        return data

    def _write_block(self, file, page_id: int, data: bytes) -> None:
        if len(data) != self.page_size:
            raise ValueError('Toda escritura debe transferir una pagina completa')
        file.seek(page_id * self.page_size)
        if file.write(data) != self.page_size:
            raise OSError('Escritura incompleta')
        self.counter.disk_writes += 1

    def _meta_bytes(self, h: Header) -> bytes:
        raw = self.META.pack(self.MAGIC, self.VERSION, h.page_size, h.main_pages,
                             h.total_pages, h.active_records, h.overflow_records,
                             h.overflow_tail, h.k_threshold, h.fill_factor,
                             h.min_fill_factor, h.next_id, h.schema_hash)
        return raw.ljust(self.page_size, b'\0')

    def _write_header(self) -> None:
        self._write_block(self.file, 0, self._meta_bytes(self.h))

    def _read_header(self) -> Header:
        values = self.META.unpack_from(self._read_block(self.file, 0))
        if values[0] != self.MAGIC or values[1] != self.VERSION:
            raise ValueError('Cabecera corrupta')
        return Header(*values[2:])

    def _pack_record(self, rec: _Record) -> bytes:
        return self.RECORD.pack(*rec.values, rec.key[1], rec.active, rec.prev, rec.next)

    def _unpack_record(self, data: bytes) -> _Record:
        unpacked = self.RECORD.unpack(data)
        values = tuple(unpacked[:-4])
        token, active, prev, next = unpacked[-4:]
        return _Record((values[self.key_index], token), values, active, prev, next)

    def _new_page(self, page_id: int, kind: int, prev: int = -1, next: int = -1):
        capacity = (self.page_size - self.PAGE.size) // (self.RECORD_SIZE + 2)
        return {'id': page_id, 'prev': prev, 'next': next, 'kind': kind,
                'head': -1, 'tail': -1, 'overflow_count': 0,
                'records': [None] * capacity}

    def _encode_page(self, page: dict) -> bytes:
        records = page['records']
        directory = sorted((i for i, r in enumerate(records) if r is not None),
                           key=lambda i: records[i].key)
        base = self.PAGE.size + 2 * len(records)
        free = next((base + i * self.RECORD_SIZE for i, r in enumerate(records)
                     if r is None), self.page_size)
        data = bytearray(self.page_size)
        self.PAGE.pack_into(data, 0, page['id'], len(directory), free,
                            page['next'], page['prev'], page['kind'],
                            page['head'], page['tail'], page['overflow_count'])
        for i in range(len(records)):
            struct.pack_into('<H', data, self.PAGE.size + 2 * i,
                             directory[i] if i < len(directory) else 65535)
        for i, rec in enumerate(records):
            if rec is not None:
                offset = base + i * self.RECORD_SIZE
                data[offset:offset + self.RECORD_SIZE] = self._pack_record(rec)
        return bytes(data)

    def _decode_page(self, data: bytes, page_id: int) -> dict:
        pid, count, free, next_id, prev_id, kind, head, tail, overflow = self.PAGE.unpack_from(data)
        if pid != page_id or count > self.capacity or kind not in (self.MAIN, self.OVERFLOW, self.RUN):
            raise ValueError('Cabecera de pagina corrupta')
        page = self._new_page(pid, kind, prev_id, next_id)
        page.update(head=head, tail=tail, overflow_count=overflow)
        base = self.PAGE.size + self.capacity * 2
        keys = []
        seen = set()
        for i in range(count):
            slot = struct.unpack_from('<H', data, self.PAGE.size + 2 * i)[0]
            if slot >= self.capacity or slot in seen:
                raise ValueError('Directorio de slots corrupto')
            seen.add(slot)
            offset = base + slot * self.RECORD_SIZE
            rec = self._unpack_record(data[offset:offset + self.RECORD_SIZE])
            if not rec.active:
                raise ValueError('Slot activo inconsistente')
            page['records'][slot] = rec
            keys.append(rec.key)
        if keys != sorted(keys):
            raise ValueError('Directorio desordenado')
        expected_free = next((base + i * self.RECORD_SIZE for i in range(self.capacity)
                              if i not in seen), self.page_size)
        if free != expected_free:
            raise ValueError('Puntero de espacio libre invalido')
        return page

    def _read_page(self, page_id: int) -> dict:
        if not 1 <= page_id < self.h.total_pages:
            raise ValueError('Puntero de pagina fuera del archivo')
        return self._decode_page(self._read_block(self.file, page_id), page_id)

    def _write_page(self, page: dict) -> None:
        self._write_block(self.file, page['id'], self._encode_page(page))

    def _ordered(self, page: dict):
        return sorted(((_RID(page['id'], i), rec) for i, rec in enumerate(page['records'])
                       if rec is not None), key=lambda pair: pair[1].key)

    def _get(self, pointer: int, cache: dict):
        rid = _RID.decode(pointer)
        if rid is None or rid.slot_number >= self.capacity:
            raise ValueError('RID invalido')
        if rid.page_id not in cache:
            cache[rid.page_id] = self._read_page(rid.page_id)
        rec = cache[rid.page_id]['records'][rid.slot_number]
        if rec is None:
            raise ValueError('RID apunta a un slot vacio')
        return rid, rec

    def _owner(self, key: tuple, cache: dict) -> dict:
        low, high = 1, self.h.main_pages
        while low < high:
            mid = (low + high + 1) // 2
            page = self._read_page(mid)
            items = self._ordered(page)
            first_key = items[0][1].key if items else None
            overflow_min = self._page_min(page)
            if overflow_min is not None:
                first_key = overflow_min if first_key is None else min(first_key, overflow_min)
            if first_key is None or key < first_key:
                high = mid - 1
            else:
                low = mid
        page = self._read_page(low)
        cache[low] = page
        return page

    def _page_min(self, page: dict):
        if page['head'] != -1:
            return self._get(page['head'], {})[1].key
        return None

    def _locate(self, key: tuple, page: dict, cache: dict):
        for rid, rec in self._ordered(page):
            if rec.key == key:
                return rid, rec
        pointer = page['head']
        scan_cache = {}
        for _ in range(page['overflow_count']):
            page_id = _RID.decode(pointer).page_id
            if page_id not in scan_cache:
                scan_cache.clear()
            rid, rec = self._get(pointer, scan_cache)
            if rec.key == key:
                cache[rid.page_id] = scan_cache[rid.page_id]
                return rid, rec
            if rec.key > key:
                break
            pointer = rec.next
        return None

    def _begin(self):
        if self.file is None or self.file.closed:
            raise ValueError('Archivo cerrado')
        return time.perf_counter(), self.counter.disk_reads, self.counter.disk_writes

    def _end(self, start, operation):
        self.last_metrics = {'operation': operation,
                             'disk_reads': self.counter.disk_reads - start[1],
                             'disk_writes': self.counter.disk_writes - start[2],
                             'execution_ms': (time.perf_counter() - start[0]) * 1000}

    def search_with_rid(self, key: tuple):
        start = self._begin()
        cache = {}
        result = self._locate(key, self._owner(key, cache), cache)
        self._end(start, 'search')
        return result

    def search(self, key: tuple) -> Optional[_Record]:
        result = self.search_with_rid(key)
        return result[1] if result is not None else None

    def read_rid(self, rid: _RID) -> Optional[_Record]:
        start = self._begin()
        if not 0 <= rid.slot_number < self.capacity:
            raise ValueError('Slot invalido')
        rec = self._read_page(rid.page_id)['records'][rid.slot_number]
        self._end(start, 'read_rid')
        return rec

    def _overflow_slot(self, cache):
        if self.h.overflow_tail != -1:
            pid = self.h.overflow_tail
            if pid not in cache:
                cache[pid] = self._read_page(pid)
            page = cache[pid]
            for slot, rec in enumerate(page['records']):
                if rec is None:
                    return _RID(pid, slot)
        pid = self.h.total_pages
        page = self._new_page(pid, self.OVERFLOW, self.h.overflow_tail)
        if self.h.overflow_tail != -1:
            cache[self.h.overflow_tail]['next'] = pid
        cache[pid] = page
        self.h.overflow_tail = pid
        self.h.total_pages += 1
        return _RID(pid, 0)

    def insert(self, rec: _Record) -> bool:
        self._pack_record(rec)
        start = self._begin()
        cache = {}
        page = self._owner(rec.key, cache)
        if self._locate(rec.key, page, cache) is not None:
            self._end(start, 'insert')
            return False
        new = replace(rec, active=True, prev=-1, next=-1)
        free = next((i for i, r in enumerate(page['records']) if r is None), None)
        dirty = {page['id']}
        if free is not None:
            page['records'][free] = new
        else:
            previous, pointer = -1, page['head']
            scan_cache = {}
            while pointer != -1:
                page_id = _RID.decode(pointer).page_id
                if page_id not in scan_cache:
                    scan_cache.clear()
                rid, current = self._get(pointer, scan_cache)
                if current.key > new.key:
                    break
                previous, pointer = pointer, current.next
            old_tail = self.h.overflow_tail
            target = self._overflow_slot(cache)
            dirty.add(target.page_id)
            if old_tail != -1:
                dirty.add(old_tail)
            new.prev, new.next = previous, pointer
            cache[target.page_id]['records'][target.slot_number] = new
            if previous != -1:
                rid, current = self._get(previous, cache)
                current.next = target.encode()
                dirty.add(rid.page_id)
            else:
                page['head'] = target.encode()
            if pointer != -1:
                rid, current = self._get(pointer, cache)
                current.prev = target.encode()
                dirty.add(rid.page_id)
            else:
                page['tail'] = target.encode()
            page['overflow_count'] += 1
            self.h.overflow_records += 1
        self.h.active_records += 1
        for pid in sorted(dirty):
            self._write_page(cache[pid])
        self._write_header()
        if self.auto_reorganize and self.h.overflow_records >= self.h.k_threshold:
            self.reorganize()
        self._end(start, 'insert')
        return True

    def delete(self, key: tuple) -> bool:
        start = self._begin()
        cache = {}
        page = self._owner(key, cache)
        found = self._locate(key, page, cache)
        if found is None:
            self._end(start, 'delete')
            return False
        rid, rec = found
        dirty = {page['id'], rid.page_id}
        if rid.page_id > self.h.main_pages:
            for pointer, attr, value in ((rec.prev, 'next', rec.next), (rec.next, 'prev', rec.prev)):
                if pointer != -1:
                    neighbor, current = self._get(pointer, cache)
                    setattr(current, attr, value)
                    dirty.add(neighbor.page_id)
            if rec.prev == -1:
                page['head'] = rec.next
            if rec.next == -1:
                page['tail'] = rec.prev
            page['overflow_count'] -= 1
            self.h.overflow_records -= 1
        cache[rid.page_id]['records'][rid.slot_number] = None
        self.h.active_records -= 1
        for pid in sorted(dirty):
            self._write_page(cache[pid])
        self._write_header()
        if not any(page['records']) or (self.auto_reorganize and
                self.h.active_records < self.h.main_pages * self.capacity * self.h.min_fill_factor):
            self.reorganize()
        self._end(start, 'delete')
        return True

    def _page_records(self, page: dict):
        main = iter(self._ordered(page))
        item = next(main, None)
        pointer = page['head']
        cached_pid, cached_page = -1, None
        traversed = 0
        current = None
        while item is not None or pointer != -1:
            if pointer != -1 and current is None:
                rid = _RID.decode(pointer)
                if rid.page_id != cached_pid:
                    cached_page = self._read_page(rid.page_id)
                    cached_pid = rid.page_id
                current = cached_page['records'][rid.slot_number]
                traversed += 1
                if current is None or traversed > page['overflow_count']:
                    raise ValueError('Cadena overflow corrupta')
            if item is not None and (current is None or item[1].key < current.key):
                yield item[1]
                item = next(main, None)
            else:
                yield current
                pointer = current.next
                current = None

    def iter_records(self, a: Optional[tuple] = None, b: Optional[tuple] = None) -> Iterator[_Record]:
        if a is not None and b is not None and a > b:
            return
        first = 1 if a is None else self._owner(a, {})['id']
        for pid in range(first, self.h.main_pages + 1):
            for rec in self._page_records(self._read_page(pid)):
                if b is not None and rec.key > b:
                    return
                if a is None or rec.key >= a:
                    yield replace(rec)

    def rangeSearch(self, a: tuple, b: tuple) -> list[_Record]:
        start = self._begin()
        result = list(self.iter_records(a, b))
        self._end(start, 'range_search')
        return result

    range_search = rangeSearch

    def _replace_from_sorted(self, records: Iterable[_Record]):
        parent = str(Path(self.filename).resolve().parent)
        fd, temp_path = tempfile.mkstemp(prefix='.sequential-', suffix='.bin', dir=parent)
        target = max(1, round(self.capacity * self.h.fill_factor))
        total, pid, previous_key = 0, 1, None
        try:
            with os.fdopen(fd, 'w+b', buffering=0) as output:
                self._write_block(output, 0, bytes(self.page_size))
                page = self._new_page(pid, self.MAIN)
                count = 0
                unique_indexes = [i for i, col in enumerate(self.table_meta.columns)
                                  if self.table_meta.is_unique_column(col['name'])]
                seen = {i: set() for i in unique_indexes if i != self.key_index}
                previous_value = None
                has_previous = False
                for original in records:
                    for i in unique_indexes:
                        value = original.values[i]
                        if i == self.key_index:
                            if has_previous and value == previous_value:
                                raise ValueError('Clave duplicada en columna UNIQUE')
                        else:
                            if value in seen[i]:
                                raise ValueError('Clave duplicada en columna UNIQUE')
                            seen[i].add(value)
                    previous_value = original.values[self.key_index]
                    has_previous = True
                    rec = replace(original, active=True, prev=-1, next=-1)
                    self._pack_record(rec)
                    if previous_key is not None and rec.key <= previous_key:
                        raise ValueError(f'Clave duplicada o desordenada: {rec.key}')
                    previous_key = rec.key
                    if count == target:
                        page['next'] = pid + 1
                        self._write_block(output, pid, self._encode_page(page))
                        pid += 1
                        page = self._new_page(pid, self.MAIN, pid - 1)
                        count = 0
                    page['records'][count] = rec
                    count += 1
                    total += 1
                self._write_block(output, pid, self._encode_page(page))
                new_h = replace(self.h, main_pages=pid, total_pages=pid + 1,
                                active_records=total, overflow_records=0, overflow_tail=-1)
                self._write_block(output, 0, self._meta_bytes(new_h))
                os.fsync(output.fileno())
            self.file.close()
            try:
                os.replace(temp_path, self.filename)
            finally:
                self.file = open(self.filename, 'r+b', buffering=0)
            self.h = new_h
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def reorganize(self) -> None:
        self.reorganizations += 1
        start = self._begin()
        self._replace_from_sorted(self.iter_records())
        self._end(start, 'reorganize')

    _rebuild = reorganize

    def _write_run(self, path, records):
        with open(path, 'w+b', buffering=0) as output:
            pid, count = 0, 0
            page = self._new_page(pid, self.RUN)
            for rec in records:
                if count == self.capacity:
                    self._write_block(output, pid, self._encode_page(page))
                    pid += 1
                    page = self._new_page(pid, self.RUN)
                    count = 0
                page['records'][count] = replace(rec, active=True, prev=-1, next=-1)
                count += 1
            if count:
                self._write_block(output, pid, self._encode_page(page))

    def _read_run(self, path):
        with open(path, 'rb', buffering=0) as source:
            size = os.fstat(source.fileno()).st_size
            if size % self.page_size:
                raise ValueError('Run incompleto')
            for pid in range(size // self.page_size):
                page = self._decode_page(self._read_block(source, pid), pid)
                for _, rec in self._ordered(page):
                    yield rec

    def bulk_insert(self, records: Iterable[_Record], buffer_size: int = 8 * 1024 * 1024,
                    page_size: Optional[int] = None, replace_existing: bool = False):
        start = self._begin()
        if page_size is not None and page_size != self.page_size:
            raise ValueError('page_size debe coincidir con el archivo')
        if buffer_size < 3 * self.page_size:
            raise ValueError('buffer_size debe permitir al menos tres paginas')
        chunk_limit = max(1, buffer_size // (self.RECORD_SIZE + 2048))
        parent = str(Path(self.filename).resolve().parent)
        with tempfile.TemporaryDirectory(prefix='.sequential-sort-', dir=parent) as directory:
            runs, chunk = [], []
            for rec in records:
                self._pack_record(rec)
                chunk.append(replace(rec, active=True, prev=-1, next=-1))
                if len(chunk) >= chunk_limit:
                    path = os.path.join(directory, f'run-{len(runs)}')
                    self._write_run(path, sorted(chunk, key=lambda r: r.key))
                    runs.append(path)
                    chunk.clear()
            if chunk:
                path = os.path.join(directory, f'run-{len(runs)}')
                self._write_run(path, sorted(chunk, key=lambda r: r.key))
                runs.append(path)
                chunk.clear()
            phase = 0
            while len(runs) > 1:
                merged = []
                for i in range(0, len(runs), 2):
                    if i + 1 == len(runs):
                        merged.append(runs[i])
                        continue
                    path = os.path.join(directory, f'merge-{phase}-{i}')
                    left, right = self._read_run(runs[i]), self._read_run(runs[i + 1])
                    try:
                        self._write_run(path, heapq.merge(left, right, key=lambda r: r.key))
                    finally:
                        left.close()
                        right.close()
                    os.unlink(runs[i])
                    os.unlink(runs[i + 1])
                    merged.append(path)
                runs = merged
                phase += 1
            incoming = self._read_run(runs[0]) if runs else iter(())
            existing = self.iter_records() if not replace_existing else iter(())
            try:
                self._replace_from_sorted(heapq.merge(existing, incoming, key=lambda r: r.key))
            finally:
                if hasattr(incoming, 'close'):
                    incoming.close()
                if hasattr(existing, 'close'):
                    existing.close()
        self._end(start, 'bulk_insert')

    def stats(self) -> dict:
        metadata = asdict(self.h)
        metadata.pop('schema_hash')
        return {**metadata, 'records_per_page': self.capacity,
                'main_fill_ratio': (self.h.active_records - self.h.overflow_records) /
                                   (self.h.main_pages * self.capacity),
                'counter': self.counter.snapshot()}

    def validate(self) -> dict:
        start = self._begin()
        h = self._read_header()
        if h != self.h:
            raise ValueError('Metadatos en memoria y disco inconsistentes')
        total, overflow, last = 0, 0, None
        for pid in range(1, h.main_pages + 1):
            page = self._read_page(pid)
            if page['kind'] != self.MAIN or page['prev'] != (pid - 1 if pid > 1 else -1) or page['next'] != (pid + 1 if pid < h.main_pages else -1):
                raise ValueError('Enlace de paginas principales invalido')
            pointer, prev, count = page['head'], -1, 0
            while pointer != -1:
                rid, rec = self._get(pointer, {})
                if rid.page_id <= h.main_pages or rec.prev != prev or count >= page['overflow_count']:
                    raise ValueError('Enlace overflow invalido')
                prev, pointer = pointer, rec.next
                count += 1
            if count != page['overflow_count'] or prev != page['tail']:
                raise ValueError('Contador overflow invalido')
            overflow += count
            for rec in self._page_records(page):
                if last is not None and rec.key <= last:
                    raise ValueError('Orden global o unicidad invalida')
                last = rec.key
                total += 1
        physical_overflow = 0
        for pid in range(h.main_pages + 1, h.total_pages):
            page = self._read_page(pid)
            if page['kind'] != self.OVERFLOW:
                raise ValueError('Tipo de pagina overflow invalido')
            if page['prev'] != (pid - 1 if pid > h.main_pages + 1 else -1) or page['next'] != (pid + 1 if pid < h.total_pages - 1 else -1):
                raise ValueError('Cadena fisica overflow invalida')
            physical_overflow += sum(r is not None for r in page['records'])
        if total != h.active_records or overflow != h.overflow_records or overflow != physical_overflow:
            raise ValueError('Conteo global invalido')
        self._end(start, 'validate')
        return {'valid': True, 'records': total, 'overflow_records': overflow,
                'pages': h.total_pages}

    def close(self):
        if self.file is not None and not self.file.closed:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()



class SequentialFile(StorageFile):
    def __init__(self, table_meta: TableMetadata, filepath: str, page_size: int,
                 key_column: Optional[str] = None, k_threshold: int = 128,
                 fill_factor: float = 0.75, min_fill_factor: float = 0.5,
                 auto_reorganize: bool = True):
        if not table_meta.columns:
            raise ValueError('El esquema debe contener columnas')
        self.table_meta = table_meta
        self.filepath = os.fspath(filepath)
        self.PAGE_SIZE = page_size
        self.key_column = key_column or table_meta.primary_key or table_meta.columns[0]['name']
        self.key_index = table_meta.col_index(self.key_column)
        self._record_struct = struct.Struct(table_meta.struct_format)
        self._counter = DiskCounter()
        super().__init__()
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        self._pages = _SequentialPages(
            table_meta, self.filepath, self.key_column, k_threshold=k_threshold,
            min_fill_factor=min_fill_factor, page_size=page_size,
            fill_factor=fill_factor, counter=self._counter,
            auto_reorganize=auto_reorganize
        )
        self.BLOCK_FACTOR = self._pages.capacity
        self.RECORD_SIZE = self._pages.RECORD_SIZE

    @property
    def page_reads(self) -> int:
        return self._counter.disk_reads

    @page_reads.setter
    def page_reads(self, value: int) -> None:
        self._counter.disk_reads = value

    @property
    def reorganizations(self) -> int:
        return self._pages.reorganizations

    @property
    def page_writes(self) -> int:
        return self._counter.disk_writes

    @page_writes.setter
    def page_writes(self, value: int) -> None:
        self._counter.disk_writes = value

    def _ensure_open(self) -> None:
        self._pages._begin()

    def _normalize(self, record: RawRecord) -> RawRecord:
        if len(record) != len(self.table_meta.columns):
            raise ValueError('Cantidad de columnas incorrecta')
        raw = self.table_meta.normalize_tuple(record)
        raw = self._record_struct.unpack(self._record_struct.pack(*raw))
        if any(isinstance(value, float) and not math.isfinite(value) for value in raw):
            raise ValueError('Los valores FLOAT deben ser finitos')
        return raw

    def _normalized_key(self, column: str, key: Key):
        value = self.table_meta.normalize_value(column, key)
        col = self.table_meta.columns[self.table_meta.col_index(column)]
        codec = struct.Struct('=' + col['fmt'])
        value = codec.unpack(codec.pack(value))[0]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('La clave debe ser finita')
        return value

    def _new_record(self, raw: RawRecord) -> _Record:
        token = self._pages.h.next_id
        if token >= 2 ** 63 - 1:
            raise OverflowError('Se agotaron los identificadores')
        self._pages.h.next_id += 1
        return _Record((raw[self.key_index], token), raw)

    def _matching(self, expression: Expression) -> Iterator[_Record]:
        self._ensure_open()
        expression = self._normalize_expression(expression)
        column = None
        literal = None
        if (isinstance(expression, Condition)
                and expression.operator == ComparisonOperator.EQ):
            left, right = expression.left, expression.right
            if isinstance(left, ColumnRef) and isinstance(right, LiteralValue):
                column, literal = left.name, right.value
            elif isinstance(left, LiteralValue) and isinstance(right, ColumnRef):
                column, literal = right.name, left.value
        if column == self.key_column:
            value = self._normalized_key(self.key_column, literal)
            records = self._pages.iter_records((value, -1), (value, 2 ** 63 - 1))
        else:
            records = self._pages.iter_records()
        for record in records:
            if evaluate_expression(record.values, expression, self.table_meta):
                yield record

    def _normalize_expression(self, expression: Expression) -> Expression:
        if isinstance(expression, Condition):
            left, right = expression.left, expression.right
            if isinstance(left, ColumnRef) and isinstance(right, LiteralValue):
                right = LiteralValue(self._normalized_key(left.name, right.value))
            elif isinstance(left, LiteralValue) and isinstance(right, ColumnRef):
                left = LiteralValue(self._normalized_key(right.name, left.value))
            return Condition(left, expression.operator, right)
        if isinstance(expression, LogicalExpression):
            return LogicalExpression(expression.operator,
                                     tuple(self._normalize_expression(operand)
                                           for operand in expression.operands))
        raise TypeError('Expresion no soportada')

    def insert(self, record: RawRecord) -> RID:
        self._ensure_open()
        raw = self._normalize(record)
        for col in self.table_meta.columns:
            if self.table_meta.is_unique_column(col['name']):
                expression = Condition(ColumnRef(col['name']), ComparisonOperator.EQ,
                                       LiteralValue(raw[self.table_meta.col_index(col['name'])]))
                if next(self._matching(expression), None) is not None:
                    raise ValueError(f"Clave duplicada: {col['name']}")
        item = self._new_record(raw)
        self._pages.insert(item)
        rid, _ = self._pages.search_with_rid(item.key)
        return rid.page_id, rid.slot_number

    def search(self, expression: Expression) -> CleanRecord | None:
        record = next(self._matching(expression), None)
        return self.table_meta.clean_tuple(record.values) if record is not None else None

    def search_all(self, expression: Expression) -> list[CleanRecord]:
        return [self.table_meta.clean_tuple(record.values)
                for record in self._matching(expression)]

    def range_search(self, key_column: str, begin_key: Key | None,
                     end_key: Key | None, include_begin: bool = True,
                     include_end: bool = True) -> list[CleanRecord]:
        self._ensure_open()
        index = self.table_meta.col_index(key_column)
        begin = None if begin_key is None else self._normalized_key(key_column, begin_key)
        end = None if end_key is None else self._normalized_key(key_column, end_key)
        if begin is not None and end is not None and begin > end:
            return []
        if key_column == self.key_column:
            low = None if begin is None else (begin, -1)
            high = None if end is None else (end, 2 ** 63 - 1)
            records = self._pages.iter_records(low, high)
        else:
            records = self._pages.iter_records()
        result = []
        for record in records:
            value = record.values[index]
            if begin is not None and (value < begin or (value == begin and not include_begin)):
                continue
            if end is not None and (value > end or (value == end and not include_end)):
                continue
            result.append(record.values)
        result.sort(key=lambda raw: raw[index])
        return [self.table_meta.clean_tuple(raw) for raw in result]

    def range_search_raw(self, key_column: str, begin_key: Key | None,
                         end_key: Key | None, include_begin: bool = True,
                         include_end: bool = True) -> list[RawRecord]:
        """Igual que range_search pero devuelve RawRecord sin limpiar."""
        self._ensure_open()
        index = self.table_meta.col_index(key_column)
        begin = None if begin_key is None else self._normalized_key(key_column, begin_key)
        end = None if end_key is None else self._normalized_key(key_column, end_key)
        if begin is not None and end is not None and begin > end:
            return []
        if key_column == self.key_column:
            low = None if begin is None else (begin, -1)
            high = None if end is None else (end, 2 ** 63 - 1)
            records = self._pages.iter_records(low, high)
        else:
            records = self._pages.iter_records()
        result = []
        for record in records:
            value = record.values[index]
            if begin is not None and (value < begin or (value == begin and not include_begin)):
                continue
            if end is not None and (value > end or (value == end and not include_end)):
                continue
            result.append(record.values)
        result.sort(key=lambda raw: raw[index])
        return result

    def delete(self, expression: Expression) -> RawRecord | None:
        record = next(self._matching(expression), None)
        if record is None:
            return None
        self._pages.delete(record.key)
        return record.values

    def delete_all(self, expression: Expression) -> list[RawRecord]:
        records = list(self._matching(expression))
        for record in records:
            self._pages.delete(record.key)
        return [record.values for record in records]

    def read(self, rid: RID) -> RawRecord | None:
        self._ensure_open()
        record = self._pages.read_rid(_RID(*rid))
        return record.values if record is not None else None

    def iter_records(self) -> Iterator[RawRecord]:
        self._ensure_open()
        for record in self._pages.iter_records():
            yield record.values

    def bulk_insert(self, records: Iterable[RawRecord],
                    buffer_size: int = 8 * 1024 * 1024,
                    replace_existing: bool = False) -> None:
        self._ensure_open()
        previous = self._pages.h.next_id
        incoming = (self._new_record(self._normalize(raw)) for raw in records)
        try:
            self._pages.bulk_insert(incoming, buffer_size, replace_existing=replace_existing)
        except BaseException:
            self._pages.h.next_id = previous
            raise

    def _csv_records(self, source_path: str, delimiter: str) -> Iterator[RawRecord]:
        with open(source_path, newline='', encoding='utf-8-sig') as source:
            reader = csv.reader(source, delimiter=delimiter)
            first = next(reader, None)
            if first is None:
                return
            names = [col['name'] for col in self.table_meta.columns]
            if [value.strip() for value in first] != names:
                yield self._normalize(first)
            for row in reader:
                yield self._normalize(row)

    @staticmethod
    def bulk_load(table_meta: TableMetadata, filepath: str, page_size: int,
                  source_path: str, delimiter: str = ',') -> SequentialFile:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'xb'):
            pass
        storage = None
        try:
            storage = SequentialFile(table_meta, filepath, page_size)
            storage.rebulk_load(source_path, delimiter)
            return storage
        except BaseException:
            if storage is not None:
                storage.close()
            os.unlink(filepath)
            raise

    def rebulk_load(self, source_path: str, delimiter: str = ',') -> None:
        self.bulk_insert(self._csv_records(source_path, delimiter), replace_existing=True)

    def reset(self) -> None:
        self._ensure_open()
        self._pages._replace_from_sorted(iter(()))

    def count(self) -> int:
        self._ensure_open()
        return self._pages.h.active_records

    def reorganize(self) -> None:
        self._ensure_open()
        self._pages.reorganize()

    def validate(self) -> dict:
        self._ensure_open()
        return self._pages.validate()

    def stats(self) -> dict:
        self._ensure_open()
        return self._pages.stats()

    def close(self) -> None:
        self._pages.close()

    def __enter__(self) -> SequentialFile:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
