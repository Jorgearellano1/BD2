from __future__ import annotations

import csv
import os
import struct
import tempfile
from collections import defaultdict
from typing import BinaryIO, Iterable, Iterator

from engine.metadata import TableMetadata, RawRecord, RID, Key, CleanRecord
from engine.conditions import Expression, evaluate_expression
from storageFile import StorageFile


class HeapFile(StorageFile):
    FILE_HEADER_FORMAT: str = '=qiq'# free_head, total_pages, total_records
    FILE_HEADER_SIZE: int = struct.calcsize(FILE_HEADER_FORMAT)


    # metadata del record
    # -1: activo
    # -2: eliminado y fin de free list
    # >= 0: eliminado y apunta al siguiente free
    RECORD_META_FORMAT: str = 'q'
    RECORD_META_SIZE: int = struct.calcsize(RECORD_META_FORMAT)


    PAGE_HEADER_FORMAT: str = 'i'
    PAGE_HEADER_SIZE: int = struct.calcsize(PAGE_HEADER_FORMAT)

    table_meta: TableMetadata
    filepath: str
    PAGE_SIZE: int
    RECORD_SIZE: int
    BLOCK_FACTOR: int
    file: BinaryIO | None
    page_reads: int
    page_writes: int

    def __init__(self, table_meta: TableMetadata, filepath: str, page_size: int) -> None:

        super().__init__()

        self.table_meta = table_meta
        self.filepath = filepath
        self.PAGE_SIZE = page_size

        self.RECORD_SIZE = (self.RECORD_META_SIZE + self.table_meta.record_size )

        self.BLOCK_FACTOR = (self.PAGE_SIZE - self.PAGE_HEADER_SIZE) // self.RECORD_SIZE

        # crear directorio contenedor si es necesario
        parent_dir = os.path.dirname(self.filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # crear archivo
        if not os.path.exists(self.filepath):

            with open(self.filepath, 'wb') as f:
                # free_head = -1
                # total_pages = 0
                # total_records = 0
                f.write(struct.pack( self.FILE_HEADER_FORMAT, -1, 0, 0))

        # descriptor abierto durante toda la vida del objeto
        self.file = open(self.filepath, 'r+b')

    def close(self) -> None:
        """Cierra el archivo del heap. Idempotente."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def is_empty(self) -> bool:
        """True si el HeapFile no tiene registros activos (total_records == 0)."""
        _, _, total_records = self._read_header()
        return total_records == 0

    def count(self) -> int:
        """Numero de registros activos, leido del header (sin recorrer paginas)."""
        _, _, total_records = self._read_header()
        return total_records

    def reset(self) -> None:
        """Deja el HeapFile vacio conservando el descriptor abierto."""
        f = self.file
        f.seek(0)
        f.truncate()
        # free_head = -1
        # total_pages = 0
        # total_records = 0
        f.write(struct.pack(self.FILE_HEADER_FORMAT, -1, 0, 0))

    def _iter_active_records(self) -> Iterator[tuple[RID, RawRecord]]:
        """Aplana iter_pages_with_records() a pares (RID, registro activo)."""
        for page_records in self.iter_pages_with_records():
            yield from page_records

    def _resolve_key(self, key_column: str, key: Key) -> tuple[int, int | float | bytes]:
        """Resuelve el indice de la columna y normaliza la clave de busqueda."""
        key_idx = self.table_meta.col_index(key_column)
        target  = self.table_meta.normalize_value(key_column, key)
        return key_idx, target

    def search(self, expression: Expression) -> CleanRecord | None:
        """Busca el primer registro activo que cumpla la expresion."""
        for _, record in self._iter_active_records():
            if evaluate_expression(record, expression, self.table_meta):
                return self.table_meta.clean_tuple(record)
        return None

    def search_all(self, expression: Expression) -> list[CleanRecord]:
        """Retorna todos los registros activos que cumplan la expresion."""
        results: list[CleanRecord] = []
        for _, record in self._iter_active_records():
            if evaluate_expression(record, expression, self.table_meta):
                cleaned = self.table_meta.clean_tuple(record)
                if cleaned is not None:
                    results.append(cleaned)
        return results

    def range_search(
        self,
        key_column: str,
        begin_key: Key | None,
        end_key: Key | None,
        include_begin: bool = True,
        include_end: bool = True,
    ) -> list[CleanRecord]:
        """
        Registros con key_column dentro del rango, ordenados por esa columna.
        None en un limite significa ausencia de limite.
        include_begin/include_end indican si los limites son inclusivos.
        """
        key_idx = self.table_meta.col_index(key_column)

        if begin_key is not None:
            _, begin = self._resolve_key(key_column, begin_key)
        else:
            begin = None
        if end_key is not None:
            _, end = self._resolve_key(key_column, end_key)
        else:
            end = None

        results: list[CleanRecord] = []
        for _, record in self._iter_active_records():
            record_key = record[key_idx]

            if begin is not None:
                if include_begin:
                    if record_key < begin:
                        continue
                elif record_key <= begin:
                    continue

            if end is not None:
                if include_end:
                    if record_key > end:
                        continue
                elif record_key >= end:
                    continue

            cleaned = self.table_meta.clean_tuple(record)
            if cleaned is not None:
                results.append(cleaned)

        results.sort(key=lambda r: r[key_idx])
        return results

    def delete(self, expression: Expression) -> RawRecord | None:
        """Elimina el primer registro activo que cumpla la expresion."""
        for rid, record in self._iter_active_records():
            if evaluate_expression(record, expression, self.table_meta):
                return self.remove(rid)
        return None

    def delete_all(self, expression: Expression) -> list[RawRecord]:
        """Elimina todos los registros activos que cumplan la expresion."""
        rids: list[RID] = []
        for rid, record in self._iter_active_records():
            if evaluate_expression(record, expression, self.table_meta):
                rids.append(rid)

        removed: list[RawRecord] = []
        for rid in rids:
            deleted = self.remove(rid)
            if deleted is not None:
                removed.append(deleted)
        return removed

    #header
    def _read_header(self) -> tuple[int, int, int]:
        self.file.seek(0)
        return struct.unpack(self.FILE_HEADER_FORMAT, self.file.read(self.FILE_HEADER_SIZE))

    def _write_header(self, free_head: int, total_pages: int, total_records: int) -> None:
        self.file.seek(0)
        self.file.write(struct.pack(self.FILE_HEADER_FORMAT, free_head, total_pages, total_records))

    ## page
    def _page_offset(self, page_id: int) -> int:
        return self.FILE_HEADER_SIZE + page_id * self.PAGE_SIZE

    def _record_offset_in_page(self, slot: int) -> int:
        return self.PAGE_HEADER_SIZE + slot * self.RECORD_SIZE

    def _rid_to_offset(self, rid: RID) -> int:
        """Convierte una referencia (page_id, slot_id) a offset fisico."""
        page_id, slot_id = rid
        return self._page_offset(page_id) + self._record_offset_in_page(slot_id)

    def _offset_to_rid(self, offset: int) -> RID:
        """Convierte un offset fisico de registro a (page_id, slot_id)."""
        page_id = (offset - self.FILE_HEADER_SIZE) // self.PAGE_SIZE
        local   = offset - self._page_offset(page_id)
        slot_id = (local - self.PAGE_HEADER_SIZE) // self.RECORD_SIZE
        return page_id, slot_id

    #insert
    def insert(self, record_tuple: RawRecord) -> RID:
        f = self.file
        f.seek(0)
        free_head, total_pages, total_records = struct.unpack(
            self.FILE_HEADER_FORMAT, f.read(self.FILE_HEADER_SIZE)
        )

        #usar free list
        if free_head != -1:

            f.seek(free_head)

            meta_raw = f.read(self.RECORD_META_SIZE)
            next_free, = struct.unpack(self.RECORD_META_FORMAT, meta_raw)

            #payload usuario
            payload = struct.pack(self.table_meta.struct_format, *record_tuple)
            #metadata activa
            meta = struct.pack(self.RECORD_META_FORMAT, -1)

            #escribir record
            f.seek(free_head)
            f.write(meta + payload)

            # actualizar free list
            if next_free == -2:
                next_free = -1

            f.seek(0)
            f.write(struct.pack(
                self.FILE_HEADER_FORMAT, next_free, total_pages, total_records + 1
            ))
            return self._offset_to_rid(free_head)

        #append simple

        # primera pagina
        if total_pages == 0:
            page_id = 0
            self._write_page(f, page_id, [record_tuple])
            f.seek(0)
            f.write(struct.pack(self.FILE_HEADER_FORMAT, -1, 1, 1))

            return (0, 0)

        #ultima pagina
        last_page = total_pages - 1

        f.seek(self._page_offset(last_page))

        page_data = bytearray(f.read(self.PAGE_SIZE))
        self.page_reads += 1
        num_records = struct.unpack(self.PAGE_HEADER_FORMAT, page_data[:self.PAGE_HEADER_SIZE])[0]

        #cabe en pagina
        if num_records < self.BLOCK_FACTOR:
            local_offset = self._record_offset_in_page(num_records)
            meta = struct.pack(self.RECORD_META_FORMAT, -1)

            payload = struct.pack(
                self.table_meta.struct_format,
                *record_tuple
            )

            full_record = meta + payload

            page_data[local_offset:local_offset + self.RECORD_SIZE] = full_record

            self._write_page_raw(f, last_page, page_data, num_records + 1)

            f.seek(0)
            f.write(struct.pack(
                self.FILE_HEADER_FORMAT, free_head, total_pages, total_records + 1
            ))

            return (last_page, num_records)

        #nueva pagina
        else:
            new_page = total_pages

            self._write_page(f, new_page, [record_tuple])
            f.seek(0)
            f.write(struct.pack(
                self.FILE_HEADER_FORMAT, -1, total_pages + 1, total_records + 1
            ))

            return (new_page, 0)
            
    def remove(self, rid: RID) -> RawRecord | None:
        offset = self._rid_to_offset(rid)

        f = self.file
        f.seek(0)
        free_head, total_pages, total_records = struct.unpack(
            self.FILE_HEADER_FORMAT, f.read(self.FILE_HEADER_SIZE)
        )

        f.seek(offset)

        raw = f.read(self.RECORD_SIZE)

        next_free, = struct.unpack(self.RECORD_META_FORMAT, raw[:self.RECORD_META_SIZE])

        # ya eliminado
        if next_free != -1:
            return None

        #recuperar record
        payload = raw[self.RECORD_META_SIZE:]
        deleted_record = struct.unpack(self.table_meta.struct_format, payload)

        #metadata nueva
        new_next = free_head if free_head != -1 else -2
        new_meta = struct.pack(self.RECORD_META_FORMAT, new_next)

        f.seek(offset)
        f.write(new_meta)

        # actualizar free list
        f.seek(0)
        f.write(struct.pack(
            self.FILE_HEADER_FORMAT, offset, total_pages, total_records - 1
        ))

        return deleted_record

    def read(self, rid: RID) -> RawRecord | None:
        offset = self._rid_to_offset(rid)

        f = self.file
        f.seek(offset)
        raw = f.read(self.RECORD_SIZE)

        next_free, = struct.unpack(self.RECORD_META_FORMAT, raw[:self.RECORD_META_SIZE])

        #liminado
        if next_free != -1:
            return None

        payload = raw[self.RECORD_META_SIZE:]

        return struct.unpack(self.table_meta.struct_format,payload)

    #read varios
    def read_many(self, rids: list[RID]) -> list[RawRecord]:
        pages: dict[int, list[tuple[int, int]]] = defaultdict(list)

        # agrupar slots por pagina guardando la posicion original
        for position, (page_id, slot_id) in enumerate(rids):
            pages[page_id].append((position, slot_id))

        results: list[RawRecord | None] = [None] * len(rids)

        f = self.file

        for page_id, slots in pages.items():

            f.seek(self._page_offset(page_id))
            page_data = f.read(self.PAGE_SIZE)
            self.page_reads += 1

            for position, slot_id in slots:
                local = self._record_offset_in_page(slot_id)

                raw = page_data[local:local + self.RECORD_SIZE]
                next_free, = struct.unpack(self.RECORD_META_FORMAT, raw[:self.RECORD_META_SIZE])

                # eliminado
                if next_free != -1:
                    continue

                payload = raw[self.RECORD_META_SIZE:]
                record = struct.unpack(self.table_meta.struct_format, payload)
                results[position] = record
        return [r for r in results if r is not None]
    

    def _bulk_insert(
        self,
        records: Iterable[RawRecord],
        reset: bool = False,
    ) -> Iterator[tuple[RawRecord, RID]]:
        """
        Nucleo fisico de la insercion masiva: escribe los registros pagina por
        pagina y entrega cada par (registro, RID) a medida que se escribe,
        sin usar insert() registro por registro.
        """
        f = self.file

        # reiniciar heap solo si se pide
        if reset:
            f.seek(0)
            f.truncate()
            # free_head = -1
            # total_pages = 0
            # total_records = 0
            f.write(struct.pack(self.FILE_HEADER_FORMAT,-1,0,0))

        # leer header actual
        f.seek(0)
        free_head, total_pages, total_records = struct.unpack(
            self.FILE_HEADER_FORMAT,
            f.read(self.FILE_HEADER_SIZE)
        )
        current_page = total_pages
        page_records: list[RawRecord] = []
        written = 0
        for record_tuple in records:
            page_records.append(record_tuple)
            # pagina llena
            if len(page_records) >= self.BLOCK_FACTOR:
                self._write_page( f,current_page, page_records)

                # referencias (page_id, slot_id)
                for slot, record in enumerate(page_records):
                    yield record, (current_page, slot)
                current_page += 1
                written += len(page_records)
                page_records.clear()
        # ultima pagina parcial
        if page_records:
            self._write_page(f,current_page,page_records)

            for slot, record in enumerate(page_records):
                yield record, (current_page, slot)
            current_page += 1
            written += len(page_records)

        # actualizar header final
        f.seek(0)
        f.write(struct.pack(
            self.FILE_HEADER_FORMAT, -1, current_page, total_records + written
        ))

    def bulk_insert(self, records: Iterable[RawRecord], reset: bool = False) -> list[RID]:
        """Inserta en bloque y retorna la lista de RIDs."""
        return [rid for _, rid in self._bulk_insert(records, reset=reset)]

    def bulk_insert_with_rids(
        self,
        records: Iterable[RawRecord],
        reset: bool = False,
    ) -> Iterator[tuple[RawRecord, RID]]:
        """Inserta en bloque entregando cada par (registro, RID)."""
        return self._bulk_insert(records, reset=reset)
    
    def _write_page(self, f: BinaryIO, page_id: int, records: list[RawRecord]) -> None:

        f.seek(self._page_offset(page_id))

        # header pagina
        header = struct.pack(self.PAGE_HEADER_FORMAT, len(records))

        body: list[bytes] = []

        for record_tuple in records:
            meta = struct.pack(self.RECORD_META_FORMAT, -1)
            payload = struct.pack(self.table_meta.struct_format, *record_tuple)
            body.append(meta + payload)

        body_bytes = b''.join(body)
        padding = b'\x00' * (self.PAGE_SIZE - self.PAGE_HEADER_SIZE - len(body_bytes))

        f.write(header +body_bytes +padding)
        self.page_writes += 1

    def _write_page_raw(self, f: BinaryIO, page_id: int, page_data: bytearray, num_records: int) -> None:
        header = struct.pack(self.PAGE_HEADER_FORMAT, num_records)

        f.seek(self._page_offset(page_id))
        f.write(header + page_data[self.PAGE_HEADER_SIZE:])
        self.page_writes += 1


    def iter_pages_with_records(
        self,
    ) -> Iterator[list[tuple[RID, RawRecord]]]:
        """
        Itera sobre el HeapFile pagina por pagina.

        Cada elemento generado es una lista de pares:
            ((page_id, slot_id), registro_crudo)

        Solo se carga una pagina del HeapFile en memoria por vez.
        """
        f = self.file

        # Leer cantidad total de paginas
        f.seek(0)
        _, total_pages, _ = struct.unpack(self.FILE_HEADER_FORMAT, f.read(self.FILE_HEADER_SIZE))

        for page_id in range(total_pages):
            page_data = f.read(self.PAGE_SIZE)
            self.page_reads += 1

            if len(page_data) < self.PAGE_SIZE:
                break

            num_records = struct.unpack(self.PAGE_HEADER_FORMAT, page_data[:self.PAGE_HEADER_SIZE])[0]
            page_records: list[tuple[RID, RawRecord]] = []

            for slot in range(num_records):
                local_offset = self._record_offset_in_page(slot)
                raw = page_data[local_offset:local_offset + self.RECORD_SIZE]

                next_free, = struct.unpack( self.RECORD_META_FORMAT, raw[:self.RECORD_META_SIZE])

                # Ignorar registros eliminados
                if next_free != -1:
                    continue

                payload = raw[self.RECORD_META_SIZE:]
                record = struct.unpack( self.table_meta.struct_format, payload)

                page_records.append(((page_id, slot), record))

            yield page_records

    @staticmethod
    def _prepend_row(first_row: list[str], reader: Iterable[list[str]]) -> Iterator[list[str]]:
        """Generador que antepone first_row al iterador reader."""
        yield first_row
        yield from reader

    def _parsed_row_generator(self, rows: Iterable[list[str]]) -> Iterator[RawRecord]:
        """
        Convierte cada fila del CSV a una tupla binaria compatible con el esquema.
        Tipos soportados: INT, FLOAT, CHAR (bytes de longitud fija).
        """
        for row in rows:
            yield self.table_meta.normalize_tuple(row)

    def _matches_schema(self, row: list[str]) -> bool:
        """
        Comprueba si una fila es compatible con el esquema (no es cabecera).
        Retorna True si parece una fila de datos.
        """
        try:
            self.table_meta.normalize_tuple(row)
            return True
        except (ValueError, IndexError):
            return False

    def _csv_record_generator(self, csv_path: str, delimiter: str = ',') -> Iterator[RawRecord]:
        """
        Lee un CSV y genera RawRecords segun el esquema de la tabla.
        Detecta automaticamente si la primera fila es cabecera.
        """
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader    = csv.reader(f, delimiter=delimiter)
            first_row = next(reader, None)

            if first_row is None:
                return

            # detectar si la primera fila es cabecera
            if self._matches_schema(first_row):
                rows = self._prepend_row(first_row, reader)
            else:
                rows = reader

            yield from self._parsed_row_generator(rows)

    def rebulk_load(self, csv_path: str, delimiter: str = ',') -> None:
        """
        Reemplaza el contenido del HeapFile por el del CSV de forma segura:
        construye el nuevo Heap en un archivo temporal y hace
        os.replace(temp -> original) solo si todo termina correctamente.
        Ante un fallo, el Heap anterior queda intacto.
        """
        orig_path  = self.filepath
        parent_dir = os.path.dirname(orig_path) or '.'

        fd, tmp_path = tempfile.mkstemp(prefix='rebulk_', suffix='.heap', dir=parent_dir)
        os.close(fd)

        # cerrar/flush el descriptor actual antes de tocar nada
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

        self.filepath = tmp_path
        try:
            self.file = open(tmp_path, 'r+b')
            # consumir el generador incrementalmente (no materializar los RIDs)
            for _ in self._bulk_insert(
                self._csv_record_generator(csv_path, delimiter), reset=True
            ):
                pass

            # cerrar/flush antes de os.replace
            self.file.flush()
            self.file.close()
            self.file = None

            os.replace(tmp_path, orig_path)
            self.filepath = orig_path
            self.file = open(orig_path, 'r+b')
        except Exception:
            if self.file is not None:
                self.file.close()
                self.file = None
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.filepath = orig_path
            self.file = open(orig_path, 'r+b')
            raise

    @staticmethod
    def bulk_load(
        table_meta: TableMetadata,
        filepath: str,
        page_size: int,
        source_path: str,
        delimiter: str = ',',
    ) -> HeapFile:
        """
        Crea un HeapFile nuevo y lo carga desde CSV (semantica CREATE).
        Falla si el archivo destino ya existe, sin sobrescribirlo.
        """
        if os.path.exists(filepath):
            raise FileExistsError(f"El HeapFile ya existe: {filepath}")

        heap = HeapFile(table_meta, filepath, page_size)
        heap.rebulk_load(source_path, delimiter)
        return heap
    
