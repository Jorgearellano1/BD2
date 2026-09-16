from __future__ import annotations

import os
import struct
import heapq
import time
from typing import Any, BinaryIO, Iterator

# Un registro del archivo temporal (tupla de longitud variable,
# ej: (key, page_id, slot_id)).
Record = tuple[Any, ...]


class ExternalSort:
    """
    External Sort de dos fases (generate runs + k-way merge).

    El formato de registro es el de cada entrada del indice temporal,
    tipicamente: key_fmt + RID_FORMAT  (ej: 'iqq' para clave INT y
    referencia (page_id, slot_id)).

    La pagina temporal tiene:
        header (4 bytes): numero de registros validos en la pagina
        body:             registros contiguos de record_size bytes cada uno
        padding:          ceros hasta completar page_size bytes
    """

    record_format: str
    page_size: int
    buffer_size: int
    B: int
    record_size: int
    header_format: str
    header_size: int
    records_per_page: int
    pages_read: int
    pages_written: int

    def __init__(self, record_format: str, page_size: int, buffer_size: int) -> None:
        self.record_format = record_format
        self.page_size = page_size
        self.buffer_size = buffer_size

        # B: numero de paginas que caben en el buffer de memoria
        self.B = buffer_size // page_size

        if self.B < 2:
            raise ValueError("buffer_size debe permitir al menos 2 paginas")

        self.record_size = struct.calcsize(record_format)

        # Header: solo el numero de registros validos (4 bytes)
        self.header_format = 'i'
        self.header_size = struct.calcsize(self.header_format)

        # Registros que caben en el area de datos de cada pagina
        self.records_per_page = (self.page_size - self.header_size) // self.record_size

        self.pages_read = 0
        self.pages_written = 0

    def external_sort(self, heap_path: str, output_path: str, sort_key_index: int) -> dict[str, int | float]:
        """
        Ordena los registros de heap_path y escribe el resultado en output_path.
        Retorna estadisticas de la operacion.
        """
        start_total = time.time()

        # Fase 1: generar runs ordenados
        start_p1 = time.time()
        runs = self.generate_runs(heap_path, sort_key_index)
        time_phase1 = time.time() - start_p1

        # Fase 2: k-way merge iterativo
        start_p2 = time.time()
        k = self.B - 1  # buffers de entrada disponibles (1 reservado para salida)

        intermediate_count = 0

        while len(runs) > k:
            batch = runs[:k]
            runs = runs[k:]

            intermediate_path = f"temp_runs/inter_{intermediate_count}.bin"
            self.multiway_merge(batch, intermediate_path, sort_key_index)

            runs.append(intermediate_path)
            intermediate_count += 1

        self.multiway_merge(runs, output_path, sort_key_index)
        time_phase2 = time.time() - start_p2

        end_total = time.time()

        return {
            'runs_generated': len(runs),
            'pages_read': self.pages_read,
            'pages_written': self.pages_written,
            'time_phase1_sec': round(time_phase1, 4),
            'time_phase2_sec': round(time_phase2, 4),
            'time_total_sec': round(end_total - start_total, 4),
        }

    def generate_runs(self, heap_path: str, sort_key_index: int) -> list[str]:
        """
        Lee B paginas a la vez, las ordena en memoria y las escribe
        como archivos temporales de run ordenado.
        Retorna la lista de rutas de los runs generados.
        """
        if not os.path.exists(heap_path):
            return []

        run_paths: list[str] = []
        total_pages = os.path.getsize(heap_path) // self.page_size
        current_page = 0
        run_count = 0

        os.makedirs("temp_runs", exist_ok=True)

        with open(heap_path, 'rb') as f:
            while current_page < total_pages:
                pages_to_read = min(self.B, total_pages - current_page)
                memory_records: list[Record] = []

                for _ in range(pages_to_read):
                    page_data = f.read(self.page_size)
                    self.pages_read += 1
                    memory_records.extend(self._extract_records(page_data))
                    current_page += 1

                # Ordenar en memoria por la clave
                memory_records.sort(key=lambda x: x[sort_key_index])

                # Escribir run ordenado a disco
                run_filename = f"temp_runs/run_{run_count}.bin"
                self._write_run(run_filename, memory_records)
                run_paths.append(run_filename)
                run_count += 1

        return run_paths

    def multiway_merge(self, run_paths: list[str], output_path: str, sort_key_index: int) -> None:
        """
        Realiza un k-way merge de los runs usando un min-heap.
        Escribe el resultado ordenado en output_path.
        Usa B-1 buffers de entrada y 1 buffer de salida.
        """
        min_heap: list[tuple[Any, int, Record]] = []
        run_files: list[BinaryIO] = [open(path, 'rb') for path in run_paths]
        run_iters: list[Iterator[Record]] = []
        output_buffer: list[Record] = []

        for i, f in enumerate(run_files):
            first_page = f.read(self.page_size)
            if first_page:
                self.pages_read += 1
                records = self._extract_records(first_page)
                if records:
                    first_rec = records.pop(0)
                    heapq.heappush(min_heap, (first_rec[sort_key_index], i, first_rec))
                    run_iters.append(self._run_record_generator(f, records))
                else:
                    run_iters.append(iter([]))
            else:
                run_iters.append(iter([]))

        with open(output_path, 'wb') as out_f:
            while min_heap:
                val, run_idx, record = heapq.heappop(min_heap)
                output_buffer.append(record)

                if len(output_buffer) == self.records_per_page:
                    self._write_binary_page(out_f, output_buffer)
                    self.pages_written += 1
                    output_buffer = []

                try:
                    next_rec = next(run_iters[run_idx])
                    heapq.heappush(min_heap, (next_rec[sort_key_index], run_idx, next_rec))
                except StopIteration:
                    pass

            # Registros restantes en el buffer de salida
            if output_buffer:
                self._write_binary_page(out_f, output_buffer)

        for f in run_files:
            f.close()
        for path in run_paths:
            os.remove(path)

    def _extract_records(self, page_data: bytes) -> list[Record]:
        """Extrae los registros validos de una pagina."""
        num_records = struct.unpack(self.header_format, page_data[:self.header_size])[0]
        records: list[Record] = []

        for i in range(num_records):
            offset = self.header_size + i * self.record_size
            record_bytes = page_data[offset:offset + self.record_size]
            records.append(struct.unpack(self.record_format, record_bytes))

        return records

    def _write_run(self, path: str, records: list[Record]) -> None:
        """Escribe una lista de registros ordenados en un archivo de run."""
        with open(path, 'wb') as f:
            current_page_records: list[Record] = []
            for rec in records:
                current_page_records.append(rec)
                if len(current_page_records) == self.records_per_page:
                    self._write_binary_page(f, current_page_records)
                    self.pages_written += 1
                    current_page_records = []

            if current_page_records:
                self._write_binary_page(f, current_page_records)

    def _write_binary_page(self, file_handle: BinaryIO, records: list[Record]) -> None:
        """Serializa una lista de registros como una pagina de tamano fijo."""
        page_buffer = bytearray()

        # Header: numero de registros validos
        page_buffer.extend(struct.pack(self.header_format, len(records)))

        for rec in records:
            page_buffer.extend(struct.pack(self.record_format, *rec))

        padding = self.page_size - len(page_buffer)
        if padding > 0:
            page_buffer.extend(b'\x00' * padding)

        file_handle.write(page_buffer)

    def _run_record_generator(self, file_handle: BinaryIO, initial_records: list[Record]) -> Iterator[Record]:
        """
        Generador que entrega registros uno a uno desde un archivo de run,
        leyendo paginas del disco cuando el buffer inicial se agota.
        """
        for r in initial_records:
            yield r

        while True:
            page_data = file_handle.read(self.page_size)
            if not page_data:
                break
            self.pages_read += 1
            for r in self._extract_records(page_data):
                yield r
