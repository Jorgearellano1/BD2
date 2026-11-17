"""
Acceso a archivos por paginas de B bytes.

El modulo vectorial no puede usar el PagedFile del modulo textual porque
cada indice tiene su propio formato de cabecera, pero el acceso (seek +
read/write de bloques de tamano fijo) es el mismo. Esta clase encapsula ese
acceso y lleva la cuenta de lecturas y escrituras que el Executor expone
como metricas de E/S.
"""

from __future__ import annotations

import os


class PagedFileError(RuntimeError):
    """Error de lectura o escritura de un archivo paginado."""


class PagedFile:
    """Archivo binario accedido exclusivamente por paginas de page_size bytes."""

    def __init__(self, path: str, page_size: int, mode: str = 'rb') -> None:
        if page_size <= 0:
            raise PagedFileError('El tamano de pagina debe ser positivo.')

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
            raise PagedFileError('La pagina excede el tamano de bloque.')

        self.file.seek(page_no * self.page_size)
        self.file.write(data.ljust(self.page_size, b'\x00'))
        self.writes += 1

    def num_pages(self) -> int:
        self.file.seek(0, os.SEEK_END)
        return self.file.tell() // self.page_size

    def flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
