from __future__ import annotations

from abc import ABC, abstractmethod

from engine.metadata import Key, RawRecord, CleanRecord, TableMetadata
from engine.conditions import Expression


class StorageFile(ABC):
    """Interfaz comun para estructuras de almacenamiento, heap file, sequential file"""

    def __init__(self) -> None:
        # Estadisticas de E/S por pagina.
        self.page_reads: int = 0
        self.page_writes: int = 0

    @abstractmethod
    def insert(self, record: RawRecord) -> object:
        """Inserta un registro y retorna su identificador."""
        pass

    @abstractmethod
    def search(
        self,
        expression: Expression
    ) -> CleanRecord | None:
        """
        Busca un registro que cumpla todas las condiciones.
        """
        pass

    @abstractmethod
    def search_all(
        self,
        expression: Expression
    ) -> list[CleanRecord]:
        """
        Busca todos los registros que cumplan todas las condiciones.
        """
        pass

    @abstractmethod
    def range_search(
        self,
        key_column: str,
        begin_key: Key | None,
        end_key: Key | None,
        include_begin: bool = True,
        include_end: bool = True
    ) -> list[CleanRecord]:
        """
        Busca registros de una columna dentro del rango indicado.

        None representa ausencia de limite.
        include_begin/include_end indican si los limites son inclusivos.
        """
        pass

    @abstractmethod
    def delete(
        self,
        expression: Expression
    ) -> RawRecord | None:
        """
        Elimina un registro que cumpla todas las condiciones
        y retorna el registro eliminado.
        """
        pass

    @abstractmethod
    def delete_all(
        self,
        expression: Expression
    ) -> list[RawRecord]:
        """
        Elimina todos los registros que cumplan todas las condiciones
        y retorna los registros eliminados.
        """
        pass

    @staticmethod
    @abstractmethod
    def bulk_load(
        table_meta: TableMetadata,
        filepath: str,
        page_size: int,
        source_path: str,
        delimiter: str = ','
    ) -> StorageFile:
        """Crea un almacenamiento nuevo y lo carga desde el origen (CREATE)."""
        pass

    @abstractmethod
    def rebulk_load(
        self,
        source_path: str,
        delimiter: str = ','
    ) -> None:
        """Reemplaza el contenido actual por el del origen de forma segura."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Vacia el almacenamiento existente."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Cantidad de registros activos (sin recorrer paginas)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cierra el archivo."""
        pass

    def reset_stats(self) -> None:
        """Reinicia las estadisticas."""
        self.page_reads = 0
        self.page_writes = 0
