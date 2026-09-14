from abc import ABC, abstractmethod

from engine.catalog import Key, RawRecord, CleanRecord

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
    def search(self, key_column: str, key: Key) -> CleanRecord | None:
        """Busca un registro por columna y clave."""
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
    def delete(self, key_column: str, key: Key) -> object | None:
        """Elimina un registro por columna y clave."""
        pass

    @abstractmethod
    def bulk_load(
        self,
        source_path: str,
        delimiter: str = ','
    ) -> None:
        """Carga registros en bloque."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cierra el archivo."""
        pass

    def reset_stats(self) -> None:
        """Reinicia las estadisticas."""
        self.page_reads = 0
        self.page_writes = 0