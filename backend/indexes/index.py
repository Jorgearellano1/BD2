from abc import ABC, abstractmethod

from engine.catalog import Key, RawRecord, CleanRecord, RID


class Index(ABC):
    """Interfaz comun para estructuras de indice."""

    def __init__(self) -> None:
        # Estadisticas de E/S por pagina.
        self.page_reads: int = 0
        self.page_writes: int = 0

    # Operaciones propias del indice

    @abstractmethod
    def insert(self, key: Key, rid: RID) -> bool:
        """Inserta una clave y su RID."""
        pass

    @abstractmethod
    def search(self, key: Key) -> RID | None:
        """Busca una clave y retorna su RID."""
        pass

    @abstractmethod
    def search_all(self, key: Key) -> list[RID]:
        """Busca todos los RIDs asociados a una clave."""
        pass

    @abstractmethod
    def range_search(
        self,
        begin_key: Key | None,
        end_key: Key | None,
        include_begin: bool = True,
        include_end: bool = True
    ) -> list[RID]:
        """
        Busca y retorna los RIDs dentro del rango indicado.

        None representa ausencia de limite.
        include_begin/include_end indican si los limites son inclusivos.
        """
        pass

    @abstractmethod
    def delete(self, key: Key, rid: RID | None = None) -> RID | None:
        """
        Elimina una entrada y retorna su RID.
        Con rid=None elimina la primera entrada que coincida con key.
        Con rid dado elimina exactamente la entrada (key, rid).
        """
        pass

    # Operaciones que combinan este indice con su almacenamiento asociado
    #
    # Estas operaciones coordinan unicamente:
    #   - este indice
    #   - el almacenamiento compartido
    #
    # No sincronizan otros indices que puedan utilizar el mismo
    # almacenamiento. La coordinacion entre multiples indices corresponde
    # a una capa superior (por ejemplo, Table).

    @abstractmethod
    def bulk_load(
        self,
        source_path: str,
        delimiter: str = ','
    ) -> None:
        """
        Carga los registros en el almacenamiento y construye este indice.
        """
        pass

    @abstractmethod
    def search_record(self, key: Key) -> CleanRecord | None:
        """
        Busca en este indice y obtiene el registro correspondiente
        desde el almacenamiento.
        """
        pass

    @abstractmethod
    def search_records(self, key: Key) -> list[CleanRecord]:
        """
        Obtiene desde el almacenamiento todos los registros asociados
        a la clave encontrada por este indice.
        """
        pass

    @abstractmethod
    def range_search_records(
        self,
        begin_key: Key | None,
        end_key: Key | None,
        include_begin: bool = True,
        include_end: bool = True
    ) -> list[CleanRecord]:
        """
        Retorna los registros del almacenamiento cuyos RIDs pertenecen
        al rango indicado por este indice.

        None representa ausencia de limite.
        include_begin/include_end indican si los limites son inclusivos.
        """
        pass

    @abstractmethod
    def insert_record(self, record: RawRecord) -> RID | None:
        """
        Almacena el registro y agrega su (key, RID) a este indice.

        Si este indice es UNIQUE y la insercion viola la unicidad,
        la operacion debe evitar dejar un registro huerfano en el
        almacenamiento.
        """
        pass

    @abstractmethod
    def delete_record(self, key: Key, rid: RID | None = None) -> CleanRecord | None:
        """
        Elimina un registro de este indice y del almacenamiento asociado.

        Con rid=None elimina la primera entrada que coincida con key.
        Con rid dado elimina exactamente la entrada (key, rid).

        La operacion afecta unicamente a este indice y al almacenamiento;
        no sincroniza otros indices que compartan dicho almacenamiento.
        """
        pass

    @abstractmethod
    def build_from_storage(self) -> None:
        """
        Construye este indice a partir de los registros existentes
        en el almacenamiento asociado.

        No modifica los registros del almacenamiento ni otros indices.
        """
        pass
    # Operaciones generales del indice

    @abstractmethod
    def close(self) -> None:
        """Cierra el archivo del indice."""
        pass

    def reset_stats(self) -> None:
        """Reinicia las estadisticas."""
        self.page_reads = 0
        self.page_writes = 0