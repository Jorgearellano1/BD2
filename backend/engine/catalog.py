from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from engine.metadata import (
    IndexKind,
    IndexMetadata,
    StorageKind,
    TableEntry,
    TableMetadata,
)

if TYPE_CHECKING:
    from indexes.index import Index
    from storage.storageFile import StorageFile


CATALOG_VERSION = 1


class Catalog:
    """
    Interfaz entre el sistema y 'catalog.json'.

    Mantiene todas las tablas (TableEntry) y todos los indices
    (IndexMetadata). El JSON solo guarda metadata logica/de ubicacion:
    nombres, columnas, tipos, primary key, unique, storage, rutas e
    informacion de indices.

    No guarda registros, RIDs, paginas, nodos, buckets ni el contenido de
    las estructuras fisicas, que siguen siendo responsabilidad de cada
    archivo. El page_size es configuracion externa y no se persiste.
    """

    def __init__(
        self,
        catalog_path: str,
        storage_dir: str,
        page_size: int,
    ) -> None:
        self.catalog_path = catalog_path
        self.storage_dir = storage_dir
        self.page_size = page_size

        self._tables: dict[str, TableEntry] = {}
        self._indexes: dict[str, IndexMetadata] = {}

        self.load()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carga el catalogo desde disco. Si no existe, empieza vacio."""
        self._tables = {}
        self._indexes = {}

        if not os.path.exists(self.catalog_path):
            return

        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        version = data.get('version', CATALOG_VERSION)
        if version != CATALOG_VERSION:
            raise ValueError(
                f'Version de catalogo no soportada: {version}'
            )

        for name, raw in data.get('tables', {}).items():
            metadata = TableMetadata.from_dict(name, raw)
            self._tables[name] = TableEntry(
                name=name,
                metadata=metadata,
                storage=StorageKind(raw['storage']),
                file_path=raw['file'],
            )

        for name, raw in data.get('indexes', {}).items():
            self._indexes[name] = IndexMetadata(
                name=name,
                table=raw['table'],
                column=raw['column'],
                kind=IndexKind(raw['type']),
                file_path=raw['file'],
            )

    def save(self) -> None:
        """Guarda el catalogo en disco de forma atomica."""
        data = {
            'version': CATALOG_VERSION,
            'tables': {
                name: {
                    'storage': entry.storage.value,
                    'file': entry.file_path,
                    **entry.metadata.to_dict(),
                }
                for name, entry in self._tables.items()
            },
            'indexes': {
                name: {
                    'table': idx.table,
                    'column': idx.column,
                    'type': idx.kind.value,
                    'file': idx.file_path,
                }
                for name, idx in self._indexes.items()
            },
        }

        directory = os.path.dirname(self.catalog_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        tmp_path = self.catalog_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.catalog_path)

    # ------------------------------------------------------------------
    # Tablas
    # ------------------------------------------------------------------

    def table_exists(self, name: str) -> bool:
        return name in self._tables

    def get_table(self, name: str) -> TableEntry:
        if name not in self._tables:
            raise ValueError(f"La tabla '{name}' no existe en el catalogo")
        return self._tables[name]

    def add_table(self, entry: TableEntry) -> None:
        if entry.name in self._tables:
            raise ValueError(
                f"La tabla '{entry.name}' ya existe en el catalogo"
            )
        self._tables[entry.name] = entry
        self.save()

    def remove_table(self, name: str) -> None:
        if name not in self._tables:
            raise ValueError(f"La tabla '{name}' no existe en el catalogo")
        del self._tables[name]

        # Un indice no puede sobrevivir a su tabla.
        for index_name in [
            index_name
            for index_name, idx in self._indexes.items()
            if idx.table == name
        ]:
            del self._indexes[index_name]

        self.save()

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------

    def index_exists(self, name: str) -> bool:
        return name in self._indexes

    def get_index(self, name: str) -> IndexMetadata:
        if name not in self._indexes:
            raise ValueError(f"El indice '{name}' no existe en el catalogo")
        return self._indexes[name]

    def add_index(self, index: IndexMetadata) -> None:
        if index.name in self._indexes:
            raise ValueError(
                f"El indice '{index.name}' ya existe en el catalogo"
            )
        self._indexes[index.name] = index
        self.save()

    def remove_index(self, name: str) -> None:
        if name not in self._indexes:
            raise ValueError(f"El indice '{name}' no existe en el catalogo")
        del self._indexes[name]
        self.save()

    def get_table_indexes(self, table_name: str) -> list[IndexMetadata]:
        return [
            index
            for index in self._indexes.values()
            if index.table == table_name
        ]

    # ------------------------------------------------------------------
    # Rutas fisicas canonicas
    # ------------------------------------------------------------------

    def table_path(self, name: str, storage: StorageKind) -> str:
        extension = '.heap' if storage is StorageKind.HEAP else '.seq'
        return os.path.join(self.storage_dir, name + extension)

    def index_path(
        self,
        table: str,
        column: str,
        kind: IndexKind,
    ) -> str:
        if kind is IndexKind.BTREE:
            filename = f'{table}_{column}_bplustree.dat'
        else:
            filename = f'{table}_{column}.hash'
        return os.path.join(self.storage_dir, filename)

    # ------------------------------------------------------------------
    # Resolucion de estructuras fisicas
    # ------------------------------------------------------------------

    def open_storage(self, table_name: str) -> StorageFile:
        """Abre (o crea) la estructura fisica de una tabla."""
        entry = self.get_table(table_name)

        if entry.storage is StorageKind.HEAP:
            from storage.heapFile import HeapFile

            return HeapFile(entry.metadata, entry.file_path, self.page_size)

        from storage.sequentialFile import SequentialFile

        return SequentialFile(entry.metadata, entry.file_path, self.page_size)

    def open_index(self, index_name: str, heap) -> Index:
        """Abre el indice indicado sobre el almacenamiento compartido dado."""
        index = self.get_index(index_name)

        if index.kind is not IndexKind.BTREE:
            raise NotImplementedError(
                f"Indice {index.kind.value} no implementado: '{index_name}'"
            )

        entry = self.get_table(index.table)
        data_dir = os.path.dirname(index.file_path) or self.storage_dir

        from indexes.bplusTree import BPlusTree

        return BPlusTree(
            entry.metadata,
            index.column,
            heap,
            self.page_size,
            data_dir=data_dir,
        )
