from __future__ import annotations

import json
import os

from engine.metadata import (
    IndexKind,
    IndexMetadata,
    StorageKind,
    TableEntry,
    TableMetadata,
)
from indexes.bplusTree import BPlusTree
from indexes.index import Index
from storage.heapFile import HeapFile
from storage.sequentialFile import SequentialFile
from storage.storageFile import StorageFile


CATALOG_VERSION = 1


class Catalog:
    """
    Interfaz entre el sistema y 'catalog.json'.

    Mantiene todas las tablas (TableEntry) y todos los indices
    (IndexMetadata). El JSON solo guarda metadata logica/de ubicacion:
    nombres, columnas, tipos, primary key, unique, storage, rutas e
    informacion de indices.
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

        # Auxiliares en memoria (no se persisten en catalog.json).
        self._index_targets: dict[tuple[str, str, IndexKind], str] = {}
        self._table_indexes: dict[str, set[str]] = {}

        self.load()

    # Persistencia
    def load(self) -> None:
        """Carga el catalogo desde disco. Si no existe, empieza vacio."""
        self._tables = {}
        self._indexes = {}
        self._index_targets = {}
        self._table_indexes = {}

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
            index = IndexMetadata(
                name=name,
                table=raw['table'],
                column=raw['column'],
                kind=IndexKind(raw['type']),
                file_path=raw['file'],
            )

            target = (index.table, index.column, index.kind)
            existing = self._index_targets.get(target)
            if existing is not None:
                raise ValueError(
                    f"Catalogo inconsistente: '{name}' y '{existing}' son "
                    f"indices {index.kind.value} sobre "
                    f"'{index.table}({index.column})'"
                )

            self._indexes[name] = index
            self._index_targets[target] = name
            self._table_indexes.setdefault(index.table, set()).add(name)

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

    # Tablas
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
        for index_name in self._table_indexes.pop(name, set()):
            index = self._indexes.pop(index_name, None)
            if index is None:
                continue
            target = (index.table, index.column, index.kind)
            if self._index_targets.get(target) == index_name:
                del self._index_targets[target]

        self.save()

    # Indices
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

        target = (index.table, index.column, index.kind)
        existing = self._index_targets.get(target)
        if existing is not None:
            raise ValueError(
                f"Ya existe un indice {index.kind.value} sobre "
                f"'{index.table}({index.column})': '{existing}'"
            )

        self._indexes[index.name] = index
        self._index_targets[target] = index.name
        self._table_indexes.setdefault(index.table, set()).add(index.name)
        self.save()

    def remove_index(self, name: str) -> None:
        if name not in self._indexes:
            raise ValueError(f"El indice '{name}' no existe en el catalogo")

        index = self._indexes.pop(name)

        target = (index.table, index.column, index.kind)
        if self._index_targets.get(target) == name:
            del self._index_targets[target]

        names = self._table_indexes.get(index.table)
        if names is not None:
            names.discard(name)
            if not names:
                del self._table_indexes[index.table]

        self.save()

    def get_table_indexes(self, table_name: str) -> list[IndexMetadata]:
        names = self._table_indexes.get(table_name)
        if not names:
            return []
        return [self._indexes[name] for name in sorted(names)]

    def find_index(
        self,
        table: str,
        column: str,
        kind: IndexKind,
    ) -> IndexMetadata | None:
        """
        Busca un indice compatible por (table, column, kind) en O(1).
        No decide reutilizacion; solo informa.
        """
        name = self._index_targets.get((table, column, kind))
        if name is None:
            return None
        return self._indexes[name]

    def rename_index(self, old_name: str, new_name: str) -> None:
        """
        Cambia solo el nombre logico de un indice conservando table, column,
        kind y file_path. No toca archivos fisicos ni recalcula file_path.
        """
        if old_name == new_name:
            return

        if old_name not in self._indexes:
            raise ValueError(
                f"El indice '{old_name}' no existe en el catalogo"
            )

        if new_name in self._indexes:
            raise ValueError(
                f"El indice '{new_name}' ya existe en el catalogo"
            )

        index = self._indexes[old_name]
        renamed = IndexMetadata(
            name=new_name,
            table=index.table,
            column=index.column,
            kind=index.kind,
            file_path=index.file_path,
        )

        del self._indexes[old_name]
        self._indexes[new_name] = renamed

        target = (index.table, index.column, index.kind)
        self._index_targets[target] = new_name

        names = self._table_indexes.setdefault(index.table, set())
        names.discard(old_name)
        names.add(new_name)

        self.save()

    # Rutas fisicas canonicas
    def table_path(self, name: str, storage: StorageKind) -> str:
        extension = '.heap' if storage is StorageKind.HEAP else '.seq'
        return os.path.join(self.storage_dir, name + extension)

    def index_path(
        self,
        table: str,
        column: str,
        kind: IndexKind,
    ) -> str:
        """
        Ruta canonica de un indice. BTREE -> .dat, HASH -> .hash.
        HASH queda contemplado aunque su integracion este pendiente.
        """
        if kind is IndexKind.BTREE:
            filename = f'{table}_{column}_bplustree.dat'
        else:
            filename = f'{table}_{column}.hash'
        return os.path.join(self.storage_dir, filename)

    # Resolucion de estructuras fisicas
    def open_storage(self, table_name: str) -> StorageFile:
        """Abre (o crea) la estructura fisica de una tabla."""
        entry = self.get_table(table_name)

        if entry.storage is StorageKind.HEAP:
            return HeapFile(entry.metadata, entry.file_path, self.page_size)

        return SequentialFile(entry.metadata, entry.file_path, self.page_size)

    def open_index(self, index_name: str, heap) -> Index:
        """
        Abre el indice indicado sobre el almacenamiento compartido dado.

        Solo BTREE esta integrado. HASH ya tiene metadata y ruta, pero
        permanece pendiente de integracion en el catalogo.
        """
        index = self.get_index(index_name)

        if index.kind is not IndexKind.BTREE:
            raise NotImplementedError(
                f"Indice {index.kind.value} no implementado: '{index_name}'"
            )

        entry = self.get_table(index.table)

        return BPlusTree(
            entry.metadata,
            index.column,
            heap,
            self.page_size,
            index_path=index.file_path,
        )
