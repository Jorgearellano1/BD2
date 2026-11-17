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


CATALOG_VERSION = 1


class Catalog:
    """
    Interfaz entre el sistema y 'catalog.json'.

    Mantiene todas las tablas (TableEntry) y todos los indices
    (IndexMetadata). El JSON solo guarda metadata logica/de ubicacion:
    nombres, columnas, tipos, primary key, unique, storage, rutas e
    informacion de indices.

    No abre estructuras fisicas; esa responsabilidad corresponde al Executor.
    """


    def __init__(
        self,
        catalog_path: str,
        storage_dir: str,
    ) -> None:
        self.catalog_path = catalog_path
        self.storage_dir = storage_dir

        self._tables: dict[str, TableEntry] = {}
        self._indexes: dict[str, IndexMetadata] = {}

        # Auxiliares en memoria (no se persisten en catalog.json).
        self._index_targets: dict[
            tuple[str, str | tuple[str, str], IndexKind],
            str
        ] = {}

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
                spatial_columns=(
                    tuple(raw['spatial_columns'])
                    if raw.get('spatial_columns')
                    else None
                ),
            )

            if index.kind is IndexKind.RTREE:
                if (
                    index.spatial_columns is None
                    or len(index.spatial_columns) != 2
                ):
                    raise ValueError(
                        f"Catalogo inconsistente: el indice R-Tree "
                        f"'{name}' debe tener dos columnas espaciales"
                    )

                target = (
                    index.table,
                    index.spatial_columns,
                    index.kind
                )
                column_description = (
                    f"{index.spatial_columns[0]}, "
                    f"{index.spatial_columns[1]}"
                )

            else:
                target = (index.table, index.column, index.kind)
                column_description = index.column

            existing = self._index_targets.get(target)

            if existing is not None:
                raise ValueError(
                    f"Catalogo inconsistente: '{name}' y '{existing}' son "
                    f"indices {index.kind.value} sobre "
                    f"'{index.table}({column_description})'"
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
                    'spatial_columns': (
                        list(idx.spatial_columns)
                        if idx.spatial_columns is not None
                        else None
                    ),
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

    def tables(self) -> list[TableEntry]:
        """Todas las tablas ordenadas por nombre."""
        return [self._tables[name] for name in sorted(self._tables)]

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

    def update_table(self, entry: TableEntry) -> None:
        """Reemplaza la metadata de una tabla existente."""
        if entry.name not in self._tables:
            raise ValueError(
                f"La tabla '{entry.name}' no existe en el catalogo"
            )
        self._tables[entry.name] = entry
        self.save()

    
    def remove_table(self, name: str) -> None:
        if name not in self._tables:
            raise ValueError(
                f"La tabla '{name}' no existe en el catalogo"
            )

        del self._tables[name]

        # Un indice no puede sobrevivir a su tabla.
        for index_name in self._table_indexes.pop(name, set()):
            index = self._indexes.pop(index_name, None)

            if index is None:
                continue

            if index.kind is IndexKind.RTREE:
                target = (
                    index.table,
                    index.spatial_columns,
                    index.kind,
                )
            else:
                target = (
                    index.table,
                    index.column,
                    index.kind,
                )

            if self._index_targets.get(target) == index_name:
                del self._index_targets[target]

        self.save()


    # Indices
    def index_exists(self, name: str) -> bool:
        return name in self._indexes

    def indexes(self) -> list[IndexMetadata]:
        """Todos los indices ordenados por nombre."""
        return [self._indexes[name] for name in sorted(self._indexes)]

    def get_index(self, name: str) -> IndexMetadata:
        if name not in self._indexes:
            raise ValueError(f"El indice '{name}' no existe en el catalogo")
        return self._indexes[name]


    
    def add_index(self, index: IndexMetadata) -> None:
        if index.name in self._indexes:
            raise ValueError(
                f"El indice '{index.name}' ya existe en el catalogo"
            )

        if index.kind is IndexKind.RTREE:
            if (
                index.spatial_columns is None
                or len(index.spatial_columns) != 2
            ):
                raise ValueError(
                    "Un indice R-Tree debe tener exactamente "
                    "dos columnas espaciales"
                )

            target = (
                index.table,
                index.spatial_columns,
                index.kind
            )
            column_description = (
                f"{index.spatial_columns[0]}, {index.spatial_columns[1]}"
            )

        else:
            target = (index.table, index.column, index.kind)
            column_description = index.column

        existing = self._index_targets.get(target)

        if existing is not None:
            raise ValueError(
                f"Ya existe un indice {index.kind.value} sobre "
                f"'{index.table}({column_description})': '{existing}'"
            )

        self._indexes[index.name] = index
        self._index_targets[target] = index.name
        self._table_indexes.setdefault(index.table, set()).add(index.name)

        self.save()



    def remove_index(self, name: str) -> None:
        if name not in self._indexes:
            raise ValueError(
                f"El indice '{name}' no existe en el catalogo"
            )

        index = self._indexes.pop(name)

        if index.kind is IndexKind.RTREE:
            target = (
                index.table,
                index.spatial_columns,
                index.kind
            )
        else:
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
        column: str | tuple[str, str],
        kind: IndexKind,
    ) -> IndexMetadata | None:
        """
        Busca un indice compatible por (table, column, kind) en O(1).
        Para R-Tree, column contiene las dos columnas espaciales.
        """
        name = self._index_targets.get((table, column, kind))

        if name is None:
            return None

        return self._indexes[name]


    
    
    def rename_index(self, old_name: str, new_name: str) -> None:
        """
        Cambia solo el nombre logico de un indice conservando table, column,
        kind, file_path y spatial_columns.
        No toca archivos fisicos ni recalcula file_path.
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
            spatial_columns=index.spatial_columns,
        )

        del self._indexes[old_name]
        self._indexes[new_name] = renamed

        if index.kind is IndexKind.RTREE:
            target = (
                index.table,
                index.spatial_columns,
                index.kind,
            )
        else:
            target = (
                index.table,
                index.column,
                index.kind,
            )

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
        column: str | tuple[str, str],
        kind: IndexKind,
    ) -> str:
        """Obtiene la ruta del archivo fisico correspondiente a un indice."""

        if kind is IndexKind.BTREE:
            filename = f'{table}_{column}_bplustree.dat'

        elif kind is IndexKind.HASH:
            filename = f'{table}_{column}.hash'

        elif kind is IndexKind.RTREE:
            if not isinstance(column, tuple) or len(column) != 2:
                raise ValueError(
                    "Un indice R-Tree requiere exactamente dos columnas espaciales"
                )

            first_column, second_column = column
            filename = (
                f'{table}_{first_column}_{second_column}_rtree.dat'
            )

        else:
            raise ValueError(f'Tipo de indice no soportado: {kind}')

        return os.path.join(self.storage_dir, filename)


