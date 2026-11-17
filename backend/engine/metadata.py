from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, NotRequired, TypedDict


class StorageKind(str, Enum):
    """Tipo de almacenamiento fisico de una tabla."""
    HEAP = 'HEAP'
    SEQUENTIAL = 'SEQUENTIAL'


class IndexKind(str, Enum):
    """Tipo de estructura de un indice."""
    BTREE = 'BTREE'
    HASH = 'HASH'
    INVERTED = 'INVERTED'


class Column(TypedDict):
    """Definicion de una columna del esquema de una tabla."""
    name: str
    type: str
    fmt: str
    size: int
    unique: NotRequired[bool]


# Una clave : INT (int) , FLOAT  o  CHAR (bytes de longitud fija).
Key = int | float | bytes | str

# Un registro crudo tal como lo devuelve struct.unpack
# (tupla de longitud variable).
RawRecord = tuple[Any, ...]

# Un registro ya pasado por clean_tuple
# (CHAR -> str, FLOAT -> float, INT -> int).
CleanRecord = tuple[str | int | float, ...]

# Referencia fisica de un registro dentro de un HeapFile: (page_id, slot_id).
RID = tuple[int, int]
RID_FORMAT = 'qq'
RID_SIZE = struct.calcsize(RID_FORMAT)


# Tipos logicos con longitud fija. CHAR se resuelve a partir de su tamano.
_FIXED_FMT: dict[str, str] = {
    'INT': 'i',
    'FLOAT': 'f',
}


class TableMetadata:
    """
    Metadata de una tabla con esquema de registros de longitud fija.

    Cada columna debe ser un diccionario con:
        name  : str  : nombre de la columna
        type  : str  : 'INT', 'FLOAT' o 'CHAR'
        fmt   : str  : formato struct (ej: 'i', 'f', '30s')
        size  : int  : tamano en bytes
        unique: bool : (opcional) True si la columna tiene UNIQUE

    Una columna es unica si es la primary_key o si tiene unique=True.

    Representa unicamente el esquema y la serializacion logica de los
    registros. No conoce archivos, indices, catalogo ni paginas.
    """

    name: str
    columns: list[Column]
    primary_key: str | None
    struct_format: str
    record_size: int

    def __init__(
        self,
        name: str,
        columns: list[Column],
        primary_key: str | None = None,
    ) -> None:
        self.name = name
        self.columns = [self._normalize_column(col) for col in columns]
        self.primary_key = primary_key

        # Construir struct_format desde los formatos de cada columna.
        self.struct_format = '=' + ''.join(col['fmt'] for col in self.columns)
        self.record_size = struct.calcsize(self.struct_format)

    @staticmethod
    def _normalize_column(col: Column) -> Column:
        """
        Completa y valida 'fmt' y 'size' de una columna a partir de su 'type'.
        Permite construir el esquema desde columnas ya resueltas o desde la
        representacion logica del catalogo (sin 'fmt').
        """
        tipo = col['type'].upper()
        size = col.get('size')
        fmt = col.get('fmt')

        if not fmt or size is None:
            if tipo == 'CHAR':
                if size is None or size <= 0:
                    raise ValueError('CHAR requiere un tamano positivo')
                fmt = f'{size}s'
            elif tipo in _FIXED_FMT:
                fmt = _FIXED_FMT[tipo]
                size = size if size is not None else struct.calcsize(fmt)
            else:
                raise ValueError(f'Tipo de columna no soportado: {tipo}')

        normalized: Column = {
            'name': col['name'],
            'type': tipo,
            'fmt': fmt,
            'size': size,
        }
        if col.get('unique', False):
            normalized['unique'] = True
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Representacion logica del esquema para el catalogo."""
        columns: list[dict[str, Any]] = []
        for col in self.columns:
            column: dict[str, Any] = {
                'name': col['name'],
                'type': col['type'],
            }
            if col['type'] == 'CHAR':
                column['size'] = col['size']
            column['unique'] = bool(col.get('unique', False))
            columns.append(column)

        return {
            'columns': columns,
            'primary_key': self.primary_key,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> TableMetadata:
        """Reconstruye el esquema desde la representacion logica del catalogo."""
        columns: list[Column] = []
        for col in data['columns']:
            columns.append({
                'name': col['name'],
                'type': col['type'],
                'fmt': '',
                'size': col.get('size'),
                'unique': bool(col.get('unique', False)),
            })
        return cls(name, columns, data.get('primary_key'))

    def col_index(self, col_name: str) -> int:
        """Devuelve el indice (0-based) de una columna por nombre."""
        for i, col in enumerate(self.columns):
            if col['name'] == col_name:
                return i

        raise ValueError(
            f"Columna '{col_name}' no encontrada en '{self.name}'"
        )

    def is_unique_column(self, column_name: str) -> bool:
        """
        True si la columna debe ser unica: es la primary_key o tiene UNIQUE
        (Column.unique == True).
        """
        if self.primary_key is not None and column_name == self.primary_key:
            return True
        col = self.columns[self.col_index(column_name)]
        return bool(col.get('unique', False))

    def normalize_value(self, column_name: str, value: Any) -> int | float | bytes:
        """
        Convierte un valor Python a su representacion binaria segun el esquema.
          - INT   : int(value)
          - FLOAT : float(value)
          - CHAR  : bytes de longitud fija (truncado y rellenado con b'\\x00')
        """
        col  = self.columns[self.col_index(column_name)]
        tipo = col['type'].upper()

        if tipo == 'INT':
            return int(value)
        if tipo == 'FLOAT':
            return float(value)
        if tipo == 'CHAR':
            size = col['size']
            if isinstance(value, str):
                value = value.encode('utf-8')
            if len(value) < size:
                value = value + b'\x00' * (size - len(value))
            return value[:size]

        raise ValueError(f"Tipo de columna no soportado: {tipo}")

    def normalize_tuple(self, values: list[str]) -> RawRecord:
        """
        Convierte una fila de valores al RawRecord equivalente segun el esquema.
        """
        normalized: list[Any] = []
        for i, col in enumerate(self.columns):
            value = values[i]
            if isinstance(value, str):
                value = value.strip()
            normalized.append(self.normalize_value(col['name'], value))
        return tuple(normalized)

    def clean_tuple(
        self,
        raw_tuple: RawRecord | None,
    ) -> CleanRecord | None:
        """
        Convierte una tupla binaria a valores Python legibles:
          - CHAR (bytes): str sin padding nulo
          - FLOAT       : redondeado a 2 decimales
          - INT         : sin cambio

        Devuelve None si raw_tuple es None.
        """
        if not raw_tuple:
            return None

        cleaned: list[Any] = []

        for col, val in zip(self.columns, raw_tuple):
            tipo = col['type'].upper()

            if tipo == 'CHAR':
                cleaned.append(
                    val.decode('utf-8', errors='replace').rstrip('\x00')
                )
            elif tipo == 'FLOAT':
                cleaned.append(round(val, 2))
            else:  # INT u otros numericos
                cleaned.append(val)

        return tuple(cleaned)


@dataclass
class TableEntry:
    """Una tabla dentro del catalogo: esquema + ubicacion fisica."""
    name: str
    metadata: TableMetadata
    storage: StorageKind
    file_path: str


@dataclass
class IndexMetadata:
    """Un indice dentro del catalogo: tabla/columna + ubicacion fisica."""
    name: str
    table: str
    column: str
    kind: IndexKind
    file_path: str
    # Opciones de WITH (...) de CREATE INDEX; solo las usa INVERTED.
    options: dict[str, Any] | None = None
