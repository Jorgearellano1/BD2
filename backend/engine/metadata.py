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
    RTREE = 'RTREE'
    INVERTED = 'INVERTED'
    IVF = 'IVF'


class Column(TypedDict):
    """Definicion de una columna del esquema de una tabla."""
    name: str
    type: str
    fmt: str
    size: int
    unique: NotRequired[bool]
    dim: NotRequired[int]


# Una clave : INT (int) , FLOAT  o  CHAR (bytes de longitud fija).
Key = int | float | bytes | str

# Un registro crudo tal como lo devuelve struct.unpack
# (tupla de longitud variable).
RawRecord = tuple[Any, ...]

# Un registro ya pasado por clean_tuple
# (CHAR -> str, FLOAT -> float, INT -> int, VECTOR -> tuple de floats).
CleanRecord = tuple[str | int | float | tuple[float, ...], ...]

# Referencia fisica de un registro dentro de un HeapFile: (page_id, slot_id).
RID = tuple[int, int]
RID_FORMAT = 'qq'
RID_SIZE = struct.calcsize(RID_FORMAT)


# Tipos logicos con longitud fija. CHAR se resuelve a partir de su tamano.
_FIXED_FMT: dict[str, str] = {
    'INT': 'i',
    'FLOAT': 'f',
    'DOUBLE': 'd',
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
        self.columns = [
            self._normalize_column(col)
            for col in columns
        ]
        self.primary_key = primary_key

        # Construir struct_format desde los formatos de cada columna.
        self.struct_format = '=' + ''.join(
            col['fmt'] for col in self.columns
        )

        # Calcular el tamaño total del registro.
        self.record_size = struct.calcsize(self.struct_format)

    @staticmethod
    def _normalize_column(col: Column) -> Column:
        """
        Completa y valida 'fmt' y 'size' de una columna a partir de su 'type'.

        Permite construir el esquema desde columnas ya resueltas o desde la
        representación lógica del catálogo (sin 'fmt').

        Para VECTOR el 'size' del catalogo es la dimension y el 'size'
        interno es la cantidad de bytes (4 por componente float32).
        """
        tipo = col['type'].upper()
        size = col.get('size')
        fmt = col.get('fmt')

        if tipo == 'CHAR':
            if size is None or size <= 0:
                raise ValueError('CHAR requiere un tamaño positivo')

            fmt = f'{size}s'
            size = struct.calcsize(fmt)

        elif tipo == 'VECTOR':
            dimension = col.get('dim', size)

            if dimension is None or dimension <= 0:
                raise ValueError('VECTOR requiere una dimension positiva')

            # Se guarda como un bloque de D float32 para que cada columna
            # ocupe exactamente un campo del struct del registro.
            fmt = f'{4 * dimension}s'
            size = 4 * dimension

        elif tipo in _FIXED_FMT:
            fmt = _FIXED_FMT[tipo]
            expected_size = struct.calcsize(fmt)

            if size is not None and size != expected_size:
                raise ValueError(
                    f'{tipo} requiere un tamaño de {expected_size} bytes'
                )

            size = expected_size

        else:
            raise ValueError(f'Tipo de columna no soportado: {tipo}')

        normalized: Column = {
            'name': col['name'],
            'type': tipo,
            'fmt': fmt,
            'size': size,
        }

        if tipo == 'VECTOR':
            normalized['dim'] = dimension

        if col.get('unique', False):
            normalized['unique'] = True

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """
        Representación lógica del esquema para el catálogo.

        Guarda:
            - name
            - type
            - size (solo para CHAR)
            - unique

        Los formatos físicos de INT, FLOAT y DOUBLE
        se reconstruyen mediante _normalize_column().
        """
        columns: list[dict[str, Any]] = []

        for col in self.columns:
            column: dict[str, Any] = {
                'name': col['name'],
                'type': col['type'],
            }

            if col['type'] == 'CHAR':
                column['size'] = col['size']

            elif col['type'] == 'VECTOR':
                column['size'] = col['dim']

            column['unique'] = bool(col.get('unique', False))
            columns.append(column)

        return {
            'columns': columns,
            'primary_key': self.primary_key,
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: dict[str, Any],
    ) -> TableMetadata:
        """
        Reconstruye el esquema desde la representación lógica del catálogo.

        Los formatos físicos se regeneran mediante _normalize_column(),
        incluyendo DOUBLE con formato 'd' y tamaño de 8 bytes.
        """
        columns: list[Column] = []

        for col in data['columns']:
            columns.append({
                'name': col['name'],
                'type': col['type'],
                'fmt': '',
                'size': col.get('size'),
                'unique': bool(col.get('unique', False)),
            })

        return cls(
            name,
            columns,
            data.get('primary_key'),
        )

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

    def normalize_value(self, column_name: str, value: Any) -> int | float | bytes | tuple[float, ...]:
        """
        Convierte un valor Python a su representación binaria según el esquema.
            - INT    : int(value)
            - FLOAT  : float(value), almacenamiento de 32 bits
            - DOUBLE : float(value), almacenamiento de 64 bits
            - CHAR   : bytes de longitud fija (truncado y rellenado con b'\\x00')
            - VECTOR : tupla de D floats
        """
        col = self.columns[self.col_index(column_name)]
        tipo = col['type'].upper()

        if tipo == 'INT':
            return int(value)

        if tipo in ('FLOAT', 'DOUBLE'):
            return float(value)

        if tipo == 'CHAR':
            size = col['size']

            if isinstance(value, str):
                value = value.encode('utf-8')

            if len(value) < size:
                value = value + b'\x00' * (size - len(value))

            return value[:size]

        if tipo == 'VECTOR':
            from vectorindex.vectors import (
                VectorError,
                parse_vector,
                vector_to_bytes,
            )

            if isinstance(value, str):
                try:
                    vector = parse_vector(value)
                except VectorError as exc:
                    raise ValueError(str(exc)) from exc
            elif isinstance(value, (list, tuple)):
                vector = tuple(float(item) for item in value)
            else:
                raise ValueError(
                    f"La columna '{column_name}' espera un vector '[a, b, ...]'"
                )

            dimension = col['dim']

            if len(vector) != dimension:
                raise ValueError(
                    f"La columna '{column_name}' espera {dimension} "
                    f"componentes y recibio {len(vector)}"
                )

            return vector_to_bytes(vector)

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
            - CHAR   : str sin padding nulo
            - FLOAT  : redondeado a 2 decimales
            - DOUBLE : conserva su precisión
            - INT    : sin cambio
            - VECTOR : tuple de floats

        Devuelve None si raw_tuple es None.
        """
        if raw_tuple is None:
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
            elif tipo == 'VECTOR':
                from vectorindex.vectors import vector_from_bytes

                values = vector_from_bytes(val, col['dim'])
                cleaned.append(tuple(round(item, 6) for item in values))
            else:
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
    spatial_columns: tuple[str, str] | None = None
    # Opciones de WITH (...) de CREATE INDEX; solo las usa INVERTED.
    options: dict[str, Any] | None = None
