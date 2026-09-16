from __future__ import annotations

import struct
from typing import Any, NotRequired, TypedDict


PAGE_SIZE: int = 8192

# Cada columna del esquema especifica directamente 'fmt' y 'size',
# por lo que TYPE_MAP solo se necesita como referencia para tipos sin longitud fija.
TYPE_MAP: dict[str, dict[str, str | int]] = {
    'INT':   {'fmt': 'i', 'size': 4},
    'FLOAT': {'fmt': 'f', 'size': 4},
    # CHAR usa el campo 'fmt' de la columna directamente
    # (ej: '30s', '20s', '10s')
}


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



class TableMetadata:
    """
    Metadata de una tabla con esquema de registros de longitud fija.

    Cada columna debe ser un diccionario con:
        name  : str  : nombre de la columna
        type  : str  : 'INT', 'FLOAT' o 'CHAR'
        fmt   : str  : formato struct (ej: 'i', 'f', '30s')
        size  : int  : tamano en bytes
        unique: bool : (opcional) True si la columna tiene UNIQUE

    La primary_key puede ser None (tabla sin PK). Una columna es unica si es
    la primary_key o si tiene unique=True.
    """

    name: str
    columns: list[Column]
    primary_key: str | None
    struct_format: str
    record_size: int
    block_factor: int

    def __init__(
        self,
        name: str,
        columns: list[Column],
        primary_key: str | None = None,
    ) -> None:
        self.name = name
        self.columns = columns
        self.primary_key = primary_key

        # Construir struct_format desde los formatos de cada columna.
        self.struct_format = '=' + ''.join(col['fmt'] for col in self.columns)
        self.record_size = struct.calcsize(self.struct_format)

        # Registros que caben en una pagina, descontando el header de pagina.
        self.block_factor = (PAGE_SIZE - 4) // self.record_size

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