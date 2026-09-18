from __future__ import annotations

import struct
import os
import tempfile
from typing import BinaryIO, Iterable, Iterator, overload

from engine.metadata import TableMetadata, Column, RawRecord, CleanRecord
from engine.metadata import RID, RID_FORMAT, RID_SIZE, Key
from storage.heapFile import HeapFile
from algorithms.external_sort import ExternalSort
from indexes.index import Index


# Entrada del nivel de hojas / archivos temporales: (clave, RID).
Entry = tuple[Key, RID]
# Entrada de un nivel durante la construccion bottom-up: (first_key, offset de nodo).
LevelEntry = tuple[Key, int]

# Limite por defecto (bytes) para la metadata de un nivel (LevelEntry) en RAM
# durante el bulk build. Independiente de PAGE_SIZE y de sort_buffer_size.
DEFAULT_LEVEL_MEMORY_LIMIT: int = 64 * 1024 * 1024


class _DuplicateKeyError(ValueError):
    """Clave duplicada en una columna UNIQUE."""


class BPlusNode:
    """
    Representa un nodo del B+ Tree en memoria.

    Layout de pagina (PAGE_SIZE bytes):
        HEADER  : node_type(B) | is_root(B) | num_keys(i) | parent_ptr(q) | next_leaf(q)
        BODY    : claves + punteros de hijo (interno) / pares (clave, RID) (hoja)
        PADDING : ceros hasta completar PAGE_SIZE

    Para nodos internos:
        body = [k0..k_{n-1}] [p0..p_n]
    Para hojas:
        body = [(k0,(page_id,slot_id)) .. (k_{n-1},(page_id,slot_id))]
    """

    HEADER_FORMAT: str = 'BBiqq'   # node_type, is_root, num_keys, parent_ptr, next_leaf
    HEADER_SIZE: int   = struct.calcsize(HEADER_FORMAT)
    INT_SIZE: int      = struct.calcsize('i')
    PTR_FORMAT: str    = 'q'
    PTR_SIZE: int      = struct.calcsize(PTR_FORMAT)

    key_fmt: str
    key_size: int
    ORDER: int
    PAGE_SIZE: int
    node_type: int
    is_root: int
    num_keys: int
    parent_ptr: int
    next_leaf: int
    keys: list[Key]
    pointers: list[int | RID]

    def __init__(
        self,
        key_fmt: str,
        key_size: int,
        order: int,
        page_size: int,
        node_type: int = 1,
        is_root: int = 0,
        num_keys: int = 0,
        parent_ptr: int = -1,
        next_leaf: int = -1,
    ) -> None:
        self.key_fmt   = key_fmt
        self.key_size  = key_size
        self.ORDER     = order      # maximo de claves por nodo
        self.PAGE_SIZE = page_size

        self.node_type  = node_type   # 0 = interno, 1 = hoja
        self.is_root    = is_root
        self.num_keys   = num_keys
        self.parent_ptr = parent_ptr
        self.next_leaf  = next_leaf

        self.keys     = []
        self.pointers = []

    def pack_header(self) -> bytes:
        return struct.pack(
            self.HEADER_FORMAT,
            self.node_type,
            self.is_root,
            self.num_keys,
            self.parent_ptr,
            self.next_leaf,
        )

    @staticmethod
    def unpack_header(data: bytes) -> tuple[int, int, int, int, int]:
        return struct.unpack(BPlusNode.HEADER_FORMAT, data)

    def pack(self) -> bytes:
        header_data  = self.pack_header()
        MAX_BODY_SIZE = self.PAGE_SIZE - self.HEADER_SIZE
        chunks = []

        if self.node_type == 0:  # nodo interno
            # claves
            for i in range(self.num_keys):
                chunks.append(struct.pack(self.key_fmt, self.keys[i]))
            # padding claves
            chunks.append(b'\x00' * ((self.ORDER - self.num_keys) * self.key_size))
            # punteros
            for i in range(self.num_keys + 1):
                chunks.append(struct.pack(self.PTR_FORMAT, self.pointers[i]))
            # padding punteros
            chunks.append(
                b'\x00' * ((self.ORDER + 1 - (self.num_keys + 1)) * self.PTR_SIZE)
            )
        else:  # hoja
            entry_format = '=' + self.key_fmt + RID_FORMAT
            for i in range(self.num_keys):
                page_id, slot_id = self.pointers[i]
                chunks.append(
                    struct.pack(entry_format, self.keys[i], page_id, slot_id)
                )
            current_size  = self.num_keys * (self.key_size + RID_SIZE)
            padding_size  = MAX_BODY_SIZE - current_size
            if padding_size > 0:
                chunks.append(b'\x00' * padding_size)

        body_data = b''.join(chunks)
        if len(body_data) < MAX_BODY_SIZE:
            body_data += b'\x00' * (MAX_BODY_SIZE - len(body_data))

        return header_data + body_data[:MAX_BODY_SIZE]

    @classmethod
    def unpack(cls, data: bytes, key_fmt: str, key_size: int, order: int, page_size: int) -> BPlusNode:
        node_type, is_root, num_keys, parent_ptr, next_leaf = struct.unpack_from(
            cls.HEADER_FORMAT, data, 0
        )

        node = cls(
            key_fmt, key_size, order, page_size,
            node_type, is_root, num_keys, parent_ptr, next_leaf,
        )

        body_start = cls.HEADER_SIZE
        ptr_format = cls.PTR_FORMAT

        if node.node_type == 0:  # interno
            keys_end = body_start + num_keys * key_size
            node.keys = list(
                struct.unpack_from(key_fmt * num_keys, data, body_start)
            )
            ptr_start = keys_end + (order - num_keys) * key_size
            node.pointers = list(
                struct.unpack_from(ptr_format * (num_keys + 1), data, ptr_start)
            )
        else:  # hoja
            entry_format = '=' + key_fmt + RID_FORMAT
            entry_size   = key_size + RID_SIZE
            entries_end  = body_start + num_keys * entry_size
            if num_keys > 0:
                keys, page_ids, slot_ids = zip(
                    *struct.iter_unpack(
                        entry_format, memoryview(data)[body_start:entries_end]
                    )
                )
            else:
                keys, page_ids, slot_ids = (), (), ()
            node.keys = list(keys)
            node.pointers = list(zip(page_ids, slot_ids))

        return node


class BPlusTree(Index):
    """
    B+ Tree con clave generica (INT o CHAR) y registros almacenados en un
    HeapFile compartido.

    Metodos estaticos de creacion:
        CSV       -> bulk_load()          (crea un indice nuevo)
        CSV       -> bulk_load_multiple() (crea varios indices nuevos, 1 pasada)
        HeapFile  -> build_from_storage() (crea un indice nuevo)
    Operaciones de instancia sobre un indice existente:
        rebuild_from_storage() (reconstruye el indice desde el HeapFile)
        clear()                (vacia solo el indice)

    El HeapFile es compartido: BPlusTree no lo crea ni lo cierra. No existe
    BPlusTree.rebulk_load(); la recarga completa de Heap + indices corresponde
    a una capa superior.

    Consultas:
        search(key) entrega RID; search_record(key) entrega el registro

    Parametros del constructor:
        table_meta      : TableMetadata con el esquema del registro
        key_column      : nombre de la columna clave (INT o CHAR)
        heap            : HeapFile compartido donde viven los registros
        data_dir        : directorio donde se guarda el archivo .dat del indice
        sort_buffer_size: bytes de RAM disponibles para ExternalSort
    """

    # Header del archivo del arbol: (root_ptr, free_head, total_nodes, total_entries)
    HEADER_FORMAT: str = '=qqiq'
    HEADER_SIZE: int   = struct.calcsize(HEADER_FORMAT)

    # Header de las paginas temporales usadas en bulk_load
    TEMP_PAGE_HEADER_FORMAT: str = 'i'
    TEMP_PAGE_HEADER_SIZE: int   = struct.calcsize(TEMP_PAGE_HEADER_FORMAT)

    table_meta: TableMetadata
    key_column: str
    data_dir: str
    PAGE_SIZE: int
    sort_buffer_size: int
    level_memory_limit: int
    key_fmt: str
    key_size: int
    LEAF_ORDER: int
    INTERNAL_ORDER: int
    filename: str
    heap_path: str
    heap: HeapFile
    file: BinaryIO | None
    root: int
    is_unique: bool

    @staticmethod
    def _index_filename(table_meta: TableMetadata, key_column: str, data_dir: str) -> str:
        """Ruta canonica del archivo de indice para (tabla, columna, data_dir)."""
        return os.path.join(
            data_dir, f"{table_meta.name}_{key_column}_bplustree.dat"
        )

    def __init__(
        self,
        table_meta: TableMetadata,
        key_column: str,
        heap: HeapFile,
        page_size: int,
        data_dir: str = 'data',
        sort_buffer_size: int = 128 * 1024 * 1024,
        level_memory_limit: int = DEFAULT_LEVEL_MEMORY_LIMIT,
    ) -> None:
        super().__init__()

        self.table_meta  = table_meta
        self.key_column  = key_column
        self.data_dir    = data_dir
        self.PAGE_SIZE   = page_size

        self.sort_buffer_size = sort_buffer_size
        self.level_memory_limit = level_memory_limit

        os.makedirs(self.data_dir, exist_ok=True)

        # Determinar formato y tamano de la clave desde el esquema
        key_col_def: Column | None = None
        for col in table_meta.columns:
            if col['name'] == key_column:
                key_col_def = col
                break
        if key_col_def is None:
            raise ValueError(f"Columna clave '{key_column}' no encontrada en el esquema")

        self.key_fmt  = key_col_def['fmt']
        self.key_size = key_col_def['size']

        # Unicidad: la columna es la primary_key o tiene UNIQUE.
        self.is_unique = self.table_meta.is_unique_column(key_column)

        # Capacidades independientes por tipo de nodo (calculadas con el tamano
        # de pagina): las hojas almacenan (clave, RID) y los nodos internos
        # almacenan claves + punteros de hijo.
        self.LEAF_ORDER = (
            self.PAGE_SIZE - BPlusNode.HEADER_SIZE
        ) // (self.key_size + RID_SIZE)
        self.INTERNAL_ORDER = (
            self.PAGE_SIZE - BPlusNode.HEADER_SIZE - BPlusNode.PTR_SIZE
        ) // (self.key_size + BPlusNode.PTR_SIZE)
        if self.LEAF_ORDER < 2 or self.INTERNAL_ORDER < 2:
            raise ValueError(
                "Capacidad demasiado pequena para PAGE_SIZE y tipo de clave dados"
            )

        # Rutas de archivos
        self.filename  = self._index_filename(
            self.table_meta, self.key_column, self.data_dir
        )
        self.heap_path = heap.filepath

        # HeapFile compartido: el B+ Tree no lo crea ni lo cierra
        self.heap = heap

        # El archivo del indice se abre si ya existe; si no, se crea de forma
        # perezosa en insert/delete/_ensure_tree_file. Los metodos estaticos
        # bulk_load()/bulk_load_multiple()/build_from_storage() crean indices
        # nuevos y no operan sobre este objeto.
        if os.path.exists(self.filename):
            self.file = open(self.filename, 'r+b')
        else:
            self.file = None

    def _ensure_tree_file(self) -> None:
        """Crea el archivo del indice (header + raiz hoja vacia) si no existe."""
        if self.file is not None:
            return
        with open(self.filename, 'wb') as f:
            root_ptr    = self.HEADER_SIZE
            free_head   = -1
            total_nodes = 1
            f.write(struct.pack(self.HEADER_FORMAT, root_ptr, free_head, total_nodes, 0))

            root = BPlusNode(
                self.key_fmt, self.key_size, self.LEAF_ORDER, self.PAGE_SIZE,
                node_type=1, is_root=1,
            )
            root.keys     = []
            root.pointers = []
            root.num_keys = 0
            f.write(root.pack())
        self.file = open(self.filename, 'r+b')
        self.root = self.HEADER_SIZE

    def close(self) -> None:
        """Cierra el archivo del indice. Idempotente. No crea ni cierra el HeapFile."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def count(self) -> int:
        """Numero de entradas (key, RID) del arbol, leido del header."""
        if self.file is None:
            return 0
        _, _, _, total_entries = self._read_header(self.file)
        return total_entries

    def validate_unique(self) -> bool:
        """
        Verifica que las claves actuales del indice sean todas unicas.
        Recorre las hojas en orden por next_leaf comparando cada clave con la
        anterior. No construye otro indice ni recorre el HeapFile.
        """
        if self.file is None:
            return True

        f    = self.file
        node = self._leftmost_leaf(f)
        previous: Key | None = None

        while True:
            for key in node.keys:
                if self._compare_keys(previous, key) == 0:
                    return False
                previous = key
            if node.next_leaf == -1:
                break
            node = self._read_node(f, node.next_leaf)

        return True

    def clear(self) -> None:
        """Vacia solo este indice (arbol vacio y valido en disco)."""
        self._reset_tree_file()
        self._init_empty_tree()

    def _storage_stats_snapshot(self) -> tuple[int, int]:
        """Snapshot del I/O acumulado en el HeapFile asociado."""
        return self.heap.page_reads, self.heap.page_writes

    def _accumulate_storage_stats(self, before: tuple[int, int]) -> None:
        """Suma a este indice los deltas de I/O del HeapFile (sin doble conteo)."""
        self.page_reads  += self.heap.page_reads  - before[0]
        self.page_writes += self.heap.page_writes - before[1]

    def _read_header(self, f: BinaryIO) -> tuple[int, int, int, int]:
        f.seek(0)
        return struct.unpack(self.HEADER_FORMAT, f.read(self.HEADER_SIZE))

    def _write_header(self, f: BinaryIO, root_ptr: int, free_head: int, total_nodes: int, total_entries: int) -> None:
        f.seek(0)
        f.write(struct.pack(self.HEADER_FORMAT, root_ptr, free_head, total_nodes, total_entries))

    def _read_node(self, f: BinaryIO, offset: int) -> BPlusNode:
        f.seek(offset)
        data = f.read(self.PAGE_SIZE)
        self.page_reads += 1
        return BPlusNode.unpack(data, self.key_fmt, self.key_size, self.INTERNAL_ORDER, self.PAGE_SIZE)

    def _write_node(self, f: BinaryIO, node: BPlusNode, offset: int) -> None:
        f.seek(offset)
        f.write(node.pack())
        self.page_writes += 1

    @overload
    def _normalize_key(self, key: None) -> None: ...

    @overload
    def _normalize_key(self, key: Key) -> Key: ...

    def _normalize_key(self, key: Key | None) -> Key | None:
        """
        Normaliza una clave a su representacion canonica.
        Para CHAR: bytes de longitud fija (key_size).
        Para INT: int.
        """
        if key is None:
            return None
        return self.table_meta.normalize_value(self.key_column, key)

    def _compare_keys(self, a: Key | None, b: Key | None) -> int:
        """
        Compara dos claves genericas.
        Retorna: -1 si a < b | 0 si a == b | 1 si a > b
        """
        a = self._normalize_key(a)
        b = self._normalize_key(b)
        if a is None and b is None:
            return 0
        if a is None:
            return -1
        if b is None:
            return 1
        return (a > b) - (a < b)

    def _binary_search_internal(self, node: BPlusNode, key: Key) -> int:
        """Devuelve el indice del puntero por donde bajar en un nodo interno."""
        left, right = 0, node.num_keys - 1
        result = node.num_keys

        while left <= right:
            mid = (left + right) // 2
            if self._compare_keys(node.keys[mid], key) <= 0:
                left = mid + 1
            else:
                result = mid
                right  = mid - 1

        return min(result, len(node.pointers) - 1)

    def _child_index_leftmost(self, node: BPlusNode, key: Key) -> int:
        """
        Indice del hijo mas a la izquierda que podria contener key
        (primer separador >= key). Necesario para delete(key, rid) cuando la
        clave se repite entre hojas.
        """
        left, right = 0, node.num_keys - 1
        result = node.num_keys

        while left <= right:
            mid = (left + right) // 2
            if self._compare_keys(node.keys[mid], key) < 0:
                left = mid + 1
            else:
                result = mid
                right  = mid - 1

        return min(result, len(node.pointers) - 1)

    def _binary_search_leaf(self, node: BPlusNode, key: Key) -> int:
        """Busca una clave exacta en una hoja. Retorna indice o -1."""
        left, right = 0, node.num_keys - 1
        result = -1

        while left <= right:
            mid       = (left + right) // 2
            cmp_value = self._compare_keys(node.keys[mid], key)
            if cmp_value == 0:
                result = mid
                right  = mid - 1
            elif cmp_value > 0:
                right = mid - 1
            else:
                left = mid + 1

        return result

    def _min_keys_leaf(self) -> int:
        return (self.LEAF_ORDER + 1) // 2

    def _min_keys_internal(self) -> int:
        return (self.INTERNAL_ORDER + 2) // 2 - 1

    def _find_first_leaf(self, f: BinaryIO, key: Key) -> tuple[BPlusNode, int]:
        """Devuelve (nodo_hoja, indice_inicio) donde podria estar la primera key."""
        key = self._normalize_key(key)
        root_ptr, _, _, _ = self._read_header(f)
        node = self._read_node(f, root_ptr)

        while node.node_type == 0:
            child_idx = self._child_index_leftmost(node, key)
            node = self._read_node(f, node.pointers[child_idx])

        return node, self._lower_bound_leaf(node, key)

    def _upper_bound_leaf(self, node: BPlusNode, key: Key) -> int:
        left, right = 0, node.num_keys

        while left < right:
            mid = (left + right) // 2
            if self._compare_keys(node.keys[mid], key) <= 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _lower_bound_leaf(self, node: BPlusNode, key: Key) -> int:
        left, right = 0, node.num_keys

        while left < right:
            mid = (left + right) // 2
            if self._compare_keys(node.keys[mid], key) < 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _leftmost_leaf(self, f: BinaryIO) -> BPlusNode:
        """Devuelve la hoja mas a la izquierda del arbol."""
        root_ptr, _, _, _ = self._read_header(f)
        node = self._read_node(f, root_ptr)
        while node.node_type == 0:
            node = self._read_node(f, node.pointers[0])
        return node

    def search(self, key: Key) -> RID | None:
        """Operacion propia del indice: retorna el RID de la clave dada, o None."""
        if self.file is None:
            return None
        f = self.file
        node, idx = self._find_first_leaf(f, key)

        if idx < node.num_keys and self._compare_keys(node.keys[idx], key) == 0:
            return node.pointers[idx]

        # caso borde: key == separador interno; _find_first_leaf pudo quedar en
        # la hoja inmediatamente anterior, con la clave al inicio de next_leaf
        if idx >= node.num_keys and node.next_leaf != -1:
            nxt = self._read_node(f, node.next_leaf)
            if nxt.num_keys > 0 and self._compare_keys(nxt.keys[0], key) == 0:
                return nxt.pointers[0]

        return None

    def search_all(self, key: Key) -> list[RID]:
        """Retorna lista de RIDs de todos los registros con la clave dada."""
        return self.range_search(key, key)

    def search_record(self, key_value: Key) -> CleanRecord | None:
        """
        Busqueda puntual por clave.
        Retorna la tupla del primer registro encontrado (limpia), o None.
        """
        before = self._storage_stats_snapshot()
        try:
            rid = self.search(key_value)
            if rid is None:
                return None
            record = self.heap.read(rid)
            return self.table_meta.clean_tuple(record)
        finally:
            self._accumulate_storage_stats(before)

    def search_records(self, key: Key) -> list[CleanRecord]:
        """Retorna lista de registros limpios para la clave dada."""
        before = self._storage_stats_snapshot()
        try:
            rids = self.search_all(key)
            if not rids:
                return []
            records = self.heap.read_many(rids)
            return [
                cleaned
                for cleaned in (self.table_meta.clean_tuple(r) for r in records)
                if cleaned is not None
            ]
        finally:
            self._accumulate_storage_stats(before)

    def range_search(
        self,
        begin_key: Key | None,
        end_key: Key | None,
        include_begin: bool = True,
        include_end: bool = True,
    ) -> list[RID]:
        """
        Operacion propia del indice: retorna los RIDs del rango indicado.
        None en un limite significa ausencia de limite.
        include_begin/include_end indican si los limites son inclusivos.
        """
        if self.file is None:
            return []
        if (
            begin_key is not None
            and end_key is not None
            and self._compare_keys(begin_key, end_key) > 0
        ):
            begin_key, end_key = end_key, begin_key

        f = self.file

        # hoja inicial
        if begin_key is None:
            node = self._leftmost_leaf(f)
            idx  = 0
        else:
            node, idx = self._find_first_leaf(f, begin_key)
            if not include_begin:
                # avanzar a la primera clave > begin (puede cruzar hojas)
                while True:
                    idx = self._upper_bound_leaf(node, begin_key)
                    if idx < node.num_keys or node.next_leaf == -1:
                        break
                    node = self._read_node(f, node.next_leaf)

        results = []

        while True:
            start = idx

            # sin limite derecho: toda la hoja pertenece al rango
            if end_key is None:
                results.extend(node.pointers[start:])

                if node.next_leaf == -1:
                    break
                node = self._read_node(f, node.next_leaf)
                idx  = 0
                continue

            # toda la cola de la hoja pertenece al rango
            if node.num_keys > 0:
                cmp_last = self._compare_keys(node.keys[-1], end_key)
                in_range = cmp_last <= 0 if include_end else cmp_last < 0
                if in_range:
                    results.extend(node.pointers[start:])

                    if node.next_leaf == -1:
                        break
                    node = self._read_node(f, node.next_leaf)
                    idx  = 0
                    continue

            # el limite derecho cae dentro de esta hoja
            if include_end:
                end = self._upper_bound_leaf(node, end_key)
            else:
                end = self._lower_bound_leaf(node, end_key)
            results.extend(node.pointers[start:end])
            break

        return results

    def range_search_records(
        self,
        begin_key: Key | None,
        end_key: Key | None,
        include_begin: bool = True,
        include_end: bool = True,
    ) -> list[CleanRecord]:
        """
        Busqueda por rango de clave.
        None en un limite significa ausencia de limite.
        Retorna lista de tuplas (limpias) con registros en el rango.
        """
        before = self._storage_stats_snapshot()
        try:
            rids = self.range_search(begin_key, end_key, include_begin, include_end)
            if not rids:
                return []
            records = self.heap.read_many(rids)
            return [
                cleaned
                for cleaned in (self.table_meta.clean_tuple(r) for r in records)
                if cleaned is not None
            ]
        finally:
            self._accumulate_storage_stats(before)

    def insert(self, key: Key, pointer: RID) -> bool:
        """
        Inserta el par (key, RID) en el B+ Tree.
        pointer es la referencia (page_id, slot_id) del registro en el almacenamiento.
        Retorna True si inserta; False si la clave ya existe en una columna
        UNIQUE (la primary_key o una columna con unique=True).
        """
        if self.is_unique and self.search(key) is not None:
            return False

        self._ensure_tree_file()
        key = self._normalize_key(key)

        f = self.file
        root_ptr, _, _, total_entries = self._read_header(f)
        result = self._insert_recursive(f, key, pointer, root_ptr)

        final_root_ptr = root_ptr
        if result is not None:
            promoted_key, new_child_ptr, old_root, new_child = result
            old_root_ptr = root_ptr

            new_root = BPlusNode(
                self.key_fmt, self.key_size, self.INTERNAL_ORDER, self.PAGE_SIZE,
                node_type=0, is_root=1,
            )
            new_root.keys     = [promoted_key]
            new_root.pointers = [old_root_ptr, new_child_ptr]
            new_root.num_keys = 1
            new_root.parent_ptr = -1

            # el antiguo root y el nuevo hijo ya vienen del split (en memoria)
            new_root_ptr = self._allocate_node(f)
            self._write_node(f, new_root, new_root_ptr)

            old_root.is_root    = 0
            old_root.parent_ptr = new_root_ptr
            self._write_node(f, old_root, old_root_ptr)

            new_child.parent_ptr = new_root_ptr
            self._write_node(f, new_child, new_child_ptr)

            final_root_ptr = new_root_ptr

        _, free_head, total_nodes, _ = self._read_header(f)
        self._write_header(f, final_root_ptr, free_head, total_nodes, total_entries + 1)

        return True

    def insert_record(self, record_tuple: RawRecord) -> RID | None:
        """
        Inserta un registro completo:
          1. Almacena en el HeapFile, obtiene su RID
          2. Extrae la clave del registro
          3. Inserta (key, RID) en el arbol
        Retorna el RID, o None si falla por duplicidad en una columna UNIQUE
        (eliminando el registro ya almacenado para no dejar huerfanos).
        """
        before  = self._storage_stats_snapshot()
        try:
            rid     = self.heap.insert(record_tuple)
            key_idx = self.table_meta.col_index(self.key_column)
            key     = record_tuple[key_idx]
            if not self.insert(key, rid):
                self.heap.remove(rid)
                return None
            return rid
        finally:
            self._accumulate_storage_stats(before)

    def _insert_in_leaf(self, node: BPlusNode, key: Key, pointer: RID) -> None:
        left, right = 0, node.num_keys
        while left < right:
            mid = (left + right) // 2
            if self._compare_keys(node.keys[mid], key) <= 0:
                left = mid + 1
            else:
                right = mid
        node.keys.insert(left, key)
        node.pointers.insert(left, pointer)
        node.num_keys += 1

    def _insert_in_internal(self, node: BPlusNode, key: Key, pointer: int) -> None:
        left, right = 0, node.num_keys
        while left < right:
            mid = (left + right) // 2
            if self._compare_keys(node.keys[mid], key) <= 0:
                left = mid + 1
            else:
                right = mid
        node.keys.insert(left, key)
        node.pointers.insert(left + 1, pointer)
        node.num_keys += 1

    def _insert_recursive(self, f: BinaryIO, key: Key, pointer: RID, node_ptr: int) -> tuple[Key, int, BPlusNode, BPlusNode] | None:
        node = self._read_node(f, node_ptr)

        if node.node_type == 1:  # hoja
            self._insert_in_leaf(node, key, pointer)
            if node.num_keys > self.LEAF_ORDER:
                return self._split_leaf(f, node, node_ptr)
            self._write_node(f, node, node_ptr)
            return None
        else:  # interno
            child_idx = self._binary_search_internal(node, key)
            child_ptr = node.pointers[child_idx]
            result    = self._insert_recursive(f, key, pointer, child_ptr)
            if result is not None:
                promoted_key, new_child_ptr, _, _ = result
                self._insert_in_internal(node, promoted_key, new_child_ptr)
                if node.num_keys > self.INTERNAL_ORDER:
                    return self._split_internal(f, node, node_ptr)
                self._write_node(f, node, node_ptr)
            return None

    def _split_leaf(self, f: BinaryIO, node: BPlusNode, node_ptr: int) -> tuple[Key, int, BPlusNode, BPlusNode]:
        mid = node.num_keys // 2

        new_node = BPlusNode(
            self.key_fmt, self.key_size, self.LEAF_ORDER, self.PAGE_SIZE,
            node_type=1, is_root=0,
        )
        new_node.parent_ptr = node.parent_ptr

        new_node_ptr = self._allocate_node(f)

        new_node.keys     = node.keys[mid:]
        new_node.pointers = node.pointers[mid:]
        new_node.num_keys = len(new_node.keys)
        new_node.next_leaf = node.next_leaf

        node.keys     = node.keys[:mid]
        node.pointers = node.pointers[:mid]
        node.num_keys = len(node.keys)
        node.next_leaf = new_node_ptr

        self._write_node(f, node, node_ptr)
        self._write_node(f, new_node, new_node_ptr)

        return new_node.keys[0], new_node_ptr, node, new_node

    def _split_internal(self, f: BinaryIO, node: BPlusNode, node_ptr: int) -> tuple[Key, int, BPlusNode, BPlusNode]:
        mid          = node.num_keys // 2
        promoted_key = node.keys[mid]

        new_node = BPlusNode(
            self.key_fmt, self.key_size, self.INTERNAL_ORDER, self.PAGE_SIZE,
            node_type=0, is_root=0,
        )
        new_node.parent_ptr = node.parent_ptr

        new_node_ptr = self._allocate_node(f)

        new_node.keys     = node.keys[mid + 1:]
        new_node.pointers = node.pointers[mid + 1:]
        new_node.num_keys = len(new_node.keys)

        for child_ptr in new_node.pointers:
            child = self._read_node(f, child_ptr)
            child.parent_ptr = new_node_ptr
            self._write_node(f, child, child_ptr)

        node.keys     = node.keys[:mid]
        node.pointers = node.pointers[:mid + 1]
        node.num_keys = len(node.keys)

        self._write_node(f, node, node_ptr)
        self._write_node(f, new_node, new_node_ptr)

        return promoted_key, new_node_ptr, node, new_node

    def delete(self, key: Key, rid: RID | None = None) -> RID | None:
        """
        Elimina una entrada del arbol.
        - delete(key): elimina la primera entrada que coincida con la clave.
        - delete(key, rid): elimina exactamente la entrada (key, rid).
        Retorna el RID eliminado, o None si no existe.
        """
        self._ensure_tree_file()
        key = self._normalize_key(key)

        f = self.file
        root_ptr, _, _, total_entries = self._read_header(f)
        if rid is None:
            deleted = self._delete_recursive(f, key, root_ptr)
        else:
            deleted = self._delete_by_rid(f, key, rid)
        if deleted is None:
            return None

        final_root_ptr = root_ptr
        root = self._read_node(f, root_ptr)
        # si la raiz interna queda vacia, bajar un nivel
        if root.num_keys == 0 and root.node_type == 0:
            old_root_ptr = root_ptr
            new_root_ptr = root.pointers[0]

            new_root = self._read_node(f, new_root_ptr)
            new_root.is_root    = 1
            new_root.parent_ptr = -1
            self._write_node(f, new_root, new_root_ptr)

            # la raiz anterior deja de usarse, liberarla
            self._free_node(f, old_root_ptr)

            final_root_ptr = new_root_ptr

        _, free_head, total_nodes, _ = self._read_header(f)
        self._write_header(f, final_root_ptr, free_head, total_nodes, total_entries - 1)

        return deleted

    def delete_record(self, key: Key, rid: RID | None = None) -> CleanRecord | None:
        before = self._storage_stats_snapshot()
        try:
            deleted_rid = self.delete(key, rid)

            if deleted_rid is None:
                return None

            deleted = self.heap.remove(deleted_rid)

            if deleted is None:
                return None

            return self.table_meta.clean_tuple(deleted)
        finally:
            self._accumulate_storage_stats(before)

    def delete_all(self, key: Key) -> list[RID]:
        """
        Elimina todas las entradas con la clave dada y retorna sus RIDs.

        Hace un solo descenso a la hoja mas a la izquierda que puede contener la
        clave, recorre las hojas por next_leaf eliminando el tramo de claves
        iguales y reutiliza los rebalanceos existentes.
        """
        self._ensure_tree_file()
        key = self._normalize_key(key)

        f = self.file
        root_ptr, _, _, total_entries = self._read_header(f)

        # descenso unico a la hoja mas a la izquierda que puede contener la clave
        node_ptr = root_ptr
        node     = self._read_node(f, node_ptr)
        while node.node_type == 0:
            child_idx = self._child_index_leftmost(node, key)
            node_ptr  = node.pointers[child_idx]
            node      = self._read_node(f, node_ptr)

        removed: list[RID] = []

        while True:
            i = self._lower_bound_leaf(node, key)

            if i < node.num_keys and self._compare_keys(node.keys[i], key) == 0:
                j = self._upper_bound_leaf(node, key)

                removed.extend(node.pointers[i:j])

                old_first_key = node.keys[0]
                del node.keys[i:j]
                del node.pointers[i:j]
                node.num_keys -= (j - i)
                new_first_key = node.keys[0] if node.num_keys > 0 else None

                survived = self._fixup_delete_in_leaf(
                    f, node, node_ptr, old_first_key, new_first_key
                )

                if not survived:
                    # la hoja se fusiono con la izquierda: continuar por su
                    # next_leaf (el objeto en memoria conserva ese enlace)
                    next_ptr = node.next_leaf
                    if next_ptr == -1:
                        break
                    node_ptr = next_ptr
                    node     = self._read_node(f, next_ptr)
                else:
                    # la hoja sobrevivio y su contenido en memoria esta al dia;
                    # solo si aun contiene la clave hay que refrescar parent_ptr
                    # para el proximo _fixup_delete_in_leaf.
                    j2 = self._lower_bound_leaf(node, key)
                    if j2 < node.num_keys and self._compare_keys(node.keys[j2], key) == 0:
                        node = self._read_node(f, node_ptr)
                continue

            if node.num_keys > 0 and self._compare_keys(node.keys[-1], key) > 0:
                break
            if node.next_leaf == -1:
                break

            next_ptr  = node.next_leaf
            next_node = self._read_node(f, next_ptr)
            if next_node.num_keys > 0 and self._compare_keys(next_node.keys[0], key) > 0:
                break

            node_ptr = next_ptr
            node     = next_node

        # reduccion de raiz (mismo mecanismo que delete()); root_ptr ya se conoce
        root = self._read_node(f, root_ptr)
        while root.num_keys == 0 and root.node_type == 0:
            old_root_ptr = root_ptr
            new_root_ptr = root.pointers[0]

            new_root = self._read_node(f, new_root_ptr)
            new_root.is_root    = 1
            new_root.parent_ptr = -1
            self._write_node(f, new_root, new_root_ptr)

            self._free_node(f, old_root_ptr)

            root_ptr = new_root_ptr
            root     = self._read_node(f, root_ptr)

        _, free_head, total_nodes, _ = self._read_header(f)
        self._write_header(
            f, root_ptr, free_head, total_nodes, total_entries - len(removed)
        )

        return removed

    def delete_records(self, key: Key) -> list[CleanRecord]:
        """
        Elimina de este indice y del almacenamiento compartido todos los
        registros asociados a la clave. No sincroniza otros indices.
        """
        before = self._storage_stats_snapshot()
        try:
            rids = self.delete_all(key)

            removed: list[CleanRecord] = []
            for rid in rids:
                raw = self.heap.remove(rid)
                if raw is None:
                    continue
                cleaned = self.table_meta.clean_tuple(raw)
                if cleaned is not None:
                    removed.append(cleaned)
            return removed
        finally:
            self._accumulate_storage_stats(before)

    def _delete_recursive(self, f: BinaryIO, key: Key, node_ptr: int, parent_ptr: int | None = None, index_in_parent: int | None = None) -> RID | None:
        node = self._read_node(f, node_ptr)

        if node.node_type == 1:  # hoja
            old_first_key = node.keys[0] if node.num_keys > 0 else None
            node_ptr, node, removed = self._remove_from_leaf(f, node, key, node_ptr)

            if removed is None:
                return None

            new_first_key = node.keys[0] if node.num_keys > 0 else None
            if (
                old_first_key is not None
                and new_first_key is not None
                and self._compare_keys(old_first_key, new_first_key) != 0
            ):
                self._update_parent_separator_after_delete(
                    f, parent_ptr, index_in_parent, old_first_key, new_first_key
                )

            MIN_KEYS = self._min_keys_leaf()
            if node.num_keys < MIN_KEYS and parent_ptr is not None:
                self._rebalance_leaf(f, node, node_ptr, parent_ptr, index_in_parent)
            else:
                self._write_node(f, node, node_ptr)

            return removed

        else:  # interno
            child_idx = self._binary_search_internal(node, key)
            child_ptr = node.pointers[child_idx]
            deleted   = self._delete_recursive(f, key, child_ptr, node_ptr, child_idx)
            if deleted is None:
                return None

            node     = self._read_node(f, node_ptr)
            MIN_KEYS = self._min_keys_internal()
            if node.num_keys < MIN_KEYS and parent_ptr is not None:
                self._rebalance_internal(f, node_ptr, parent_ptr, index_in_parent)
            return deleted

    def _remove_from_leaf(self, f: BinaryIO, node: BPlusNode, key: Key, node_ptr: int) -> tuple[int, BPlusNode, RID | None]:
        current_node = node
        current_ptr  = node_ptr

        while True:
            idx = self._binary_search_leaf(current_node, key)

            if idx != -1:
                removed_rid = current_node.pointers[idx]
                del current_node.keys[idx]
                del current_node.pointers[idx]
                current_node.num_keys -= 1
                return current_ptr, current_node, removed_rid

            if current_node.num_keys > 0 and self._compare_keys(current_node.keys[-1], key) > 0:
                break
            if current_node.next_leaf == -1:
                break

            next_ptr  = current_node.next_leaf
            next_node = self._read_node(f, next_ptr)
            if next_node.num_keys > 0 and self._compare_keys(next_node.keys[0], key) > 0:
                break

            current_ptr  = next_ptr
            current_node = next_node

        return node_ptr, node, None

    def _delete_by_rid(self, f: BinaryIO, key: Key, rid: RID) -> RID | None:
        """
        Elimina exactamente la entrada (key, rid).
        La clave puede repetirse entre hojas, asi que se escanea desde la hoja
        mas a la izquierda que podria contenerla y se reajusta usando el camino
        real del nodo (via parent_ptr).
        """
        root_ptr, _, _, _ = self._read_header(f)
        node_ptr = root_ptr
        node     = self._read_node(f, node_ptr)
        while node.node_type == 0:
            child_idx = self._child_index_leftmost(node, key)
            node_ptr  = node.pointers[child_idx]
            node      = self._read_node(f, node_ptr)

        while True:
            idx = self._binary_search_leaf(node, key)

            if idx != -1:
                i = idx
                while (
                    i < node.num_keys
                    and self._compare_keys(node.keys[i], key) == 0
                ):
                    if node.pointers[i] == rid:
                        removed       = node.pointers[i]
                        old_first_key = node.keys[0]
                        del node.keys[i]
                        del node.pointers[i]
                        node.num_keys -= 1
                        new_first_key = node.keys[0] if node.num_keys > 0 else None
                        self._fixup_delete_in_leaf(
                            f, node, node_ptr, old_first_key, new_first_key
                        )
                        return removed
                    i += 1

            if node.num_keys > 0 and self._compare_keys(node.keys[-1], key) > 0:
                break
            if node.next_leaf == -1:
                break

            next_ptr  = node.next_leaf
            next_node = self._read_node(f, next_ptr)
            if next_node.num_keys > 0 and self._compare_keys(next_node.keys[0], key) > 0:
                break

            node_ptr = next_ptr
            node     = next_node

        return None

    def _fixup_delete_in_leaf(self, f: BinaryIO, leaf: BPlusNode, leaf_ptr: int, old_first_key: Key | None, new_first_key: Key | None) -> bool:
        """
        Reajusta separadores y rebalancea tras eliminar una hoja localizada por
        escaneo. Usa parent_ptr para subir nivel a nivel aplicando los mismos
        rebalanceos que _delete_recursive, sin reconstruir el camino completo.

        Retorna True si la hoja sigue existiendo en leaf_ptr y False si fue
        fusionada con su hermano izquierdo (quedo liberada).
        """
        parent_ptr = leaf.parent_ptr
        if parent_ptr == -1:
            # la hoja es la raiz
            self._write_node(f, leaf, leaf_ptr)
            return True

        parent     = self._read_node(f, parent_ptr)
        leaf_index = parent.pointers.index(leaf_ptr)

        if (
            old_first_key is not None
            and new_first_key is not None
            and self._compare_keys(old_first_key, new_first_key) != 0
        ):
            self._update_parent_separator_after_delete(
                f, parent_ptr, leaf_index, old_first_key, new_first_key, parent
            )

        if leaf.num_keys < self._min_keys_leaf():
            survived = self._rebalance_leaf(
                f, leaf, leaf_ptr, parent_ptr, leaf_index, parent
            )
        else:
            self._write_node(f, leaf, leaf_ptr)
            survived = True

        # underflow hacia arriba; la primera iteracion reutiliza el padre cargado
        loaded = parent
        while parent_ptr != -1:
            if loaded is None:
                parent = self._read_node(f, parent_ptr)
            else:
                parent = loaded
                loaded = None

            gp_ptr = parent.parent_ptr
            if gp_ptr == -1:
                break
            if parent.num_keys < self._min_keys_internal():
                gp       = self._read_node(f, gp_ptr)
                gp_index = gp.pointers.index(parent_ptr)
                self._rebalance_internal(
                    f, parent_ptr, gp_ptr, gp_index, parent, gp
                )
                parent_ptr = gp_ptr
            else:
                break

        return survived

    def _remove_from_internal(self, node: BPlusNode, index: int) -> None:
        """Elimina la clave en 'index' y su puntero hijo derecho."""
        del node.keys[index]
        del node.pointers[index + 1]
        node.num_keys -= 1

    def _update_parent_separator_after_delete(self, f: BinaryIO, parent_ptr: int | None, child_index: int | None, old_key: Key | None, new_key: Key | None, parent: BPlusNode | None = None) -> None:
        current_parent_ptr  = parent_ptr
        current_child_index = child_index
        loaded = parent

        while current_parent_ptr not in (-1, None):
            if loaded is None:
                node = self._read_node(f, current_parent_ptr)
            else:
                node   = loaded
                loaded = None   # solo la primera iteracion reutiliza el nodo ya cargado

            if current_child_index > 0:
                sep_idx = current_child_index - 1
                if sep_idx < len(node.keys):
                    node.keys[sep_idx] = new_key
                    self._write_node(f, node, current_parent_ptr)
                break

            old_parent_ptr     = current_parent_ptr
            current_parent_ptr = node.parent_ptr
            if current_parent_ptr == -1:
                break

            grandparent = self._read_node(f, current_parent_ptr)
            try:
                current_child_index = grandparent.pointers.index(old_parent_ptr)
            except ValueError:
                break
            loaded = grandparent

    def _rebalance_leaf(self, f: BinaryIO, node: BPlusNode, node_ptr: int, parent_ptr: int, index_in_parent: int, parent: BPlusNode | None = None) -> bool:
        if parent is None:
            parent = self._read_node(f, parent_ptr)
        MIN_KEYS = self._min_keys_leaf()

        # sin hermanos (padre con un unico hijo): no hay redistribucion ni merge.
        # La reduccion de raiz posterior convierte a este nodo en la nueva raiz.
        if len(parent.pointers) < 2:
            self._write_node(f, node, node_ptr)
            return True

        left_ptr  = None
        left      = None
        right_ptr = None
        right     = None

        # redistribuir con hermano izquierdo: solo si puede aportar lo suficiente
        if index_in_parent > 0:
            left_ptr = parent.pointers[index_in_parent - 1]
            left     = self._read_node(f, left_ptr)

            needed    = MIN_KEYS - node.num_keys
            available = left.num_keys - MIN_KEYS

            if needed > 0 and available >= needed:
                amount = min(needed, available)
                node.keys[0:0]     = left.keys[-amount:]
                node.pointers[0:0] = left.pointers[-amount:]
                del left.keys[-amount:]
                del left.pointers[-amount:]
                left.num_keys -= amount
                node.num_keys += amount
                parent.keys[index_in_parent - 1] = node.keys[0]
                self._write_node(f, left, left_ptr)
                self._write_node(f, node, node_ptr)
                self._write_node(f, parent, parent_ptr)
                return True

        # redistribuir con hermano derecho: solo si puede aportar lo suficiente
        if index_in_parent < len(parent.pointers) - 1:
            right_ptr = parent.pointers[index_in_parent + 1]
            right     = self._read_node(f, right_ptr)

            needed    = MIN_KEYS - node.num_keys
            available = right.num_keys - MIN_KEYS

            if needed > 0 and available >= needed:
                amount = min(needed, available)
                node.keys.extend(right.keys[:amount])
                node.pointers.extend(right.pointers[:amount])
                del right.keys[:amount]
                del right.pointers[:amount]
                right.num_keys -= amount
                node.num_keys += amount
                parent.keys[index_in_parent] = right.keys[0]
                self._write_node(f, right, right_ptr)
                self._write_node(f, node, node_ptr)
                self._write_node(f, parent, parent_ptr)
                return True

        # merge con hermano izquierdo (reutiliza el hermano ya cargado)
        if index_in_parent > 0:
            if left is None:
                left_ptr = parent.pointers[index_in_parent - 1]
                left     = self._read_node(f, left_ptr)
            left.keys.extend(node.keys)
            left.pointers.extend(node.pointers)
            left.num_keys  = len(left.keys)
            left.next_leaf = node.next_leaf
            self._write_node(f, left, left_ptr)
            self._remove_from_internal(parent, index_in_parent - 1)
            if left.num_keys > 0:
                self._update_parent_separator_after_delete(
                    f,
                    parent_ptr,
                    index_in_parent - 1,
                    parent.keys[index_in_parent - 1] if index_in_parent - 1 < len(parent.keys) else None,
                    left.keys[0],
                    parent,
                )
            self._write_node(f, parent, parent_ptr)
            self._free_node(f, node_ptr)
            return False
        else:
            # merge con hermano derecho (reutiliza el hermano ya cargado)
            if right is None:
                right_ptr = parent.pointers[index_in_parent + 1]
                right     = self._read_node(f, right_ptr)
            node.keys.extend(right.keys)
            node.pointers.extend(right.pointers)
            node.num_keys  = len(node.keys)
            node.next_leaf = right.next_leaf
            self._write_node(f, node, node_ptr)
            self._remove_from_internal(parent, index_in_parent)
            if node.num_keys > 0:
                self._update_parent_separator_after_delete(
                    f,
                    parent_ptr,
                    index_in_parent,
                    parent.keys[index_in_parent] if index_in_parent < len(parent.keys) else None,
                    node.keys[0],
                    parent,
                )
            self._write_node(f, parent, parent_ptr)
            self._free_node(f, right_ptr)
        return True

    def _rebalance_internal(self, f: BinaryIO, node_ptr: int, parent_ptr: int, index_in_parent: int, node: BPlusNode | None = None, parent: BPlusNode | None = None) -> None:
        if node is None:
            node = self._read_node(f, node_ptr)
        if parent is None:
            parent = self._read_node(f, parent_ptr)
        MIN_KEYS = self._min_keys_internal()

        # sin hermanos (padre con un unico hijo): no hay redistribucion ni merge.
        # La reduccion de raiz posterior se encarga de colapsar el nivel.
        if len(parent.pointers) < 2:
            return

        left_ptr  = None
        left      = None
        right_ptr = None
        right     = None

        # redistribuir con hermano izquierdo
        if index_in_parent > 0:
            left_ptr = parent.pointers[index_in_parent - 1]
            left     = self._read_node(f, left_ptr)
            if left.num_keys > MIN_KEYS:
                borrowed_key = left.keys.pop(-1)
                borrowed_ptr = left.pointers.pop(-1)
                left.num_keys -= 1
                node.keys.insert(0, parent.keys[index_in_parent - 1])
                node.pointers.insert(0, borrowed_ptr)
                borrowed_child = self._read_node(f, borrowed_ptr)
                borrowed_child.parent_ptr = node_ptr
                self._write_node(f, borrowed_child, borrowed_ptr)
                node.num_keys += 1
                parent.keys[index_in_parent - 1] = borrowed_key
                self._write_node(f, left, left_ptr)
                self._write_node(f, node, node_ptr)
                self._write_node(f, parent, parent_ptr)
                return

        # redistribuir con hermano derecho
        if index_in_parent < len(parent.pointers) - 1:
            right_ptr = parent.pointers[index_in_parent + 1]
            right     = self._read_node(f, right_ptr)
            if right.num_keys > MIN_KEYS:
                borrowed_key = right.keys.pop(0)
                borrowed_ptr = right.pointers.pop(0)
                right.num_keys -= 1
                node.keys.append(parent.keys[index_in_parent])
                node.pointers.append(borrowed_ptr)
                borrowed_child = self._read_node(f, borrowed_ptr)
                borrowed_child.parent_ptr = node_ptr
                self._write_node(f, borrowed_child, borrowed_ptr)
                node.num_keys += 1
                parent.keys[index_in_parent] = borrowed_key
                self._write_node(f, right, right_ptr)
                self._write_node(f, node, node_ptr)
                self._write_node(f, parent, parent_ptr)
                return

        # merge con hermano izquierdo (reutiliza el hermano ya cargado)
        if index_in_parent > 0:
            if left is None:
                left_ptr = parent.pointers[index_in_parent - 1]
                left     = self._read_node(f, left_ptr)
            left.keys.append(parent.keys[index_in_parent - 1])
            left.keys.extend(node.keys)
            left.pointers.extend(node.pointers)
            left.num_keys = len(left.keys)
            for child_ptr in node.pointers:
                child = self._read_node(f, child_ptr)
                child.parent_ptr = left_ptr
                self._write_node(f, child, child_ptr)
            self._write_node(f, left, left_ptr)
            self._remove_from_internal(parent, index_in_parent - 1)
            self._write_node(f, parent, parent_ptr)
            self._free_node(f, node_ptr)
        else:
            # merge con hermano derecho (reutiliza el hermano ya cargado)
            if right is None:
                right_ptr = parent.pointers[index_in_parent + 1]
                right     = self._read_node(f, right_ptr)
            node.keys.append(parent.keys[index_in_parent])
            node.keys.extend(right.keys)
            node.pointers.extend(right.pointers)
            node.num_keys = len(node.keys)
            for child_ptr in right.pointers:
                child = self._read_node(f, child_ptr)
                child.parent_ptr = node_ptr
                self._write_node(f, child, child_ptr)
            self._write_node(f, node, node_ptr)
            self._remove_from_internal(parent, index_in_parent)
            self._write_node(f, parent, parent_ptr)
            self._free_node(f, right_ptr)

    @staticmethod
    def bulk_load(
        table_meta: TableMetadata,
        heap: HeapFile,
        key_column: str,
        page_size: int,
        source_path: str,
        delimiter: str = ',',
        data_dir: str = 'data',
        sort_buffer_size: int = 128 * 1024 * 1024,
        level_memory_limit: int = DEFAULT_LEVEL_MEMORY_LIMIT,
    ) -> BPlusTree:
        """
        Metodo estatico que crea un indice B+ Tree nuevo cargando el CSV en el
        HeapFile recibido (ya creado y vacio) y construyendo el indice en una
        sola pasada. No crea ni cierra el HeapFile; delega toda la carga en
        bulk_load_multiple(). Falla si el archivo del indice ya existe (CREATE).
        No existe BPlusTree.rebulk_load().
        """
        before = (heap.page_reads, heap.page_writes)

        indexes = BPlusTree.bulk_load_multiple(
            table_meta,
            heap,
            [key_column],
            page_size,
            source_path,
            delimiter=delimiter,
            data_dir=data_dir,
            sort_buffer_size=sort_buffer_size,
            level_memory_limit=level_memory_limit,
        )
        index = indexes[0]
        index._accumulate_storage_stats(before)
        return index

    @staticmethod
    def bulk_load_multiple(
        table_meta: TableMetadata,
        heap: HeapFile,
        key_columns: list[str],
        page_size: int,
        source_path: str,
        delimiter: str = ',',
        data_dir: str = 'data',
        sort_buffer_size: int = 128 * 1024 * 1024,
        level_memory_limit: int = DEFAULT_LEVEL_MEMORY_LIMIT,
    ) -> list[BPlusTree]:
        """
        Metodo estatico que crea varios indices B+ Tree nuevos de una misma
        tabla en UNA sola pasada del CSV/Heap:
          1. Escribe los registros en el HeapFile recibido (ya creado y vacio)
             y obtiene (record, RID) una sola vez
          2. Distribuye (key, RID) a un temporal por columna en esa misma pasada
          3. Ordena cada temporal con ExternalSort
          4. Construye cada B+ Tree con la logica existente (valida UNIQUE)
        No crea ni cierra el HeapFile (es compartido y lo aporta el llamador).
        Exige que el HeapFile este vacio y que NINGUN archivo de indice destino
        exista (CREATE): si alguno existe lanza FileExistsError sin abrirlo ni
        modificarlo.
        Ante CUALQUIER fallo durante la construccion revierte el HeapFile
        (queda vacio), descarta los indices parciales y elimina los temporales.
        El I/O del Heap (una sola pasada) se contabiliza una vez en el propio
        Heap; cada indice contabiliza unicamente su I/O de nodos.
        """
        if not key_columns:
            raise ValueError("key_columns no puede estar vacio")
        if len(set(key_columns)) != len(key_columns):
            raise ValueError("key_columns contiene columnas repetidas")

        for column in key_columns:
            table_meta.col_index(column)

        if not heap.is_empty():
            raise ValueError(
                "El HeapFile ya contiene datos; no se puede hacer bulk_load"
            )

        # CREATE: ningun indice destino debe existir (no se abre ni modifica)
        for column in key_columns:
            filename = BPlusTree._index_filename(table_meta, column, data_dir)
            if os.path.exists(filename):
                raise FileExistsError(f"El indice ya existe: {filename}")

        indexes = [
            BPlusTree(
                table_meta, column, heap, page_size, data_dir,
                sort_buffer_size, level_memory_limit,
            )
            for column in key_columns
        ]

        temp_unsorted = [
            os.path.join(data_dir, f'tmp_unsorted_{column}.idx')
            for column in key_columns
        ]
        temp_sorted = [
            os.path.join(data_dir, f'tmp_sorted_{column}.idx')
            for column in key_columns
        ]

        built = 0
        try:
            # El StorageFile escribe los registros y entrega (record, RID) en la misma pasada
            records_with_rids = heap.bulk_insert_with_rids(
                heap._csv_record_generator(source_path, delimiter),
                reset=True,
            )

            totals = BPlusTree._generate_multiple_index_entries(
                indexes, temp_unsorted, records_with_rids
            )

            for i, index in enumerate(indexes):
                if totals[i] > 0:
                    index._external_sort_index(temp_unsorted[i], temp_sorted[i])

            for i, index in enumerate(indexes):
                index._reset_tree_file()
                if totals[i] == 0:
                    index._init_empty_tree()
                else:
                    index.build_bulk_tree(temp_sorted[i], totals[i])
                built += 1
        except Exception:
            # rollback ante cualquier fallo: Heap vacio y sin indices parciales
            heap.reset()
            for index in indexes[:built + 1]:
                index._discard_partial_index()
            raise
        finally:
            for path in temp_unsorted + temp_sorted:
                if os.path.exists(path):
                    os.remove(path)

        return indexes

    def _build_from_heap_inplace(self) -> None:
        """
        Construye este indice a partir de los registros activos del HeapFile,
        en una sola pasada. Unica implementacion del build desde Heap: la usan
        build_from_storage() (indice nuevo) y rebuild_from_storage() (existente).
        Rechaza duplicados si el indice es UNIQUE (no modifica el HeapFile).
        """
        before = self._storage_stats_snapshot()
        try:
            temp_unsorted = os.path.join(self.data_dir, f'tmp_unsorted_{self.key_column}.idx')
            temp_sorted = os.path.join( self.data_dir, f'tmp_sorted_{self.key_column}.idx')

            try:
                # Leer el StorageFile (una sola pasada) y generar (key, RID)
                total_entries = self._generate_index_entries(
                    self._iter_storage_records_with_rids(), temp_unsorted
                )
                if total_entries > 0:
                    # Ordenar las entradas
                    self._external_sort_index(temp_unsorted, temp_sorted)

                # Recien ahora se toca el archivo del indice
                self._reset_tree_file()
                if total_entries == 0:
                    self._init_empty_tree()
                else:
                    # Construir el arbol
                    try:
                        self.build_bulk_tree(temp_sorted, total_entries)
                    except _DuplicateKeyError:
                        self._discard_partial_index()
                        raise
            finally:
                if os.path.exists(temp_unsorted):
                    os.remove(temp_unsorted)
                if os.path.exists(temp_sorted):
                    os.remove(temp_sorted)
        finally:
            self._accumulate_storage_stats(before)

    def rebuild_from_storage(self) -> None:
        """
        Reconstruye este indice desde el HeapFile de forma segura: construye en
        un archivo temporal y hace os.replace(temp -> original) solo si todo
        termina correctamente. Ante un fallo, el indice previo queda intacto.
        """
        if not os.path.exists(self.filename):
            self._build_from_heap_inplace()
            return

        orig_name  = self.filename
        parent_dir = os.path.dirname(orig_name) or '.'

        fd, tmp_path = tempfile.mkstemp(prefix='rebuild_', suffix='.dat', dir=parent_dir)
        os.close(fd)

        # cerrar/flush el descriptor actual antes de tocar nada
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

        self.filename = tmp_path
        try:
            self.file = open(tmp_path, 'r+b')
            self._build_from_heap_inplace()

            # cerrar/flush antes de os.replace
            self.file.flush()
            self.file.close()
            self.file = None

            os.replace(tmp_path, orig_name)
            self.filename = orig_name
            self.file = open(orig_name, 'r+b')
        except Exception:
            if self.file is not None:
                self.file.close()
                self.file = None
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.filename = orig_name
            self.file = open(orig_name, 'r+b')
            raise

    @staticmethod
    def build_from_storage(
        table_meta: TableMetadata,
        heap: HeapFile,
        key_column: str,
        page_size: int,
        data_dir: str = 'data',
        sort_buffer_size: int = 128 * 1024 * 1024,
        level_memory_limit: int = DEFAULT_LEVEL_MEMORY_LIMIT,
    ) -> BPlusTree:
        """
        Crea un indice B+ Tree nuevo a partir de los registros activos del
        HeapFile. Falla si el archivo del indice ya existe, sin modificarlo.
        """
        filename = BPlusTree._index_filename(table_meta, key_column, data_dir)
        if os.path.exists(filename):
            raise FileExistsError(f"El indice ya existe: {filename}")

        index = BPlusTree(
            table_meta, key_column, heap, page_size, data_dir,
            sort_buffer_size, level_memory_limit,
        )
        index._build_from_heap_inplace()
        return index

    def _iter_storage_records_with_rids(self) -> Iterator[tuple[RawRecord, RID]]:
        """Adapta iter_pages_with_records() a pares (record, RID)."""
        for page_records in self.heap.iter_pages_with_records():
            for rid, record in page_records:
                yield record, rid

    def _init_empty_tree(self) -> None:
        """Crea el archivo del arbol con una raiz hoja vacia."""
        self.file.seek(0)
        self.file.truncate()
        self.file.write(struct.pack(self.HEADER_FORMAT, self.HEADER_SIZE, -1, 1, 0))
        root = BPlusNode(
            self.key_fmt, self.key_size, self.LEAF_ORDER, self.PAGE_SIZE,
            node_type=1, is_root=1,
        )
        self.file.write(root.pack())
        self.root = self.HEADER_SIZE

    def _reset_tree_file(self) -> None:
        """Reinicia unicamente el archivo del B+ Tree."""
        if self.file is not None:
            self.file.close()
            self.file = None

        with open(self.filename, 'wb') as f:
            f.write(struct.pack(
                    self.HEADER_FORMAT,
                    self.HEADER_SIZE,  # root_ptr inicial
                    -1,                # free_head
                    0,                 # total_nodes
                    0,                 # total_entries
                )
            )

        self.file = open(self.filename, 'r+b')

    def _flush_index_page(self, f: BinaryIO, entries: list[Entry]) -> None:
        """Escribe una pagina de entradas (key, RID) al archivo temporal."""
        entry_format = '=' + self.key_fmt + RID_FORMAT
        packed       = bytearray()

        packed.extend(struct.pack(self.TEMP_PAGE_HEADER_FORMAT, len(entries)))

        for key, (page_id, slot_id) in entries:
            packed.extend(struct.pack(entry_format, key, page_id, slot_id))

        remaining = self.PAGE_SIZE - len(packed)
        if remaining > 0:
            packed.extend(b'\x00' * remaining)

        f.write(packed)

    def _generate_index_entries(
        self,
        records_with_rids: Iterable[tuple[RawRecord, RID]],
        temp_path: str,
    ) -> int:
        """
        Escribe pares (key, RID) en el archivo temporal temp_path por
        paginas, extrayendo la key de cada registro con la key_column del indice.
        Retorna el numero total de entradas generadas.
        """
        key_idx      = self.table_meta.col_index(self.key_column)
        entry_format = '=' + self.key_fmt + RID_FORMAT
        entry_size   = struct.calcsize(entry_format)

        usable_bytes = self.PAGE_SIZE - self.TEMP_PAGE_HEADER_SIZE
        max_entries  = usable_bytes // entry_size

        buffer: list[Entry] = []
        total_entries = 0

        with open(temp_path, 'wb') as f:
            for record_tuple, rid in records_with_rids:
                key = self._normalize_key(record_tuple[key_idx])
                buffer.append((key, rid))
                total_entries += 1
                if len(buffer) >= max_entries:
                    self._flush_index_page(f, buffer)
                    buffer.clear()

            if buffer:
                self._flush_index_page(f, buffer)

        return total_entries

    @staticmethod
    def _generate_multiple_index_entries(
        indexes: list[BPlusTree],
        unsorted_paths: list[str],
        records_with_rids: Iterable[tuple[RawRecord, RID]],
    ) -> list[int]:
        """
        Escribe un archivo temporal (key, RID) por cada indice, extrayendo las
        claves de todas las columnas en UNA sola pasada de records_with_rids.
        Retorna la cantidad de entradas generadas por indice.
        """
        key_idxs = [
            index.table_meta.col_index(index.key_column) for index in indexes
        ]

        max_entries: list[int] = []
        for index in indexes:
            entry_format = '=' + index.key_fmt + RID_FORMAT
            entry_size   = struct.calcsize(entry_format)
            usable_bytes = index.PAGE_SIZE - index.TEMP_PAGE_HEADER_SIZE
            max_entries.append(usable_bytes // entry_size)

        files   = [open(path, 'wb') for path in unsorted_paths]
        buffers: list[list[Entry]] = [[] for _ in indexes]
        totals  = [0 for _ in indexes]

        try:
            for record_tuple, rid in records_with_rids:
                for i, index in enumerate(indexes):
                    key = index._normalize_key(record_tuple[key_idxs[i]])
                    buffers[i].append((key, rid))
                    totals[i] += 1
                    if len(buffers[i]) >= max_entries[i]:
                        index._flush_index_page(files[i], buffers[i])
                        buffers[i].clear()

            for i, index in enumerate(indexes):
                if buffers[i]:
                    index._flush_index_page(files[i], buffers[i])
        finally:
            for f in files:
                f.close()

        return totals

    def _iter_index_entries(self, path: str) -> Iterator[Entry]:
        """Itera sobre los pares (key, RID) del archivo temporal ordenado."""
        entry_format    = '=' + self.key_fmt + RID_FORMAT
        entry_size      = struct.calcsize(entry_format)
        pages_per_buffer = max(1, self.sort_buffer_size // self.PAGE_SIZE)
        buffer_size     = pages_per_buffer * self.PAGE_SIZE

        with open(path, 'rb') as f:
            while True:
                buffer = f.read(buffer_size)
                if not buffer:
                    break
                for page_start in range(0, len(buffer), self.PAGE_SIZE):
                    page = buffer[page_start:page_start + self.PAGE_SIZE]
                    if len(page) < self.PAGE_SIZE:
                        break
                    num_records = struct.unpack(
                        self.TEMP_PAGE_HEADER_FORMAT,
                        page[:self.TEMP_PAGE_HEADER_SIZE],
                    )[0]
                    pos = self.TEMP_PAGE_HEADER_SIZE
                    for _ in range(num_records):
                        chunk = page[pos:pos + entry_size]
                        if len(chunk) < entry_size:
                            break
                        key, page_id, slot_id = struct.unpack(entry_format, chunk)
                        yield key, (page_id, slot_id)
                        pos += entry_size

    def _external_sort_index(self, input_path: str, output_path: str) -> None:
        """Ordena el archivo de entradas usando ExternalSort."""
        entry_format = '=' + self.key_fmt + RID_FORMAT
        sorter = ExternalSort(
            record_format=entry_format,
            page_size=self.PAGE_SIZE,
            buffer_size=self.sort_buffer_size,
        )
        sorter.external_sort(
            heap_path=input_path,
            output_path=output_path,
            sort_key_index=0,
        )

    def _discard_partial_index(self) -> None:
        """Cierra y elimina el archivo de indice parcial tras un fallo."""
        if self.file is not None:
            self.file.close()
            self.file = None
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def _allocate_node(self, f: BinaryIO) -> int:
        """Reserva espacio para un nuevo nodo (reusa de free list si es posible) y actualiza el header."""
        root_ptr, free_head, total_nodes, total_entries = self._read_header(f)
        if free_head != -1:
            ptr = free_head
            f.seek(ptr)
            next_free = struct.unpack(
                BPlusNode.PTR_FORMAT, f.read(BPlusNode.PTR_SIZE)
            )[0]
            self._write_header(f, root_ptr, next_free, total_nodes, total_entries)
            return ptr
        else:
            ptr = self.HEADER_SIZE + total_nodes * self.PAGE_SIZE
            self._write_header(f, root_ptr, free_head, total_nodes + 1, total_entries)
            return ptr

    def _free_node(self, f: BinaryIO, offset: int) -> None:
        """Agrega una pagina vacia a la free list."""
        root_ptr, free_head, total_nodes, total_entries = self._read_header(f)
        f.seek(offset)
        f.write(struct.pack(BPlusNode.PTR_FORMAT, free_head))
        self._write_header(f, root_ptr, offset, total_nodes, total_entries)

    @staticmethod
    def _count_partition(node_count: int, max_children: int) -> int:
        """Numero de nodos de un nivel con node_count hijos (misma regla que el reparto)."""
        if node_count <= max_children:
            return 1
        return (node_count + max_children - 1) // max_children

    @staticmethod
    def _iter_partition(
        node_count: int,
        max_children: int,
        min_children: int,
    ) -> Iterator[int]:
        """
        Genera (sin materializar listas) cuantos hijos recibe cada nodo de un
        nivel, con exactamente la misma distribucion que la version anterior:
        los primeros (num_nodes - 2) nodos van llenos a max_children y los dos
        ultimos se reparten el resto en partes (casi) iguales, de modo que
        ambos cumplen [min_children, max_children]. Un unico nodo se devuelve
        tal cual (sera la raiz, con regla especial).
        """
        if node_count <= max_children:
            yield node_count
            return

        num_nodes = (node_count + max_children - 1) // max_children
        full      = num_nodes - 2
        rest      = node_count - full * max_children
        left      = (rest + 1) // 2
        right     = rest - left

        if (
            left < min_children or left > max_children
            or right < min_children or right > max_children
        ):
            raise ValueError(
                f"reparto de nivel invalido: full={full} left={left} right={right} "
                f"(min={min_children}, max={max_children})"
            )

        for _ in range(full):
            yield max_children
        yield left
        yield right

    def _compute_level_counts(self, total_entries: int) -> list[int]:
        """
        Determina de antemano el numero de nodos de cada nivel (hojas -> raiz).
        counts[k] = nodos del nivel k; counts[-1] == 1 (raiz). Sin I/O ni while
        exploratorio (aritmetica O(log N)).
        """
        counts = [self._count_partition(total_entries, self.LEAF_ORDER)]
        while counts[-1] > 1:
            counts.append(self._count_partition(counts[-1], self.INTERNAL_ORDER + 1))
        return counts

    def _flush_level_page(self, f: BinaryIO, entries: list[LevelEntry]) -> None:
        """Escribe una pagina de LevelEntry (key, offset) al temporal de nivel."""
        entry_format = '=' + self.key_fmt + BPlusNode.PTR_FORMAT
        packed       = bytearray()

        packed.extend(struct.pack(self.TEMP_PAGE_HEADER_FORMAT, len(entries)))

        for key, offset in entries:
            packed.extend(struct.pack(entry_format, key, offset))

        remaining = self.PAGE_SIZE - len(packed)
        if remaining > 0:
            packed.extend(b'\x00' * remaining)

        f.write(packed)

    def _iter_level_entries(self, path: str) -> Iterator[LevelEntry]:
        """Itera en streaming los (key, offset) del temporal de nivel."""
        entry_format     = '=' + self.key_fmt + BPlusNode.PTR_FORMAT
        entry_size       = struct.calcsize(entry_format)
        pages_per_buffer = max(1, self.sort_buffer_size // self.PAGE_SIZE)
        buffer_size      = pages_per_buffer * self.PAGE_SIZE

        with open(path, 'rb') as f:
            while True:
                buffer = f.read(buffer_size)
                if not buffer:
                    break
                for page_start in range(0, len(buffer), self.PAGE_SIZE):
                    page = buffer[page_start:page_start + self.PAGE_SIZE]
                    if len(page) < self.PAGE_SIZE:
                        break
                    num_records = struct.unpack(
                        self.TEMP_PAGE_HEADER_FORMAT,
                        page[:self.TEMP_PAGE_HEADER_SIZE],
                    )[0]
                    pos = self.TEMP_PAGE_HEADER_SIZE
                    for _ in range(num_records):
                        chunk = page[pos:pos + entry_size]
                        if len(chunk) < entry_size:
                            break
                        key, offset = struct.unpack(entry_format, chunk)
                        yield key, offset
                        pos += entry_size

    def _store_level(
        self,
        entries: Iterator[LevelEntry],
        count: int,
        tag: str,
    ) -> tuple[str, object]:
        """
        Almacena la metadata de un nivel. Unica condicion de decision:
            count * entry_cost <= level_memory_limit  -> RAM
        si no, escribe a un temporal que el nivel superior consumira una vez.
        """
        entry_cost = self.key_size + BPlusNode.PTR_SIZE

        if count * entry_cost <= self.level_memory_limit:
            return ('ram', list(entries))

        path         = os.path.join(
            self.data_dir, f'tmp_level_{self.key_column}_{tag}.bin'
        )
        entry_format = '=' + self.key_fmt + BPlusNode.PTR_FORMAT
        entry_size   = struct.calcsize(entry_format)
        usable_bytes = self.PAGE_SIZE - self.TEMP_PAGE_HEADER_SIZE
        max_entries  = usable_bytes // entry_size

        try:
            with open(path, 'wb') as f:
                buffer: list[LevelEntry] = []
                for entry in entries:
                    buffer.append(entry)
                    if len(buffer) >= max_entries:
                        self._flush_level_page(f, buffer)
                        buffer.clear()
                if buffer:
                    self._flush_level_page(f, buffer)
        except Exception:
            if os.path.exists(path):
                os.remove(path)
            raise

        return ('file', path)

    def _iter_level(self, ref: tuple[str, object]) -> Iterator[LevelEntry]:
        """Itera la metadata de un nivel (en RAM o desde temporal)."""
        kind, payload = ref
        if kind == 'ram':
            yield from payload                      # type: ignore[misc]
        else:
            yield from self._iter_level_entries(payload)   # type: ignore[arg-type]

    def _remove_level(self, ref: tuple[str, object]) -> None:
        """Elimina el temporal de un nivel (si lo hubiera)."""
        kind, payload = ref
        if kind == 'file' and os.path.exists(payload):   # type: ignore[arg-type]
            os.remove(payload)                           # type: ignore[arg-type]

    def _build_leaf_level(
        self,
        f: BinaryIO,
        sorted_path: str,
        total_entries: int,
        leaf_count: int,
    ) -> Iterator[LevelEntry]:
        """
        Construye el nivel de hojas y emite (first_key, leaf_offset) por hoja.
        Reparte con _iter_partition (sin listas globales) y enlaza hojas por
        next_leaf sin preasignar todos los offsets (solo el actual y el siguiente).
        Si el indice es UNIQUE, valida duplicados en esta misma pasada.
        """
        sizes   = self._iter_partition(
            total_entries, self.LEAF_ORDER, self._min_keys_leaf()
        )
        entries = self._iter_index_entries(sorted_path)

        previous: Key | None = None
        first    = True
        group    = 0

        remaining_in_group = next(sizes)
        current_offset     = self._allocate_node(f)
        current_leaf = BPlusNode(
            self.key_fmt, self.key_size, self.LEAF_ORDER, self.PAGE_SIZE,
            node_type=1, is_root=1 if leaf_count == 1 else 0,
        )
        next_offset = self._allocate_node(f) if leaf_count > 1 else -1
        current_leaf.next_leaf = next_offset

        try:
            for key, rid in entries:
                if self.is_unique and not first and self._compare_keys(previous, key) == 0:
                    raise _DuplicateKeyError(
                        f"clave duplicada en columna unica: {key!r}"
                    )
                previous = key
                first    = False

                if remaining_in_group == 0:
                    self._write_node(f, current_leaf, current_offset)
                    yield current_leaf.keys[0], current_offset

                    group += 1
                    remaining_in_group = next(sizes)
                    current_offset     = next_offset
                    next_offset = (
                        self._allocate_node(f) if group < leaf_count - 1 else -1
                    )
                    current_leaf = BPlusNode(
                        self.key_fmt, self.key_size, self.LEAF_ORDER, self.PAGE_SIZE,
                        node_type=1,
                    )
                    current_leaf.next_leaf = next_offset

                current_leaf.keys.append(key)
                current_leaf.pointers.append(rid)
                current_leaf.num_keys += 1
                remaining_in_group -= 1
        finally:
            # liberar el archivo temporal si se aborta por duplicado
            entries.close()

        if current_leaf.num_keys > 0:
            self._write_node(f, current_leaf, current_offset)
            yield current_leaf.keys[0], current_offset

    def _build_internal_level(
        self,
        f: BinaryIO,
        child_ref: tuple[str, object],
        child_count: int,
        parent_count: int,
    ) -> Iterator[LevelEntry]:
        """
        Construye un nivel de nodos internos consumiendo la metadata del nivel
        hijo (RAM o temporal) y emite (first_key, node_offset) por nodo.
        El reparto se genera con _iter_partition (sin listas globales) y el
        numero de nodos es parent_count (ya conocido de antemano).
        """
        sizes = self._iter_partition(
            child_count, self.INTERNAL_ORDER + 1, self._min_keys_internal() + 1
        )
        child_entries = self._iter_level(child_ref)

        for size in sizes:
            current_node = BPlusNode(
                self.key_fmt, self.key_size, self.INTERNAL_ORDER, self.PAGE_SIZE,
                node_type=0, is_root=1 if parent_count == 1 else 0,
            )
            current_offset = self._allocate_node(f)
            first_child = True
            current_subtree_min: Key | None = None

            for _ in range(size):
                key, ptr = next(child_entries)

                child = self._read_node(f, ptr)
                child.parent_ptr = current_offset
                self._write_node(f, child, ptr)

                if first_child:
                    current_node.pointers.append(ptr)
                    current_subtree_min = key
                    first_child = False
                else:
                    current_node.keys.append(key)
                    current_node.pointers.append(ptr)
                    current_node.num_keys += 1

            self._write_node(f, current_node, current_offset)
            yield current_subtree_min, current_offset

    def build_bulk_tree(self, sorted_path: str, total_entries: int) -> None:
        """
        Construye el arbol bottom-up desde los pares ordenados.
        Los conteos de cada nivel se calculan de antemano (sin while
        exploratorio). La metadata de cada nivel vive en RAM si cabe en
        level_memory_limit, o en un temporal que se consume una sola vez y se
        elimina.
        """
        f = self.file

        if total_entries == 0:
            self._init_empty_tree()
            return

        counts = self._compute_level_counts(total_entries)

        ref = self._store_level(
            self._build_leaf_level(f, sorted_path, total_entries, counts[0]),
            counts[0],
            "leaf",
        )
        try:
            for level in range(1, len(counts)):
                new_ref = self._store_level(
                    self._build_internal_level(
                        f, ref, counts[level - 1], counts[level]
                    ),
                    counts[level],
                    str(level),
                )
                self._remove_level(ref)
                ref = new_ref

            # counts[-1] == 1: el unico LevelEntry del ultimo nivel es la raiz
            _, root_ptr = next(self._iter_level(ref))
        finally:
            self._remove_level(ref)

        _, free_head, total_nodes, _ = self._read_header(f)
        self._write_header(f, root_ptr, free_head, total_nodes, total_entries)
        self.root = root_ptr

    