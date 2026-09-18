from typing import Optional, List, Tuple, Dict, Any

from storage.heapFile import HeapFile
from engine.metadata import RID, Key, TableMetadata

class HashEntry:

    """
    Una entrada del índice hash.

    Guarda:

        key -> RID del registro dentro del HeapFile
    """

    def __init__(
        self,
        key: Key,
        rid: RID
    ):
        self.key = key
        self.rid = rid

    def __eq__(self, other):
        """
        Dos entradas son iguales si tienen
        la misma key y el mismo RID.
        """

        if not isinstance(other, HashEntry):
            return NotImplemented

        return (
            self.key == other.key
            and self.rid == other.rid
        )

    def __repr__(self):
        """
        Representación útil para debugging.
        """

        return (
            f"HashEntry("
            f"key={self.key!r}, "
            f"rid={self.rid!r}"
            f")"
        )

    def serialize(self):
        """
        Convierte la entrada a una estructura
        que pueda almacenarse posteriormente.

        Por ahora usamos un diccionario.
        """

        return {
            "key": self.key,
            "rid": self.rid
        }

    @classmethod
    def deserialize(cls, data):
        """
        Reconstruye un HashEntry a partir
        de los datos serializados.
        """

        return cls(
            key=data["key"],
            rid=data["rid"]
        )

class HashBucket:

    """
    Bucket del Extendible Hashing.

    Un bucket contiene entradas (key, RID).
    """

    def __init__(
        self,
        bucket_id: int,
        local_depth: int,
        capacity: int
    ):
        self.bucket_id = bucket_id
        self.local_depth = local_depth
        self.capacity = capacity
        self.entries: List[HashEntry] = []

    
    # Estado del bucket

    def is_full(self) -> bool:
        """
        Indica si el bucket alcanzó su capacidad máxima.
        """
        return len(self.entries) >= self.capacity

    def is_empty(self) -> bool:
        """
        Indica si el bucket no contiene entradas.
        """
        return len(self.entries) == 0

    def size(self) -> int:
        """
        Cantidad de entradas almacenadas en el bucket.
        """
        return len(self.entries)

    def has_space(self) -> bool:
        """
        Indica si todavía existe espacio para otra entrada.
        """
        return len(self.entries) < self.capacity

    # Operaciones sobre entradas

    def insert(
        self,
        entry: HashEntry
    ) -> bool:
        """
        Inserta una entrada si existe espacio.

        Retorna:
            True  -> si se insertó
            False -> si el bucket está lleno
        """

        if self.is_full():
            return False

        self.entries.append(entry)

        return True

    def remove(
        self,
        key: Key,
        rid: Optional[RID] = None
    ) -> bool:
        """
        Elimina una entrada.

        Si rid es None:
            elimina la primera entrada con esa key.

        Si rid tiene valor:
            elimina exactamente la entrada key + rid.

        Retorna:
            True  -> si se eliminó
            False -> si no se encontró
        """

        for i, entry in enumerate(self.entries):

            if entry.key != key:
                continue

            if rid is not None and entry.rid != rid:
                continue

            self.entries.pop(i)

            return True

        return False

    def find(
        self,
        key: Key
    ) -> Optional[HashEntry]:
        """
        Busca la primera entrada asociada a una key.

        Retorna la entrada encontrada o None.
        """

        for entry in self.entries:

            if entry.key == key:
                return entry

        return None

    def find_all(
        self,
        key: Key
    ) -> List[HashEntry]:
        """
        Busca todas las entradas asociadas a una key.
        """

        result = []

        for entry in self.entries:

            if entry.key == key:
                result.append(entry)

        return result

    def contains(
        self,
        key: Key,
        rid: Optional[RID] = None
    ) -> bool:
        """
        Comprueba si existe una entrada.

        Si rid es None:
            busca cualquier entrada con esa key.

        Si rid tiene valor:
            busca exactamente key + rid.
        """

        for entry in self.entries:

            if entry.key != key:
                continue

            if rid is not None and entry.rid != rid:
                continue

            return True

        return False

    def clear(self):
        """
        Elimina todas las entradas del bucket.
        """

        self.entries.clear()

    # --------------------------------------------------
    # Persistencia
    # --------------------------------------------------

    def serialize(self):
        """
        Convierte el bucket en una estructura serializable.
        """

        return {
            "bucket_id": self.bucket_id,
            "local_depth": self.local_depth,
            "capacity": self.capacity,
            "entries": [
                entry.serialize()
                for entry in self.entries
            ]
        }

    @classmethod
    def deserialize(
        cls,
        data
    ):
        """
        Reconstruye un HashBucket desde datos serializados.
        """

        bucket = cls(
            bucket_id=data["bucket_id"],
            local_depth=data["local_depth"],
            capacity=data["capacity"]
        )

        bucket.entries = [
            HashEntry.deserialize(entry_data)
            for entry_data in data["entries"]
        ]

        return bucket

    # Debug

    def __repr__(self):
        """
        Representación del bucket para debugging.
        """

        return (
            f"HashBucket("
            f"id={self.bucket_id}, "
            f"ld={self.local_depth}, "
            f"capacity={self.capacity}, "
            f"entries={self.entries}"
            f")"
        )

    def print_bucket(self):
        """
        Imprime información legible del bucket.
        """

        print(
            f"Bucket {self.bucket_id} "
            f"(LD={self.local_depth}, "
            f"Capacity={self.capacity}, "
            f"Size={self.size()})"
        )

        if self.is_empty():
            print("  [vacío]")
            return

        for entry in self.entries:
            print(
                f"  {entry.key} -> {entry.rid}"
            )


class ExtendibleHash:
    """
    Índice Extendible Hashing asociado a un HeapFile.

    El HeapFile es dueño de los registros.
    Este índice solamente mantiene:

        key -> RID

    Ejemplo:

        150 -> RID(page=20, slot=7)
    """

    def __init__(
        self,
        meta: TableMetadata,
        key_column: str,
        heap_file: HeapFile,
        index_path: str,
        bucket_capacity: int = 50,
        global_depth: int = 1
    ):
        super().__init__()

        self.meta = meta
        self.heap_file = heap_file
        self.key_column = key_column
        self.index_path = index_path

        self.bucket_capacity = bucket_capacity
        self.global_depth = global_depth

        self.directory: List[HashBucket] = []
        self.buckets: Dict[int, HashBucket] = {}
        self.next_bucket_id = 0

        self.is_open = False
        self.is_dirty = False

        self.inserts = 0
        self.deletes = 0
        self.searches = 0

        self.bucket_splits = 0
        self.bucket_merges = 0
        self.directory_doublings = 0
        self.directory_shrinks = 0

        self.page_reads = 0
        self.page_writes = 0


    # INICIALIZACIÓN

    def create(self):
        self.directory = []
        self.buckets = {}
        self.next_bucket_id = 0

        cantidad_buckets = 2 ** self.global_depth

        for _ in range(cantidad_buckets):
            bucket = HashBucket(
                bucket_id=self.next_bucket_id,
                local_depth=self.global_depth,
                capacity=self.bucket_capacity
            )

            self.buckets[bucket.bucket_id] = bucket
            self.directory.append(bucket)
            self.next_bucket_id += 1

        self.is_open = True
        self.is_dirty = True


    def open(self):
        """
        Abre un índice hash existente desde disco.
        """
        pass


    def close(self):
        """
        Guarda cambios y cierra el índice.
        """
        if not self.is_open:
            return

        self.flush()
        self.is_open = False


    def flush(self):
        """
        Escribe en disco los cambios pendientes.
        """
        if not self.is_open:
            return

        if not self.is_dirty:
            return

        # Pendiente: implementar persistencia en disco
        self.is_dirty = False
    def bulk_load(
        self,
        source_path: str,
        delimiter: str = ','
    ) -> None:

        if not self.directory:
            self.create()

        records_with_rids = self.heap_file.bulk_insert_with_rids(
            self.heap_file._csv_record_generator(
                source_path,
                delimiter
            ),
            reset=True
        )

        key_idx = self.meta.col_index(
            self.key_column
        )

        for record, rid in records_with_rids:
            key = self.meta.normalize_value(
                self.key_column,
                record[key_idx]
            )

            self.insert(key, rid)
    # HASH
    

    def _hash(
        self,
        key: Key,
        depth: Optional[int] = None
    ) -> int:
        """
        Calcula el hash utilizando los bits necesarios.
        """
        if depth is None:
            depth = self.global_depth

        # Convertimos la clave en un valor entero
        if isinstance(key, int):
            value = key
        else:
            value = hash(key)

        # Usamos solamente los 'depth' bits menos significativos
        mask = (1 << depth) - 1

        return value & mask


    def _directory_index(
        self,
        key: Key
    ) -> int:
        """
        Obtiene el índice del directorio correspondiente
        a una clave.
        """
        return self._hash(key, self.global_depth)


    def _get_bucket(
        self,
        key: Key
    ) -> HashBucket:
        """
        Obtiene el bucket donde debería encontrarse la clave.
        """

        # Calculamos el índice usando los bits correspondientes
        # al Global Depth actual.
        directory_index = self._hash(
            key,
            self.global_depth
        )

        # El directorio tiene exactamente 2^global_depth posiciones.
        # Por seguridad, nos aseguramos de que el índice esté
        # dentro del rango válido del directorio.
        directory_index = directory_index % len(self.directory)

        return self.directory[directory_index]

    # BUCKETS
    def _create_bucket(
        self,
        local_depth: int
    ) -> HashBucket:
        """
        Crea un nuevo bucket.
        """
        bucket = HashBucket(
            bucket_id=self.next_bucket_id,
            local_depth=local_depth,
            capacity=self.bucket_capacity
        )

        self.buckets[bucket.bucket_id] = bucket
        self.next_bucket_id += 1

        return bucket


    def _get_bucket_by_id(
        self,
        bucket_id: int
    ) -> Optional[HashBucket]:
        return self.buckets.get(bucket_id)


    def _remove_bucket(
        self,
        bucket_id: int
    ):
        self.buckets.pop(bucket_id, None)
    # DIRECTORY

    def _initialize_directory(self):
        """
        Crea el directorio inicial.

        GD = 1:
            0 -> bucket 0
            1 -> bucket 1
        """
        self.directory = []
        self.buckets = {}
        self.next_bucket_id = 0

        for _ in range(2 ** self.global_depth):
            bucket = self._create_bucket(self.global_depth)
            self.directory.append(bucket)


    def _double_directory(self):
        """
        Duplica el directorio cuando:
            local_depth == global_depth
        """
        old_directory = self.directory.copy()

        # El directorio pasa de 2^GD a 2^(GD+1)
        self.directory = old_directory + old_directory

        self.global_depth += 1
        self.directory_doublings += 1

        self.is_dirty = True


    def _shrink_directory(self):
        """
        Reduce el directorio cuando ya no es necesario
        mantener su tamaño actual.
        """
        # Mientras GD > 1 podemos intentar reducir el directorio.
        while self.global_depth > 1:

            half = 2 ** (self.global_depth - 1)

            # Las dos mitades deben apuntar a los mismos buckets.
            can_shrink = True

            for i in range(half):
                if self.directory[i] is not self.directory[i + half]:
                    can_shrink = False
                    break

            if not can_shrink:
                break

            self.directory = self.directory[:half]
            self.global_depth -= 1
            self.directory_shrinks += 1

        self.is_dirty = True


    def _update_directory_after_split(
        self,
        old_bucket: HashBucket,
        new_bucket: HashBucket
    ):
        """
        Actualiza las entradas del directorio después
        de dividir un bucket.
        """
        # El nuevo bucket tiene una profundidad local
        # mayor que el bucket original.
        new_local_depth = new_bucket.local_depth

        # Bit que diferencia al bucket antiguo del nuevo.
        split_bit = 1 << (new_local_depth - 1)

        for i in range(len(self.directory)):
            # Solo modificamos las entradas que actualmente
            # apuntan al bucket que acabamos de dividir.
            if self.directory[i] is old_bucket:

                # Si el bit correspondiente está activo,
                # esta entrada debe apuntar al nuevo bucket.
                if i & split_bit:
                    self.directory[i] = new_bucket

        self.is_dirty = True

    # INSERT

    def insert(
        self,
        key: Key,
        rid: RID
    ) -> None:
        """
        Inserta:
            key -> RID
        """
        entry = HashEntry(key, rid)
        self.insert_entry(entry)


    def insert_entry(
        self,
        entry: HashEntry
    ) -> None:
        """
        Inserta una entrada key -> RID.

        Si el bucket está lleno, intenta dividirlo.
        Si después del split la clave sigue cayendo en un
        bucket lleno (por ejemplo, muchas claves repetidas),
        aumenta la capacidad de ese bucket para evitar un
        ciclo infinito de splits.
        """

        bucket = self._get_bucket(entry.key)

        # --------------------------------------------------
        # Si ya hay espacio, insertamos directamente
        # --------------------------------------------------

        if bucket.has_space():

            if not bucket.insert(entry):
                raise RuntimeError(
                    "El bucket indicó tener espacio "
                    "pero no permitió la inserción."
                )

            self.inserts += 1
            self.is_dirty = True

            return

        # --------------------------------------------------
        # Bucket lleno
        # --------------------------------------------------

        max_attempts = 20

        for attempt in range(max_attempts):

            bucket = self._get_bucket(entry.key)

            # Puede que un split anterior haya dejado
            # espacio en este bucket.
            if bucket.has_space():

                if not bucket.insert(entry):
                    raise RuntimeError(
                        "Error insertando entrada después "
                        "de verificar espacio."
                    )

                self.inserts += 1
                self.is_dirty = True

                return

            # --------------------------------------------------
            # Intentar split
            # --------------------------------------------------

            old_size = bucket.size()
            old_capacity = bucket.capacity
            old_local_depth = bucket.local_depth

            self._split_bucket(bucket)

            # --------------------------------------------------
            # Obtener nuevamente el bucket
            # --------------------------------------------------

            bucket = self._get_bucket(entry.key)

            # --------------------------------------------------
            # Si ahora hay espacio, insertar
            # --------------------------------------------------

            if bucket.has_space():

                if not bucket.insert(entry):
                    raise RuntimeError(
                        "El bucket tiene espacio pero "
                        "la inserción falló."
                    )

                self.inserts += 1
                self.is_dirty = True

                return

            # --------------------------------------------------
            # El split no consiguió liberar espacio
            # --------------------------------------------------

            """
            Esto puede ocurrir cuando existen muchas entradas
            con exactamente la misma key.

            Ejemplo:

                26 -> RID1
                26 -> RID2
                26 -> RID3
                ...
                
            El hash de todas es idéntico, por lo que dividir
            el bucket no consigue separar esas entradas.
            """

            if (
                bucket.size() >= old_size
                and
                bucket.capacity == old_capacity
                and
                bucket.local_depth == old_local_depth + 1
            ):
                # Aumentamos capacidad para permitir
                # múltiples RIDs con la misma clave.
                bucket.capacity = max(
                    bucket.capacity * 2,
                    bucket.size() + 1
                )

                self.is_dirty = True

                print(
                    f"⚠️ Bucket {bucket.bucket_id}: "
                    f"muchas claves repetidas. "
                    f"Capacidad aumentada a "
                    f"{bucket.capacity}",
                    flush=True
                )

        # --------------------------------------------------
        # Protección contra ciclo infinito
        # --------------------------------------------------

        raise RuntimeError(
            "No fue posible insertar la entrada después de "
            f"{max_attempts} intentos. "
            f"key={entry.key}, rid={entry.rid}, "
            f"bucket={bucket.bucket_id}, "
            f"LD={bucket.local_depth}, "
            f"GD={self.global_depth}, "
            f"size={bucket.size()}, "
            f"capacity={bucket.capacity}"
        )


    def insert_many(
        self,
        entries: List[Tuple[Key, RID]]
    ) -> None:
        """
        Inserta múltiples entradas:
            [(key1, rid1), (key2, rid2), ...]
        """
        for key, rid in entries:
            self.insert(key, rid)

    # SPLIT
    def _split_bucket(
        self,
        bucket: HashBucket
    ) -> HashBucket:
        """
        Divide un bucket lleno.

        Si las entradas pueden separarse mediante el hash,
        se realiza un split normal.

        Si todas las entradas tienen la misma distribución de hash
        y no pueden separarse, se evita realizar splits infinitos.
        Esto es importante para índices sobre columnas con muchos
        valores repetidos, como Age.
        """

        # --------------------------------------------------
        # 1. Si LD == GD, primero duplicamos el directorio
        # --------------------------------------------------

        if bucket.local_depth == self.global_depth:
            self._double_directory()

        # --------------------------------------------------
        # 2. Guardamos las entradas originales
        # --------------------------------------------------

        old_entries = bucket.entries.copy()

        # --------------------------------------------------
        # 3. Calculamos la nueva profundidad local
        # --------------------------------------------------

        new_local_depth = bucket.local_depth + 1

        # --------------------------------------------------
        # 4. Creamos temporalmente el nuevo bucket
        # --------------------------------------------------

        new_bucket = self._create_bucket(
            new_local_depth
        )

        # --------------------------------------------------
        # 5. Aumentamos LD del bucket original
        # --------------------------------------------------

        bucket.local_depth = new_local_depth

        # --------------------------------------------------
        # 6. Actualizamos el directorio
        # --------------------------------------------------

        self._update_directory_after_split(
            bucket,
            new_bucket
        )

        # --------------------------------------------------
        # 7. Vaciamos el bucket original
        # --------------------------------------------------

        bucket.clear()

        # --------------------------------------------------
        # 8. Intentamos redistribuir
        # --------------------------------------------------

        bucket_entries = []
        new_bucket_entries = []

        for entry in old_entries:

            target_bucket = self._get_bucket(
                entry.key
            )

            if target_bucket is bucket:
                bucket_entries.append(entry)

            elif target_bucket is new_bucket:
                new_bucket_entries.append(entry)

            else:
                # Seguridad:
                # si el directorio apunta a otro bucket,
                # usamos el bucket correspondiente.
                target_bucket.entries.append(entry)

        # --------------------------------------------------
        # 9. Comprobamos si realmente hubo separación
        # --------------------------------------------------

        if (
            len(bucket_entries) == len(old_entries)
            or
            len(new_bucket_entries) == len(old_entries)
        ):
            """
            Todas las entradas terminaron en el mismo bucket.

            Esto ocurre cuando tenemos muchas claves repetidas,
            por ejemplo:

                Age = 25
                Age = 25
                Age = 25
                ...

            Como el hash depende de la key, todas producen
            exactamente el mismo hash.

            En ese caso NO podemos seguir dividiendo el bucket.
            """

            # Restauramos todas las entradas en el bucket original.
            bucket.clear()

            for entry in old_entries:
                bucket.entries.append(entry)

            # Eliminamos el bucket que no se pudo utilizar.
            self._remove_bucket(
                new_bucket.bucket_id
            )

            # Restauramos las referencias del directorio
            # que apuntaban al nuevo bucket.
            for i, directory_bucket in enumerate(
                self.directory
            ):
                if directory_bucket is new_bucket:
                    self.directory[i] = bucket

            # El split realmente no produjo una separación.
            self.bucket_splits -= 1

            # Aumentamos temporalmente la capacidad para permitir
            # múltiples registros con la misma key.
            if bucket.capacity <= len(bucket.entries):
                bucket.capacity = max(
                    bucket.capacity * 2,
                    len(bucket.entries) + 1
                )

            self.is_dirty = True

            return bucket

        # --------------------------------------------------
        # 10. Insertamos las entradas redistribuidas
        # --------------------------------------------------

        for entry in bucket_entries:
            bucket.entries.append(entry)

        for entry in new_bucket_entries:
            new_bucket.entries.append(entry)

        # --------------------------------------------------
        # 11. Verificación de seguridad
        # --------------------------------------------------

        total_after = (
            len(bucket.entries)
            +
            len(new_bucket.entries)
        )

        if total_after != len(old_entries):
            raise RuntimeError(
                "Error durante la redistribución del bucket: "
                f"antes={len(old_entries)}, "
                f"después={total_after}"
            )

        # --------------------------------------------------
        # 12. Estadísticas
        # --------------------------------------------------

        self.bucket_splits += 1
        self.is_dirty = True

        return new_bucket


    def _redistribute_entries(
        self,
        old_bucket: HashBucket,
        new_bucket: HashBucket
    ):
        """
        Redistribuye las entradas entre los dos buckets.
        """

        # Guardamos las entradas actuales.
        entries = old_bucket.entries.copy()

        # Vaciamos el bucket original.
        old_bucket.clear()

        # Las volvemos a colocar según el nuevo bit
        # de profundidad local.
        for entry in entries:

            bit = self._bucket_local_bit(
                entry.key,
                old_bucket.local_depth
            )

            if bit == 0:
                old_bucket.insert(entry)
            else:
                new_bucket.insert(entry)

        self.is_dirty = True


    def _bucket_local_bit(
        self,
        key: Key,
        local_depth: int
    ) -> int:
        """
        Obtiene el bit utilizado para decidir
        entre los dos buckets después de un split.
        """

        # Calculamos el hash usando la profundidad local.
        hash_value = self._hash(key, local_depth)

        # El bit que nos interesa es el último bit
        # de ese hash.
        return hash_value & 1
    
    # SEARCH
    
    def search(
        self,
        key: Key
    ) -> Optional[RID]:
        """
        Busca una clave y devuelve un RID.
        """
        entry = self.search_entry(key)

        if entry is None:
            return None

        return entry.rid


    def search_entry(
        self,
        key: Key
    ) -> Optional[HashEntry]:
        """
        Busca una clave y devuelve su HashEntry.
        """
        bucket = self._get_bucket(key)

        self.searches += 1

        entry = bucket.find(key)

        return entry


    def search_all(
        self,
        key: Key
    ) -> List[RID]:
        """
        Devuelve todos los RIDs asociados a una clave.
        """
        entries = self.search_all_entries(key)

        return [entry.rid for entry in entries]


    def search_all_entries(
        self,
        key: Key
    ) -> List[HashEntry]:
        """
        Devuelve todas las entradas asociadas a una clave.
        """
        bucket = self._get_bucket(key)

        self.searches += 1

        return bucket.find_all(key)

    
    # SEARCH + HEAPFILE
    

    def search_record(self, key: Key) -> CleanRecord | None:
        """
        Busca un registro por su clave usando el índice Hash.
        """

        # 1. Obtener el bucket correspondiente a la clave
        bucket = self._get_bucket(key)

        # 2. Buscar la entrada dentro del bucket
        entry = bucket.find(key)

        # 3. Si no existe la clave
        if entry is None:
            return None

        # 4. Obtener el RID asociado
        rid = entry.rid

        # 5. Leer el registro completo desde el HeapFile
        record = self.heap_file.read(rid)

        # 6. Si el registro ya no existe
        if record is None:
            return None

        # 7. Convertir a CleanRecord
        return self.meta.clean_tuple(record)


    def search_records(
        self,
        key: Key
    ) -> List[CleanRecord]:
        """
        Busca todos los RIDs asociados a una clave
        y recupera los registros correspondientes
        desde el HeapFile.
        """

        # 1. Obtener todos los RIDs de la clave
        rids = self.search_all(key)

        # 2. Lista donde guardaremos los registros
        records = []

        # 3. Recuperar cada registro desde el HeapFile
        for rid in rids:

            record = self.heap_file.read(rid)

            # Puede ocurrir que el RID ya no exista
            if record is None:
                continue

            # Convertir RawRecord -> CleanRecord
            clean_record = self.meta.clean_tuple(record)

            records.append(clean_record)

        return records


    
    # DELETE

    def delete(
        self,
        key: Key,
        rid: Optional[RID] = None
    ) -> bool:
        """
        Elimina una entrada key -> RID.
        """
        bucket = self._get_bucket(key)

        if not bucket.remove(key, rid):
            return False

        self.deletes += 1
        self.is_dirty = True

        return True


    def delete_all(
        self,
        key: Key
    ) -> int:
        """
        Elimina todas las entradas asociadas a key.
        """
        bucket = self._get_bucket(key)

        entries = bucket.find_all(key)

        if not entries:
            return 0

        # Eliminamos todas las entradas de esa clave.
        for entry in entries:
            bucket.remove(entry.key, entry.rid)

        self.deletes += len(entries)
        self.is_dirty = True

        return len(entries)

    # MERGE
    def _can_merge(
        self,
        bucket: HashBucket
    ) -> bool:
        """
        Determina si un bucket puede fusionarse
        con su bucket buddy.
        """
        buddy = self._get_buddy_bucket(bucket)

        if buddy is None:
            return False

        # Para fusionarse ambos buckets deben tener
        # la misma profundidad local.
        if buddy.local_depth != bucket.local_depth:
            return False

        # La suma de sus entradas debe caber en un solo bucket.
        if bucket.size() + buddy.size() > bucket.capacity:
            return False

        return True


    def _get_buddy_bucket(
        self,
        bucket: HashBucket
    ) -> Optional[HashBucket]:
        """
        Obtiene el bucket buddy de un bucket.
        """

        # Un bucket con LD=0 no tiene buddy que permita
        # una fusión adicional.
        if bucket.local_depth <= 0:
            return None

        # Buscamos una posición del directorio que apunte
        # al bucket.
        bucket_index = None

        for i, directory_bucket in enumerate(self.directory):
            if directory_bucket is bucket:
                bucket_index = i
                break

        if bucket_index is None:
            return None

        # El buddy se obtiene cambiando el bit correspondiente
        # a la profundidad local.
        buddy_index = bucket_index ^ (1 << (bucket.local_depth - 1))

        if buddy_index >= len(self.directory):
            return None

        buddy = self.directory[buddy_index]

        if buddy is bucket:
            return None

        return buddy


    def _merge_bucket(
        self,
        bucket: HashBucket
    ) -> Optional[HashBucket]:
        """
        Fusiona un bucket con su buddy.
        """

        buddy = self._get_buddy_bucket(bucket)

        if buddy is None:
            return None

        if not self._can_merge(bucket):
            return None

        # El bucket que vamos a conservar será el buddy.
        # Primero movemos las entradas del bucket hacia él.
        for entry in bucket.entries:
            buddy.insert(entry)

        # El bucket resultante tiene una profundidad local menor.
        buddy.local_depth -= 1

        # Todas las posiciones del directorio que apuntaban
        # al bucket eliminado deben apuntar al buddy.
        for i in range(len(self.directory)):
            if self.directory[i] is bucket:
                self.directory[i] = buddy

        # Eliminamos el bucket de la colección de buckets.
        self._remove_bucket(bucket.bucket_id)

        self.bucket_merges += 1
        self.is_dirty = True

        return buddy


    def _merge_after_delete(
        self,
        bucket: HashBucket
    ):
        """
        Intenta realizar merges después de eliminar
        una entrada.
        """

        current_bucket = bucket

        while current_bucket is not None:

            # Si no se puede fusionar, terminamos.
            if not self._can_merge(current_bucket):
                break

            merged_bucket = self._merge_bucket(current_bucket)

            if merged_bucket is None:
                break

            # Después de fusionar, el bucket resultante
            # podría volver a fusionarse con otro buddy.
            current_bucket = merged_bucket

        # Después de los merges intentamos reducir
        # el tamaño del directorio si es posible.
        self._shrink_directory()

    # HEAPFILE INTEGRATION
    def insert_record(
        self,
        record
    ) -> RID:
        """
        Inserta un registro mediante el HeapFile
        y después actualiza el índice.
        """

        rid = self.heap_file.insert(record)

        column_index = self.meta.col_index(
            self.key_column
        )

        key = record[column_index]

        self.insert(key, rid)

        return rid


    def delete_record(
        self,
        rid: RID
    ) -> bool:
        """
        Elimina la entrada del índice asociada
        a un RID.
        """

        # Recorremos los buckets buscando el RID.
        for bucket in self.buckets.values():

            for entry in bucket.entries:

                if entry.rid == rid:
                    key = entry.key

                    # Eliminamos exactamente key + RID.
                    deleted = self.delete(key, rid)

                    if deleted:
                        self._merge_after_delete(bucket)

                    return deleted

        return False


    def update_record(
        self,
        old_key: Key,
        new_key: Key,
        rid: RID
    ) -> None:
        """
        Actualiza el índice cuando cambia la clave
        de un registro.
        """

        # Primero eliminamos la relación anterior:
        # old_key -> RID
        deleted = self.delete(old_key, rid)

        if not deleted:
            return

        # Luego insertamos la nueva relación:
        # new_key -> RID
        self.insert(new_key, rid)

    # BULK LOAD

    def build_from_storage(self):
        """
        Construye el índice recorriendo todos los
        registros existentes del HeapFile.
        """

        self.create()

        column_index = self.meta.col_index(self.key_column)

        iterator = self.heap_file._iter_active_records()

        for rid, record in iterator:
            key = self.meta.normalize_value(
                self.key_column,
                record[column_index]
            )

            self.insert(key, rid)

    def bulk_insert(
        self,
        entries: List[Tuple[Key, RID]]
    ):
        """
        Inserta múltiples pares:
            (key, RID)
        """

        for key, rid in entries:
            self.insert(key, rid)
    # PERSISTENCIA
    def _write_header(self):
        """
        Escribe la información principal del índice.
        """
        import json

        data = {
            "global_depth": self.global_depth,
            "bucket_capacity": self.bucket_capacity,
            "next_bucket_id": self.next_bucket_id
        }

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f)


    def _read_header(self):
        """
        Lee la información principal del índice.
        """
        import json

        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.global_depth = data["global_depth"]
        self.bucket_capacity = data["bucket_capacity"]
        self.next_bucket_id = data["next_bucket_id"]


    def _write_directory(self):
        """
        Escribe el directorio del índice.
        """
        import json

        data = self._serialize_directory()

        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write("\n")
            json.dump(data, f)


    def _read_directory(self):
        """
        Lee el directorio del índice.
        """
        # La implementación completa depende del formato final
        # que utilicemos para almacenar header, directory y buckets.
        pass


    def _write_bucket(
        self,
        bucket: HashBucket
    ):
        """
        Escribe un bucket en disco.
        """
        import json

        with open(
            self.index_path,
            "a",
            encoding="utf-8"
        ) as f:
            f.write("\n")
            json.dump(bucket.serialize(), f)


    def _read_bucket(
        self,
        bucket_id: int
    ) -> HashBucket:
        """
        Lee un bucket específico desde disco.
        """
        # La implementación completa depende del formato final
        # de persistencia.
        bucket = self._get_bucket_by_id(bucket_id)

        if bucket is None:
            raise KeyError(
                f"Bucket {bucket_id} no existe"
            )

        return bucket


    def _write_all_buckets(self):
        """
        Escribe todos los buckets en disco.
        """
        for bucket in self.buckets.values():
            self._write_bucket(bucket)


    def _read_all_buckets(self):
        """
        Reconstruye todos los buckets.
        """
        # Los buckets se reconstruirán cuando implementemos
        # el formato definitivo de lectura.
        pass


    def _serialize_directory(self):
        """
        Convierte el directorio a una estructura
        serializable.

        El directorio guarda IDs de buckets,
        no copias de los buckets.
        """
        return [
            bucket.bucket_id
            for bucket in self.directory
        ]


    def _deserialize_directory(
        self,
        data
    ):
        """
        Reconstruye el directorio utilizando los
        bucket_id almacenados.
        """
        self.directory = []

        for bucket_id in data:
            bucket = self._get_bucket_by_id(bucket_id)

            if bucket is None:
                raise KeyError(
                    f"Bucket {bucket_id} no existe"
                )

            self.directory.append(bucket)

    # ESTADÍSTICAS

    def get_stats(self) -> Dict[str, Any]:
        """
        Devuelve estadísticas del índice Extendible Hashing.
        """
        return {
            "global_depth": self.global_depth,
            "bucket_capacity": self.bucket_capacity,
            "num_buckets": len(self.buckets),
            "directory_size": len(self.directory),
            "next_bucket_id": self.next_bucket_id,

            "inserts": self.inserts,
            "deletes": self.deletes,
            "searches": self.searches,

            "bucket_splits": self.bucket_splits,
            "bucket_merges": self.bucket_merges,
            "directory_doublings": self.directory_doublings,
            "directory_shrinks": self.directory_shrinks,

            "page_reads": self.page_reads,
            "page_writes": self.page_writes,

            "is_open": self.is_open,
            "is_dirty": self.is_dirty,
        }


    def reset_stats(self):
        """
        Reinicia las estadísticas del índice.
        """
        self.inserts = 0
        self.deletes = 0
        self.searches = 0

        self.bucket_splits = 0
        self.bucket_merges = 0
        self.directory_doublings = 0
        self.directory_shrinks = 0

        self.page_reads = 0
        self.page_writes = 0
    # DEBUG / VISUALIZACIÓN
    def print_buckets(self):
        """
        Muestra un resumen de los buckets sin imprimir
        miles de líneas en la terminal.
        """

        print("\nBuckets:")
        print("-" * 60)

        print(f"Total de buckets : {self.bucket_count}")
        print(f"Global Depth     : {self.global_depth}")
        print(f"Max Local Depth  : {self.max_local_depth}")

        # Mostrar solamente los primeros 5 buckets
        # como ejemplo de la estructura.
        limit = 5

        print("\nPrimeros buckets:")

        for i, bucket in enumerate(self.buckets.values()):

            if i >= limit:
                break

            print(
                f"  Bucket {bucket.bucket_id} "
                f"(LD={bucket.local_depth}, "
                f"Entries={bucket.size()})"
            )

        if self.bucket_count > limit:
            print(
                f"  ... y {self.bucket_count - limit} "
                f"buckets más."
            )


    def print_structure(self):
        """
        Muestra únicamente un resumen de la estructura
        del Extendible Hash.
        """

        print("\n" + "=" * 50)
        print("      EXTENDIBLE HASH STRUCTURE")
        print("=" * 50)

        print(f"Global Depth      : {self.global_depth}")
        print(f"Bucket Capacity   : {self.bucket_capacity}")
        print(f"Number of Buckets : {len(self.buckets)}")
        print(f"Directory Size    : {len(self.directory)}")

        if self.buckets:
            local_depths = [bucket.local_depth for bucket in self.buckets.values()]
            sizes = [bucket.size() for bucket in self.buckets.values()]

            print(f"Local Depth       : {min(local_depths)} - {max(local_depths)}")
            print(f"Entries           : {sum(sizes):,}")
            print(f"Bucket Size       : {min(sizes)} - {max(sizes)}")

        print("=" * 50)

    def print_stats(self):
        """
        Muestra las estadísticas del índice.
        """
        stats = self.get_stats()

        print("\n=== HASH INDEX STATS ===")

        for name, value in stats.items():
            print(f"{name}: {value}")
    # VALIDACIÓN
    def validate(self) -> bool:
        """
        Comprueba que la estructura del Extendible Hash
        sea consistente.
        """
        return (
            self._validate_directory()
            and self._validate_buckets()
            and self._validate_local_depths()
            and self._validate_entries()
        )


    def _validate_directory(self) -> bool:
        """
        Comprueba que el directorio sea consistente.
        """
        # El tamaño del directorio debe ser 2^GD.
        if len(self.directory) != 2 ** self.global_depth:
            return False

        # Todas las referencias del directorio deben apuntar
        # a buckets que realmente existan.
        for bucket in self.directory:
            if bucket.bucket_id not in self.buckets:
                return False

        return True


    def _validate_buckets(self) -> bool:
        """
        Comprueba que los buckets sean consistentes.
        """
        for bucket_id, bucket in self.buckets.items():

            # El ID del bucket debe coincidir con su clave
            # dentro del diccionario.
            if bucket.bucket_id != bucket_id:
                return False

            # Un bucket nunca puede superar su capacidad.
            if bucket.size() > bucket.capacity:
                return False

            # La capacidad debe ser válida.
            if bucket.capacity <= 0:
                return False

        return True


    def _validate_local_depths(self) -> bool:
        """
        Comprueba que las profundidades locales sean válidas.
        """
        for bucket in self.buckets.values():

            if bucket.local_depth < 0:
                return False

            if bucket.local_depth > self.global_depth:
                return False

        return True


    def _validate_entries(self) -> bool:
        """
        Comprueba que las entradas de los buckets
        sean consistentes.
        """
        for bucket in self.buckets.values():

            for entry in bucket.entries:

                if entry.key is None:
                    return False

                if entry.rid is None:
                    return False

        return True

    # PROPIEDADES / INFORMACIÓN
    @property
    def bucket_count(self) -> int:
        return len(self.buckets)


    @property
    def directory_size(self) -> int:
        return len(self.directory)


    @property
    def entry_count(self) -> int:
        return sum(
            bucket.size()
            for bucket in self.buckets.values()
        )


    @property
    def load_factor(self) -> float:
        if self.bucket_count == 0:
            return 0.0

        total_capacity = self.bucket_count * self.bucket_capacity

        return self.entry_count / total_capacity


    @property
    def max_local_depth(self) -> int:

        if not self.buckets:
            return 0

        return max(
            bucket.local_depth
            for bucket in self.buckets.values()
        )

    @property
    def filename(self) -> str:
        """
        Ruta del archivo donde se almacena el índice Hash.
        Compatible con el main y con la interfaz usada por B+ Tree.
        """
        return self.index_path