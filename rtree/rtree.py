import heapq
import os
import struct
from rtree.rtreeNode import MBR, LeafEntry, InternalEntry, RTreeNode

HEADER_FORMAT = "<8sIIiIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b"RTREE001"
VERSION = 1

class RTree:
    """Indice espacial R-Tree persistente en disco."""

    def __init__(
        self,
        table_meta,
        spatial_columns,
        heap,
        page_size,
        file_path
    ):
        """Inicializa la estructura y los recursos del índice R-Tree."""

        if len(spatial_columns) != 2:
            raise ValueError(
                "El R-Tree requiere exactamente dos columnas espaciales."
            )

        if page_size <= 0:
            raise ValueError("El tamaño de página debe ser positivo.")

        self.table_meta = table_meta
        self.spatial_columns = tuple(spatial_columns)
        self.heap = heap
        self.page_size = page_size
        self.file_path = file_path

        # Capacidad máxima segura considerando entradas de hoja.
        header_size = struct.calcsize("<Bii")
        entry_size = struct.calcsize("<dddd") + struct.calcsize("<II")

        self.max_entries = (page_size - header_size) // entry_size

        if self.max_entries < 2:
            raise ValueError(
                "El tamaño de página es insuficiente para almacenar "
                "al menos dos entradas."
            )

        self.file = None

        self.root_page_id = -1
        self.next_page_id = 1
        self.height = 0
        self.entry_count = 0

        self._page_reads = 0
        self._page_writes = 0

    # Inicializacion y persistencia

    def create(self):
        """Crea el archivo físico del índice R-Tree."""

        if os.path.exists(self.file_path):
            raise FileExistsError(
                f"El archivo del índice ya existe: {self.file_path}"
            )

        directory = os.path.dirname(self.file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w+b"):
            pass

        self.root_page_id = -1
        self.next_page_id = 0
        self.height = 0
        self.entry_count = 0

        self._page_reads = 0
        self._page_writes = 0

        self._initialize_file()

    def open(self):
        """Abre un índice R-Tree existente y recupera su cabecera."""

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"No existe el archivo del índice: {self.file_path}"
            )

        self.file = open(self.file_path, "r+b")
        self._read_header()

    def close(self):
        """Cierra el archivo físico del índice R-Tree."""

        if self.file is not None and not self.file.closed:
            self.file.flush()
            self.file.close()

        self.file = None

    def _initialize_file(self):
        """Inicializa la cabecera del archivo R-Tree."""

        self.root_page_id = -1
        self.next_page_id = 1
        self.height = 0
        self.entry_count = 0

        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            self.page_size,
            self.root_page_id,
            self.next_page_id,
            self.height,
            self.entry_count
        )

        with open(self.file_path, "r+b") as file:
            file.seek(0)
            file.write(header)

            remaining = self.page_size - HEADER_SIZE
            file.write(b"\x00" * remaining)

            file.flush()

    def _read_header(self):
        """Lee y valida la cabecera del archivo R-Tree."""

        self.file.seek(0)

        header_data = self.file.read(HEADER_SIZE)

        if len(header_data) != HEADER_SIZE:
            raise ValueError("La cabecera del R-Tree está incompleta.")

        (
            magic,
            version,
            page_size,
            root_page_id,
            next_page_id,
            height,
            entry_count
        ) = struct.unpack(HEADER_FORMAT, header_data)

        if magic != MAGIC:
            raise ValueError("El archivo no corresponde a un índice R-Tree.")

        if version != VERSION:
            raise ValueError("Versión del archivo R-Tree no compatible.")

        if page_size != self.page_size:
            raise ValueError("El tamaño de página no coincide con el configurado.")

        self.root_page_id = root_page_id
        self.next_page_id = next_page_id
        self.height = height
        self.entry_count = entry_count

        self._page_reads += 1

    def _write_header(self):
        """Escribe la cabecera actualizada del archivo R-Tree."""

        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            VERSION,
            self.page_size,
            self.root_page_id,
            self.next_page_id,
            self.height,
            self.entry_count
        )

        self.file.seek(0)
        self.file.write(header)

        remaining = self.page_size - HEADER_SIZE
        self.file.write(b"\x00" * remaining)

        self.file.flush()

        self._page_writes += 1

    # Gestion de paginas

    def _read_node(self, page_id):
        """Lee una página del disco y reconstruye su nodo R-Tree."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        if page_id < 1 or page_id >= self.next_page_id:
            raise ValueError(f"Identificador de página inválido: {page_id}")

        offset = page_id * self.page_size

        self.file.seek(offset)
        page_data = self.file.read(self.page_size)

        if len(page_data) != self.page_size:
            raise IOError(f"No se pudo leer la página {page_id} completa.")

        self._page_reads += 1

        return self._deserialize_node(page_data, page_id)

    def _write_node(self, node):
        """Serializa y escribe un nodo R-Tree en su página correspondiente."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        if node.page_id is None or node.page_id < 1:
            raise ValueError("El nodo no tiene un identificador de página válido.")

        page_data = self._serialize_node(node)

        if len(page_data) > self.page_size:
            raise ValueError(
                f"El nodo excede el tamaño de página de {self.page_size} bytes."
            )

        if len(page_data) < self.page_size:
            page_data += b"\x00" * (self.page_size - len(page_data))

        offset = node.page_id * self.page_size

        self.file.seek(offset)
        self.file.write(page_data)
        self.file.flush()

        self._page_writes += 1

    def _allocate_page(self):
        """Reserva un nuevo identificador de página para el R-Tree."""

        page_id = self.next_page_id

        self.next_page_id += 1

        offset = page_id * self.page_size

        self.file.seek(offset)
        self.file.write(b"\x00" * self.page_size)
        self.file.flush()

        self._page_writes += 1

        return page_id

    def _serialize_node(self, node):
        """Convierte un nodo R-Tree en una representación binaria."""

        node_header_format = "<Bii"
        node_header = struct.pack(
            node_header_format,
            1 if node.is_leaf else 0,
            len(node.entries),
            node.parent_page_id if node.parent_page_id is not None else -1
        )

        data = bytearray(node_header)

        for entry in node.entries:

            mbr_data = struct.pack(
                "<dddd",
                entry.mbr.xmin,
                entry.mbr.ymin,
                entry.mbr.xmax,
                entry.mbr.ymax
            )

            data.extend(mbr_data)

            if node.is_leaf:
                rid_page, rid_slot = entry.rid

                entry_data = struct.pack(
                    "<II",
                    rid_page,
                    rid_slot
                )
            else:
                entry_data = struct.pack(
                    "<I",
                    entry.child_page_id
                )

            data.extend(entry_data)

        if len(data) > self.page_size:
            raise ValueError(
                f"El nodo ocupa {len(data)} bytes y supera "
                f"el tamaño de página de {self.page_size} bytes."
            )

        return bytes(data)

    def _deserialize_node(self, data, page_id):
        """Reconstruye un nodo R-Tree a partir de su representación binaria."""

        node_header_format = "<Bii"
        node_header_size = struct.calcsize(node_header_format)

        is_leaf, entry_count, parent_page_id = struct.unpack(
            node_header_format,
            data[:node_header_size]
        )

        if is_leaf not in (0, 1):
            raise ValueError("Tipo de nodo R-Tree inválido.")

        if entry_count < 0:
            raise ValueError("Cantidad de entradas inválida.")

        node = RTreeNode(
            is_leaf=bool(is_leaf),
            page_id=page_id,
            parent_page_id=(
                None if parent_page_id == -1 else parent_page_id
            )
        )

        offset = node_header_size

        for _ in range(entry_count):

            mbr_size = struct.calcsize("<dddd")

            xmin, ymin, xmax, ymax = struct.unpack(
                "<dddd",
                data[offset:offset + mbr_size]
            )

            offset += mbr_size

            mbr = MBR(xmin, ymin, xmax, ymax)

            if node.is_leaf:
                entry_size = struct.calcsize("<II")

                rid_page, rid_slot = struct.unpack(
                    "<II",
                    data[offset:offset + entry_size]
                )

                entry = LeafEntry(mbr, (rid_page, rid_slot))

            else:
                entry_size = struct.calcsize("<I")

                child_page_id = struct.unpack(
                    "<I",
                    data[offset:offset + entry_size]
                )[0]

                entry = InternalEntry(mbr, child_page_id)

            offset += entry_size
            node.add_entry(entry)

        return node

    # Operaciones geometricas

    def _calculate_mbr(self, entries):
        """Calcula el MBR mínimo que contiene todas las entradas."""

        if not entries:
            return None

        xmin = entries[0].mbr.xmin
        ymin = entries[0].mbr.ymin
        xmax = entries[0].mbr.xmax
        ymax = entries[0].mbr.ymax

        for entry in entries[1:]:
            xmin = min(xmin, entry.mbr.xmin)
            ymin = min(ymin, entry.mbr.ymin)
            xmax = max(xmax, entry.mbr.xmax)
            ymax = max(ymax, entry.mbr.ymax)

        return MBR(xmin, ymin, xmax, ymax)

    def _calculate_node_mbr(self, node):
        """Calcula el MBR que engloba todas las entradas de un nodo."""

        if node is None or node.is_empty():
            return None

        return self._calculate_mbr(node.entries)

    def _choose_leaf(self, mbr):
        """Selecciona la hoja que requiere la menor ampliación de su MBR."""

        if self.root_page_id == -1:
            return None

        current_node = self._read_node(self.root_page_id)

        while not current_node.is_leaf:

            best_entry = None
            best_enlargement = float("inf")
            best_area = float("inf")

            for entry in current_node.entries:

                enlargement = entry.mbr.enlargement(mbr)
                area = entry.mbr.area()

                if (
                    enlargement < best_enlargement
                    or (
                        enlargement == best_enlargement
                        and area < best_area
                    )
                ):
                    best_entry = entry
                    best_enlargement = enlargement
                    best_area = area

            current_node = self._read_node(best_entry.child_page_id)

        return current_node

    # Insercion y division

    def insert(self, mbr, rid):
        """Inserta una entrada espacial en el R-Tree."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        if not isinstance(mbr, MBR):
            raise TypeError("La entrada debe contener un objeto MBR.")

        if len(rid) != 2 or rid[0] < 0 or rid[1] < 0:
            raise ValueError(
                "El RID debe contener una página y un slot válidos."
            )

        if mbr.xmin > mbr.xmax or mbr.ymin > mbr.ymax:
            raise ValueError("Las coordenadas del MBR no son válidas.")

        new_entry = LeafEntry(mbr, rid)

        if self.root_page_id == -1:

            root = RTreeNode(
                is_leaf=True,
                page_id=self._allocate_page(),
                parent_page_id=None
            )

            root.add_entry(new_entry)

            self.root_page_id = root.page_id
            self.height = 1

            self._write_node(root)

        else:

            leaf = self._choose_leaf(mbr)

            leaf.add_entry(new_entry)

            if leaf.is_full(self.max_entries):
                self._split_leaf(leaf)
            else:
                self._write_node(leaf)
                self._adjust_tree(leaf)

        self.entry_count += 1
        self._write_header()

        return True

    def _insert_entry(self, node, entry):
        """Inserta una entrada en un nodo y actualiza su MBR en disco."""

        if node is None:
            raise ValueError("No se puede insertar una entrada en un nodo inexistente.")

        if node.is_full():
            raise OverflowError("El nodo no tiene capacidad para otra entrada.")

        if node.is_leaf and not isinstance(entry, LeafEntry):
            raise TypeError("Un nodo hoja solamente admite entradas LeafEntry.")

        if not node.is_leaf and not isinstance(entry, InternalEntry):
            raise TypeError("Un nodo interno solamente admite entradas InternalEntry.")

        node.add_entry(entry)

        self._write_node(node)

        return node

    def _adjust_tree(self, node):
        """Actualiza los MBR de los nodos antecesores hasta llegar a la raíz."""

        if node is None:
            return

        current_node = node

        while current_node.parent_page_id is not None:

            current_mbr = self._calculate_node_mbr(current_node)

            if current_mbr is None:
                break

            parent = self._read_node(current_node.parent_page_id)

            parent_entry = None

            for entry in parent.entries:
                if entry.child_page_id == current_node.page_id:
                    parent_entry = entry
                    break

            if parent_entry is None:
                raise ValueError(
                    f"No se encontró el nodo hijo {current_node.page_id} "
                    f"en su padre {parent.page_id}."
                )

            parent_entry.mbr = current_mbr

            self._write_node(parent)

            current_node = parent

        if current_node.page_id == self.root_page_id:
            self._write_node(current_node)

    def _split_node(self, node):
        """Divide un nodo según su tipo y propaga los cambios al árbol."""

        if node is None:
            raise ValueError("No se puede dividir un nodo inexistente.")

        if not node.is_full():
            raise ValueError("No se puede dividir un nodo que no está lleno.")

        if node.is_leaf:
            return self._split_leaf(node)

        return self._split_internal(node)

    


    def _split_leaf(self, node):
        """Divide una hoja mediante Quadratic Split, con trazas de diagnóstico."""

        if node is None or not node.is_leaf:
            raise ValueError("El nodo debe ser una hoja válida.")

        targets = {
            "Shusf": (60.0, 31.81),
            "Bahawalnagar": (73.25, 30.0),
        }

        entries = list(node.entries)

        if len(entries) < 2:
            raise ValueError("No hay suficientes entradas para dividir la hoja.")

        # Detectar si la hoja contiene alguno de los registros investigados.
        traced = []

        for entry in entries:
            for name, point in targets.items():
                if (
                    entry.mbr.xmin == point[0]
                    and entry.mbr.xmax == point[0]
                    and entry.mbr.ymin == point[1]
                    and entry.mbr.ymax == point[1]
                ):
                    traced.append((name, entry))

        if traced:
            print(f"\n[TRACE-SPLIT-LEAF] Nodo={node.page_id}")
            print(f"[TRACE-SPLIT-LEAF] Padre={node.parent_page_id}")
            print(f"[TRACE-SPLIT-LEAF] Entradas antes={len(entries)}")

            for name, entry in traced:
                print(f"[TRACE-SPLIT-LEAF] Contiene {name} | RID={entry.rid}")

        # Seleccionar las dos entradas que generan mayor desperdicio.
        seed_i = 0
        seed_j = 1
        max_waste = float("-inf")

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                combined_mbr = entries[i].mbr.union(entries[j].mbr)

                waste = (
                    combined_mbr.area()
                    - entries[i].mbr.area()
                    - entries[j].mbr.area()
                )

                if waste > max_waste:
                    max_waste = waste
                    seed_i = i
                    seed_j = j

        seed_1 = entries[seed_i]
        seed_2 = entries[seed_j]

        remaining = [
            entry
            for i, entry in enumerate(entries)
            if i != seed_i and i != seed_j
        ]

        # Crear el nuevo nodo hoja.
        new_node = RTreeNode(
            is_leaf=True,
            page_id=self._allocate_page(),
            parent_page_id=node.parent_page_id
        )

        node.entries.clear()

        node.add_entry(seed_1)
        new_node.add_entry(seed_2)

        max_entries = self.max_entries
        min_entries = max(1, int(max_entries * 0.4 + 0.999999))

        # Distribuir las entradas restantes.
        while remaining:

            if len(node.entries) + len(remaining) == min_entries:
                for entry in remaining:
                    node.add_entry(entry)
                break

            if len(new_node.entries) + len(remaining) == min_entries:
                for entry in remaining:
                    new_node.add_entry(entry)
                break

            mbr_1 = self._calculate_node_mbr(node)
            mbr_2 = self._calculate_node_mbr(new_node)

            best_index = 0
            best_difference = float("-inf")

            for i, entry in enumerate(remaining):
                enlargement_1 = mbr_1.enlargement(entry.mbr)
                enlargement_2 = mbr_2.enlargement(entry.mbr)

                difference = abs(enlargement_1 - enlargement_2)

                if difference > best_difference:
                    best_difference = difference
                    best_index = i

            entry = remaining.pop(best_index)

            enlargement_1 = mbr_1.enlargement(entry.mbr)
            enlargement_2 = mbr_2.enlargement(entry.mbr)

            if (
                enlargement_1 < enlargement_2
                or (
                    enlargement_1 == enlargement_2
                    and mbr_1.area() <= mbr_2.area()
                )
            ):
                node.add_entry(entry)
            else:
                new_node.add_entry(entry)

        # Verificar distribución.
        if traced:
            print(
                f"[TRACE-SPLIT-LEAF] Nodo original={node.page_id} "
                f"entradas={len(node.entries)} "
                f"MBR={self._calculate_node_mbr(node)}"
            )

            print(
                f"[TRACE-SPLIT-LEAF] Nodo nuevo={new_node.page_id} "
                f"entradas={len(new_node.entries)} "
                f"MBR={self._calculate_node_mbr(new_node)}"
            )

            for name, entry in traced:
                if entry in node.entries:
                    destination = "ORIGINAL"
                elif entry in new_node.entries:
                    destination = "NUEVO"
                else:
                    destination = "PERDIDO"

                print(
                    f"[TRACE-SPLIT-LEAF] {name} "
                    f"RID={entry.rid} destino={destination}"
                )

        # Guardar ambos nodos.
        self._write_node(node)
        self._write_node(new_node)

        # Actualizar el padre o crear una nueva raíz.
        if node.parent_page_id is None:

            self._create_new_root(node, new_node)

        else:

            parent = self._read_node(node.parent_page_id)

            for parent_entry in parent.entries:
                if parent_entry.child_page_id == node.page_id:
                    parent_entry.mbr = self._calculate_node_mbr(node)
                    break
            else:
                raise ValueError(
                    "No se encontró la hoja dentro de su padre."
                )

            new_entry = InternalEntry(
                self._calculate_node_mbr(new_node),
                new_node.page_id
            )

            parent.add_entry(new_entry)

            self._write_node(parent)

            if parent.is_full(self.max_entries):
                self._split_internal(parent)
            else:
                self._adjust_tree(parent)

            if traced:
                print(
                    f"[TRACE-SPLIT-LEAF] Padre actualizado={parent.page_id}"
                )

                parent_mbr = self._calculate_node_mbr(parent)

                for name, point in targets.items():
                    contains = (
                        parent_mbr is not None
                        and parent_mbr.xmin <= point[0] <= parent_mbr.xmax
                        and parent_mbr.ymin <= point[1] <= parent_mbr.ymax
                    )

                    print(
                        f"[TRACE-SPLIT-LEAF] Padre contiene "
                        f"{name}: {contains}"
                    )

        if traced:
            print("[TRACE-SPLIT-LEAF] División finalizada\n")

        return new_node


    def _split_internal(self, node):
        """Divide un nodo interno mediante Quadratic Split."""

        if node is None or node.is_leaf:
            raise ValueError("El nodo debe ser un nodo interno válido.")

        entries = list(node.entries)

        if len(entries) < 2:
            raise ValueError("No hay suficientes entradas para dividir el nodo.")

        # Seleccionar las dos semillas con mayor desperdicio.
        seed_i = 0
        seed_j = 1
        max_waste = float("-inf")

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                combined_mbr = entries[i].mbr.union(entries[j].mbr)

                waste = (
                    combined_mbr.area()
                    - entries[i].mbr.area()
                    - entries[j].mbr.area()
                )

                if waste > max_waste:
                    max_waste = waste
                    seed_i = i
                    seed_j = j

        seed_1 = entries[seed_i]
        seed_2 = entries[seed_j]

        remaining = [
            entry
            for i, entry in enumerate(entries)
            if i != seed_i and i != seed_j
        ]

        # Crear el nuevo nodo interno.
        new_node = RTreeNode(
            is_leaf=False,
            page_id=self._allocate_page(),
            parent_page_id=node.parent_page_id
        )

        node.entries.clear()

        node.add_entry(seed_1)
        new_node.add_entry(seed_2)

        # Calcular la ocupación mínima permitida.
        max_entries = self.max_entries
        min_entries = max(1, int(max_entries * 0.4 + 0.999999))

        # Distribuir las entradas restantes.
        while remaining:

            if len(node.entries) + len(remaining) == min_entries:
                for entry in remaining:
                    node.add_entry(entry)
                break

            if len(new_node.entries) + len(remaining) == min_entries:
                for entry in remaining:
                    new_node.add_entry(entry)
                break

            mbr_1 = self._calculate_node_mbr(node)
            mbr_2 = self._calculate_node_mbr(new_node)

            best_index = 0
            best_difference = float("-inf")

            for i, entry in enumerate(remaining):
                enlargement_1 = mbr_1.enlargement(entry.mbr)
                enlargement_2 = mbr_2.enlargement(entry.mbr)

                difference = abs(enlargement_1 - enlargement_2)

                if difference > best_difference:
                    best_difference = difference
                    best_index = i

            entry = remaining.pop(best_index)

            enlargement_1 = mbr_1.enlargement(entry.mbr)
            enlargement_2 = mbr_2.enlargement(entry.mbr)

            if (
                enlargement_1 < enlargement_2
                or (
                    enlargement_1 == enlargement_2
                    and mbr_1.area() <= mbr_2.area()
                )
            ):
                node.add_entry(entry)
            else:
                new_node.add_entry(entry)

        # Actualizar los padres de los nodos hijos redistribuidos.
        for entry in node.entries:
            child = self._read_node(entry.child_page_id)
            child.parent_page_id = node.page_id
            self._write_node(child)

        for entry in new_node.entries:
            child = self._read_node(entry.child_page_id)
            child.parent_page_id = new_node.page_id
            self._write_node(child)

        # Persistir los nodos divididos.
        self._write_node(node)
        self._write_node(new_node)

        # Actualizar el padre o crear una nueva raíz.
        if node.parent_page_id is None:
            self._create_new_root(node, new_node)

        else:
            parent = self._read_node(node.parent_page_id)

            for parent_entry in parent.entries:
                if parent_entry.child_page_id == node.page_id:
                    parent_entry.mbr = self._calculate_node_mbr(node)
                    break
            else:
                raise ValueError(
                    "No se encontró el nodo interno dentro de su padre."
                )

            new_entry = InternalEntry(
                self._calculate_node_mbr(new_node),
                new_node.page_id
            )

            parent.add_entry(new_entry)

            self._write_node(parent)

            if parent.is_full(self.max_entries):
                self._split_internal(parent)
            else:
                self._adjust_tree(parent)

        return new_node
        

 

    def _create_new_root(self, left, right):
        """Crea una nueva raíz que conecta los dos nodos resultantes de una división."""

        new_root = RTreeNode(
            is_leaf=False,
            page_id=self._allocate_page(),
            parent_page_id=None
        )

        left.parent_page_id = new_root.page_id
        right.parent_page_id = new_root.page_id

        left_entry = InternalEntry(
            self._calculate_node_mbr(left),
            left.page_id
        )

        right_entry = InternalEntry(
            self._calculate_node_mbr(right),
            right.page_id
        )

        new_root.add_entry(left_entry)
        new_root.add_entry(right_entry)

        self._write_node(left)
        self._write_node(right)
        self._write_node(new_root)

        self.root_page_id = new_root.page_id
        self.height += 1

        self._write_header()

        return new_root

    # Busqueda espacial

    def search_intersects(self, query_mbr):
        """Busca los registros cuyos MBR intersectan con el MBR de consulta."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        if not isinstance(query_mbr, MBR):
            raise TypeError("La consulta debe ser un objeto MBR.")

        if self.root_page_id == -1:
            return []

        results = []

        self._search_recursive(
            self.root_page_id,
            query_mbr,
            results
        )

        return results


    def _search_recursive(self, page_id, query_mbr, results):
        """Recorre las ramas cuyos MBR intersectan con la consulta."""

        node = self._read_node(page_id)

        for entry in node.entries:

            intersects = entry.mbr.intersects(query_mbr)

            if not intersects:
                continue

            if node.is_leaf:

                results.append(entry.rid)

            else:

                self._search_recursive(
                    entry.child_page_id,
                    query_mbr,
                    results
                )

    # Busqueda por proximidad

    def nearest_neighbors(self, x, y, k):
        """Obtiene los K registros más cercanos a un punto mediante Best-First Search."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        if k <= 0:
            raise ValueError("K debe ser un entero positivo.")

        if not isinstance(k, int):
            raise TypeError("K debe ser un entero.")

        if self.root_page_id == -1:
            return []

        priority_queue = []
        candidates = []

        sequence = 0

        root = self._read_node(self.root_page_id)

        for entry in root.entries:

            distance = entry.mbr.mindist(x, y)

            if root.is_leaf:
                heapq.heappush(
                    priority_queue,
                    (distance, sequence, "leaf", entry.rid)
                )
            else:
                heapq.heappush(
                    priority_queue,
                    (distance, sequence, "internal", entry.child_page_id)
                )

            sequence += 1

        while priority_queue:

            distance, _, entry_type, value = heapq.heappop(priority_queue)

            if len(candidates) == k:
                worst_distance = -candidates[0][0]

                if distance > worst_distance:
                    break

            if entry_type == "leaf":

                rid = value
                candidate = (-distance, -rid[0], -rid[1], rid)

                if len(candidates) < k:
                    heapq.heappush(candidates, candidate)

                else:
                    worst = candidates[0]

                    if candidate > worst:
                        heapq.heapreplace(candidates, candidate)

            else:

                node = self._read_node(value)

                for entry in node.entries:

                    child_distance = entry.mbr.mindist(x, y)

                    if len(candidates) == k:
                        worst_distance = -candidates[0][0]

                        if child_distance > worst_distance:
                            continue

                    if node.is_leaf:
                        heapq.heappush(
                            priority_queue,
                            (
                                child_distance,
                                sequence,
                                "leaf",
                                entry.rid
                            )
                        )
                    else:
                        heapq.heappush(
                            priority_queue,
                            (
                                child_distance,
                                sequence,
                                "internal",
                                entry.child_page_id
                            )
                        )

                    sequence += 1

        results = [
            (candidate[3], -candidate[0])
            for candidate in candidates
        ]

        results.sort(key=lambda item: item[1])

        return results

    # Eliminacion

    def delete(self, mbr, rid):
        """Elimina una entrada espacial y reorganiza el R-Tree."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        if not isinstance(mbr, MBR):
            raise TypeError("La entrada debe contener un objeto MBR.")

        if not isinstance(rid, tuple) or len(rid) != 2:
            raise ValueError("El RID debe contener una página y un slot.")

        if self.root_page_id == -1:
            return False

        leaf = self._find_leaf(self.root_page_id, mbr, rid)

        if leaf is None:
            return False

        entry_to_delete = None

        for entry in leaf.entries:

            same_mbr = (
                entry.mbr.xmin == mbr.xmin
                and entry.mbr.ymin == mbr.ymin
                and entry.mbr.xmax == mbr.xmax
                and entry.mbr.ymax == mbr.ymax
            )

            if entry.rid == rid and same_mbr:
                entry_to_delete = entry
                break

        if entry_to_delete is None:
            return False

        leaf.remove_entry(entry_to_delete)

        self._write_node(leaf)

        self.entry_count -= 1

        if leaf.page_id == self.root_page_id:

            if leaf.is_empty():
                self.root_page_id = -1
                self.height = 0

        else:
            self._condense_tree(leaf)

        self._write_header()

        return True

    def _find_leaf(self, page_id, mbr, rid):
        """Busca la hoja que contiene una entrada identificada por su MBR y RID."""

        node = self._read_node(page_id)

        if node.is_leaf:

            for entry in node.entries:

                same_mbr = (
                    entry.mbr.xmin == mbr.xmin
                    and entry.mbr.ymin == mbr.ymin
                    and entry.mbr.xmax == mbr.xmax
                    and entry.mbr.ymax == mbr.ymax
                )

                if same_mbr and entry.rid == rid:
                    return node

            return None

        for entry in node.entries:

            if entry.mbr.intersects(mbr):

                result = self._find_leaf(
                    entry.child_page_id,
                    mbr,
                    rid
                )

                if result is not None:
                    return result

        return None

    def _condense_tree(self, node):
        """Reorganiza el R-Tree después de una eliminación."""

        if node is None:
            return

        orphaned_entries = []

        leaf_header_size = struct.calcsize("<Bii")
        leaf_entry_size = struct.calcsize("<ddddII")

        internal_header_size = struct.calcsize("<Bii")
        internal_entry_size = struct.calcsize("<ddddI")

        leaf_capacity = (
            self.page_size - leaf_header_size
        ) // leaf_entry_size

        internal_capacity = (
            self.page_size - internal_header_size
        ) // internal_entry_size

        leaf_min_entries = max(1, int(leaf_capacity * 0.4 + 0.999999))
        internal_min_entries = max(1, int(internal_capacity * 0.4 + 0.999999))

        def collect_leaf_entries(page_id):
            """Recopila las entradas hoja de un subárbol."""

            current = self._read_node(page_id)

            if current.is_leaf:
                orphaned_entries.extend(current.entries)
                return

            for entry in current.entries:
                collect_leaf_entries(entry.child_page_id)

        current_node = node

        while current_node.page_id != self.root_page_id:

            parent = self._read_node(current_node.parent_page_id)

            min_entries = (
                leaf_min_entries
                if current_node.is_leaf
                else internal_min_entries
            )

            if len(current_node.entries) < min_entries:

                collect_leaf_entries(current_node.page_id)

                parent_entry = None

                for entry in parent.entries:
                    if entry.child_page_id == current_node.page_id:
                        parent_entry = entry
                        break

                if parent_entry is None:
                    raise ValueError(
                        "No se encontró el nodo hijo dentro de su padre."
                    )

                parent.remove_entry(parent_entry)
                self._write_node(parent)

            else:

                current_mbr = self._calculate_node_mbr(current_node)

                for entry in parent.entries:
                    if entry.child_page_id == current_node.page_id:
                        entry.mbr = current_mbr
                        break
                else:
                    raise ValueError(
                        "No se encontró el nodo hijo dentro de su padre."
                    )

                self._write_node(current_node)
                self._write_node(parent)

            current_node = parent

        # Ajustar la raíz cuando queda con un único hijo.
        root = self._read_node(self.root_page_id)

        if not root.is_leaf and len(root.entries) == 1:

            new_root_page_id = root.entries[0].child_page_id
            new_root = self._read_node(new_root_page_id)

            new_root.parent_page_id = None

            self.root_page_id = new_root_page_id
            self.height -= 1

            self._write_node(new_root)

        elif not root.entries:

            self.root_page_id = -1
            self.height = 0

        else:
            self._write_node(root)

        # Ajustar el contador antes de reinsertar las entradas huérfanas.
        self.entry_count -= len(orphaned_entries)

        for entry in orphaned_entries:
            self.insert(entry.mbr, entry.rid)

        self._write_header()

    # Construccion del indice

    
    @classmethod
    def build_from_storage(
        cls,
        table_meta,
        heap,
        spatial_columns,
        page_size,
        index_path
    ):
        """Construye un índice R-Tree utilizando los registros del HeapFile."""

        import time

        if len(spatial_columns) != 2:
            raise ValueError("Se requieren dos columnas espaciales.")

        latitude_column, longitude_column = spatial_columns

        latitude_index = table_meta.col_index(latitude_column)
        longitude_index = table_meta.col_index(longitude_column)

        rtree = cls(
            table_meta=table_meta,
            spatial_columns=spatial_columns,
            heap=heap,
            page_size=page_size,
            file_path=index_path
        )

        rtree.create()
        rtree.open()

        processed = 0
        start_time = time.perf_counter()

        try:
            print("[R-TREE] Iniciando construcción del índice...")

            for page_records in heap.iter_pages_with_records():
                for rid, record in page_records:

                    latitude = float(record[latitude_index])
                    longitude = float(record[longitude_index])

                    mbr = MBR(
                        longitude,
                        latitude,
                        longitude,
                        latitude
                    )

                    rtree.insert(mbr, rid)
                    processed += 1

                    if processed % 10000 == 0:
                        elapsed = time.perf_counter() - start_time
                        print(
                            f"[R-TREE] Procesados: {processed:,} registros "
                            f"| Tiempo: {elapsed:.2f}s"
                        )

            rtree._write_header()

            elapsed = time.perf_counter() - start_time

            print(
                f"[R-TREE] Construcción finalizada: "
                f"{processed:,} registros en {elapsed:.2f}s"
            )

            return rtree

        except Exception:
            rtree.close()
            raise
    # Estadisticas

    def count(self):
        """Devuelve la cantidad de entradas espaciales del R-Tree."""

        return self.entry_count

    def clear(self):
        """Elimina todas las entradas y reinicializa el índice R-Tree."""

        if self.file is None or self.file.closed:
            raise RuntimeError("El archivo R-Tree no está abierto.")

        self.root_page_id = -1
        self.next_page_id = 1
        self.height = 0
        self.entry_count = 0

        self.file.seek(0)
        self.file.truncate(0)

        self.file.write(b"\x00" * self.page_size)
        self.file.flush()

        self._page_writes += 1

        self._write_header()

    @property
    def page_reads(self):
        """Devuelve la cantidad de lecturas de páginas realizadas."""
        return self._page_reads

    @property
    def page_writes(self):
        """Devuelve la cantidad de escrituras de páginas realizadas."""
        return self._page_writes
