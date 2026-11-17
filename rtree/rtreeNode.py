

class MBR:
    """
    Minimum Bounding Rectangle.
    Rectángulo envolvente mínimo de un objeto espacial.
    """

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def area(self):
        """
        Calcula el área del rectángulo.
        """
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        return width * height

    def union(self, other):
        """
        Devuelve un nuevo MBR que contiene ambos rectángulos.
        """
        xmin = min(self.xmin, other.xmin)
        ymin = min(self.ymin, other.ymin)
        xmax = max(self.xmax, other.xmax)
        ymax = max(self.ymax, other.ymax)

        return MBR(xmin, ymin, xmax, ymax)

    def enlargement(self, other):
        """
        Calcula cuánto aumentaría el área al incluir otro MBR.
        """
        combined_mbr = self.union(other)

        return combined_mbr.area() - self.area()

    def intersects(self, other):
        """
        Comprueba si dos MBR se intersectan.
        """
        if self.xmax < other.xmin:
            return False

        if self.xmin > other.xmax:
            return False

        if self.ymax < other.ymin:
            return False

        if self.ymin > other.ymax:
            return False

        return True

    def mindist(self, x, y):
        """
        Calcula la distancia mínima entre un punto y el MBR.
        """
        dx = max(self.xmin - x, 0, x - self.xmax)
        dy = max(self.ymin - y, 0, y - self.ymax)

        return (dx**2 + dy**2)**0.5

    def __repr__(self):
        return (
            f"MBR({self.xmin}, {self.ymin}, "
            f"{self.xmax}, {self.ymax})"
        )



class LeafEntry:
    """
    Entrada de un nodo hoja.
    Almacena un MBR y el RID del registro.
    """

    def __init__(self, mbr, rid=None):
        self.mbr = mbr
        self.rid = rid


class InternalEntry:
    """
    Entrada de un nodo interno.
    Almacena un MBR y el identificador de la página hija.
    """

    def __init__(self, mbr, child_page_id=None):
        self.mbr = mbr
        self.child_page_id = child_page_id




class RTreeNode:
    """
    Nodo del R-Tree.
    Puede ser hoja o interno.
    """

    def __init__(self, is_leaf=True, page_id=None, parent_page_id=None):
        self.is_leaf = is_leaf
        self.entries = []
        self.page_id = page_id
        self.parent_page_id = parent_page_id

    def add_entry(self, entry):
        self.entries.append(entry)

    def remove_entry(self, entry):
        self.entries.remove(entry)

    def is_empty(self):
        return len(self.entries) == 0

    def is_full(self, max_entries):
        return len(self.entries) >= max_entries

    def __repr__(self):
        node_type = "Leaf" if self.is_leaf else "Internal"
        return f"RTreeNode(type={node_type}, page_id={self.page_id}, entries={len(self.entries)})"

