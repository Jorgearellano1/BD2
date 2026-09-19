from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from engine.metadata import RID


@dataclass
class Bitmap:
    """
    Conjunto de RIDs representado como page_id -> mascara de slots.

    El bit slot_id de la mascara indica que el RID (page_id, slot_id)
    pertenece al conjunto.
    """

    pages: dict[int, int] = field(default_factory=dict)

    def add(self, rid: RID) -> None:
        """Marca un RID en el bitmap."""
        page_id, slot_id = rid
        self.pages[page_id] = self.pages.get(page_id, 0) | (1 << slot_id)

    def contains(self, rid: RID) -> bool:
        """True si el RID esta marcado."""
        page_id, slot_id = rid
        return bool(self.pages.get(page_id, 0) & (1 << slot_id))

    def iter_rids(self) -> Iterator[RID]:
        """Recorre los RIDs marcados (slots en orden ascendente por pagina)."""
        for page_id, mask in self.pages.items():
            current = mask
            while current:
                lsb = current & -current
                slot_id = lsb.bit_length() - 1
                current ^= lsb
                yield (page_id, slot_id)

    def __and__(self, other: Bitmap) -> Bitmap:
        """Interseccion de dos bitmaps."""
        if not isinstance(other, Bitmap):
            return NotImplemented

        pages: dict[int, int] = {}
        for page_id, mask in self.pages.items():
            other_mask = other.pages.get(page_id)
            if other_mask is None:
                continue
            combined = mask & other_mask
            if combined:
                pages[page_id] = combined
        return Bitmap(pages)

    def __or__(self, other: Bitmap) -> Bitmap:
        """Union de dos bitmaps."""
        if not isinstance(other, Bitmap):
            return NotImplemented

        pages: dict[int, int] = dict(self.pages)
        for page_id, mask in other.pages.items():
            combined = pages.get(page_id, 0) | mask
            if combined:
                pages[page_id] = combined
        return Bitmap(pages)
