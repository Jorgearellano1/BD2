from __future__ import annotations

import csv
import os
import struct
import tempfile
from collections import defaultdict
from typing import BinaryIO, Iterable, Iterator

from engine.catalog import TableMetadata, RawRecord, RID, Key, CleanRecord
from engine.conditions import Expression, evaluate_expression
from storageFile import StorageFile


class HeapFile(StorageFile):
    pass