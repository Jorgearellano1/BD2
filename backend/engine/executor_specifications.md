# Especificación del Executor

## 1. Objetivo

El Executor recibe planes producidos por el Planner y los ejecuta sobre las
estructuras físicas del sistema, manteniendo la consistencia entre el
almacenamiento y todos sus índices.

La vía normal es `Session` (§21), que planifica y ejecuta sentencia por
sentencia sobre un mismo `ExecutionContext`. `Executor.execute(ProgramPlan)`
es la API batch: planifica todo el `Program` antes de ejecutar y solo es válida
si la metadata de las tablas e índices referenciados ya está registrada en el
catálogo.

El flujo general es:

```text
AST
 │
 ▼
SemanticVisitor
 │
 ▼
Session
 │  por sentencia
 ▼
PlannerVisitor
 │
 ▼
ProgramPlan (una sentencia)
 ├── statements: list[PlanNode]
 └── statement_infos: list[StatementInfo]
 │
 ▼
Executor.execute_statement
 │
 ▼
ExecutionOutput
 ├── explains: list[StatementExplain]
 └── results: list[StatementResult]
```

La ejecución es determinista y no introduce costos, estadísticas ni
selección de candidatos; esas decisiones pertenecen al Planner.

---

## 2. Alcance y responsabilidades

El Executor no vuelve a validar lo que ya garantizan las etapas anteriores.

| Etapa | Validaciones |
|---|---|
| Parser | sintaxis, tokens y estructura de las sentencias |
| SemanticVisitor | existencia de tabla/índice/columna, tipos, PRIMARY KEY, UNIQUE, DOUBLE, SEQUENTIAL sin UNIQUE ni FORCE, nombre de índice libre |
| Planner | aplicabilidad de índices, FORCE, residual, rango solo sobre PRIMARY KEY, `EmptyResultPlan` para predicado falso, plan físico |
| Executor | estado físico y datos: rutas ocupadas, duplicados reales de PK/UNIQUE, B+Tree existente para `(tabla, columna)`, HASH no soportado, cabecera y tipos inferidos del CSV, consistencia Heap + índices y rollback |

El Executor no repite comprobaciones de existencia de tabla o columna, de
compatibilidad de tipos, ni de reglas FORCE o de elección de access path.

---

## 3. Contexto de ejecución

```python
@dataclass
class ExecutionContext:
    catalog: Catalog
    page_size: int
    sort_buffer_size: int = 128 * 1024 * 1024
    level_memory_limit: int = 64 * 1024 * 1024

    storage_cache: dict[str, StorageFile] = field(default_factory=dict)
    index_cache: dict[str, Index] = field(default_factory=dict)

    next_node_id: int = 1
    node_metrics: dict[int, NodeMetrics] = field(default_factory=dict)

    current_statement_id: int | None = None
    transaction: RollbackJournal | None = None

    io_reads: int = 0
    io_writes: int = 0
```

En una `Session` este contexto se comparte entre todas las sentencias del
`Program`: las cachés permanecen abiertas, `next_node_id` y las métricas siguen
acumulando, y el rollback continúa siendo por sentencia. `close()` se ejecuta
una sola vez al terminar el `Program`.

### 3.1. Claves de caché

Las cachés usan el nombre lógico de los recursos:

```text
storage_cache
    key = nombre logico de tabla

index_cache
    key = nombre logico de indice
```

El nombre lógico identifica la estructura dentro del Executor y del `Catalog`;
`TableEntry.file_path` e `IndexMetadata.file_path` identifican la ubicación
física y se usan únicamente para abrir el recurso.

El `Catalog` no expone operaciones de apertura. El Executor abre directamente
con la metadata del catálogo, la ruta y el `page_size` del contexto.

### 3.2. Apertura perezosa

```text
_get_storage(nombre)
    -> storage_cache o abrir con metadata + file_path + page_size

_get_index(nombre)
    -> index_cache o abrir con metadata + file_path + page_size
```

El cierre de estructuras es responsabilidad exclusiva del Executor. Un
`BPlusTree` nunca cierra el `HeapFile` compartido.

---

## 4. Métricas

### 4.1. Snapshots de contadores

```python
@dataclass(frozen=True)
class CounterSnapshot:
    reads: int
    writes: int
```

Los contadores de cada recurso (`page_reads`, `page_writes`) son
acumulativos. Nunca se reinician entre nodos; cada operación se mide por
diferencia:

```text
delta = after - before
```

### 4.2. OperationMetrics

```python
@dataclass
class OperationMetrics:
    resource_type: str
    resource_name: str
    operation: str
    elapsed_ms: float
    pages_read: int
    pages_written: int
    entries_affected: int = 0
```

`resource_type` puede ser `HeapFile`, `SequentialFile`, `BPlusTree` o
`ExtendibleHash`. `operation` identifica la llamada (`insert`, `search`,
`search_all`, `range_search`, `read_many`, `read_bitmap`, `delete`,
`remove_many`, `build`, `reset`, `bulk_load`, `seq_scan`, `seq_scan_rids`,
`delete_all`).

Semántica de `entries_affected` según la operación:

```text
insert / delete              -> 1 por registro afectado
remove_many / bulk_load      -> registros eliminados / cargados
build                        -> entradas construidas (count del indice)
search / range_search        -> RIDs encontrados
seq_scan / range_search      -> registros devueltos
seq_scan_rids                -> RIDs devueltos
read_many / read_bitmap      -> registros recuperados
reset / delete_all           -> registros eliminados
```

`build` y `bulk_load` registran su duración real con `perf_counter`. En la
rama de `CREATE TABLE FROM CSV` con columnas UNIQUE, `bulk_load_multiple`
construye el Heap y los índices en una sola llamada: el tiempo se asigna al
`bulk_load` del `HeapFile` y cada `build` de índice reporta sus entradas con
`elapsed_ms = 0`.

Una `OperationMetrics` representa una operación atómica dentro de un nodo; no
tiene hijos y no dispone de `self_time_ms`. No se resta del tiempo ni de las
páginas propias del nodo: forma parte de su trabajo directo.

### 4.3. NodeMetrics

```python
@dataclass
class NodeMetrics:
    node_id: int
    node_type: str
    elapsed_ms: float
    self_time_ms: float
    rows_out: int
    pages_read: int
    pages_written: int
    self_pages_read: int
    self_pages_written: int
    operations: list[OperationMetrics]
    children: list[NodeMetrics]
```

Semántica temporal:

```text
elapsed_ms
    = tiempo total del nodo, incluidos sus hijos

self_time_ms
    = elapsed_ms - suma(elapsed_ms de los hijos)
```

Semántica de E/S:

```text
pages_read / pages_written
    = E/S total atribuible al nodo, incluidos sus hijos

self_pages_read / self_pages_written
    = pages_* - suma(pages_* de los hijos)
```

### 4.4. Cálculo de E/S sin doble conteo

`BPlusTree.build_from_storage` acumula en sus contadores la E/S del Heap
compartido. Para esa operación:

```text
heap_delta  = delta del HeapFile
index_delta = delta del BPlusTree

index_owned = index_delta - heap_delta   (no negativo)
```

La operación de índice registra `index_owned`; el nodo suma
`heap_delta + index_owned`. En `BPlusTree.bulk_load_multiple` el Heap y cada
índice contabilizan su propia E/S por separado.

---

## 5. Tipado de registros

```text
Nodos de acceso
    -> RawRecord (records_clean=False)

FilterPlan / FilterRidsPlan
    -> preservan records_clean del hijo

ProjectPlan
    -> CleanRecord (records_clean=True)

SELECT * al cerrar la sentencia
    -> clean_tuple sobre los RawRecord
```

Los accesos entregan `RawRecord` porque `evaluate_expression`, las claves de
índice y las comparaciones trabajan con la representación binaria normalizada
(CHAR en `bytes`, FLOAT sin redondear). La limpieza se realiza en
`ProjectPlan` o, para `SELECT *`, al construir `StatementResult`.

`StatementResult.rows` es siempre una lista de `CleanRecord`.

---

## 6. ExecutionValue

```python
@dataclass
class ExecutionValue:
    records: list[RawRecord] | list[CleanRecord] | None = None
    rids: list[RID] | None = None
    bitmap: Bitmap | None = None
    affected_rows: int | None = None
    output_columns: list[str] | None = None
    records_clean: bool = False
```

Contratos de salida por nodo:

| Nodo | Produce |
|---|---|
| `SeqScanPlan` | records |
| `SequentialRangeScanPlan` | records |
| `SeqScanRidsPlan` | rids |
| `IndexScanPlan` | rids |
| `IndexRangeScanPlan` | rids |
| `BitmapIndexScanPlan` | bitmap |
| `BitmapIndexRangeScanPlan` | bitmap |
| `BitmapAndPlan` / `BitmapOrPlan` | bitmap |
| `HeapFetchPlan` / `BitmapHeapScanPlan` | records |
| `FilterPlan` | records |
| `FilterRidsPlan` | rids |
| `ProjectPlan` | CleanRecords |
| `EmptyResultPlan` | vacío |
| `InsertPlan` | affected_rows = 1 |
| `DeletePlan` | registros eliminados y affected_rows |
| `SequentialDeletePlan` | registros eliminados y affected_rows |
| `CreateTablePlan` / `CreateIndexPlan` / `AddUniquePlan` | affected_rows = 0 |

---

## 7. Accesos

### 7.1. SeqScanPlan

```text
predicate = None
    -> recorre el almacenamiento y devuelve todos los RawRecord

predicate = Expression
    -> recorre el almacenamiento y evalua la expresion completa
       durante el scan, devolviendo los RawRecord que cumplen
```

El predicado pertenece al nodo: no se genera un `FilterPlan` posterior. No se
usa `search_all()` porque esa API devuelve `CleanRecord`.

### 7.2. SequentialRangeScanPlan

```text
storage.range_search_raw(
    key_column=node.column,
    begin_key=node.range.lower,
    end_key=node.range.upper,
    include_begin=node.range.include_lower,
    include_end=node.range.include_upper,
)
```

Corresponde a un rango sobre la PRIMARY KEY de una tabla `SequentialFile`.
Produce `RawRecord`. El residual, si existe, se aplica con `FilterPlan`.

### 7.3. SeqScanRidsPlan

Exclusivo de `HeapFile`. Recorre `iter_pages_with_records()` y produce RIDs.
No se utiliza con `SequentialFile`.

### 7.4. IndexScanPlan

```text
index.search_all(node.key)
```

Se usa `search_all` incluso para B+Tree, porque un índice no único puede tener
varios RIDs por clave. No lee el Heap.

### 7.5. IndexRangeScanPlan

```text
index.range_search(lower, upper, include_lower, include_upper)
```

Solo aplica a BTREE. No lee el Heap.

### 7.6. BitmapIndexScanPlan y BitmapIndexRangeScanPlan

Ejecutan la búsqueda correspondiente y construyen un `Bitmap` con los RIDs
obtenidos.

### 7.7. BitmapAndPlan y BitmapOrPlan

```text
resultado = bitmap_1 & bitmap_2 & ...    (AND)
resultado = bitmap_1 | bitmap_2 | ...    (OR)
```

No realizan E/S física directa.

---

## 8. Materialización

### 8.1. HeapFetchPlan

```text
hijo con rids   -> heap.read_many(rids)
hijo con bitmap -> heap.read_bitmap(bitmap)
```

Produce `RawRecord`. No aplica filtros.

### 8.2. BitmapHeapScanPlan

```text
bitmap -> heap.read_bitmap(bitmap) -> RawRecord
```

---

## 9. Residual y proyección

### 9.1. FilterPlan

Evalúa exclusivamente el residual sobre registros:

```text
record cumple node.predicate -> evaluate_expression(record, predicate, table_meta)
```

`FilterPlan` solo existe cuando un access path previo resolvió parte del
WHERE. Nunca representa el WHERE completo.

### 9.2. FilterRidsPlan

```text
RID[] o Bitmap
    -> heap.read_many_with_rids(rids)
    -> evaluate_expression(record, predicate, table_meta)
    -> RID[] supervivientes
```

El registro se usa internamente; el resultado lógico son RIDs.

### 9.3. ProjectPlan

```text
RawRecord
    -> clean_tuple
    -> seleccion de columnas
    -> CleanRecord
```

No realiza E/S.

### 9.4. EmptyResultPlan

Devuelve un resultado vacío sin abrir Heap ni índices.

---

## 10. INSERT

### 10.1. SequentialFile

```text
record = table_meta.normalize_tuple(values)
storage.insert(record)
```

`SequentialFile` comprueba las columnas únicas definidas en su metadata. El
Executor no actualiza índices ni expone RIDs.

### 10.2. HeapFile

```text
1. normalizar el registro
2. prevalidar UNIQUE: buscar la clave en cada indice unico
3. heap.insert(record) -> rid
4. insertar (key, rid) en todos los indices de la tabla
5. si algun insert devuelve False (clave duplicada), deshacer
```

`BPlusTree.insert` devuelve `False` cuando la clave ya existe en una columna
única; no lanza excepción. El Executor comprueba el valor de retorno.

No se utiliza `insert_record`, porque mezcla Heap e índice y el Executor
necesita controlar todos los índices de forma atómica.

---

## 11. DELETE

### 11.1. HeapFile

```text
1. child -> RIDs finales
2. read_many_with_rids(rids) -> (RID, RawRecord)
3. eliminar (key, rid) de TODOS los indices de la tabla
4. snapshot fisico del Heap
5. heap.remove_many(rids)
6. commit: descartar snapshot
```

Rollback:

```text
fallo al eliminar indices
    -> reinsertar las entradas ya eliminadas
    -> el Heap no fue modificado

fallo en remove_many
    -> restaurar el snapshot fisico del Heap
    -> reinsertar las entradas de indice eliminadas
```

El child puede ser `SeqScanRidsPlan`, `IndexScanPlan`, `IndexRangeScanPlan`,
`BitmapAndPlan`, `BitmapOrPlan` o `FilterRidsPlan`. Si es `EmptyResultPlan` no
se modifica nada y `affected_rows` es 0.

### 11.2. SequentialFile

```text
predicate != None -> storage.delete_all(predicate)
predicate = None  -> iter_records() + storage.reset()
```

Se guarda un snapshot físico antes de la operación; ante un fallo se restaura
el archivo y se reabre la estructura. No se usan RIDs ni `DeletePlan`,
`SeqScanRidsPlan` o `FilterRidsPlan`.

---

## 12. DDL

### 12.1. CREATE TABLE

Prevalidaciones físicas:

```text
la ruta del archivo no debe existir
```

Tabla con `SequentialFile`:

```text
sin CSV -> SequentialFile(metadata, file_path, page_size)
con CSV -> SequentialFile(...) + rebulk_load(csv, delimiter)
```

Tabla con `HeapFile`:

```text
con CSV y columnas unicas
    -> HeapFile vacio + BPlusTree.bulk_load_multiple(...)

con CSV y sin columnas unicas
    -> HeapFile vacio + rebulk_load(csv, delimiter)

sin CSV
    -> HeapFile vacio
       + BPlusTree vacio por columna unica (build_from_storage)
```

Los índices automáticos corresponden a PRIMARY KEY y UNIQUE, sin duplicados.
Su nombre lógico canónico es `f"{tabla}_{columna}_bplustree"` y su ruta se
obtiene con `Catalog.index_path(tabla, columna, BTREE)`.

Orden de registro:

```text
1. construir las estructuras fisicas
2. catalog.add_table(entry)
3. catalog.add_index(index_metadata) por cada indice
4. almacenar en las caches
```

Rollback: cerrar estructuras, eliminar archivos creados y revertir las
entradas del catálogo que se hayan registrado.

### 12.2. CREATE INDEX

```text
HASH                       -> UnsupportedIndexError
BTREE ya existente en la
columna                     -> ExecutorError mencionando el indice actual
archivo destino existente   -> ExecutorError
```

Si las validaciones pasan:

```text
BPlusTree.build_from_storage(...)
    -> catalog.add_index(...)
    -> index_cache
```

`CREATE INDEX` no reutiliza, no renombra y no reemplaza índices existentes.

### 12.3. ALTER TABLE ADD UNIQUE

Solo para `HeapFile`.

```text
si la columna ya es UNIQUE            -> lo garantiza el SemanticVisitor
si ya existe un BTREE en la columna   -> ExecutorError
```

Proceso:

```text
1. metadata temporal con unique=True
2. BPlusTree.build_from_storage(...)
       (falla si hay duplicados)
3. catalog.add_index(...)
4. catalog.update_table(entry con unique=True)
```

Ante un fallo se cierra el índice, se elimina el archivo y se revierte la
metadata.

---

## 13. CREATE TABLE FROM CSV sin columnas

La inferencia del esquema se realiza en el Executor:

```text
1. La primera fila del CSV es obligatoriamente la cabecera.
   Si no existe, error.
2. La PRIMARY KEY indicada en la sentencia es obligatoria y su nombre
   debe coincidir exactamente con una columna de la cabecera.
3. Tipos por columna:
       todos los valores enteros     -> INT
       todos los valores numericos   -> FLOAT
       caso contrario                -> CHAR(n),
                                        n = longitud maxima en bytes
4. Con el TableMetadata inferido:
       HeapFile    -> HeapFile + BPlusTree.bulk_load_multiple([PRIMARY KEY])
       Sequential  -> SequentialFile + rebulk_load
5. PRIMARY KEY duplicada -> error y rollback total
   (sin tabla en el catalogo y sin archivos)
```

No se utiliza el scanner SQL para la inferencia.

---

## 14. StatementInfo y resultados

El Planner adjunta `StatementInfo` a cada sentencia del programa:

```python
@dataclass(frozen=True)
class StatementInfo:
    statement_id: int
    statement_type: str
    table: str | None
    select_columns: list[str] | None
```

`select_columns = None` significa `SELECT *`. Esta información permite
construir el resultado cuando el árbol no contiene la tabla o las columnas,
por ejemplo con `EmptyResultPlan`.

### 14.1. StatementResult

```python
@dataclass
class StatementResult:
    statement_id: int
    statement_type: str
    columns: list[str]
    rows: list[CleanRecord]
    affected_rows: int | None
    message: str | None = None
```

```text
SELECT  -> rows = resultado, affected_rows = None
INSERT  -> rows = [], affected_rows = 1
DELETE  -> rows = [], affected_rows = cantidad eliminada
DDL     -> rows = [], affected_rows = 0
```

### 14.2. StatementExplain y ExecutionOutput

```python
@dataclass
class StatementExplain:
    statement_id: int
    statement_type: str
    root: NodeMetrics | None
    success: bool = True
    error: str | None = None
    error_type: str | None = None

@dataclass
class ExecutionOutput:
    explains: list[StatementExplain]
    results: list[StatementResult]
    success: bool = True
    error: str | None = None
    error_type: str | None = None
    failed_statement_id: int | None = None
```

`results` solo contiene las sentencias exitosas; la asociación con la sentencia
original se hace por `statement_id`. Ante un fallo, `explains` incluye la
sentencia fallida con `success=False`, `success=False` y `failed_statement_id`
indica dónde se detuvo el `Program`. `error_type` es el nombre de la clase del
error (por ejemplo `ExecutorError`, `UnsupportedIndexError`, `PlannerError`).

---

## 15. Transacciones y rollback

Cada sentencia mutante tiene su propia transacción.

```text
S1 OK
S2 OK
S3 ERROR
    -> rollback de S3
    -> S1 y S2 permanecen
    -> S4, S5, ... no se ejecutan
```

La `Session` no lanza la excepción: devuelve un `ExecutionOutput` parcial con
`explains` de S1, S2 y S3 (esta última con `success=False`), `results` de S1 y
S2, `success=False`, `error`, `error_type` y `failed_statement_id=3`.

```python
@dataclass
class UndoAction:
    description: str
    undo: Callable[[], None]

class RollbackJournal:
    def add(self, description, undo) -> None: ...
    def rollback(self) -> None: ...
    def commit(self) -> None: ...
```

El rollback ejecuta las acciones en orden inverso. Para DELETE se combinan
acciones de undo en memoria con snapshots físicos de archivo.

---

## 16. Cierre de recursos

```text
close()
    close_all_indexes()
    close_all_storages()
```

`execute` lo invoca siempre en su bloque `finally`. En una `Session` se invoca
una sola vez al terminar el `Program` completo. Los índices no cierran el
`HeapFile` compartido. El cierre es idempotente.

---

## 17. Errores

```text
ExecutorError
    reglas fisicas: ruta ocupada, BTREE existente, PK/UNIQUE duplicadas,
    archivo de indice existente

UnsupportedIndexError (ExecutorError)
    indices HASH no soportados

RollbackError (RuntimeError)
    fallo al revertir una sentencia
```

En la `Session` estos errores no escapan: `execute_statement` los captura por
sentencia y los reporta en el `ExecutionOutput` con `error`, `error_type` y
`failed_statement_id`. `Executor.execute` los propaga como salida parcial.

---

## 18. rows_out por nodo

| Nodo | rows_out |
|---|---|
| `ProgramPlan` | sentencias ejecutadas |
| `CreateTablePlan` / `CreateIndexPlan` / `AddUniquePlan` | 0 |
| `InsertPlan` | 1 |
| `SeqScanPlan` | registros producidos |
| `SeqScanRidsPlan` | RIDs producidos |
| `SequentialRangeScanPlan` | registros producidos |
| `SequentialDeletePlan` | registros eliminados |
| `IndexScanPlan` / `IndexRangeScanPlan` | RIDs encontrados |
| `BitmapIndexScanPlan` / `BitmapIndexRangeScanPlan` | bits marcados |
| `BitmapAndPlan` / `BitmapOrPlan` | cardinalidad final |
| `HeapFetchPlan` / `BitmapHeapScanPlan` | registros recuperados |
| `FilterPlan` | registros supervivientes |
| `FilterRidsPlan` | RIDs supervivientes |
| `EmptyResultPlan` | 0 |
| `ProjectPlan` | filas proyectadas |
| `DeletePlan` | registros eliminados |

---

## 19. Ejemplos de referencia

### 19.1. SELECT Heap con índice y residual

```sql
SELECT name
FROM users
WHERE age = 20
  AND city = 'Lima';
```

```text
ProjectPlan(name)
└── FilterPlan(city = 'Lima')
    └── HeapFetchPlan
        └── IndexScanPlan(idx_age, 20)
```

### 19.2. SELECT Heap sin índice

```sql
SELECT name
FROM users
WHERE age = 20
  AND city = 'Lima';
```

```text
ProjectPlan(name)
└── SeqScanPlan(predicate = edad = 20 AND city = 'Lima')
```

### 19.3. SELECT Sequential con rango de PRIMARY KEY

```sql
SELECT name
FROM users
WHERE id >= 20
  AND id < 30;
```

```text
ProjectPlan(name)
└── SequentialRangeScanPlan(id, [20, 30))
```

### 19.4. SELECT Sequential con rango de PRIMARY KEY y residual

```sql
SELECT name
FROM users
WHERE id >= 20
  AND id < 30
  AND city = 'Lima';
```

```text
ProjectPlan(name)
└── FilterPlan(city = 'Lima')
    └── SequentialRangeScanPlan(id, [20, 30))
```

### 19.5. DELETE Heap

```sql
DELETE FROM users
WHERE age = 20
  AND city = 'Lima';
```

```text
DeletePlan
└── FilterRidsPlan(city = 'Lima')
    └── IndexScanPlan(age = 20)
```

### 19.6. DELETE Sequential

```sql
DELETE FROM users
WHERE city = 'Lima';
```

```text
SequentialDeletePlan(predicate = city = 'Lima')
```

---

## 20. Restricciones de implementación

```text
El Executor NO abre almacenamiento para EmptyResultPlan
El Executor NO usa RIDs en accesos SequentialFile
El Executor NO usa insert_record ni delete_record de BPlusTree
El Executor NO revalida reglas del Parser, SemanticVisitor ni Planner
El Executor NO introduce costos, estadisticas ni candidatos
El Executor NO modifica el plan ni crea PlanNode nuevos
```

El Executor únicamente:

```text
abre estructuras fisicas de forma perezosa
ejecuta los PlanNode
resuelve RawRecord y CleanRecord
mantiene Heap e indices consistentes
registra metricas de tiempo y E/S
aplica rollback por sentencia
cierra todos los recursos
```

---

## 21. Session

La `Session` es el punto de entrada para ejecutar un `Program` completo con DDL
y DML mezclados.

```python
class Session:
    def __init__(self, catalog, page_size,
                 sort_buffer_size=128 * 1024 * 1024,
                 level_memory_limit=64 * 1024 * 1024) -> None: ...
    def execute(self, program: Program) -> ExecutionOutput: ...
```

Flujo:

```text
1. SemanticVisitor(catalog).visit_program(program)
       valida todo el Program antes de ejecutar nada
2. ExecutionContext y Executor compartidos
3. por cada sentencia, con id global 1..n:
       PlannerVisitor(catalog).visit_program(Program([sentencia]))
       Executor.execute_statement(statement, info)
4. close() en finally
```

La clave es que el Planner se invoca después de que el Executor haya
registrado las sentencias anteriores. Por eso una sentencia posterior puede
usar tablas creadas desde CSV, columnas inferidas, autoíndices de PRIMARY KEY
y UNIQUE, e índices declarados en el mismo `Program`.

Semántica de errores (la `Session` nunca lanza):

```text
error semántico en la sentencia S
    -> no se ejecuta ninguna
    -> success=False, failed_statement_id=S, explains=[], results=[]

error de planificación o ejecución en S3
    -> rollback de S3
    -> S1 y S2 permanecen
    -> S4, S5, ... no se ejecutan
    -> success=False, error, error_type, failed_statement_id=3
    -> explains = [S1, S2, S3], results = [S1, S2]
```

`Executor.execute(program)` sigue disponible como API batch: planifica todo el
`Program` antes de ejecutar y solo es válido si la metadata de las tablas e
índices referenciados ya está registrada en el catálogo. Ante un fallo devuelve
también una salida parcial (`success=False`).

Interfaz interna del Executor:

```text
execute_statement(statement, info) -> (StatementExplain, StatementResult | None)
    abre RollbackJournal, ejecuta, hace commit o rollback; None si falla
execute(program) -> ExecutionOutput
    recorre el Program, corta en el primer fallo y cierra los recursos
```
