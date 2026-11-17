# Especificación del Planner

## 1. Objetivo

El Planner transforma un AST validado semánticamente en un árbol de ejecución compuesto por `PlanNode`.

La planificación es **determinista** y está controlada por las cláusulas `FORCE`. No se utilizan costos, estadísticas, candidatos ni selección basada en estimaciones.

Flujo general:

```text
Program (AST)
 │
 ▼
SemanticVisitor
 │
 ▼
Session (por sentencia)
 │
 ▼
PlannerVisitor
 │
 ├── AST de expresión → conditions.Expression
 ├── análisis del WHERE
 ├── construcción del access path
 ├── aplicación de FORCE
 └── construcción del plan final
 │
 ▼
PlanNode
 │
 ▼
Executor
```

El Planner **no abre** `HeapFile`, `SequentialFile`, B+ Tree ni Hash. El catálogo aporta únicamente metadata y la apertura de estructuras físicas corresponde al Executor.

La planificación de un `Program` con DDL se realiza sentencia por sentencia a través de `Session` (ver `executor_specifications.md`), de modo que cada plan se construye contra el catálogo ya actualizado por las sentencias anteriores. El uso batch de `PlannerVisitor` (`visit_program` con todo el `Program`) requiere que la metadata de las tablas e índices referenciados ya esté registrada en el catálogo. Un `PlannerError` de una sentencia no relanza: la `Session` lo captura y lo reporta en el `ExecutionOutput` con `success=False`, `error_type='PlannerError'` y `failed_statement_id`.

---

## 2. Arquitectura

Debe existir un único visitor:

```python
PlannerExpression = Operand | Expression

class PlannerVisitor(Visitor[PlanNode, PlannerExpression]):
    ...
```

La separación de tipos es:

```text
Statement
    → PlanNode

Expression AST
    → Operand | Expression
```

No se debe introducir un `ConditionVisitor`.

### AST relevante para SELECT y DELETE

Las sentencias `SELECT` y `DELETE` contienen la información necesaria para controlar el access path mediante `FORCE`.

La forma conceptual de `SelectStm` es:

```python
@dataclass
class SelectStm(Stm):
    columns: list[str] | None
    table: str
    force_indexes: list[str] | None = None
    force_seqscan: bool = False
    force_all_indexes: bool = False
    where: Exp | None = None
```

La forma conceptual de `DeleteStm` es:

```python
@dataclass
class DeleteStm(Stm):
    table: str
    force_indexes: list[str] | None = None
    force_seqscan: bool = False
    force_all_indexes: bool = False
    where: Exp | None = None
```

`FORCE` se aplica antes de `WHERE` en la gramática de ambas sentencias. En particular, `DELETE` sigue la forma:

```text
DELETE FROM ID [ForceClause] [WhereClause] ";"
```

El Planner recibe estos campos ya separados por el AST y no necesita volver a interpretar la sintaxis SQL.

### Conversión de expresiones

```text
IdExp("age")
    → ColumnRef("age")

LiteralExp(20)
    → LiteralValue(20)

age >= 20
    → Condition(
           ColumnRef("age"),
           ComparisonOperator.GE,
           LiteralValue(20)
       )

A AND B
    → LogicalExpression(
           LogicalOperator.AND,
           (A, B)
       )

A OR B
    → LogicalExpression(
           LogicalOperator.OR,
           (A, B)
       )

NOT A
    → LogicalExpression(
           LogicalOperator.NOT,
           (A,)
       )
```

---

## 3. Normalización de condiciones

La representación interna de una condición debe ser canónica cuando una comparación tiene una columna y un literal.

```text
columna operador literal
```

Por tanto:

```text
20 = age
```

se normaliza a:

```text
age = 20
```

y:

```text
20 < age
```

se normaliza a:

```text
age > 20
```

El operador debe invertirse correctamente:

```text
20 = age   → age = 20
20 != age  → age != 20
20 < age   → age > 20
20 <= age  → age >= 20
20 > age   → age < 20
20 >= age  → age <= 20
```

La normalización se realiza para simplificar la construcción del access path. `conditions.py` puede evaluar ambas orientaciones, pero el Planner debe trabajar con una única forma canónica.

---

## 4. AccessResult

La construcción de acceso utiliza una estructura auxiliar:

```python
@dataclass
class AccessResult:
    plan: PlanNode
    used_indexes: set[str]
    residual: Expression | None
```

Representa:

```text
plan
    → access path construido

used_indexes
    → índices que realmente participan

residual
    → parte del WHERE que todavía debe evaluarse sobre
      los registros recuperados
```

`AccessResult` no representa candidatos ni alternativas de optimización.

Es una estructura **auxiliar e intermedia del Planner**. No forma parte del árbol final de ejecución. Una vez construido el plan final, `residual` se transforma en `FilterPlan` para `SELECT` o en `FilterRidsPlan` para `DELETE`.

---

## 5. Simplificación y propagación de constantes

Antes de construir el access path, el Planner debe simplificar las expresiones que puedan evaluarse completamente durante la planificación.

## 5.1. Expresiones constantes

Cuando ambos operandos de una comparación son literales, la condición puede evaluarse inmediatamente.

Ejemplo:

```text
10 = 10
    → TRUE

10 = 20
    → FALSE
```

Lo mismo aplica a los demás operadores de comparación cuando ambos lados son constantes.

Estas expresiones no deben generar un access path indexado porque no dependen de ninguna columna de la tabla.

## 5.2. Propagación lógica

Los valores constantes deben propagarse por las expresiones lógicas para simplificar el `WHERE` completo.

Reglas mínimas:

```text
TRUE AND A   → A
FALSE AND A  → FALSE
TRUE OR A    → TRUE
FALSE OR A   → A
NOT TRUE     → FALSE
NOT FALSE    → TRUE

NOT NOT A    → A

A AND A      → A
A OR A       → A
```

Contradicciones seguras sobre una misma columna:

```text
a = X AND a = Y   → FALSE   si X != Y
a = X AND a > Y   → FALSE   si X <= Y
a = X AND a < Y   → FALSE   si X >= Y
a > X AND a < Y   → FALSE   si la interseccion del rango es vacia
```

La simplificación debe aplicarse de forma recursiva, durante la misma
recursión que construye la `Expression`. No existe un segundo recorrido
de simplificación sobre el árbol ya construido.

Ejemplo:

```text
(age > 18) AND (10 = 20)
    → (age > 18) AND FALSE
    → FALSE
```

y:

```text
(age > 18) OR (10 = 20)
    → (age > 18) OR FALSE
    → age > 18
```

Esto se realiza antes de decidir el access path, porque una expresión simplificada puede eliminar condiciones completas del `WHERE`.

## 5.3. WHERE completamente TRUE

Si el `WHERE` se reduce a `TRUE`, no queda ninguna condición restrictiva sobre columnas.

Por tanto:

```text
WHERE TRUE
    → no se necesita FilterPlan
```

El Planner construye el acceso normal a la tabla. En una tabla con `HeapFile`, esto puede ser `SeqScanPlan(predicate=None)` cuando no existe otra condición indexable. En una tabla con `SequentialFile`, el WHERE completo va en `SeqScanPlan(predicate=WHERE)` o, si hay rango de PRIMARY KEY, en `SequentialRangeScanPlan`.

`FORCE INDEX(*)` no debe inventar el uso de un índice cuando no existe ninguna condición indexable a la cual aplicar el índice.

## 5.4. WHERE completamente FALSE

Si el `WHERE` se reduce a `FALSE`, el resultado es conocido durante la planificación y **no debe realizarse ningún acceso a la tabla ni a sus índices**.

Para representar este caso se agrega:

```python
@dataclass
class EmptyResultPlan(PlanNode):
    """Plan cuyo resultado es conocido como conjunto vacio."""

    pass
```

Este nodo representa un resultado vacío sin acceder al almacenamiento físico.

Para `SELECT`:

```text
Project
└── EmptyResultPlan
```

Para `DELETE`:

```text
DeletePlan
└── EmptyResultPlan
```

`EmptyResultPlan` no lee la tabla, no abre índices y no elimina registros.

La reducción a `FALSE` se resuelve antes de construir el access path porque ya se conoce que el resultado será vacío. No existe entonces una condición indexable pendiente que deba utilizarse.

---

# 6. Tipos de acceso

### 6.1. SeqScan

```python
SeqScanPlan(table, predicate=None)
```

Recorre secuencialmente el almacenamiento y produce registros.

```text
predicate = None        -> devuelve todos los registros
predicate = Expression   -> evalua esa expresion completa durante el scan
```

Cuando ninguna condicion puede resolverse mediante un access path, el WHERE
completo va dentro de `predicate`; NO se agrega `FilterPlan`.

### 6.2. SeqScanRids

```python
SeqScanRidsPlan(table)
```

Realiza un recorrido secuencial de la tabla y produce RIDs.

Se utiliza como access path de `DELETE` sobre `HeapFile`.

```text
SeqScanRidsPlan
    → RIDs
```

Este nodo no representa una operación de borrado; `DeletePlan` sigue siendo responsable de eliminar los RIDs finales.

### 6.2.1. SequentialRangeScan

```python
SequentialRangeScanPlan(table, column, range)
```

Usa `SequentialFile.range_search(...)` y produce registros.

Solo se utiliza cuando la tabla utiliza `SequentialFile`, la condicion es un
rango y la columna es la PRIMARY KEY. No produce RIDs.

### 6.2.2. SequentialDelete

```python
SequentialDeletePlan(table, predicate)
```

Elimina directamente mediante `SequentialFile.delete_all(predicate)`.

No usa RIDs ni `DeletePlan`, `SeqScanRidsPlan` ni `FilterRidsPlan`.


### 6.4. IndexScan

Para igualdad:

```python
IndexScanPlan(
    table=...,
    index=...,
    index_kind=...,
    key=...
)
```

Produce RIDs.

### 6.3. IndexRangeScan

Para rangos BTREE:

```python
IndexRangeScanPlan(
    table=...,
    index=...,
    range=RangeSpec(...)
)
```

Produce RIDs.

### 6.5. BitmapIndexScan

Para construir un bitmap a partir de una igualdad:

```python
BitmapIndexScanPlan(...)
```

Produce un `Bitmap` de RIDs.

### 6.6. BitmapIndexRangeScan

Para construir un bitmap a partir de un rango:

```python
BitmapIndexRangeScanPlan(...)
```

Produce un `Bitmap` de RIDs.

### 6.7. BitmapAnd

Intersección de access paths:

```text
BitmapAnd
├── acceso 1
└── acceso 2
```

### 6.8. BitmapOr

Unión de access paths:

```text
BitmapOr
├── acceso 1
└── acceso 2
```

### 6.9. HeapFetch

Convierte RIDs en registros:

```text
HeapFetch
└── IndexScan
```

### 6.10. BitmapHeapScan

Convierte un bitmap de RIDs en registros:

```text
BitmapHeapScan
└── BitmapAnd / BitmapOr / BitmapIndexScan / ...
```

### 6.11. Filter

Evalúa exclusivamente un residual, es decir, la parte del WHERE que un access
path previo no resolvió. Nunca representa el WHERE completo cuando no hubo
access path:

```text
Filter
└── HeapFetch / BitmapHeapScan / SequentialRangeScanPlan
```

Cuando no existe ningun access path, el WHERE completo va dentro de
`SeqScanPlan(predicate=WHERE)`, no en un `Filter`.

### 6.12. FilterRids

Filtra candidatos identificados por RID. Recibe RIDs o un bitmap de RIDs, lee los registros asociados para evaluar el residual y devuelve únicamente los RIDs que cumplen.

```text
FilterRidsPlan
└── IndexScan / IndexRangeScan / BitmapAnd / BitmapOr / ...
```

El registro (`RawRecord`) se utiliza internamente para evaluar la condición, pero no forma parte del resultado lógico del nodo.

### 6.13. EmptyResult

Representa un resultado vacío conocido durante la planificación.

```text
EmptyResultPlan
```

No accede a la tabla ni a los índices.

### 6.14. Project

Selecciona las columnas del `SELECT`:

```text
Project
└── ...
```

No se utiliza para `SELECT *`.

---

# 7. Reglas de indexación

## 7.1. Operadores soportados

| Operador | HASH | BTREE |
|---|---:|---:|
| `=` | Sí | Sí |
| `<` | No | Sí |
| `<=` | No | Sí |
| `>` | No | Sí |
| `>=` | No | Sí |
| `!=` / `<>` | No | No |

Además:

```text
columna = columna
    → no indexable

literal = literal
    → no indexable
```

No se deben construir access paths indexados para esas condiciones.

---

## 7.2. Preferencia HASH en igualdad

Cuando una misma condición de igualdad puede ser resuelta tanto por HASH como por BTREE:

```text
a = 10

idx_a_hash   → aplicable
idx_a_btree  → aplicable
```

se debe preferir:

```text
idx_a_hash
```

No se deben ejecutar ambos para resolver la misma igualdad.

Resultado:

```text
IndexScan(idx_a_hash, 10)
```

y no:

```text
BitmapAnd
├── idx_a_hash
└── idx_a_btree
```

Esta preferencia es determinista y no depende de un modelo de costos.

---

## 7.3. Índices sobre la misma columna con condiciones distintas

Dos índices sobre la misma columna sí pueden participar cuando resuelven **condiciones diferentes**.

Ejemplo:

```sql
WHERE a = 10
  AND a >= 5
```

con:

```text
idx_a_hash
idx_a_btree
```

puede construirse:

```text
BitmapAnd
├── BitmapIndexScan(idx_a_hash, a = 10)
└── BitmapIndexRangeScan(idx_a_btree, a >= 5)
```

Aquí:

```text
HASH
→ resuelve a = 10

BTREE
→ resuelve a >= 5
```

Los índices no están duplicando exactamente la misma operación.

---

# 8. Rangos

## 8.1. Un solo límite

```text
a >= 20
```

con BTREE:

```text
IndexRangeScan
└── RangeSpec(
        lower=20,
        upper=None,
        include_lower=True,
        include_upper=True
    )
```

```text
a > 20
```

produce:

```text
RangeSpec(
    lower=20,
    upper=None,
    include_lower=False,
    include_upper=True
)
```

Análogamente:

```text
a <= 50
a < 50
```

usan el límite superior.

---

## 8.2. Dos límites sobre la misma columna

Las condiciones:

```text
a >= 10
AND
a <= 20
```

se deben fusionar en un único rango:

```text
IndexRangeScan
└── RangeSpec(
        lower=10,
        upper=20,
        include_lower=True,
        include_upper=True
    )
```

No se debe generar:

```text
BitmapAnd
├── a >= 10
└── a <= 20
```

cuando ambas condiciones pertenecen a la misma columna y el BTREE puede representarlas como un único rango.

---

# 9. AND

`AND` permite separar condiciones indexables y residuales.

Para:

```text
A AND B AND C
```

donde:

```text
A → indexable
B → indexable
C → no indexable
```

se construye:

```text
BitmapAnd
├── A
└── B
```

y:

```text
residual = C
```

### Ejemplo

```sql
WHERE age = 20
  AND city = 'Lima'
```

si solo existe índice sobre `age`:

```text
Filter(city = 'Lima')
└── HeapFetch
    └── IndexScan(idx_age, 20)
```

La condición no indexable no invalida el uso de un índice dentro de un `AND`.

---

# 10. OR

`OR` solo puede convertirse en un acceso indexado si **todas sus ramas necesarias son completamente indexables**.

Para:

```text
A OR B
```

si:

```text
A → access indexado completo
B → access indexado completo
```

se permite:

```text
BitmapOr
├── A
└── B
```

### Ejemplo

```sql
WHERE departamento = 'IT'
   OR departamento = 'HR'
```

con índice sobre `departamento`:

```text
BitmapOr
├── BitmapIndexScan(idx_departamento, 'IT')
└── BitmapIndexScan(idx_departamento, 'HR')
```

---

## 10.1. OR parcialmente indexable

Si:

```text
A → indexable
B → no indexable
```

no se debe construir:

```text
BitmapOr
└── A
```

porque eso cambia la semántica de la consulta.

El OR completo queda como residual.

Sin otro acceso indexado exterior:

```text
SeqScan
```

Con otro predicado indexable mediante `AND`, ese predicado exterior sí puede utilizarse.

### Ejemplo

```sql
WHERE c = 30
  AND (a = 10 OR b = 20)
```

con solo `idx_c`:

```text
Filter(
    (a = 10 OR b = 20)
)
└── HeapFetch
    └── IndexScan(idx_c, 30)
```

El OR permanece completo como residual.

---

# 11. NOT

`NOT` no se empuja hacia el índice.

Ejemplo:

```text
NOT a = 20
```

no produce:

```text
IndexScan(a != 20)
```

ni un rango equivalente.

Se conserva como residual:

```text
NOT(a = 20)
```

`NOT` no se empuja nunca al índice. La única simplificación permitida es
`NOT NOT A → A` (sección 5.2): una vez eliminada la doble negación, `A`
puede indexarse porque ya no hay ningún `NOT` en la expresión. Esto no es
pushdown de `NOT`.

Sin embargo, otras condiciones indexables fuera del `NOT` sí pueden utilizarse.

### Ejemplo

```sql
WHERE b = 10
  AND NOT a = 20
```

con solo `idx_a`:

```text
idx_a
→ no es utilizable
```

porque el índice está debajo de `NOT`.

Por tanto:

```text
FORCE INDEX(*)
→ PlannerError
```

porque no existe ningún índice utilizable.

Sin `FORCE`:

```text
SeqScan
```

---

# 12. Reglas de FORCE

Una tabla con `SequentialFile` no admite ninguna clausula FORCE. Esa
prohibicion se valida en el analisis semantico, antes del Planner.

## 12.1. FORCE SEQSCAN

Siempre:

```text
FORCE SEQSCAN
    → SeqScan
```

Los índices existentes se ignoran.

No existe un `ForceSeqScanPlan`.

---

## 12.2. FORCE INDEX(idx1, idx2, ...)

Solo los índices indicados pueden participar.

Reglas:

```text
cada índice especificado debe ser utilizable
todos los índices especificados deben participar
otras condiciones pueden quedar como residual
no existe fallback a SeqScan
```

Si alguno no puede utilizarse:

```text
→ PlannerError
```

### Ejemplo

```sql
WHERE a = 10
```

con:

```text
idx_a_hash
idx_a_btree
```

y:

```sql
FORCE INDEX(idx_a_hash)
```

se utiliza:

```text
idx_a_hash
```

---

### Ejemplo de índice no utilizable

```sql
WHERE a >= 10
```

con:

```text
idx_a_hash
idx_a_btree
```

y:

```sql
FORCE INDEX(idx_a_hash)
```

produce:

```text
PlannerError
```

porque HASH no soporta `>=`.

Aunque `idx_a_btree` sí pudiera utilizarse, no puede reemplazar al índice explícitamente forzado.

---

## 12.3. Índices forzados redundantes

Si dos índices forzados sobre la misma columna resuelven exactamente la misma condición y uno domina al otro para ese operador, no existe un uso independiente correcto de ambos.

La redundancia se evalúa globalmente sobre el uso de los índices en el
árbol lógico completo, no por condición aislada. Dos índices HASH/BTREE
sobre la misma columna pueden participar cuando resuelven condiciones
distintas, aunque estén en ramas separadas de un `OR` (sección 12.4).
Solo hay redundancia cuando no existe ninguna asignación posible de los
índices forzados a condiciones diferentes.

Resultado:

```text
→ PlannerError
```

### Ejemplo

```sql
WHERE a = 10
```

con:

```text
idx_a_hash
idx_a_btree
```

y:

```sql
FORCE INDEX(idx_a_hash, idx_a_btree)
```

produce:

```text
PlannerError
```

porque:

```text
HASH → a = 10
BTREE → a = 10
```

resuelven la misma igualdad y HASH domina a BTREE.

No se debe fabricar:

```text
BitmapAnd
├── HASH
└── BTREE
```

solo para aparentar que ambos participaron.

---

## 12.4. Índices forzados sobre condiciones diferentes

Sí pueden participar ambos cuando resuelven restricciones diferentes.

### Ejemplo

```sql
WHERE a = 10
  AND a >= 5
```

con:

```text
idx_a_hash
idx_a_btree
```

y:

```sql
FORCE INDEX(idx_a_hash, idx_a_btree)
```

es válido:

```text
BitmapAnd
├── BitmapIndexScan(idx_a_hash, 10)
└── BitmapIndexRangeScan(idx_a_btree, [5, +∞))
```

porque cada índice está participando en una condición diferente.

---

# 13. FORCE INDEX(*)

`FORCE INDEX(*)` exige utilizar **todos los índices que sean aplicables al `WHERE`**, siempre que su uso sea correcto y no exista una regla explícita que los descarte por dominancia, redundancia o una representación equivalente del mismo acceso.

En particular:

```text
si un índice es aplicable a una condición del WHERE
    → debe participar en el access path
```

La excepción principal es la preferencia por HASH en igualdad: si HASH y BTREE pueden resolver exactamente la misma condición `a = 10`, HASH domina a BTREE y se utiliza solo HASH para esa condición.

Reglas:

```text
usar todos los índices aplicables al WHERE
usar cada índice cuando su participación sea correcta
preferir HASH sobre BTREE para igualdad cuando ambos resuelven la misma condición
no usar índices redundantes para la misma condición
las condiciones no indexables pueden quedar como residual
no es necesario usar todos los índices físicos de la tabla
si ningún índice puede utilizarse → PlannerError
```

`FORCE INDEX(*)` no significa:

```text
usar físicamente todos los índices existentes en la tabla
```

Significa:

```text
usar todos los índices de la tabla que sean aplicables al WHERE
salvo los que una regla explícita descarte por dominancia,
redundancia o representación equivalente del mismo acceso.
```

---

## 13.1. Ejemplo: igualdad con HASH y BTREE

```sql
WHERE a = 10
```

con:

```text
idx_a_hash
idx_a_btree
```

resultado:

```text
IndexScan(idx_a_hash, 10)
```

No se utiliza BTREE en paralelo para la misma igualdad.

---

## 13.2. Ejemplo: condición diferente

```sql
WHERE a = 10
  AND a >= 5
```

con:

```text
idx_a_hash
idx_a_btree
```

resultado posible:

```text
BitmapAnd
├── HASH(a = 10)
└── BTREE(a >= 5)
```

---

## 13.3. Ejemplo con residual

```sql
WHERE c = 30
  AND (a = 10 OR b = 20)
```

con:

```text
idx_c
```

resultado:

```text
Filter(
    a = 10 OR b = 20
)
└── HeapFetch
    └── IndexScan(idx_c, 30)
```

`FORCE INDEX(*)` funciona porque existe al menos un índice utilizable.

---

## 13.4. Ejemplo sin ningún índice utilizable

```sql
WHERE b = 10
  AND NOT a = 20
```

con únicamente:

```text
idx_a
```

resultado:

```text
PlannerError
```

porque:

```text
idx_a
→ está debajo de NOT
→ no puede utilizarse
```

y no existe otro índice aplicable.

---

# 14. Sin FORCE

La construcción indexada inicial debe seguir las mismas reglas de aplicabilidad que `FORCE INDEX(*)`.

Por tanto, cuando un índice sea aplicable y no exista una excepción de dominancia, redundancia o representación equivalente, debe utilizarse también sin `FORCE`.

Reglas:

```text
intentar access path indexado
usar todos los índices aplicables
preferir HASH frente a BTREE para la misma igualdad
combinar índices aplicables sobre condiciones diferentes
permitir residual
si ningún índice puede utilizarse → SeqScan
```

La diferencia con `FORCE INDEX(*)` es solamente el comportamiento ante la ausencia total de un índice utilizable:

```text
FORCE INDEX(*)
    → PlannerError

SIN FORCE
    → SeqScan
```

---

# 15. SELECT

## 15.1. Sin índice

```sql
SELECT id, nombre
FROM empleados
WHERE nombre = 'Ada';
```

sin índice sobre `nombre`:

```text
Project(id, nombre)
└── Filter(nombre = 'Ada')
    └── SeqScan(empleados)
```

---

## 15.2. Igualdad indexada

```sql
SELECT id, nombre
FROM empleados
WHERE id = 25;
```

con `idx_id`:

```text
Project(id, nombre)
└── HeapFetch
    └── IndexScan(idx_id, 25)
```

---

## 15.3. Índice + residual

```sql
SELECT *
FROM empleados
WHERE age = 20
  AND city = 'Lima';
```

con índice solo sobre `age`:

```text
Filter(city = 'Lima')
└── HeapFetch
    └── IndexScan(idx_age, 20)
```

El índice localiza candidatos. El residual se evalúa sobre registros completos.

---

## 15.4. Rango

```sql
SELECT *
FROM empleados
WHERE salario >= 5000
  AND salario <= 8000;
```

con `idx_salario` BTREE:

```text
HeapFetch
└── IndexRangeScan
    └── RangeSpec(
            lower=5000,
            upper=8000,
            include_lower=True,
            include_upper=True
        )
```

---

## 15.5. AND de índices

```sql
SELECT *
FROM empleados
WHERE departamento = 'IT'
  AND salario >= 5000;
```

con:

```text
idx_departamento → HASH
idx_salario      → BTREE
```

resultado:

```text
BitmapHeapScan
└── BitmapAnd
    ├── BitmapIndexScan(idx_departamento, 'IT')
    └── BitmapIndexRangeScan(idx_salario, [5000, +∞))
```

---

## 15.6. OR de índices

```sql
SELECT *
FROM empleados
WHERE departamento = 'IT'
   OR departamento = 'HR';
```

resultado:

```text
BitmapHeapScan
└── BitmapOr
    ├── BitmapIndexScan(idx_departamento, 'IT')
    └── BitmapIndexScan(idx_departamento, 'HR')
```

---

# 16. DELETE

`DELETE` reutiliza la misma construcción de access path que `SELECT`, pero mantiene los RIDs hasta la etapa final porque esos RIDs son los que se utilizan para realizar el borrado físico.

El plan debe ser:

```text
DeletePlan
└── access path
```

Si existe una condición residual, esta se representa explícitamente mediante `FilterRidsPlan`:

```text
DeletePlan
└── FilterRidsPlan
    └── access path
```

`FilterRidsPlan` recibe RIDs o un bitmap de RIDs, lee los registros asociados para evaluar el `predicate` y devuelve únicamente los RIDs que cumplen. Internamente utiliza los datos (`RawRecord`) para evaluar la condición, pero su resultado sigue siendo un conjunto de RIDs.

`DeletePlan` recibe los RIDs finales, realiza el borrado físico y devuelve los `RawRecord` que fueron eliminados.

No se debe utilizar un `FilterPlan` para el residual de `DELETE`, porque `FilterPlan` trabaja con registros completos como entrada y salida, mientras que `DELETE` necesita conservar los RIDs.

### Ejemplo simple

```sql
DELETE FROM empleados
WHERE id = 25;
```

resultado:

```text
DeletePlan
└── IndexScan(idx_id, 25)
```

### Ejemplo de DELETE secuencial

Si la tabla utiliza `SequentialFile`, el Planner usa `SequentialDeletePlan` en
una sola operacion; no usa RIDs:

```sql
DELETE FROM empleados
WHERE city = 'Lima';
```

```text
SequentialDeletePlan(empleados, city = 'Lima')
```

`SequentialFile.delete_all(predicate)` realiza el borrado directamente.
No se genera `DeletePlan`, `SeqScanRidsPlan` ni `FilterRidsPlan`.

### Ejemplo con residual

```sql
DELETE FROM empleados
WHERE age = 20
  AND city = 'Lima';
```

con índice sobre `age`:

```text
DeletePlan
└── FilterRidsPlan
    └── IndexScan(idx_age, 20)
```

`FilterRidsPlan` obtiene los registros asociados a los RIDs candidatos, evalúa `city = 'Lima'` y devuelve solo los RIDs que cumplen.

Luego `DeletePlan` elimina esos RIDs y devuelve los registros eliminados.

---

# 17. Construcción final de SELECT

La secuencia debe ser:

```text
WHERE
 ↓
simplificación / propagación de constantes
 ↓
si WHERE == FALSE:
    EmptyResultPlan
 ↓
si no:
    _build_access()
 ↓
AccessResult
 ↓
si el acceso produce RIDs/bitmap:
    HeapFetch / BitmapHeapScan
 ↓
si existe residual:
    FilterPlan
 ↓
si columns != None:
    ProjectPlan
```

Ejemplo:

```text
Project
└── Filter
    └── HeapFetch
        └── BitmapAnd
            ├── index A
            └── index B
```

Para `SELECT *`:

```text
Filter
└── HeapFetch
    └── ...
```

o simplemente:

```text
HeapFetch
└── ...
```

cuando no existe residual.

---

# 18. Construcción final de DELETE

La secuencia lógica es:

```text
WHERE
 ↓
_build_access()
 ↓
AccessResult
 ↓
si existe residual:
    FilterRidsPlan(
        table=...,
        predicate=access.residual,
        child=access.plan
    )
 ↓
DeletePlan(
    table=...,
    child=plan_final
)
```

Si el `WHERE` se reduce a `FALSE`:

```text
DeletePlan
└── EmptyResultPlan
```

Si no existe residual:

```text
DeletePlan
└── access.plan
```

En una tabla `SequentialFile`, para `DELETE`, se usa `SequentialDeletePlan` directamente (nunca `SeqScanRidsPlan`).

Si existe residual:

```text
DeletePlan
└── FilterRidsPlan
    └── access.plan
```

`FilterRidsPlan` usa los registros asociados a los RIDs para evaluar la condición, pero devuelve solo los RIDs supervivientes. `DeletePlan` recibe esos RIDs finales, realiza el borrado y devuelve los `RawRecord` eliminados.

Nunca:

```text
DeletePlan
└── FilterPlan
```

porque `FilterPlan` trabaja con registros completos y `DELETE` necesita conservar los RIDs para borrar físicamente.

---

# 19. Ejemplos de comportamiento obligatorio

## Caso A — igualdad con HASH y BTREE

```text
WHERE a = 10

idx_a_hash
idx_a_btree
```

Resultado:

```text
HASH
```

---

## Caso B — igualdad forzada a ambos índices

```text
WHERE a = 10

FORCE INDEX(idx_a_hash, idx_a_btree)
```

Resultado:

```text
PlannerError
```

Motivo estructural:

```text
ambos resuelven la misma condición
HASH domina a BTREE
no existe una participación independiente correcta del BTREE
```

---

## Caso C — HASH y BTREE para condiciones diferentes

```text
WHERE a = 10
  AND a >= 5

FORCE INDEX(idx_a_hash, idx_a_btree)
```

Resultado:

```text
BitmapAnd
├── HASH → a = 10
└── BTREE → a >= 5
```

---

## Caso D — HASH no puede hacer rango

```text
WHERE a >= 10

FORCE INDEX(idx_a_hash)
```

Resultado:

```text
PlannerError
```

---

## Caso E — `FORCE INDEX(*)` con una condición residual

```text
WHERE c = 30
  AND (a = 10 OR b = 20)

idx_c
```

Resultado:

```text
Filter(a = 10 OR b = 20)
└── HeapFetch
    └── IndexScan(idx_c, 30)
```

No hay error.

---

## Caso F — `FORCE INDEX(*)` y ningún índice utilizable

```text
WHERE b = 10
  AND NOT a = 20

idx_a
```

Resultado:

```text
PlannerError
```

---

## Caso G — sin FORCE para el caso anterior

```text
WHERE b = 10
  AND NOT a = 20

idx_a
```

Resultado:

```text
SeqScan
```

---

## Caso H — igualdad y condición adicional sobre la misma columna

```text
WHERE a = 10
  AND a >= 5

idx_a_hash
idx_a_btree
```

Sin `FORCE`:

```text
BitmapAnd
├── HASH → a = 10
└── BTREE → a >= 5
```

No se ejecutan HASH y BTREE dos veces para la misma igualdad.

---

## Caso I — dos límites de un mismo rango

```text
WHERE a >= 10
  AND a <= 20
```

con BTREE:

```text
IndexRangeScan
└── [10, 20]
```

No se generan dos búsquedas independientes.

---

## Caso J — OR parcialmente indexable

```text
WHERE a = 10
   OR b = 20
```

solo con `idx_a`:

```text
OR completo → residual
```

No:

```text
BitmapOr
└── idx_a
```

porque eso no representa:

```text
(a = 10) OR (b = 20)
```

correctamente.

---

# 20. Reglas de error

El Planner debe lanzar `PlannerError` cuando:

```text
FORCE INDEX(idx1, ...)
    → algún índice especificado no puede participar

FORCE INDEX(idx1, idx2, ...)
    → existen índices redundantes que resuelven exactamente
      la misma condición y uno domina al otro

FORCE INDEX(*)
    → ningún índice puede utilizarse
```

El Planner no debe lanzar `PlannerError` solamente porque:

```text
exista una condición residual
```

Ejemplo válido:

```text
idx_c
+
(a = 10 OR b = 20)
```

si `idx_c` sí puede utilizarse.

---

# 21. Contratos de salida de los PlanNode

```text
SeqScanPlan
    → RawRecords (con predicate opcional)

SeqScanRidsPlan
    → RIDs

SequentialRangeScanPlan
    → RawRecords

SequentialDeletePlan
    → RawRecords eliminados

IndexScanPlan
    → RIDs

IndexRangeScanPlan
    → RIDs

BitmapIndexScanPlan
    → Bitmap/RIDs

BitmapIndexRangeScanPlan
    → Bitmap/RIDs

BitmapAndPlan / BitmapOrPlan
    → Bitmap/RIDs

HeapFetchPlan / BitmapHeapScanPlan
    → RawRecords

FilterPlan
    → RawRecords

FilterRidsPlan
    → RIDs

ProjectPlan
    → registros proyectados

DeletePlan
    → RawRecords eliminados

EmptyResultPlan
    → resultado vacío
```

`FilterRidsPlan` utiliza `RawRecord` internamente para evaluar su predicado, pero su salida lógica continúa siendo un conjunto de RIDs.

---

# 22. Restricciones de implementación

El Planner:

```text
NO abre almacenamiento
NO abre índices
NO ejecuta búsquedas físicas
NO calcula costos
NO usa estadísticas
NO genera candidatos alternativos
NO selecciona por costo
NO introduce ConditionVisitor
```

El Planner solamente:

```text
lee metadata del catálogo
traduce expresiones
determina indexabilidad
construye access paths
aplica las reglas de FORCE
arma SELECT/DELETE
```

Además, cuando la tabla utiliza `SequentialFile`, el access path del Planner es: `SeqScanPlan(predicate=WHERE)` o `SeqScanPlan(predicate=None)` si no hay WHERE; un rango valido sobre la PRIMARY KEY puede usar `SequentialRangeScanPlan`; el `DELETE` usa `SequentialDeletePlan`. No se usan RIDs para los accesos SELECT sobre `SequentialFile`.

El Executor será responsable de:

```text
abrir estructuras físicas
ejecutar los PlanNode
leer Heap
leer índices
evaluar filtros residuales
realizar DELETE
```

---

# 23. Resumen de reglas

```text
FORCE SEQSCAN
    → SeqScan siempre


Tabla con SequentialFile
    → no admite FORCE (validado en el SemanticVisitor)
    → no usa RIDs: su API entrega registros directamente
    → SELECT sin WHERE -> SeqScanPlan(predicate=None)
    → SELECT con WHERE sin rango de PK -> SeqScanPlan(predicate=WHERE)
    → SELECT con rango de PK -> SequentialRangeScanPlan (residual opcional en FilterPlan)
    → DELETE -> SequentialDeletePlan(predicate)
    → no usa RIDs, ni SeqScanRidsPlan, ni FilterRidsPlan


FORCE INDEX(idx1, idx2, ...)
    → solo los índices especificados
    → todos deben poder participar
    → todos deben participar
    → residual permitido
    → índice no utilizable → PlannerError
    → índices redundantes resolviendo exactamente la misma condición
      con uno dominante → PlannerError


FORCE INDEX(*)
    → usar todos los índices aplicables al WHERE
    → usar cada índice cuando su participación sea correcta
    → HASH preferido frente a BTREE para la misma igualdad
    → no usar índices redundantes para la misma condición
    → residual permitido
    → no exige usar todos los índices físicos de la tabla
    → ningún índice aplicable → PlannerError


SIN FORCE
    → mismas reglas de aplicabilidad que FORCE INDEX(*)
    → usar los índices aplicables
    → HASH preferido frente a BTREE para la misma igualdad
    → residual permitido
    → ningún índice aplicable → SeqScan
```

Simplificación de expresiones:

```text
literal operador literal
    → evaluar durante la planificación

TRUE AND A   → A
FALSE AND A  → FALSE
TRUE OR A    → TRUE
FALSE OR A   → A
NOT TRUE     → FALSE
NOT FALSE    → TRUE

WHERE → FALSE
    → EmptyResultPlan
    → sin acceso a tabla ni índices
```

Reglas de expresiones:

```text
AND
    → combinar accesos indexados
    → permitir residual

OR
    → BitmapOr solo si todo el OR es indexable
    → OR parcial → residual

NOT
    → nunca pushdown al índice
    → residual

a >= x AND a <= y
    → un único IndexRangeScan

igualdad:
    HASH > BTREE

misma columna + condiciones diferentes
    → pueden participar múltiples índices

misma condición + HASH y BTREE
    → no duplicar acceso
```

---

# 24. SequentialFile

`SequentialFile` nunca trabaja con RIDs a nivel de su API: entrega registros
directamente. Por eso el Planner no debe usar RIDs para los accesos sobre una
tabla secuencial, ni `SeqScanRidsPlan`, ni `FilterRidsPlan`, ni `DeletePlan`.

Ademas, una tabla `SequentialFile` no admite ninguna clausula `FORCE`
(`SEQSCAN`, `INDEX(lista)` o `INDEX(*)`). Esa prohibicion se valida en el
analisis semantico, antes del Planner.

## 24.1. Reglas del Planner

```text
SELECT sin WHERE
    -> SeqScanPlan(table, predicate=None)

SELECT con WHERE que no puede reducirse a rango de PRIMARY KEY
    -> SeqScanPlan(table, predicate=WHERE completo)

SELECT con rango valido sobre la PRIMARY KEY
    -> SequentialRangeScanPlan(table, column=PK, range)

SELECT con rango de PRIMARY KEY + condiciones sobrantes
    -> FilterPlan(residual)
         -> SequentialRangeScanPlan(...)

WHERE FALSE / contradiccion
    -> EmptyResultPlan (sin acceso)

DELETE sin WHERE
    -> SequentialDeletePlan(table, predicate=None)

DELETE con WHERE
    -> SequentialDeletePlan(table, predicate=WHERE completo)

DELETE WHERE FALSE / contradiccion
    -> EmptyResultPlan
```

Un WHERE completo que ejecuta `SeqScanPlan` no es residual: va dentro de
`predicate` con `residual=None`. `FilterPlan` solo representa el residual que
un access path previo dejo pendiente.

`SequentialRangeScanPlan` se usa unicamente para rangos de los operadores
`<`, `<=`, `>`, `>=` sobre la PRIMARY KEY. Una igualdad sobre la PK
permanece como `SeqScanPlan(predicate=...)`.

## 24.2. Ejemplos obligatorios

```text
users(id PRIMARY KEY, name, age, city, active)  -- SequentialFile

SELECT * FROM users;
    -> SeqScanPlan(predicate=None)

SELECT * FROM users WHERE city = 'Lima';
    -> SeqScanPlan(predicate=(city='Lima'))

SELECT * FROM users WHERE age >= 20 AND age < 30;
    -> SeqScanPlan(predicate=(age>=20 AND age<30))   (age no es PK)

SELECT * FROM users WHERE id >= 20 AND id < 30;
    -> SequentialRangeScanPlan(id, [20,30))

SELECT name FROM users
WHERE id >= 20 AND id < 30 AND city = 'Lima';
    -> Project(name)
         -> Filter(city='Lima')
              -> SequentialRangeScanPlan(id, [20,30))

SELECT * FROM users WHERE id < 20 OR id > 60;
    -> SeqScanPlan(predicate=(id<20 OR id>60))       (OR no se convierte en rangos)

SELECT * FROM users WHERE NOT (id >= 20);
    -> SeqScanPlan(predicate=NOT(id>=20))

SELECT * FROM users WHERE id = 20;
    -> SeqScanPlan(predicate=(id=20))                (igualdad PK no es rango)

DELETE FROM users;
    -> SequentialDeletePlan(predicate=None)

DELETE FROM users WHERE city = 'Lima';
    -> SequentialDeletePlan(predicate=(city='Lima'))

DELETE FROM users WHERE id >= 20 AND id < 30;
    -> SequentialDeletePlan(predicate=(id>=20 AND id<30))   (nunca rango)

DELETE FROM users WHERE 1 = 0;
    -> EmptyResultPlan
```

## 24.3. Invariantes estructurales

Regla general: ninguna operacion sobre una tabla `SequentialFile`
(CREATE TABLE, INSERT, SELECT, DELETE) puede producir en su arbol de
ejecucion un nodo basado en RIDs. En particular, nunca aparecen:

```text
SeqScanRidsPlan
IndexScanPlan
IndexRangeScanPlan
BitmapIndexScanPlan
BitmapIndexRangeScanPlan
BitmapAndPlan / BitmapOrPlan
HeapFetchPlan
BitmapHeapScanPlan
FilterRidsPlan
DeletePlan
```

Invariantes concretos:

```text
1. SeqScanPlan sin WHERE       -> predicate=None
2. SeqScanPlan con WHERE       -> predicate=WHERE completo y residual=None
3. FilterPlan solo tras un access path que redujo el WHERE
   (IndexScan, IndexRangeScan, BitmapHeapScan, HeapFetch o
    SequentialRangeScanPlan) y solo con residual
4. nunca FilterPlan(WHERE completo) -> SeqScanPlan
5. nunca SeqScanPlan(predicate=WHERE) -> FilterPlan(mismo WHERE)
6. SequentialRangeScanPlan solo si column == PRIMARY KEY
7. SequentialRangeScanPlan nunca para DELETE
8. SequentialFile DELETE nunca usa SeqScanRidsPlan ni FilterRidsPlan
9. SequentialFile SELECT nunca usa RIDs
10. CREATE TABLE / INSERT sobre SequentialFile no producen nodos de acceso
```

## 24.4. API fisica asociada

```text
SeqScanPlan(predicate)          -> recorrido secuencial; evalua predicate si existe
SequentialRangeScanPlan         -> SequentialFile.range_search(key_column, begin, end, ...)
SequentialDeletePlan(predicate) -> SequentialFile.delete_all(predicate)
```


## 25. Modulo vectorial

### 25.1. Entradas

```text
CREATE TABLE t (... v VECTOR(D) ...) USING HEAP;
CREATE INDEX idx ON t(v) USING IVF
    WITH (lists = C, metric = 'L2' | 'COSINE' | 'IP', iterations = k, seed = s);
SET ivf.probes = p;
SELECT ..., v <-> '[...]' AS d FROM t [WHERE ...] ORDER BY d ASC LIMIT k;
```

`<->` es distancia L2, `<=>` es distancia coseno (1 - cos) y `<#>` es producto
interno. La metrica del indice debe coincidir con el operador de la consulta.

### 25.2. Reglas

```text
1. Una sola distancia vectorial por SELECT, con alias obligatorio o 'distancia'.
2. La columna debe ser VECTOR y la tabla HEAP.
3. ORDER BY puede usar el alias de la distancia o una columna real.
4. LIMIT se empuja al Top-K del IVF cuando no hay filtro residual ni orden
   por otra columna.
5. Con WHERE relacional:
       indice escalar disponible -> PRE_FILTER (RID permitidos antes del IVF)
       sin indice escalar         -> POST_FILTER
6. Sin WHERE relacional: INDEX_ONLY.
7. Sin indice IVF o FORCE SEQSCAN: SEQ_SCAN con k-NN lineal exacto.
8. FORCE INDEX(...) solo acepta indices IVF de la misma tabla y columna.
9. SET ivf.probes se guarda en el ExecutionContext y afecta a las sentencias
   siguientes del mismo Program.
```

### 25.3. Planes

```text
VectorScanPlan(table, column, index, query, operator, top_k,
               strategy, prefilter)
    index=None -> escaneo exacto
    index=nombre -> IVF con probes del contexto

VectorFetchPlan(table, column, query, operator, alias,
                child, residual, limit)
    recupera los RID en orden de distancia y aplica post-filtro

VectorProjectPlan(table, columns, alias, position, child)
    intercala la distancia en la lista de salida

SetPlan(name, value)
    modifica ivf.probes
```

### 25.4. Invariantes

```text
1. PRE_FILTER -> prefilter != None y residual puede ser None
2. POST_FILTER -> prefilter == None y residual == WHERE relacional completo
3. INDEX_ONLY -> sin WHERE relacional
4. SEQ_SCAN -> index == None
5. top_k != None solo si no hay residual ni ORDER BY por otra columna
6. VectorFetchPlan siempre recibe VectorScanPlan
7. La metrica del indice y el operador deben coincidir
```
