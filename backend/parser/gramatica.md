# Gramática EBNF LL(1)

`{}` = repetición
`[]` = opcional

```ebnf
Program ::= { Statement }


Statement ::= CREATE CreateStmtRest
            | INSERT InsertStmtRest
            | ALTER AlterStmtRest
            | SELECT SelectStmtRest
            | DELETE DeleteStmtRest


CreateStmtRest ::= TABLE ID CreateTableBody
                 | INDEX ID ON ID "(" ID ")"
                   USING IndexType ";"


CreateTableBody ::= "(" ColumnDefList ")"
                    [ UsingClause ]
                    [ CsvClause ]
                    ";"

                  | CsvClause
                    [ UsingClause ]
                    PrimaryKeyClause
                    ";"


ColumnDefList ::= ColumnDef { "," ColumnDef }

ColumnDef ::= ID Type [ ColumnConstraint ]

ColumnConstraint ::= PRIMARY KEY
                   | UNIQUE


Type ::= INT
       | FLOAT
       | DOUBLE
       | CHAR "(" INT_LITERAL ")"


UsingClause ::= USING StorageType

StorageType ::= HEAP
              | SEQUENTIAL


CsvClause ::= FROM CSV "(" STRING_LITERAL
              [ "," DELIMITER "=" STRING_LITERAL ]
              ")"


PrimaryKeyClause ::= PRIMARY KEY "(" ID ")"


InsertStmtRest ::= INTO ID
                   VALUES "(" ValueList ")"
                   ";"

ValueList ::= Value { "," Value }

Value ::= INT_LITERAL
        | REAL_LITERAL
        | STRING_LITERAL


IndexType ::= BTREE
            | HASH


AlterStmtRest ::= TABLE ID
                  ADD UNIQUE "(" ID ")"
                  ";"


SelectStmtRest ::= SelectList
                   FROM ID
                   [ ForceClause ]
                   [ WhereClause ]
                   ";"

SelectList ::= "*"
             | ColumnList

ColumnList ::= ID { "," ID }


ForceClause ::= FORCE ForceClauseRest

ForceClauseRest ::= INDEX "(" IndexList ")"
                  | SEQSCAN

IndexList ::= ID { "," ID }


WhereClause ::= WHERE Condition

Condition ::= OrCondition

OrCondition ::= AndCondition
                { OR AndCondition }

AndCondition ::= NotCondition
                 { AND NotCondition }

NotCondition ::= NOT NotCondition
               | PrimaryCondition

PrimaryCondition ::= "(" Condition ")"
                   | Operand ComparisonOperator Operand


Operand ::= ID
          | INT_LITERAL
          | REAL_LITERAL
          | STRING_LITERAL


ComparisonOperator ::= "="
                     | "!="
                     | "<>"
                     | "<"
                     | "<="
                     | ">"
                     | ">="


DeleteStmtRest ::= FROM ID
                   [ WhereClause ]
                   ";"
```

### 

1. El nombre de una tabla debe ser único dentro de la base de datos. No puede existir la misma tabla como HEAP y SEQUENTIAL. El usuario no indica el tipo de almacenamiento al hacer SELECT; el catálogo y el planner ya conocen cómo está almacenada.

2. CREATE TABLE puede definir el esquema explícitamente, definir el esquema y cargar un CSV, o inferir el esquema desde un CSV indicando obligatoriamente la primary key. Si no se especifica USING, se asume HEAP. El delimitador del CSV es coma por defecto, pero puede cambiarse.

3. Los tipos permitidos son INT, FLOAT, DOUBLE y CHAR(n). No se incluye BOOL ni BOOLEAN.

4. Una tabla puede tener una sola PRIMARY KEY. La PRIMARY KEY implica UNIQUE. Se pueden tener varias restricciones UNIQUE. UNIQUE es una restricción y no es lo mismo que CREATE INDEX.

5. INSERT permite insertar los valores de las columnas en el orden definido por la tabla. Los valores pueden ser enteros, reales o cadenas.

6. Los índices explícitos se crean con CREATE INDEX y pueden ser BTREE o HASH. Los índices explícitos se permiten para HEAP. SEQUENTIAL no usa CREATE INDEX como mecanismo expuesto al usuario.

7. No existe una sentencia BULK INSERT independiente; la carga masiva se realiza mediante CREATE TABLE ... FROM CSV.

8. SEQUENTIAL es un método de almacenamiento físico ordenado. Puede aprovechar su orden para ciertas búsquedas, especialmente sobre su clave de ordenamiento.

9. SELECT puede consultar todas las columnas o una lista específica de columnas.

10. Para una condición de igualdad, el planner puede usar un índice BTREE o HASH mediante IndexScan. Si no existe un índice aplicable, usa SeqScan.

11. Para condiciones de rango, BTREE puede utilizarse mediante IndexRangeScan. HASH no sirve para búsquedas de rango. Si no hay un índice BTREE aplicable, se usa SeqScan.

12. FORCE INDEX permite especificar un conjunto de índices candidatos para el planner. No significa necesariamente que todos los índices indicados deban ejecutarse.

13. WHERE admite los operadores de comparación igual, distinto, menor, menor o igual, mayor y mayor o igual.

14. WHERE admite NOT, AND y OR, con precedencia NOT mayor que AND y AND mayor que OR. También se permiten paréntesis para agrupar condiciones.

15. La gramática permite comparar una columna con un valor, una columna con otra columna o un valor con una columna. Sin embargo, la lógica actual de índices está pensada principalmente para condiciones columna-operador-valor. Las comparaciones columna-columna pueden resolverse mediante SeqScan.

16. DELETE puede eliminar todos los registros de una tabla o utilizar una condición WHERE. Su elección de ejecución sigue la misma lógica general del planner: índice aplicable cuando corresponde y SeqScan en caso contrario.

17. Cada tabla HEAP tiene un único HeapFile. Varios índices pueden compartir ese mismo HeapFile.

18. El B+ Tree almacena pares de clave y RID. El RID está formado por page_id y slot_number.

19. El B+ Tree no debe cerrar el HeapFile compartido. Cada estructura debe cerrar únicamente los archivos que le pertenecen.

20. El almacenamiento utiliza páginas de tamaño fijo y acceso mediante seek, read y write. No se usa pickle ni se carga todo el archivo de una sola vez.

21. Las lecturas y escrituras de páginas deben medirse mediante DiskCounter.

22. Las cargas masivas deben procesarse de forma streaming y con memoria acotada. El objetivo es evitar estructuras en memoria que crezcan proporcionalmente con todo el dataset.
