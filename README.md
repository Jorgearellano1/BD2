# BD2

Motor de base de datos con almacenamiento en Heap y Sequential, índices B+ Tree
y Hash extensible, parser SQL propio y ejecución basada en planes.

## Requisitos

- Python 3.10 o superior (probado en 3.14)
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

El motor (parser, engine, storage, indexes) usa únicamente la biblioteca
estándar. El API HTTP (`backend/api/`) requiere `fastapi` y `uvicorn`, ya
incluidos en `requirements.txt`.

## Estructura

```text
main.py             punto de entrada del API (uvicorn main:app)
backend/
  api/              API HTTP (FastAPI) sobre el motor
  parser/           scanner, parser LL(1), AST, SemanticVisitor y gramatica
  engine/           Catalog, Planner, Executor y Session
  storage/          HeapFile, SequentialFile y StorageFile
  indexes/          B+ Tree e interfaz de indices
  algorithms/       utilidades de ordenamiento y mezcla
frontend/
data/datasets/
storage/
benchmarks/results/
temp_runs/          catalogos y datos temporales de ejecucion
tests/              suite end-to-end y guiones SQL (ver tests/README.md)
```

## Arquitectura

```text
SQL
 |
 v
Scanner -> Parser -> AST
 |
 v
SemanticVisitor
 |
 v
Session
 |  por sentencia:
 |      PlannerVisitor -> PlanNode -> Executor
 v
ExecutionOutput
```

La `Session` valida semánticamente todo el `Program` y luego planifica y
ejecuta sentencia por sentencia sobre un mismo contexto. Así cada plan observa
el catálogo actualizado por las sentencias anteriores (tablas creadas desde
CSV, autoíndices e índices declarados en el mismo `Program`).

`Session.execute` nunca lanza excepciones de sentencia: devuelve un
`ExecutionOutput` con `success`, `error`, `error_type` y `failed_statement_id`,
además de `explains` y `results` parciales (solo de las sentencias exitosas).

Especificaciones:

- `backend/parser/gramatica.md`: gramática EBNF LL(1)
- `backend/engine/planner_specifications.md`: reglas de planificación
- `backend/engine/executor_specifications.md`: ejecución, métricas y Session
- `tests/README.md`: suite end-to-end

Los paquetes se importan por nombre (`engine.*`, `parser.*`, `storage.*`,
`indexes.*`) con `backend/` en `sys.path`; el parser usa importaciones
relativas internas.

## Uso

```python
import sys

sys.path.insert(0, 'backend')

from engine.catalog import Catalog
from engine.session import Session
from parser.parser import Parser
from parser.scanner import Scanner

catalog = Catalog('storage/catalog.json', 'storage/data')
sql = (
    'CREATE TABLE users (id INT PRIMARY KEY, name CHAR(10));'
    "INSERT INTO users VALUES (1, 'ana');"
    'SELECT * FROM users;'
)
program = Parser(Scanner(sql)).parse_program()
output = Session(catalog, page_size=4096).execute(program)

print(output.results[2].columns)  # ['id', 'name']
print(output.results[2].rows)     # [(1, 'ana')]
```

## API

`backend/api/` expone el motor por HTTP con FastAPI (ver Requisitos).

Ejecutar desde la raíz del proyecto:

```bash
uvicorn main:app --reload
```

`main.py` agrega `backend/` a `sys.path` y expone la app. Como alternativa:

```bash
uvicorn api.main:app --reload --app-dir backend
```

Endpoints:

- `GET /api/health` -> `{"status": "ok"}`
- `GET /api/tables` -> tablas con `storage`, `primary_key` y `columns`
- `GET /api/indexes` -> índices con `table`, `column` y `type`
- `POST /api/query` -> `{"sql": "..."}`; responde por sentencia con `success`,
  `columns`, `rows`, `affected_rows`, `message`, `error` y `explain` (métricas)

`POST /api/query` acepta un script completo (varias sentencias) y devuelve la
misma estructura que la `Session`: ante un fallo incluye `success=false`,
`error`, `error_type`, `failed_statement_id` y las sentencias ya ejecutadas.
Un lock serializa las peticiones para no mutar el catálogo ni los archivos a la
vez. El catálogo y los datos quedan en `storage/`; `PAGE_SIZE` (por defecto
`4096`) configura el tamaño de página.
