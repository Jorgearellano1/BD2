# BD2

Motor de base de datos con almacenamiento en Heap y Sequential, índices B+ Tree
y Hash extensible, parser SQL propio y ejecución basada en planes.

## Requisitos

- Python 3.10 o superior (probado en 3.14)
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

El motor relacional y el módulo vectorial usan únicamente la biblioteca
estándar. El módulo textual usa `nltk` para el stemming Snowball y el API HTTP
requiere `fastapi` y `uvicorn`; todo está incluido en `requirements.txt`.
`pytest` y `httpx` cubren la suite de pruebas. El recomendador con subida de
imágenes necesita además `torch`, `torchvision`, `opencv-python-headless`,
`Pillow` y `numpy`: `pip install -r requirements-vision.txt`.

## Estructura

```text
main.py             punto de entrada del API (uvicorn main:app)
backend/
  api/              API HTTP (FastAPI) sobre el motor
  parser/           scanner, parser LL(1), AST, SemanticVisitor y gramatica
  engine/           Catalog, Planner, Executor y Session
  storage/          HeapFile, SequentialFile y StorageFile
  indexes/          B+ Tree e interfaz de indices
  textindex/        indice invertido, BoW y ranking TF-IDF/coseno
  vectorindex/      vectores, K-Means, IVF, BoVW y metricas ANN
  algorithms/       utilidades de ordenamiento y mezcla
rtree/              R-Tree persistente en disco
frontend/
data/datasets/       datasets (incluye cars/ para el recomendador)
data/models/         pesos locales de ResNet-18
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
  `columns`, `rows`, `affected_rows`, `message`, `error` y `explain` (métricas).
  `explains` incluye el plan y las métricas de todas las sentencias del script.

`POST /api/query` acepta un script completo (varias sentencias) y devuelve la
misma estructura que la `Session`: ante un fallo incluye `success=false`,
`error`, `error_type`, `failed_statement_id` y las sentencias ya ejecutadas.
Un lock serializa las peticiones para no mutar el catálogo ni los archivos a la
vez. El catálogo y los datos quedan en `storage/`; `PAGE_SIZE` (por defecto
`4096`) configura el tamaño de página.

## Búsqueda vectorial

El tipo `VECTOR(D)` guarda `D` floats dentro del registro. Los índices IVF se
persisten en un directorio con centroides y listas paginadas, y las consultas
de similitud usan los operadores `<->` (L2), `<=>` (coseno) y `<#>` (producto
interno negado). En los tres casos, menor distancia significa mayor similitud.

```sql
CREATE TABLE multimedia (
  id INT PRIMARY KEY,
  categoria CHAR(20),
  embedding VECTOR(128)
) USING HEAP;

INSERT INTO multimedia VALUES (1, 'ropa', '[0.12, -0.34, 0.56, ...]');

CREATE INDEX idx_vec ON multimedia(embedding) USING IVF
  WITH (lists = 100, metric = 'COSINE');

SET ivf.probes = 5;

SELECT id, categoria, embedding <=> '[0.10, -0.30, 0.50, ...]' AS distancia
FROM multimedia
WHERE categoria = 'ropa'
ORDER BY distancia ASC
LIMIT 5;
```

El planificador reporta la estrategia (`INDEX_ONLY`, `PRE_FILTER`,
`POST_FILTER` o `SEQ_SCAN`) y el ejecutor separa las lecturas del IVF de las
del Heap. `SET ivf.probes` afecta a las sentencias siguientes del mismo
script, así que conviene enviarlo junto con la consulta. Los benchmarks
comparan BoVW (índice invertido) con embeddings densos y miden el compromiso
recall-latencia del IVF.

## Recomendador multimodal

La pestaña **Recomendador** del cliente web usa un dataset real de Craigslist
(12,000 vehículos, licencia CC0) con imagen, título, descripción, precio,
categoría y ubicación. Permite subir una foto o escribir una descripción,
filtrar por precio y cercanía de tienda, elegir el Top-K y compara BoVW
(SIFT + índice invertido) con embeddings densos (ResNet-18 + IVF).

El dataset procesado y los pesos ya están en el repositorio
(`data/datasets/cars/` y `data/models/resnet18.pth`).

### Puesta en marcha (dataset incluido)

```bash
pip install -r requirements.txt
pip install -r requirements-vision.txt
python benchmarks/vector/load_car_db.py   # ~3 min, crea vehiculos e indices
uvicorn main:app --reload
```

`storage/` no está commiteado: `load_car_db.py` crea la tabla `vehiculos` y sus
índices. Solo si quieres regenerar el dataset desde cero:

```bash
mkdir -p data/raw/craigslist
curl -L -o data/raw/craigslist/cars.zip   https://www.kaggle.com/api/v1/datasets/download/austinreese/craigslist-carstrucks-data
python benchmarks/vector/build_car_dataset.py --zip data/raw/craigslist/cars.zip
python benchmarks/vector/load_car_db.py
```

El zip queda en `data/raw/`, ignorado por git; el dataset procesado sí está
commiteado. El detalle de los pasos está en `benchmarks/vector/README.md`.
Endpoints: `POST /api/recommend`, `GET /api/recommend/samples` y
`/media/<imagen>`. El Top-K es un máximo (1-20): si los filtros dejan menos
candidatos, salen menos resultados.

```bash
python benchmarks/vector/run_experiment7.py --quick
python benchmarks/vector/run_experiment8.py --quick
```

El módulo también procesa audio: MFCC por ventana, BoAW sobre el índice
invertido y k-NN con DTW y poda LB_Keogh (`run_audio.py`).

## Pruebas

```bash
python -m pytest tests -q
```
