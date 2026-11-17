# Benchmark del modulo vectorial

Experimentos 7 y 8 del motor multimodal. Usan el motor real (parser, Session,
HeapFile e IVF) sobre bases temporales y guardan los resultados en
`benchmarks/results/experiment_7/` y `.../experiment_8/`.

## Dataset

```text
data/datasets/multimedia/
├── images/<clase>/*.jpg          600 imagenes (6 clases)
├── sift_descriptors.bin           78 126 descriptores SIFT de 128D
├── embeddings.bin                 600 embeddings de 512D
├── embeddings_manifest.csv        id,category,filename
└── manifest.json
```

Los binarios se leen con `struct`; el motor no depende de OpenCV ni PyTorch.
Para regenerarlos (requiere `opencv-python` y `torchvision`):

```bash
python benchmarks/vector/build_dataset.py
python benchmarks/vector/verify_dataset.py
```

El experimento 8 usa ademas `data/datasets/dry_beans.arff`
(13 687 vectores de 16 dimensiones, 7 clases).

## Experimento 7: BoVW contra embeddings densos

```bash
python benchmarks/vector/run_experiment7.py --quick
python benchmarks/vector/run_experiment7.py
```

Compara tres rutas sobre las mismas 600 imagenes:

1. **BoVW** sobre el indice invertido (`MATCH` + `COSINE_SCORE`).
2. **Dense exacto** con `FORCE SEQSCAN` y el operador `<=>`.
3. **Dense IVF** con `CREATE INDEX ... USING IVF` y `SET ivf.probes`.

Salidas:

```text
experiment_7/<fecha>-<perfil>/
├── summary.csv        precision@5/10, latencia, lecturas, tiempos de build
├── queries.csv        detalle por imagen de consulta
├── manifest.json      parametros, dataset y resumen
└── exp7_*.png         precision, latencia y lecturas
```

`--quick` reduce codebook, muestras y listas. Los embeddings y los descriptores
SIFT se cargan precalculados; `feature_ms` mide la construccion del codebook y
la cuantizacion BoVW, y en el paradigma denso solo la carga del binario.

## Experimento 8: eficiencia contra calidad en IVF

```bash
python benchmarks/vector/run_experiment8.py --quick
python benchmarks/vector/run_experiment8.py
```

Verdad base: k-NN lineal exacto. Se varia `probes` sobre los mismos queries y
se mide Recall@10, latencia y bloques leidos para N crecientes.

Salidas:

```text
experiment_8/<fecha>-<perfil>/
├── summary.csv        recall@10, latencia y lecturas por N y probes
├── queries.csv        detalle por consulta
├── manifest.json      listas, N, semilla y parametros
└── exp8_ann.png       curvas recall, latencia, lecturas y compromiso
```

## Referencia opcional con pgvector

Requiere Docker en ejecucion. La comparacion usa el mismo dataset y la misma
verdad base:

```bash
python benchmarks/vector/pgvector_reference.py --start
python benchmarks/vector/pgvector_reference.py --n 3000 --queries 20
python benchmarks/vector/pgvector_reference.py --stop
```

Escribe `benchmarks/results/experiment_8/pgvector_reference/`. Si Docker no
esta disponible, el script termina con un mensaje y no afecta al resto de los
experimentos.

## Camino de audio: MFCC, BoAW y DTW

El motor tambien procesa audio. `features.mfcc_frames_from_wav` devuelve un
descriptor de 13 dimensiones por ventana; esos descriptores se cuantizan con
el mismo K-Means y se indexan como BoAW sobre el indice invertido. La busqueda
exacta puede usar DTW, con poda LB_Keogh para Filter-and-Refine.

```bash
python benchmarks/vector/run_audio.py --quick
python benchmarks/vector/run_audio.py
```

El demo genera clips WAV sinteticos (tonos, chirp, bitono y ruido) porque el
repositorio no incluye un corpus de audio. Cualquier WAV PCM de 16 bits puede
pasar por `mfcc_frames_from_wav`. Salida en
`benchmarks/results/experiment_audio/<fecha>-<perfil>/` con `summary.csv`,
`queries.csv`, `manifest.json` y una grafica. Los experimentos 7 y 8 siguen
usando las imagenes y los embeddings densos.

## Formato de los binarios multimedia

```text
sift_descriptors.bin
    b'BD2SIFT1' | uint32 num_images | uint32 dim=128
    por imagen: uint32 count + count*128 uint8

embeddings.bin
    b'BD2EMB01' | uint32 num_images | uint32 dim=512
    num_images*512 float32 little-endian (L2 normalizados)

embeddings_manifest.csv
    id,category,filename   (mismo orden que las filas de embeddings.bin)
```

Los descriptores SIFT conservan el valor 0..255 de OpenCV; el lector los
normaliza a 0..1. Los embeddings corresponden a la salida de ResNet-18
(fc = Identity). Ambos archivos se leen con `struct`, sin OpenCV ni PyTorch.

`data/datasets/dry_beans_demo.csv` contiene 210 filas (30 por clase) del
dataset original de 13 611 filas. Se regenera con
`python benchmarks/vector/build_beans_demo.py` y es el archivo que usa el
ejemplo vectorial del frontend.

## BoVW para el ejemplo del frontend

`data/datasets/multimedia/bovw.csv` contiene las 600 imagenes con formato
`id;categoria;histograma`, donde el histograma es una secuencia de palabras
visuales (`v0012 v0012 v0007 ...`). Se genera con:

```bash
python benchmarks/vector/build_bovw_csv.py
```

El script lee `sift_descriptors.bin`, construye un codebook de 64 palabras y
escribe tambien `bovw_codebook.json` y `bovw_manifest.json`. El frontend usa
`bovw.csv` en el ejemplo "BoVW: crear tabla e indice invertido".

## Recomendador de vehiculos

Dataset real de Craigslist (Kaggle, CC0): 426k vehiculos con imagen, titulo,
descripcion, precio, categoria y latitud/longitud. Se usa un subconjunto de
12,000 con imagenes de 256x192.

```bash
mkdir -p data/raw/craigslist
curl -L -o data/raw/craigslist/cars.zip \
  https://www.kaggle.com/api/v1/datasets/download/austinreese/craigslist-carstrucks-data

pip install -r requirements-vision.txt
python benchmarks/vector/build_car_dataset.py --zip data/raw/craigslist/cars.zip
python benchmarks/vector/load_car_db.py
```

`build_car_dataset.py` descarga las imagenes, entrena el codebook BoVW, extrae
SIFT y embeddings ResNet-18 y deja todo en `data/datasets/cars/`. Tambien
guarda `data/models/resnet18.pth`, que usa el API para procesar fotos subidas.

`load_car_db.py` crea la tabla `vehiculos` en `storage/` con indices:

```text
B+ Tree     precio
R-Tree      latitud, longitud
Invertido   descripcion (ingles)
Invertido   palabras visuales BoVW
IVF         embedding (512D, coseno)
```

Luego se levanta el API y se abre la pestana **Recomendador**:

```bash
uvicorn main:app --reload
```

Se puede subir una foto o escribir una descripcion, filtrar por precio y
ciudad/radio. El resultado muestra BoVW y embeddings en paralelo, con sus
planes y metricas de I/O. Endpoints: `POST /api/recommend`,
`GET /api/recommend/samples` y `/media/<imagen>`.
