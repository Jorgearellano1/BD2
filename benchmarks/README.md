# Benchmark local de BD2

Ejecuta los cuatro experimentos sin iniciar FastAPI ni el frontend. Usa el motor real y comprueba las filas devueltas y el plan de cada consulta. No toca `storage/`: todas las bases experimentales se crean en un directorio temporal propio y se eliminan al terminar. Cada ejecución conserva sus resultados en una carpeta nueva.

## Preparación

Desde la raíz del proyecto:

```bash
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/python -m pip install -r benchmarks/requirements.txt
```

## Verificar que funciona

```bash
benchmarks/.venv/bin/python benchmarks/run_benchmarks.py --quick
```

Este perfil prueba los cuatro experimentos con 100–1000 registros y 20 búsquedas. Los gráficos dicen **PRUEBA RÁPIDA**. Sirve para verificar la integración, no para presentar resultados a escala de la rúbrica.

## Ejecutar los experimentos del informe

```bash
benchmarks/.venv/bin/python benchmarks/run_benchmarks.py
```

Puede tardar considerablemente: incluye full scans, reorganización, persistencia y validación real de resultados. Imprime el avance. No hay servidor ni servicio externo involucrado. Ctrl+C marca la ejecución como fallida y conserva las mediciones ya escritas; no presenta una ejecución incompleta como exitosa.

Opciones:

```bash
benchmarks/.venv/bin/python benchmarks/run_benchmarks.py --seed 2042 --repeats 3
```

`--repeats` controla las repeticiones de inserción y rangos. Igualdad siempre usa 1000 consultas en modo completo. La semilla fija mantiene los datos y las consultas reproducibles; se alterna el orden de las estructuras para reducir el sesgo de orden.

## Experimentos

1. **Carga masiva e inserción:** N = 1000, 10000, 50000, 100000, 250000 y 500000; Heap, Sequential sin reorganización automática, Sequential con reorganización manual final, Heap+B+ y Heap+Hash. Tres repeticiones por defecto. En cada caso se baraja el mismo lote, se carga el 50% mediante CSV, se construye el índice si corresponde y se inserta el 50% restante registro a registro. La medición incluye carga, construcción de índice y las inserciones; la variante con reorganización también incluye esa operación. Así hay inserciones sobre una base existente y overflow real. No se presenta como tiempo de N sentencias SQL INSERT individuales. Ambos Sequential parten de los mismos datos y orden; se registran los conteos de overflow antes/después. Se cierra y reabre para verificar todos los IDs.
2. **Igualdad:** 1000 claves seleccionadas con semilla sobre 100000 registros. Compara Heap, Sequential, B+ y Hash; media y desviación estándar muestral de tiempo y lecturas. Se comprueba el contenido exacto. Heap fuerza SEQSCAN; B+ y Hash fuerzan su único índice experimental. **El planner actual convierte `Student_ID = k` en SeqScan sobre Sequential**. Para medir el acceso ordenado se usa la consulta equivalente `Student_ID >= k AND Student_ID <= k`; se registra este SQL y se exige SequentialRangeScan. No se modifica el planner.
3. **Rangos:** 500000 registros, selectividades 0.1%, 1%, 5%, 10% y 25%, en Heap, Sequential y B+. Tres posiciones aleatorias por selectividad por defecto; las mismas para cada estructura. La salida se contrasta con el conjunto de IDs esperado. Hash no participa porque no admite rangos.
4. **Página:** B+ sobre 100000 registros, reconstruido para 1024, 2048, 4096 y 8192 bytes. Ejecuta las mismas 1000 búsquedas de igualdad de E2. Registra fan-out máximo calculado por la implementación, altura real (número de niveles; raíz hoja = 1), tiempo, lecturas y escrituras por consulta. Las escrituras de estas consultas pueden ser cero: no son las de construcción. La lectura de altura queda fuera de la medición.

Heap no crea un B+ automático: el esquema SQL de sus casos no declara PK. La unicidad de los IDs se comprueba al leer el CSV. Sequential sí declara PK para habilitar el acceso ordenado. En B+ y Hash se crea explícitamente el índice sobre Student_ID. Esta diferencia de configuración está separada de los datos, que son los mismos.

## Archivos de salida

En `benchmarks/results/FECHA-HORA-full/` o `...-quick/`:

- `manifest.json`: perfil, configuración, estado, entorno, semilla y huella del dataset. Solo `status: complete` indica finalización exitosa.
- `raw.csv`: mediciones individuales y SQL/plan de consultas; se escribe progresivamente.
- `summary.csv`: medias, desviaciones, cantidad de muestras y configuración.
- `exp1_summary.csv` a `exp4_summary.csv`: tablas separadas para el informe.
- `equality_keys.csv`: claves utilizadas en E2 y E4.
- Ocho gráficas: `exp1_tiempo.png`, `exp1_escrituras.png`, `exp2_lecturas.png`, `exp2_tiempo.png`, `exp3_lecturas.png`, `exp3_tiempo.png`, `exp4_estructura.png`, `exp4_io.png`.

Los nombres PNG coinciden con la plantilla LaTeX del informe. Copiar las imágenes de una ejecución **full completa** junto al `.tex`, y completar las tablas con sus CSV. No combinar tiempos de ejecuciones distintas sin explicarlo.

## Cómo interpretar las mediciones

- Se emplean los contadores existentes del motor. Son **páginas instrumentadas**, no mediciones del dispositivo físico ni fallos de caché del sistema operativo. Hay accesos de cabecera y apertura/cierre que el motor no instrumenta de forma uniforme; este script no corrige esos algoritmos. Esta limitación debe mencionarse si se exige contabilización exacta de todo acceso.
- `execution_ms` de consultas mide Session completa: validación, planificación, ejecución y cierre. Excluye parseo, HTTP, rendering y comprobación posterior de los resultados. `parse_ms` se guarda separado. E1 suma la ejecución de las fases de carga y el lote de inserciones/cierre.
- Una nueva Session abre recursos para cada consulta; la caché del sistema operativo no se vacía. Hay una consulta de calentamiento por estructura fuera de las estadísticas. No describir los resultados como pruebas de caché fría.
- La comparación E1 incluye almacenamiento e índice en los casos B+/Hash, no únicamente el archivo de índice. La carga masiva y construcción no equivalen al algoritmo de inserción incremental de un índice vacío.
- El índice Hash accede a su directorio y a sus buckets como páginas del archivo; su apertura/cierre forma parte de la Session y parte de esa E/S puede no quedar desglosada por nodo. Conviene contrastar sus contadores con los del resto de estructuras antes de afirmar ventajas de I/O físico.
- Una sola muestra tiene desviación guardada como cero por convenio; `samples` indica cuándo ocurre. El modo completo proporciona las repeticiones descritas.
- Los CSV son datos generados. Las bases de prueba temporales se eliminan automáticamente, pero los resultados y las tablas del usuario no se eliminan.

## Modulo vectorial (experimentos 7 y 8)

Los experimentos de BoVW, embeddings densos y busqueda aproximada IVF viven en
`benchmarks/vector/`. Tienen su propio README con datasets, comandos y salidas:

```bash
python benchmarks/vector/run_experiment7.py --quick
python benchmarks/vector/run_experiment8.py --quick
```

Los resultados se guardan en `benchmarks/results/experiment_7/` y
`benchmarks/results/experiment_8/`, separados de las corridas relacionales.
