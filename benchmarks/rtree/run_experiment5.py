"""Mide consultas de radio y k-NN con el R-Tree y con un recorrido secuencial."""
import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rtree.rtree import RTree
from rtree.rtreeNode import MBR
from benchmarks.rtree.rtree_multi import MultiDimensionalRTree

RESULTS = ROOT / 'benchmarks' / 'results' / 'experiment_5'
PREPARED = ROOT / 'benchmarks' / 'temp_runs' / 'experiment_5'


def load_points(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        return [(int(row['id']), float(row['longitude']), float(row['latitude']))
                for row in csv.DictReader(file)]


def show_progress(done, total, started):
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0
    remaining = (total - done) / rate if rate else 0
    filled = round(24 * done / total)
    bar = '#' * filled + '-' * (24 - filled)
    print(f'  [{bar}] {done:,}/{total:,} ({done / total:.0%}) | '
          f'{elapsed:.1f}s transcurridos | ~{remaining:.1f}s restantes', flush=True)


def build_tree(points, index_path):
    tree = RTree(None, ('longitude', 'latitude'), None, 4096, str(index_path))
    tree.create()
    tree.open()
    started = time.perf_counter()
    print(f'\nConstruyendo R-Tree para {len(points):,} ciudades', flush=True)

    for number, (city_id, longitude, latitude) in enumerate(points, 1):
        tree.insert(MBR(longitude, latitude, longitude, latitude), (city_id, 0))
        if number % 10_000 == 0 or number == len(points):
            show_progress(number, len(points), started)

    return tree, (time.perf_counter() - started) * 1000


def radius_results(points, coordinates, tree, center, radius):
    _, longitude, latitude = center
    radius_squared = radius ** 2
    expected = {city_id for city_id, x, y in points
                if (x - longitude) ** 2 + (y - latitude) ** 2 <= radius_squared}

    started = time.perf_counter()
    sequential = {city_id for city_id, x, y in points
                  if (x - longitude) ** 2 + (y - latitude) ** 2 <= radius_squared}
    sequential_ms = (time.perf_counter() - started) * 1000

    tree._page_reads = 0
    search_area = MBR(longitude - radius, latitude - radius,
                      longitude + radius, latitude + radius)
    started = time.perf_counter()
    candidates = tree.search_intersects(search_area)
    found = {rid[0] for rid in candidates
             if (coordinates[rid[0]][0] - longitude) ** 2
             + (coordinates[rid[0]][1] - latitude) ** 2 <= radius_squared}
    rtree_ms = (time.perf_counter() - started) * 1000

    return sequential_ms, rtree_ms, tree.page_reads, found == sequential == expected, len(found)


def knn_results(points, tree, center, k):
    _, longitude, latitude = center
    started = time.perf_counter()
    sequential = sorted(points, key=lambda point:
                        (point[1] - longitude) ** 2 + (point[2] - latitude) ** 2)[:k]
    sequential_ms = (time.perf_counter() - started) * 1000

    tree._page_reads = 0
    started = time.perf_counter()
    result = tree.nearest_neighbors(longitude, latitude, k)
    rtree_ms = (time.perf_counter() - started) * 1000
    found = [item[0][0] for item in result]
    expected = [point[0] for point in sequential]

    return sequential_ms, rtree_ms, tree.page_reads, set(found) == set(expected), len(found)


def measure_size(n, points, centers, output):
    coordinates = {city_id: (longitude, latitude)
                   for city_id, longitude, latitude in points}
    tree, build_ms = build_tree(points, output / f'rtree_{n}.idx')
    records = []

    for radius in (1.0, 5.0):
        for query_id, center in enumerate(centers):
            seq_ms, tree_ms, tree_io, correct, count = radius_results(
                points, coordinates, tree, center, radius)
            records.append({
                'n': n, 'consulta': 'radio', 'parametro': radius,
                'query_id': query_id, 'secuencial_ms': seq_ms, 'rtree_ms': tree_ms,
                'secuencial_io_estimado': math.ceil(n / 100), 'rtree_io': tree_io,
                'resultados': count, 'correcto': correct, 'construccion_ms': build_ms,
            })

    for k in (5, 20, 50):
        for query_id, center in enumerate(centers):
            seq_ms, tree_ms, tree_io, correct, count = knn_results(points, tree, center, k)
            records.append({
                'n': n, 'consulta': 'knn', 'parametro': k,
                'query_id': query_id, 'secuencial_ms': seq_ms, 'rtree_ms': tree_ms,
                'secuencial_io_estimado': math.ceil(n / 100), 'rtree_io': tree_io,
                'resultados': count, 'correcto': correct, 'construccion_ms': build_ms,
            })

    tree.close()
    Path(tree.file_path).unlink(missing_ok=True)
    return records


def summarize(records):
    keys = sorted({(row['n'], row['consulta'], row['parametro']) for row in records})
    summary = []
    for n, query, parameter in keys:
        group = [row for row in records if (row['n'], row['consulta'], row['parametro'])
                 == (n, query, parameter)]
        for name, time_key, io_key in (
            ('Secuencial', 'secuencial_ms', 'secuencial_io_estimado'),
            ('R-Tree', 'rtree_ms', 'rtree_io'),
        ):
            times = [row[time_key] for row in group]
            summary.append({
                'n': n, 'consulta': query, 'parametro': parameter, 'estructura': name,
                'muestras': len(times), 'tiempo_medio_ms': round(statistics.mean(times), 4),
                'tiempo_std_ms': round(statistics.stdev(times) if len(times) > 1 else 0, 4),
                'bloques_io_medios': round(statistics.mean(row[io_key] for row in group), 2),
                'resultados_correctos': all(row['correcto'] for row in group),
                'construccion_rtree_ms': round(group[0]['construccion_ms'], 2),
            })
    return summary


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path, rows):
    with path.open('w', encoding='utf-8') as file:
        file.write('% Resumen del Experimento 5\n')
        file.write('N & Consulta & Parámetro & Estructura & Muestras & Tiempo (ms) & Desv. est. & Bloques I/O & Correcto \\\\ \n\\hline\n')
        for row in rows:
            query = 'Radio' if row['consulta'] == 'radio' else 'k-NN'
            parameter = f"r={row['parametro']}" if query == 'Radio' else f"k={row['parametro']}"
            correct = 'Sí' if row['resultados_correctos'] else 'No'
            file.write(f"{row['n']} & {query} & {parameter} & {row['estructura']} & "
                       f"{row['muestras']} & {row['tiempo_medio_ms']:.4f} & "
                       f"{row['tiempo_std_ms']:.4f} & {row['bloques_io_medios']:.2f} & "
                       f"{correct} \\\\ \n")


def measure_dimensions(quick, seed):
    dimensions_to_test = [2, 4, 8, 16, 32, 64, 128]
    point_count = 500 if quick else 5_000
    query_count = 2 if quick else 5
    randomizer = random.Random(seed + 1)
    measurements = []

    for dimensions in dimensions_to_test:
        print(f'\nProbando vectores sintéticos D={dimensions}, N={point_count:,}', flush=True)
        points = [
            (point_id, tuple(randomizer.random() for _ in range(dimensions)))
            for point_id in range(point_count)
        ]
        centers = [points[index][1]
                   for index in randomizer.sample(range(point_count), query_count)]
        started = time.perf_counter()
        tree = MultiDimensionalRTree(points, dimensions)
        build_ms = (time.perf_counter() - started) * 1000
        radius = 0.15 * math.sqrt(dimensions)
        record_size = dimensions * 8 + 8
        page_capacity = max(1, (4096 - 16) // record_size)
        sequential_io = math.ceil(point_count / page_capacity)

        for query_id, center in enumerate(centers):
            started = time.perf_counter()
            expected = [point_id for point_id, vector in points
                        if MultiDimensionalRTree._point_distance(vector, center) <= radius ** 2]
            sequential_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            found = tree.radius_search(center, radius)
            rtree_ms = (time.perf_counter() - started) * 1000
            measurements.append({
                'dimension': dimensions, 'n': point_count, 'consulta': 'radio',
                'parametro': radius, 'query_id': query_id,
                'secuencial_ms': sequential_ms, 'rtree_ms': rtree_ms,
                'secuencial_io': sequential_io, 'rtree_io': tree.page_reads,
                'resultados': len(found), 'correcto': set(found) == set(expected),
                'construccion_ms': build_ms,
            })

            for k in (5, 20, 50):
                started = time.perf_counter()
                expected_knn = sorted(
                    points,
                    key=lambda item: (
                        MultiDimensionalRTree._point_distance(item[1], center), item[0]
                    ),
                )[:k]
                sequential_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                found_knn = tree.nearest_neighbors(center, k)
                rtree_ms = (time.perf_counter() - started) * 1000
                measurements.append({
                    'dimension': dimensions, 'n': point_count, 'consulta': 'knn',
                    'parametro': k, 'query_id': query_id,
                    'secuencial_ms': sequential_ms, 'rtree_ms': rtree_ms,
                    'secuencial_io': sequential_io, 'rtree_io': tree.page_reads,
                    'resultados': len(found_knn),
                    'correcto': {item[0] for item in found_knn}
                    == {item[0] for item in expected_knn},
                    'construccion_ms': build_ms,
                })

        print(f'  Índice construido en {build_ms:.1f} ms; radio={radius:.3f}', flush=True)

    return measurements


def summarize_dimensions(records):
    summary = []
    for dimensions in sorted({row['dimension'] for row in records}):
        for query in ('radio', 'knn'):
            group = [row for row in records
                     if row['dimension'] == dimensions and row['consulta'] == query]
            for name, time_key, io_key in (
                ('Secuencial', 'secuencial_ms', 'secuencial_io'),
                ('R-Tree multidimensional', 'rtree_ms', 'rtree_io'),
            ):
                times = [row[time_key] for row in group]
                summary.append({
                    'dimension': dimensions, 'consulta': query, 'estructura': name,
                    'muestras': len(times),
                    'tiempo_medio_ms': round(statistics.mean(times), 4),
                    'tiempo_std_ms': round(
                        statistics.stdev(times) if len(times) > 1 else 0, 4
                    ),
                    'bloques_io_medios': round(
                        statistics.mean(row[io_key] for row in group), 2
                    ),
                    'resultados_correctos': all(row['correcto'] for row in group),
                    'construccion_ms': round(group[0]['construccion_ms'], 2),
                })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=505)
    args = parser.parse_args()
    sizes = [1_000, 5_000] if args.quick else [10_000, 50_000, 100_000]
    profile = 'quick' if args.quick else 'full'
    data_folder = PREPARED / profile
    output = RESULTS / ('experimento_5_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        + ('_rapido' if args.quick else '_completo'))
    output.mkdir(parents=True, exist_ok=False)
    randomizer = random.Random(args.seed)
    all_records = []

    for n in sizes:
        data_path = data_folder / f'cities_{n}.csv'
        if not data_path.exists():
            raise SystemExit('Faltan datos preparados. Ejecuta build_experiment5.py primero.')
        points = load_points(data_path)
        center_count = min(10 if args.quick else 30, n)
        centers = [points[index] for index in randomizer.sample(range(n), center_count)]
        all_records.extend(measure_size(n, points, centers, output))

    dimension_records = measure_dimensions(args.quick, args.seed)
    dimension_summary = summarize_dimensions(dimension_records)
    summary = summarize(all_records)
    write_csv(output / 'mediciones_detalladas.csv', all_records)
    write_csv(output / 'tabla_resumen.csv', summary)
    write_latex(output / 'tabla_resumen.tex', summary)
    write_csv(output / 'dimensionalidad_detalle.csv', dimension_records)
    write_csv(output / 'tabla_dimensionalidad.csv', dimension_summary)
    manifest = {
        'experimento': 5, 'perfil': profile, 'dataset': 'cities.csv', 'tamanos': sizes,
        'semilla': args.seed, 'radios': [1.0, 5.0], 'valores_k': [5, 20, 50],
        'consultas_geográficas': len(all_records),
        'mediciones_dimensionalidad': len(dimension_records),
        'dimensiones_probadas': [2, 4, 8, 16, 32, 64, 128],
        'índice_multidimensional': 'R-tree empaquetado independiente para benchmark',
    }
    (output / 'configuracion.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nMediciones guardadas en: {output}', flush=True)


if __name__ == '__main__':
    main()
