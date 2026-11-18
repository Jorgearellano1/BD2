"""Crea gráficas y un informe HTML a partir de una ejecución del Experimento 5."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def read_summary(folder):
    with (folder / 'tabla_resumen.csv').open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def save_time_plot(rows, folder):
    cases = sorted({(int(row['n']), row['consulta'], row['parametro']) for row in rows})
    labels, sequential, rtree = [], [], []
    for n, query, parameter in cases:
        match = [row for row in rows if int(row['n']) == n and row['consulta'] == query
                 and row['parametro'] == parameter]
        labels.append(f'{n // 1000}k {"r=" if query == "radius" else "k="}{parameter}')
        sequential.append(float(next(row for row in match if row['estructura'] == 'Secuencial')['tiempo_medio_ms']))
        rtree.append(float(next(row for row in match if row['estructura'] == 'R-Tree')['tiempo_medio_ms']))
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * .45), 5))
    ax.plot(positions, sequential, marker='o', label='Secuencial')
    ax.plot(positions, rtree, marker='s', label='R-Tree')
    ax.set_xticks(positions, labels, rotation=50, ha='right')
    ax.set_ylabel('Tiempo medio (ms)')
    ax.set_title('Experimento 5: tiempo por consulta')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'exp5_tiempo.png', dpi=160)
    plt.close(fig)


def save_io_plot(rows, folder):
    cases = sorted({(int(row['n']), row['consulta'], row['parametro']) for row in rows})
    labels, sequential, rtree = [], [], []
    for n, query, parameter in cases:
        match = [row for row in rows if int(row['n']) == n and row['consulta'] == query
                 and row['parametro'] == parameter]
        labels.append(f'{n // 1000}k {"r=" if query == "radius" else "k="}{parameter}')
        sequential.append(float(next(row for row in match if row['estructura'] == 'Secuencial')['bloques_io_medios']))
        rtree.append(float(next(row for row in match if row['estructura'] == 'R-Tree')['bloques_io_medios']))
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * .45), 5))
    ax.plot(positions, sequential, marker='o', label='Secuencial (estimado)')
    ax.plot(positions, rtree, marker='s', label='R-Tree (contador de páginas)')
    ax.set_xticks(positions, labels, rotation=50, ha='right')
    ax.set_ylabel('Bloques I/O medios')
    ax.set_title('Experimento 5: bloques I/O por consulta')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'exp5_bloques_io.png', dpi=160)
    plt.close(fig)


def save_dimension_plot(rows, folder, metric, filename, title, ylabel):
    dimensions = sorted({int(row['dimension']) for row in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for query, query_label, marker in (
        ('radio', 'Radio', 'o'),
        ('knn', 'k-NN', 's'),
    ):
        for structure, structure_label, line_style in (
            ('Secuencial', 'Secuencial', '--'),
            ('R-Tree multidimensional', 'R-Tree multidimensional', '-'),
        ):
            values = []
            for dimension in dimensions:
                match = next(row for row in rows
                             if int(row['dimension']) == dimension
                             and row['consulta'] == query
                             and row['estructura'] == structure)
                values.append(float(match[metric]))
            ax.plot(dimensions, values, linestyle=line_style, marker=marker,
                    label=f'{query_label}: {structure_label}')

    ax.set_xscale('log', base=2)
    ax.set_xticks(dimensions, [str(value) for value in dimensions])
    ax.set_xlabel('Dimensión D')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=160)
    plt.close(fig)


def save_html(rows, folder):
    headers = list(rows[0])
    head = ''.join(f'<th>{column}</th>' for column in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{row[column]}</td>' for column in headers) + '</tr>' for row in rows)
    dimension_rows = []
    with (folder / 'tabla_dimensionalidad.csv').open(newline='', encoding='utf-8-sig') as file:
        dimension_rows = list(csv.DictReader(file))
    dimension_headers = ''.join(f'<th>{column}</th>' for column in dimension_rows[0])
    dimension_body = ''.join(
        '<tr>' + ''.join(f'<td>{row[column]}</td>' for column in dimension_rows[0]) + '</tr>'
        for row in dimension_rows
    )
    page = f'''<!doctype html><html lang="es"><meta charset="utf-8"><title>Experimento 5</title>
<style>body{{font:15px Arial,sans-serif;margin:30px;color:#1c2733}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4df;padding:7px;text-align:left}}th{{background:#eaf0f8}}tr:nth-child(even){{background:#f6f8fb}}img{{max-width:100%}}.note{{padding:12px;background:#fff3d2}}</style>
<h1>Experimento 5 — R-Tree vs. secuencial</h1><h2>Tiempo</h2><img src="exp5_tiempo.png"><h2>Bloques I/O</h2><img src="exp5_bloques_io.png">
<h2>Dimensionalidad</h2><img src="exp5_dimension_tiempo.png"><img src="exp5_dimension_io.png">
<h2>Tabla geográfica</h2><table><tr>{head}</tr>{body}</table>
<h2>Tabla de dimensionalidad</h2><table><tr>{dimension_headers}</tr>{dimension_body}</table>
<p class="note">La prueba sintética usa un R-tree empaquetado independiente del índice geográfico de la minibase. El I/O secuencial se estima a partir del tamaño del vector y una página de 4 KiB.</p></html>'''
    (folder / 'reporte_experimento_5.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path, help='carpeta de resultados; por defecto usa la ejecución más reciente')
    args = parser.parse_args()
    parent = ROOT / 'benchmarks' / 'results' / 'experiment_5'
    folders = sorted(parent.glob('experimento_5_*'), key=lambda path: path.stat().st_mtime)
    folder = args.folder.resolve() if args.folder else (folders[-1] if folders else None)
    if folder is None or not (folder / 'tabla_resumen.csv').exists():
        raise SystemExit('No hay resultados para graficar. Ejecuta run_experiment5.py primero.')
    rows = read_summary(folder)
    dimension_rows = []
    with (folder / 'tabla_dimensionalidad.csv').open(newline='', encoding='utf-8-sig') as file:
        dimension_rows = list(csv.DictReader(file))
    save_time_plot(rows, folder)
    save_io_plot(rows, folder)
    save_dimension_plot(dimension_rows, folder, 'tiempo_medio_ms',
                        'exp5_dimension_tiempo.png',
                        'Efecto de la dimensión en la latencia', 'Tiempo medio (ms)')
    save_dimension_plot(dimension_rows, folder, 'bloques_io_medios',
                        'exp5_dimension_io.png',
                        'Efecto de la dimensión en los bloques visitados', 'Bloques I/O medios')
    save_html(rows, folder)
    print(f'Gráficas e informe creados en: {folder}')


if __name__ == '__main__':
    main()
