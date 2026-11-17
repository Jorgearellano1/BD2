"""
Graficas y tablas LaTeX del Experimento 6 a partir de los CSV reales
(build.csv, summary.csv) de una ejecucion de run_experiment6.py.

Uso:
  .venv/bin/python benchmarks/text/plot_experiment6.py benchmarks/results/experiment_6/<ejecucion>

Genera en esa carpeta:
  exp6_construccion.png, exp6_tamano.png, exp6_latencia_categorias.png,
  exp6_lecturas_categorias.png, exp6_latencia_vs_n.png, exp6_lecturas_vs_n.png,
  tabla_construccion.tex, tabla_consultas.tex
Todas en escala de grises (informe en blanco y negro).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SYSTEMS = [
    ('indice', 'Índice invertido propio', '0.15', '', 'o', '-'),
    ('full_scan', 'Full Text Scan', '0.55', '//', 's', '--'),
    ('postgres_gin', 'PostgreSQL GIN', '0.85', 'xx', '^', ':'),
]
CATEGORY_ORDER = [('term', 1), ('and', 3), ('and', 5), ('or', 3), ('or', 5),
                  ('andnot', 3), ('andnot', 5), ('rank', 1), ('rank', 3), ('rank', 5)]
CATEGORY_LABEL = {'term': 'término', 'and': 'AND', 'or': 'OR', 'andnot': 'AND-NOT', 'rank': 'Top-10'}

plt.rcParams.update({
    'font.size': 9, 'axes.grid': True, 'grid.color': '0.85', 'grid.linewidth': 0.6,
    'axes.edgecolor': '0.2', 'savefig.dpi': 200, 'figure.dpi': 100,
})


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def num(value: str) -> float | None:
    return float(value) if value not in ('', None) else None


def label(kind: str, k: int) -> str:
    return f'{CATEGORY_LABEL[kind]}\n{k} t.'


def main(folder: Path) -> None:
    build = read_csv(folder / 'build.csv')
    summary = read_csv(folder / 'summary.csv')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    ns = [int(r['n']) for r in build]
    nmax = max(ns)
    has_pg = any(r.get('pg_build_ms') for r in build)

    # 1. Construccion
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(ns, [float(r['build_ms']) / 1000 for r in build], '-o', color='0.1', label='Índice invertido propio (SPIMI)')
    if has_pg:
        ax.plot(ns, [float(r['pg_build_ms']) / 1000 for r in build], ':^', color='0.45', label='PostgreSQL GIN')
    ax.set_xlabel('Documentos (N)')
    ax.set_ylabel('Tiempo de construcción (s)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'exp6_construccion.png')
    plt.close(fig)

    # 2. Tamano en disco
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    mb = 1024 * 1024
    ax.plot(ns, [int(r['postings_bytes']) / mb for r in build], '--s', color='0.35', label='postings.inv')
    ax.plot(ns, [int(r['vocab_bytes']) / mb for r in build], '-.d', color='0.5', label='vocab.dic')
    ax.plot(ns, [int(r['docs_bytes']) / mb for r in build], ':v', color='0.6', label='docs.dat')
    ax.plot(ns, [int(r['index_bytes']) / mb for r in build], '-o', color='0.05', label='Total índice propio')
    if has_pg:
        ax.plot(ns, [int(r['pg_index_bytes']) / mb for r in build], ':^', color='0.25', label='PostgreSQL GIN')
    ax.set_xlabel('Documentos (N)')
    ax.set_ylabel('Tamaño en disco (MiB)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / 'exp6_tamano.png')
    plt.close(fig)

    def value(n, system, kind, k, field):
        for r in summary:
            if (int(r['n']) == n and r['system'] == system and r['kind'] == kind and int(r['k']) == k):
                return num(r[field])
        return None

    # 3 y 4. Barras por categoria en N maximo
    for field, ylabel, name in (
        ('latency_mean_ms', 'Latencia media (ms, escala log)', 'exp6_latencia_categorias.png'),
        ('reads_mean', 'Bloques leídos (media, escala log)', 'exp6_lecturas_categorias.png'),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        width = 0.27
        present = [s for s in SYSTEMS if any(r['system'] == s[0] for r in summary)]
        for offset, (system, text, color, hatch, _, _) in enumerate(present):
            xs, ys = [], []
            for i, (kind, k) in enumerate(CATEGORY_ORDER):
                v = value(nmax, system, kind, k, field)
                if v is not None and v > 0:
                    xs.append(i + (offset - (len(present) - 1) / 2) * width)
                    ys.append(v)
            ax.bar(xs, ys, width, color=color, hatch=hatch, edgecolor='0.1', linewidth=0.5, label=text)
        ax.set_yscale('log')
        ax.set_xticks(range(len(CATEGORY_ORDER)))
        ax.set_xticklabels([label(k, n) for k, n in CATEGORY_ORDER], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f'N = {nmax:,} documentos'.replace(',', '.'), fontsize=9)
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(folder / name)
        plt.close(fig)

    # 5 y 6. Escalabilidad para ranking de 3 terminos y AND de 3 terminos
    for field, ylabel, name in (
        ('latency_mean_ms', 'Latencia media (ms, escala log)', 'exp6_latencia_vs_n.png'),
        ('reads_mean', 'Bloques leídos (media, escala log)', 'exp6_lecturas_vs_n.png'),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for ax, (kind, k) in zip(axes, (('rank', 3), ('and', 3))):
            for system, text, color, _, marker, style in SYSTEMS:
                ys = [value(n, system, kind, k, field) for n in ns]
                pts = [(n, y) for n, y in zip(ns, ys) if y]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=style, marker=marker,
                            color='0.1' if system == 'indice' else color if system != 'postgres_gin' else '0.4',
                            label=text)
            ax.set_yscale('log')
            ax.set_xlabel('Documentos (N)')
            ax.set_title(f'{CATEGORY_LABEL[kind]} de {k} términos', fontsize=9)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / name)
        plt.close(fig)

    write_tables(folder, build, summary, nmax, has_pg, value, manifest)
    print('Graficas y tablas en', folder)


def fmt(x: float | None, digits: int = 2) -> str:
    if x is None:
        return '--'
    return f'{x:,.{digits}f}'.replace(',', '\\,')


def write_tables(folder, build, summary, nmax, has_pg, value, manifest) -> None:
    lines = [
        '% Generado por benchmarks/text/plot_experiment6.py desde build.csv',
        '\\begin{tabular}{rrrrrrr' + ('rr' if has_pg else '') + '}',
        '\\hline',
        'N & Términos & Postings & Constr. (s) & Pág. escritas & Índice (MiB) & Heap (MiB)'
        + (' & GIN (s) & GIN (MiB)' if has_pg else '') + ' \\\\',
        '\\hline',
    ]
    mb = 1024 * 1024
    for r in build:
        row = [f"{int(r['n']):,}".replace(',', '\\,'), f"{int(r['terms']):,}".replace(',', '\\,'),
               f"{int(r['postings']):,}".replace(',', '\\,'), fmt(float(r['build_ms']) / 1000),
               f"{int(r['build_pages_written']):,}".replace(',', '\\,'),
               fmt(int(r['index_bytes']) / mb), fmt(int(r['heap_bytes']) / mb)]
        if has_pg:
            row += [fmt(float(r['pg_build_ms']) / 1000), fmt(int(r['pg_index_bytes']) / mb)]
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\hline', '\\end{tabular}']
    (folder / 'tabla_construccion.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    head = 'Consulta & \\multicolumn{2}{c}{Índice propio} & \\multicolumn{2}{c}{Full Text Scan}'
    cols = 'lrrrr'
    if has_pg:
        head += ' & \\multicolumn{2}{c}{PostgreSQL GIN}'
        cols += 'rr'
    lines = [
        f'% Generado desde summary.csv, N = {nmax}',
        f'\\begin{{tabular}}{{{cols}}}', '\\hline', head + ' \\\\',
        ' & ms & bloques & ms & bloques' + (' & ms & bloques' if has_pg else '') + ' \\\\', '\\hline',
    ]
    for kind, k in CATEGORY_ORDER:
        row = [f'{CATEGORY_LABEL[kind]} ({k} t.)']
        for system in ('indice', 'full_scan') + (('postgres_gin',) if has_pg else ()):
            row += [fmt(value(nmax, system, kind, k, 'latency_mean_ms')),
                    fmt(value(nmax, system, kind, k, 'reads_mean'), 1)]
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\hline', '\\end{tabular}']
    (folder / 'tabla_consultas.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
