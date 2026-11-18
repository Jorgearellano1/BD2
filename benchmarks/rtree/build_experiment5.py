"""Prepara los conjuntos de ciudades para el Experimento 5."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / 'datasets' / 'cities.csv'


def read_cities():
    with SOURCE.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def save_prefix(rows, count, destination):
    columns = ['id', 'longitude', 'latitude']
    with destination.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows[:count], 1):
            writer.writerow({name: row[name] for name in columns})
            if index % 10_000 == 0 or index == count:
                print(f'  {destination.name}: {index:,}/{count:,} ciudades')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='prepara N=1 000 y 5 000')
    args = parser.parse_args()
    sizes = [1_000, 5_000] if args.quick else [10_000, 50_000, 100_000]
    rows = read_cities()
    if len(rows) < max(sizes):
        raise SystemExit(f'El CSV tiene {len(rows):,} filas; se requieren {max(sizes):,}')
    profile = 'quick' if args.quick else 'full'
    output = ROOT / 'benchmarks' / 'temp_runs' / 'experiment_5' / profile
    output.mkdir(parents=True, exist_ok=True)
    print(f'Preparando datos ({profile}) desde {SOURCE.name}')
    for size in sizes:
        save_prefix(rows, size, output / f'cities_{size}.csv')
    print(f'Datos listos en: {output}')


if __name__ == '__main__':
    main()
