
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
CSV_PATH = ROOT_DIR / "data" / "datasets" / "cities.csv"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from engine.catalog import Catalog
from engine.session import Session
from parser.scanner import Scanner
from parser.parser import Parser

PAGE_SIZE = 4096

TABLE_NAME = "cities"
INDEX_NAME = "idx_cities_spatial"

MIN_LATITUDE = 25.0
MAX_LATITUDE = 35.0
MIN_LONGITUDE = 60.0
MAX_LONGITUDE = 75.0


def execute_sql(session, sql):
    print(f"\n[SQL]\n{sql.strip()}")

    program = Parser(Scanner(sql)).parse_program()
    output = session.execute(program)

    if not output.success:
        raise RuntimeError(
            f"Error SQL: {output.error_type}\n"
            f"Detalle: {output.error}"
        )

    return output


def execute_query(session, sql):
    start = time.perf_counter()

    output = execute_sql(session, sql)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if not output.results:
        raise RuntimeError("La consulta no devolvió un resultado.")

    rows = output.results[-1].rows

    plans = [
        explain.root
        for explain in output.explains
        if explain.root is not None
    ]

    return rows, elapsed_ms, plans


QUERY_SQL = f"""
SELECT id, name, latitude, longitude
FROM {TABLE_NAME}
WHERE latitude >= {MIN_LATITUDE}
  AND latitude <= {MAX_LATITUDE}
  AND longitude >= {MIN_LONGITUDE}
  AND longitude <= {MAX_LONGITUDE};
"""


def show_result(title, rows, elapsed_ms, plans):
    print(f"\n{title}")
    print("Registros encontrados:", len(rows))
    print(f"Tiempo: {elapsed_ms:.4f} ms")

    print("\nPlanes devueltos:")

    for plan in plans:
        print(plan)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV: {CSV_PATH}"
        )

    with TemporaryDirectory(prefix="rtree_sql_test_") as tmp:
        catalog = Catalog(
            str(Path(tmp) / "catalog.json"),
            str(Path(tmp) / "storage")
        )

        session = Session(
            catalog=catalog,
            page_size=PAGE_SIZE
        )

        create_table_sql = f"""
        CREATE TABLE {TABLE_NAME} (
            id INT PRIMARY KEY,
            name CHAR(100),
            state_id INT,
            state_code CHAR(10),
            state_name CHAR(100),
            country_id INT,
            country_code CHAR(10),
            country_name CHAR(100),
            latitude FLOAT,
            longitude FLOAT,
            wikiDataId CHAR(30)
        )
        USING HEAP
        FROM CSV ('{CSV_PATH.as_posix()}');
        """

        print("\nCREANDO TABLA")
        execute_sql(session, create_table_sql)

        rows_scan, time_scan, plans_scan = execute_query(
            session,
            QUERY_SQL
        )

        show_result(
            "CONSULTA SIN RTREE",
            rows_scan,
            time_scan,
            plans_scan
        )

        create_index_sql = f"""
        CREATE INDEX {INDEX_NAME}
        ON {TABLE_NAME} (latitude, longitude)
        USING RTREE;
        """

        print("\nCREANDO RTREE")
        execute_sql(session, create_index_sql)

        rows_rtree, time_rtree, plans_rtree = execute_query(
            session,
            QUERY_SQL
        )

        show_result(
            "CONSULTA DESPUÉS DE CREAR RTREE",
            rows_rtree,
            time_rtree,
            plans_rtree
        )

        ids_scan = {row[0] for row in rows_scan}
        ids_rtree = {row[0] for row in rows_rtree}

        missing = ids_scan - ids_rtree
        extra = ids_rtree - ids_scan

        print("\nVALIDACIÓN")
        print("Resultados secuenciales:", len(ids_scan))
        print("Resultados posteriores:", len(ids_rtree))
        print("IDs faltantes:", len(missing))
        print("IDs adicionales:", len(extra))

        if ids_scan == ids_rtree:
            print("RESULTADO: LOS RESULTADOS COINCIDEN")
        else:
            print("RESULTADO: EXISTEN DIFERENCIAS")

        print("\nCOMPARACIÓN DE TIEMPOS")
        print(f"Primera consulta: {time_scan:.4f} ms")
        print(f"Segunda consulta: {time_rtree:.4f} ms")

        if time_rtree > 0:
            print(f"Speedup: {time_scan / time_rtree:.2f}x")

        print("\nVALIDACIÓN DEL PLAN")

        plan_text = "\n".join(
            str(plan) for plan in plans_rtree
        )

        if "IndexScanPlan" in plan_text:
            print("El plan contiene IndexScanPlan.")
        elif "SeqScanPlan" in plan_text:
            print("El Planner continúa utilizando SeqScanPlan.")
            print("El R-Tree todavía no está siendo seleccionado por el Planner.")
        else:
            print("Revisar el plan devuelto por el motor.")

        if ids_scan != ids_rtree:
            raise AssertionError(
                "Las consultas devolvieron resultados diferentes."
            )

        print("\nPRUEBA FINALIZADA")


if __name__ == "__main__":
    main()
