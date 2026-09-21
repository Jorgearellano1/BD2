from __future__ import annotations

from engine.catalog import Catalog
from engine.executor import (
    ExecutionOutput,
    NodeMetrics,
    OperationMetrics,
    StatementExplain,
    StatementResult,
)
from engine.metadata import IndexMetadata, TableEntry, TableMetadata


_STATEMENT_MESSAGES = {
    'CREATE_TABLE': 'Tabla creada correctamente',
    'CREATE_INDEX': 'Indice creado correctamente',
    'ALTER_TABLE': 'UNIQUE agregada correctamente',
    'INSERT': 'INSERT ejecutado correctamente',
    'DELETE': 'DELETE ejecutado correctamente',
}


def operation_to_dict(operation: OperationMetrics) -> dict:
    return {
        'resource_type': operation.resource_type,
        'resource_name': operation.resource_name,
        'operation': operation.operation,
        'elapsed_ms': operation.elapsed_ms,
        'pages_read': operation.pages_read,
        'pages_written': operation.pages_written,
        'entries_affected': operation.entries_affected,
    }


def node_to_dict(node: NodeMetrics | None) -> dict | None:
    if node is None:
        return None
    return {
        'node_id': node.node_id,
        'node_type': node.node_type,
        'elapsed_ms': node.elapsed_ms,
        'self_time_ms': node.self_time_ms,
        'rows_out': node.rows_out,
        'pages_read': node.pages_read,
        'pages_written': node.pages_written,
        'self_pages_read': node.self_pages_read,
        'self_pages_written': node.self_pages_written,
        'operations': [operation_to_dict(item) for item in node.operations],
        'children': [node_to_dict(child) for child in node.children],
    }


def statement_to_dict(
    explain: StatementExplain,
    result: StatementResult | None,
) -> dict:
    payload = {
        'statement_id': explain.statement_id,
        'statement_type': explain.statement_type,
        'success': explain.success,
        'columns': [],
        'rows': [],
        'affected_rows': None,
        'message': None,
        'error': explain.error,
        'error_type': explain.error_type,
        'explain': node_to_dict(explain.root),
    }

    if not explain.success or result is None:
        return payload

    payload['columns'] = list(result.columns)
    payload['rows'] = [list(row) for row in result.rows]
    payload['affected_rows'] = result.affected_rows
    payload['message'] = result.message or _STATEMENT_MESSAGES.get(
        explain.statement_type
    )
    return payload


def output_to_dict(output: ExecutionOutput) -> dict:
    results = {result.statement_id: result for result in output.results}
    statements = [
        statement_to_dict(explain, results.get(explain.statement_id))
        for explain in output.explains
    ]
    return {
        'success': output.success,
        'statements': statements,
        'error': output.error,
        'error_type': output.error_type,
        'failed_statement_id': output.failed_statement_id,
    }


def column_to_dict(metadata: TableMetadata, column: dict) -> dict:
    return {
        'name': column['name'],
        'type': column['type'],
        'size': column['size'],
        'unique': metadata.is_unique_column(column['name']),
    }


def table_to_dict(entry: TableEntry) -> dict:
    return {
        'name': entry.name,
        'storage': entry.storage.value,
        'primary_key': entry.metadata.primary_key,
        'columns': [
            column_to_dict(entry.metadata, column)
            for column in entry.metadata.columns
        ],
    }


def index_to_dict(index: IndexMetadata) -> dict:
    return {
        'name': index.name,
        'table': index.table,
        'column': index.column,
        'type': index.kind.value,
    }


def catalog_tables(catalog: Catalog) -> dict:
    return {'tables': [table_to_dict(entry) for entry in catalog.tables()]}


def catalog_indexes(catalog: Catalog) -> dict:
    return {'indexes': [index_to_dict(index) for index in catalog.indexes()]}


def frontend_output(output: ExecutionOutput, program, parse_ms: float, execution_ms: float) -> dict:
    successful = [item for item in output.explains if item.success]
    result = output.results[-1] if output.results else None
    explain = successful[-1] if successful else None
    statement = program.statements[result.statement_id - 1] if result else None
    from engine.planner import PlannerVisitor
    info = PlannerVisitor.statement_info(result.statement_id, statement) if result else None

    def access_nodes(node):
        if node is None:
            return []
        children = [item for child in node.children for item in access_nodes(child)]
        return children or [node.node_type.removesuffix('Plan')]

    roots = [item.root for item in output.explains if item.root is not None]
    return {
        'success': output.success, 'error': output.error,
        'error_type': output.error_type, 'failed_statement_id': output.failed_statement_id,
        'columns': list(result.columns) if result else [],
        'rows': [list(row) for row in result.rows] if result else [],
        'affected_rows': (result.affected_rows or 0) if result else 0,
        'message': (result.message or _STATEMENT_MESSAGES.get(result.statement_type, 'Consulta completada.')) if result else '',
        'table': info.table if info else None,
        'plan': ' + '.join(dict.fromkeys(access_nodes(explain.root))) if explain else '—',
        'statements': len(output.explains),
        'metrics': {
            'disk_reads': sum(root.pages_read for root in roots),
            'disk_writes': sum(root.pages_written for root in roots),
            'parse_ms': parse_ms, 'execution_ms': execution_ms,
        },
    }
