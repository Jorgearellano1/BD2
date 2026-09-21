from __future__ import annotations

from dataclasses import replace

from parser.ast import Program

from engine.catalog import Catalog
from engine.executor import (
    ExecutionContext,
    ExecutionOutput,
    Executor,
    StatementExplain,
    StatementResult,
)
from engine.planner import PlannerVisitor
from parser.semantic_visitor import SemanticVisitor


class Session:
    """
    Ejecuta un Program completo con estado persistente entre sentencias.

    Primero valida semanticamente todo el Program. Luego planifica y
    ejecuta sentencia por sentencia sobre un mismo ExecutionContext, de
    modo que el Planner siempre observa el catalogo ya actualizado por
    las sentencias anteriores (tablas creadas desde CSV, autoindices y
    indices declarados en el mismo Program).

    Errores:
        Nunca lanza excepciones de sentencia. Un error semantico se
        reporta con success=False y failed_statement_id sin ejecutar
        nada. Un error de planificacion o ejecucion detiene el Program,
        deja confirmadas las sentencias anteriores y se reporta en el
        ExecutionOutput (success, error, error_type, failed_statement_id).
    """

    def __init__(
        self,
        catalog: Catalog,
        page_size: int,
        sort_buffer_size: int = 128 * 1024 * 1024,
        level_memory_limit: int = 64 * 1024 * 1024,
    ) -> None:
        self.catalog = catalog
        self.page_size = page_size
        self.sort_buffer_size = sort_buffer_size
        self.level_memory_limit = level_memory_limit

    def execute(self, program: Program) -> ExecutionOutput:
        try:
            SemanticVisitor(self.catalog).visit_program(program)
        except ValueError as exc:
            return self._failure(
                [],
                [],
                str(exc),
                getattr(exc, "statement_id", None),
                type(exc).__name__,
            )

        context = ExecutionContext(
            catalog=self.catalog,
            page_size=self.page_size,
            sort_buffer_size=self.sort_buffer_size,
            level_memory_limit=self.level_memory_limit,
        )
        executor = Executor(context)

        explains: list[StatementExplain] = []
        results: list[StatementResult] = []

        try:
            for statement_id, statement in enumerate(program.statements, start=1):
                try:
                    plan = PlannerVisitor(self.catalog).visit_program(
                        Program([statement])
                    )
                    info = replace(
                        plan.statement_infos[0],
                        statement_id=statement_id,
                    )
                    plan_node = plan.statements[0]
                except Exception as exc:
                    info = PlannerVisitor.statement_info(statement_id, statement)
                    explain = StatementExplain(
                        statement_id,
                        info.statement_type,
                        None,
                        False,
                        str(exc),
                        type(exc).__name__,
                    )
                    explains.append(explain)
                    return self._failure(
                        explains,
                        results,
                        explain.error,
                        statement_id,
                        explain.error_type,
                    )

                explain, result = executor.execute_statement(plan_node, info)
                explains.append(explain)

                if result is None:
                    return self._failure(
                        explains,
                        results,
                        explain.error,
                        statement_id,
                        explain.error_type,
                    )

                results.append(result)

            return ExecutionOutput(explains, results)
        finally:
            executor.close()

    @staticmethod
    def _failure(
        explains: list[StatementExplain],
        results: list[StatementResult],
        error: str | None,
        failed_statement_id: int | None,
        error_type: str | None,
    ) -> ExecutionOutput:
        return ExecutionOutput(
            explains,
            results,
            success=False,
            error=error,
            error_type=error_type,
            failed_statement_id=failed_statement_id,
        )
