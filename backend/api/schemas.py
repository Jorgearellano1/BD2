from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class QueryRequest(BaseModel):
    sql: str = Field(min_length=1)
    page_size: int = Field(default=50, ge=1, le=500)


class ColumnResponse(BaseModel):
    name: str
    type: str
    size: int
    unique: bool


class ReorganizeRequest(BaseModel):
    table: str


class RecommendRequest(BaseModel):
    image_base64: str | None = None
    description: str | None = None
    max_price: float | None = Field(default=None, ge=0)
    lat: float | None = None
    lon: float | None = None
    radius_km: float | None = Field(default=None, gt=0)
    limit: int = Field(default=5, ge=1, le=20)


class TableResponse(BaseModel):
    name: str
    storage: str
    primary_key: str | None
    columns: list[ColumnResponse]
    records: int
    indexes: list[dict]


class TablesResponse(BaseModel):
    tables: list[TableResponse]
    page_size: int


class IndexResponse(BaseModel):
    name: str
    table: str
    column: str
    type: str
    spatial_columns: list[str] | None = None
    options: dict | None = None


class IndexesResponse(BaseModel):
    indexes: list[IndexResponse]


class OperationMetricsResponse(BaseModel):
    resource_type: str
    resource_name: str
    operation: str
    elapsed_ms: float
    pages_read: int
    pages_written: int
    entries_affected: int


class NodeMetricsResponse(BaseModel):
    node_id: int
    node_type: str
    elapsed_ms: float
    self_time_ms: float
    rows_out: int
    pages_read: int
    pages_written: int
    self_pages_read: int
    self_pages_written: int
    operations: list[OperationMetricsResponse]
    children: list['NodeMetricsResponse']
    detail: str | None = None


class StatementExplainResponse(BaseModel):
    statement_id: int
    statement_type: str
    success: bool
    error: str | None = None
    table: str | None = None
    plan: str = '—'
    explain: NodeMetricsResponse | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    io_breakdown: dict[str, int] | None = None
    strategy: str | None = None


class StatementResponse(BaseModel):
    statement_id: int
    statement_type: str
    success: bool
    columns: list[str]
    rows: list[list[object]]
    affected_rows: int | None
    message: str | None
    error: str | None = None
    error_type: str | None = None
    explain: NodeMetricsResponse | None = None


class QueryResponse(BaseModel):
    success: bool
    error: str | None = None
    error_type: str | None = None
    failed_statement_id: int | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    affected_rows: int = 0
    message: str = ''
    table: str | None = None
    plan: str = '—'
    explain: NodeMetricsResponse | None = None
    explains: list[StatementExplainResponse] = Field(default_factory=list)
    index: str | None = None
    statements: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    io_breakdown: dict[str, int] | None = None
    strategy: str | None = None
    result_id: str | None = None
    page: int = 1
    total_pages: int = 1
    total_rows: int = 0
