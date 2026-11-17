const byId = (id) => document.getElementById(id);
const format = new Intl.NumberFormat('es-PE');
const decimal = new Intl.NumberFormat('es-PE', { maximumFractionDigits: 3 });
const state = { tables: [], selected: null, result: null, busy: false, connected: false };
const editor = byId('sql');
const highlight = byId('highlight');

function escapeHtml(text) {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

function highlightSql() {
  const source = editor.value;
  const tokens = /'(?:''|[^'])*'|\b(?:SELECT|FROM|WHERE|AND|OR|NOT|CREATE|TABLE|INDEX|INSERT|INTO|VALUES|DELETE|USING|HEAP|SEQUENTIAL|BTREE|HASH|ON|INT|FLOAT|CHAR|PRIMARY|KEY|UNIQUE|CSV|DELIMITER|FORCE|INVERTED|WITH|MATCH|ORDER|BY|LIMIT|ASC|DESC|AS|COSINE_SCORE)\b|\b\d+(?:\.\d+)?\b/gi;
  let result = '';
  let position = 0;
  for (const match of source.matchAll(tokens)) {
    result += escapeHtml(source.slice(position, match.index));
    const kind = match[0].startsWith("'") ? 'string' : /^\d/.test(match[0]) ? 'number' : 'keyword';
    result += `<span class="token-${kind}">${escapeHtml(match[0])}</span>`;
    position = match.index + match[0].length;
  }
  highlight.firstElementChild.innerHTML = result + escapeHtml(source.slice(position)) + '\n';
  syncScroll();
}

function syncScroll() {
  highlight.scrollTop = editor.scrollTop;
  highlight.scrollLeft = editor.scrollLeft;
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof data.detail === 'string' ? data.detail : Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(' · ') : 'No se pudo completar la operación.';
    throw new Error(message);
  }
  return data;
}

function controls() {
  byId('execute').disabled = state.busy || !state.connected || !editor.value.trim();
  byId('execute').textContent = state.busy ? 'Ejecutando…' : 'Ejecutar';
  byId('refresh').disabled = state.busy;
  editor.readOnly = state.busy;
  const table = state.tables.find((item) => item.name === state.selected);
  byId('reorganize').disabled = state.busy || !state.connected || table?.storage !== 'SEQUENTIAL';
  byId('previous').disabled = state.busy || !state.result || state.result.page <= 1;
  byId('next').disabled = state.busy || !state.result || state.result.page >= state.result.total_pages;
  byId('result-area').setAttribute('aria-busy', String(state.busy));
  document.querySelectorAll('.table-item').forEach((button) => { button.disabled = state.busy; });
}

function textElement(tag, text, className) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function renderTables() {
  const list = byId('table-list');
  list.replaceChildren();
  list.setAttribute('aria-busy', 'false');
  if (!state.tables.length) list.append(textElement('p', 'Crea una tabla con CREATE TABLE para comenzar.', 'empty'));
  for (const table of state.tables) {
    const button = textElement('button', '', 'table-item');
    button.type = 'button';
    button.setAttribute('aria-pressed', String(table.name === state.selected));
    button.append(textElement('span', table.name, 'table-name'),
      textElement('span', table.storage, 'table-kind'),
      textElement('span', `${format.format(table.records)} registros`, 'table-records'));
    button.addEventListener('click', () => {
      state.selected = table.name;
      editor.value = `SELECT * FROM ${table.name};`;
      highlightSql();
      renderTables();
      renderSchema();
      controls();
    });
    list.append(button);
  }
}

function renderSchema() {
  const schema = byId('schema');
  const table = state.tables.find((item) => item.name === state.selected);
  schema.hidden = !table;
  schema.replaceChildren();
  if (!table) return;
  schema.append(textElement('h3', 'Columnas'));
  const columns = document.createElement('ul');
  for (const col of table.columns) {
    const item = document.createElement('li');
    const type = col.type === 'CHAR' ? `CHAR(${col.size})` : col.type;
    item.append(textElement('span', col.name, 'column-name'),
      textElement('span', `${type}${col.name === table.primary_key ? ' · PK' : col.unique ? ' · UNIQUE' : ''}`, 'column-type'));
    columns.append(item);
  }
  schema.append(columns, textElement('h3', 'Índices', 'index-heading'));
  if (!table.indexes.length) schema.append(textElement('p', 'Sin índices activos', 'empty'));
  const indexes = document.createElement('ul');
  for (const index of table.indexes) {
    const item = textElement('li', '', 'index-item');
    item.append(textElement('div', index.name), textElement('div', `${index.type === 'BTREE' ? 'B+' : index.type} · ${index.column}`, 'index-type'));
    indexes.append(item);
  }
  schema.append(indexes);
}

async function loadTables() {
  try {
    const data = await request('/api/tables');
    state.tables = data.tables;
    state.connected = true;
    if (!state.tables.some((table) => table.name === state.selected)) state.selected = state.tables[0]?.name ?? null;
    byId('page-size').textContent = `Tamaño de página: ${format.format(data.page_size)} bytes`;
    renderTables();
    renderSchema();
  } catch (error) {
    state.connected = false;
    byId('table-list').replaceChildren(textElement('p', 'No se pudo conectar. Comprueba que el servidor esté encendido y pulsa Actualizar.', 'empty'));
    byId('table-list').setAttribute('aria-busy', 'false');
  }
  controls();
}

function renderExplain(node) {
  const lines = [];
  function visit(item, depth) {
    const detail = item.detail ? ` [${item.detail}]` : '';
    const io = typeof item.pages_read === 'number' ? `  (lecturas ${format.format(item.pages_read)}, escrituras ${format.format(item.pages_written ?? 0)})` : '';
    lines.push(`${'  '.repeat(depth)}${item.node_type.replace(/Plan$/, '')}${detail}${io}`);
    for (const child of item.children) visit(child, depth + 1);
  }
  if (node) visit(node, 0);
  byId('explain').textContent = node ? lines.join('\n') : 'No hay árbol de ejecución disponible para esta operación.';
}

function clearMetrics() {
  renderExplain(null);
  byId('plan').textContent = '—';
  byId('plan-detail').textContent = '';
  for (const id of ['reads', 'writes', 'parse-time', 'execution-time']) byId(id).textContent = '—';
  byId('reads-bar').style.width = '0%';
  byId('writes-bar').style.width = '0%';
}

function renderResult(data) {
  state.result = data;
  const area = byId('result-area');
  area.replaceChildren();
  const message = byId('result-message');
  message.className = 'result-message';
  message.textContent = data.message;
  message.hidden = Boolean(data.columns?.length && data.rows.length);
  if (data.success === false) {
    message.hidden = false;
    message.className = 'result-message error';
    message.textContent = data.error;
  }
  if (data.columns?.length) {
    const table = document.createElement('table');
    const head = table.createTHead().insertRow();
    for (const column of data.columns) {
      const cell = textElement('th', column);
      cell.scope = 'col';
      head.append(cell);
    }
    const body = table.createTBody();
    for (const row of data.rows) {
      const tr = body.insertRow();
      for (const value of row) {
        const cell = tr.insertCell();
        cell.textContent = value == null ? 'NULL' : String(value);
        if (typeof value === 'number') cell.className = 'numeric';
      }
    }
    area.append(table);
  }
  byId('record-count').textContent = data.columns?.length ? `${format.format(data.total_rows)} registros` : `${format.format(data.affected_rows ?? 0)} afectados`;
  byId('page-info').textContent = data.total_rows ? `Página ${format.format(data.page)} de ${format.format(data.total_pages)}` : 'Sin filas';
  byId('plan').textContent = data.plan;
  renderExplain(data.explain);
  const io = data.io_breakdown;
  byId('plan-detail').textContent = [data.table, data.index ? `Índice: ${data.index}` : null,
    data.strategy ? `Estrategia: ${data.strategy}` : null,
    io && (io.index_reads || io.table_reads) ? `Lecturas en índice: ${format.format(io.index_reads)} · en tabla: ${format.format(io.table_reads)}` : null,
    data.statements > 1 ? `Último plan · métricas de ${data.statements} sentencias` : null].filter(Boolean).join(' · ');
  const metrics = data.metrics;
  byId('reads').textContent = format.format(metrics.disk_reads);
  byId('writes').textContent = format.format(metrics.disk_writes);
  byId('parse-time').textContent = `${decimal.format(metrics.parse_ms)} ms`;
  byId('execution-time').textContent = `${decimal.format(metrics.execution_ms)} ms`;
  const maximum = Math.max(metrics.disk_reads, metrics.disk_writes, 1);
  byId('reads-bar').style.width = `${metrics.disk_reads / maximum * 100}%`;
  byId('writes-bar').style.width = `${metrics.disk_writes / maximum * 100}%`;
  area.scrollTop = 0;
  controls();
}

function showError(error) {
  state.result = null;
  byId('result-message').hidden = false;
  byId('result-message').className = 'result-message error';
  byId('result-message').textContent = error.message;
  byId('result-area').replaceChildren();
  byId('record-count').textContent = '—';
  byId('page-info').textContent = 'Sin resultados';
  byId('query-state').textContent = 'No se completó la operación.';
  clearMetrics();
}

async function run(path, body, message) {
  if (state.busy) return;
  state.busy = true;
  controls();
  byId('query-state').textContent = message;
  byId('result-message').hidden = true;
  try {
    const data = await request(path, body);
    if (data.success === false && !data.statements) throw new Error(data.error || 'No se pudo ejecutar la consulta.');
    renderResult(data);
    byId('query-state').textContent = data.success === false ? 'El script se detuvo; se muestran los últimos resultados disponibles.' : 'Operación completada.';
    if (data.table) state.selected = data.table;
  } catch (error) {
    showError(error);
  } finally {
    await loadTables();
    state.busy = false;
    controls();
  }
}

async function changePage(direction) {
  if (state.busy || !state.result) return;
  state.busy = true;
  controls();
  try {
    const result = state.result;
    renderResult(await request(`/api/results/${result.result_id}?page=${result.page + direction}`));
  } catch (error) {
    showError(error);
  } finally {
    state.busy = false;
    controls();
  }
}

function selectResultTab(name) {
  for (const tab of ['results', 'explain']) {
    const active = tab === name;
    byId(`${tab}-tab`).setAttribute('aria-selected', String(active));
    byId(`${tab}-tab`).tabIndex = active ? 0 : -1;
    byId(`${tab}-view`).hidden = !active;
  }
  byId('record-count').hidden = name !== 'results';
}

for (const name of ['results', 'explain']) {
  const tab = byId(`${name}-tab`);
  tab.addEventListener('click', () => selectResultTab(name));
  tab.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'results' : event.key === 'End' ? 'explain' : name === 'results' ? 'explain' : 'results';
    selectResultTab(next);
    byId(`${next}-tab`).focus();
  });
}

editor.addEventListener('input', () => { highlightSql(); controls(); });
editor.addEventListener('scroll', syncScroll);
editor.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    event.preventDefault();
    byId('execute').click();
  }
  if (event.key === 'Tab' && !state.busy) {
    event.preventDefault();
    editor.setRangeText('  ', editor.selectionStart, editor.selectionEnd, 'end');
    highlightSql();
  }
});
byId('execute').addEventListener('click', () => run('/api/query', { sql: editor.value, page_size: 50 }, 'Ejecutando consulta…'));
byId('reorganize').addEventListener('click', () => run('/api/tables/reorganize', { table: state.selected }, 'Reorganizando tabla…'));
byId('refresh').addEventListener('click', loadTables);
byId('previous').addEventListener('click', () => changePage(-1));
byId('next').addEventListener('click', () => changePage(1));
highlightSql();
loadTables();
