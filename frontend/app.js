const byId = (id) => document.getElementById(id);
const format = new Intl.NumberFormat('es-PE');
const decimal = new Intl.NumberFormat('es-PE', { maximumFractionDigits: 3 });
const state = { tables: [], selected: null, result: null, busy: false, connected: false };
const editor = byId('sql');
const highlight = byId('highlight');
const EXAMPLES = [
  ['P1 · Rango por clave', 'SELECT * FROM estudiantes\nWHERE Student_ID >= 100 AND Student_ID <= 150;'],
  ['P2 · Espacial: crear tabla y R-Tree', "CREATE TABLE cities (\n  id INT PRIMARY KEY, name CHAR(100), state_id INT, state_code CHAR(10),\n  state_name CHAR(100), country_id INT, country_code CHAR(10), country_name CHAR(100),\n  latitude FLOAT, longitude FLOAT, wikiDataId CHAR(30)\n) USING HEAP FROM CSV ('data/datasets/cities.csv');\nCREATE INDEX idx_cities_spatial ON cities (latitude, longitude) USING RTREE;"],
  ['P2 · Espacial: consulta por rectángulo', 'SELECT * FROM cities\nWHERE latitude >= 25 AND latitude <= 35\n  AND longitude >= 60 AND longitude <= 75;'],
  ['P2 · Texto: crear tabla e índice invertido', "CREATE TABLE docs (id INT PRIMARY KEY, libro INT, titulo CHAR(80), cuerpo CHAR(1000))\n  FROM CSV ('data/datasets/corpus_gutenberg_es.csv');\nCREATE INDEX idx_cuerpo ON docs (cuerpo) USING INVERTED WITH (scoring = 'TFIDF_COSINE');"],
  ['P2 · Texto: consulta booleana', "SELECT id, titulo FROM docs\nWHERE cuerpo MATCH 'caballero AND andante AND NOT escudero';"],
  ['P2 · Texto: Top-K por coseno', "SELECT id, titulo, COSINE_SCORE() AS relevancia FROM docs\nWHERE cuerpo MATCH 'amor guerra'\nORDER BY relevancia DESC LIMIT 10;"],
  ['P2 · Texto con filtro relacional', "SELECT id, COSINE_SCORE() AS r FROM docs\nWHERE cuerpo MATCH 'amor guerra' AND id <= 2000\nORDER BY r DESC LIMIT 5;"],
  ['P2 · Vectorial: crear tabla e índice IVF', "CREATE TABLE demo_vec (\n  id INT PRIMARY KEY, categoria CHAR(20), embedding VECTOR(4)\n) USING HEAP;\nINSERT INTO demo_vec VALUES (1, 'a', '[1.0, 0.0, 0.0, 0.0]');\nINSERT INTO demo_vec VALUES (2, 'a', '[0.9, 0.1, 0.0, 0.0]');\nINSERT INTO demo_vec VALUES (3, 'b', '[0.0, 1.0, 0.0, 0.0]');\nINSERT INTO demo_vec VALUES (4, 'b', '[0.0, 0.0, 1.0, 0.0]');\nCREATE INDEX idx_demo_vec ON demo_vec(embedding) USING IVF WITH (lists = 2, metric = 'COSINE');\nCREATE INDEX idx_demo_cat ON demo_vec(categoria) USING BTREE;"],
  ['P2 · Vectorial: similitud coseno Top-K', "SET ivf.probes = 2;\nSELECT id, categoria, embedding <=> '[1.0, 0.0, 0.0, 0.0]' AS distancia\nFROM demo_vec\nORDER BY distancia ASC\nLIMIT 2;"],
  ['P2 · Vectorial con filtro relacional', "SET ivf.probes = 2;\nSELECT id, categoria, embedding <=> '[1.0, 0.0, 0.0, 0.0]' AS distancia\nFROM demo_vec\nWHERE categoria = 'a'\nORDER BY distancia ASC\nLIMIT 2;"],
  ['P2 · Vectorial: beans desde CSV', "CREATE TABLE beans (\n  id INT PRIMARY KEY,\n  clase CHAR(20),\n  v VECTOR(16)\n) USING HEAP\n  FROM CSV ('data/datasets/dry_beans_demo.csv', DELIMITER = ';');\nCREATE INDEX idx_beans ON beans(v) USING IVF WITH (lists = 14, metric = 'L2');"],
  ['P2 · Vectorial: consulta beans (L2)', "SET ivf.probes = 5;\nSELECT id, clase, v <-> '[95737.000000,1284.958000,431.620041,283.315831,1.523459,0.754412,97477.000000,349.136269,0.765284,0.982150,0.728638,0.808897,0.004508,0.001191,0.654315,0.996822]' AS dist\nFROM beans\nORDER BY dist ASC\nLIMIT 5;"],
  ['P2 · BoVW: crear tabla e índice invertido', "CREATE TABLE imagenes_bovw (\n  id INT PRIMARY KEY,\n  categoria CHAR(20),\n  palabras CHAR(1000)\n) USING HEAP\n  FROM CSV ('data/datasets/multimedia/bovw.csv', DELIMITER = ';');\nCREATE INDEX idx_bovw ON imagenes_bovw(palabras) USING INVERTED;"],
  ['P2 · BoVW: consulta Top-K', "SELECT id, categoria, COSINE_SCORE() AS score\nFROM imagenes_bovw\nWHERE palabras MATCH 'v0003 v0003 v0003 v0059 v0059 v0059 v0006 v0006 v0006 v0030 v0030 v0030 v0032 v0032 v0032 v0034 v0034 v0034 v0044 v0044 v0044 v0005 v0005 v0005 v0009 v0009 v0009 v0020 v0020 v0020 v0021 v0021 v0021 v0040 v0040 v0040 v0048 v0048 v0048 v0053 v0053 v0053 v0062 v0062 v0062 v0000 v0000 v0000 v0008 v0008 v0008 v0010 v0010 v0010 v0015 v0015 v0015 v0018 v0018 v0018'\nORDER BY score DESC\nLIMIT 5;"],
];
const INDEX_LABELS = { BTREE: 'B+', HASH: 'Hash', RTREE: 'R-Tree', INVERTED: 'Invertido', IVF: 'IVF' };

function escapeHtml(text) {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

function highlightSql() {
  const source = editor.value;
  const tokens = /'(?:''|[^'])*'|\b(?:SELECT|FROM|WHERE|AND|OR|NOT|CREATE|TABLE|INDEX|INSERT|INTO|VALUES|DELETE|USING|HEAP|SEQUENTIAL|BTREE|HASH|RTREE|ON|INT|FLOAT|DOUBLE|CHAR|VECTOR|PRIMARY|KEY|UNIQUE|CSV|DELIMITER|FORCE|INVERTED|IVF|SET|WITH|MATCH|ORDER|BY|LIMIT|ASC|DESC|AS|COSINE_SCORE)\b|\b\d+(?:\.\d+)?\b/gi;
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
  byId('examples').disabled = state.busy;
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
    item.append(textElement('div', index.name), textElement('div', `${INDEX_LABELS[index.type] ?? index.type} · ${(index.spatial_columns ?? [index.column]).join(', ')}`, 'index-type'));
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

function explainTree(node) {
  const lines = [];
  function visit(item, depth) {
    const detail = item.detail ? ` [${item.detail}]` : '';
    const io = typeof item.pages_read === 'number' ? `  (lecturas ${format.format(item.pages_read)}, escrituras ${format.format(item.pages_written ?? 0)})` : '';
    lines.push(`${'  '.repeat(depth)}${item.node_type.replace(/Plan$/, '')}${detail}${io}`);
    for (const child of item.children) visit(child, depth + 1);
  }
  if (node) visit(node, 0);
  return lines;
}

function renderExplain(data) {
  const entries = data?.explains?.length
    ? data.explains
    : data?.explain
      ? [{ statement_id: 1, statement_type: 'Consulta', success: true, explain: data.explain, error: null }]
      : [];

  if (!entries.length) {
    byId('explain').textContent = 'No hay árbol de ejecución disponible para esta operación.';
    return;
  }

  const multiple = entries.length > 1;
  const blocks = [];

  for (const entry of entries) {
    if (multiple) {
      const status = entry.success === false ? `ERROR: ${entry.error ?? 'sin detalle'}` : 'OK';
      blocks.push(`Sentencia ${entry.statement_id} · ${entry.statement_type} · ${status}`);
      const summary = [entry.plan && `Plan: ${entry.plan}`, entry.strategy && `Estrategia: ${entry.strategy}`].filter(Boolean);
      if (summary.length) blocks.push(`  ${summary.join(' · ')}`);
    }

    if (entry.explain) {
      const tree = explainTree(entry.explain);
      blocks.push(...tree.map((line) => (multiple ? `  ${line}` : line)));
    } else {
      blocks.push('  (sin árbol de ejecución)');
    }

    if (multiple) blocks.push('');
  }

  byId('explain').textContent = blocks.join('\n').trimEnd();
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
        cell.textContent = value == null ? 'NULL' : Array.isArray(value) ? `[${value.join(', ')}]` : String(value);
        if (typeof value === 'number') cell.className = 'numeric';
      }
    }
    area.append(table);
  }
  byId('record-count').textContent = data.columns?.length ? `${format.format(data.total_rows)} registros` : `${format.format(data.affected_rows ?? 0)} afectados`;
  byId('page-info').textContent = data.total_rows ? `Página ${format.format(data.page)} de ${format.format(data.total_pages)}` : 'Sin filas';
  byId('plan').textContent = data.plan;
  renderExplain(data);
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
for (const [label, sql] of EXAMPLES) byId('examples').append(new Option(label, sql));
byId('examples').addEventListener('change', (event) => {
  if (!event.target.value) return;
  editor.value = event.target.value;
  event.target.value = '';
  highlightSql();
  controls();
  editor.focus();
});
byId('previous').addEventListener('click', () => changePage(-1));
byId('next').addEventListener('click', () => changePage(1));

const workspace = document.querySelector('.workspace');
const resizer = byId('editor-resizer');
const EDITOR_MIN_HEIGHT = 190;
const EDITOR_DEFAULT_HEIGHT = 245;
let resizeStart = null;

function editorPanelHeight() {
  return document.querySelector('.editor-panel')?.getBoundingClientRect().height ?? EDITOR_DEFAULT_HEIGHT;
}

function setEditorHeight(value, persist = true) {
  const bounds = workspace.getBoundingClientRect();
  const maximum = Math.max(EDITOR_MIN_HEIGHT, bounds.height - 300);
  const height = Math.round(Math.min(Math.max(value, EDITOR_MIN_HEIGHT), maximum));
  workspace.style.setProperty('--editor-height', `${height}px`);
  resizer.setAttribute('aria-valuenow', String(height));
  resizer.setAttribute('aria-valuemax', String(Math.round(maximum)));
  if (persist) {
    try {
      localStorage.setItem('bd2-editor-height', String(height));
    } catch (error) {
      // Sin almacenamiento disponible: el ajuste vale solo para esta sesión.
    }
  }
}

try {
  const savedHeight = Number(localStorage.getItem('bd2-editor-height'));
  if (savedHeight) setEditorHeight(savedHeight, false);
} catch (error) {
  // Sin almacenamiento disponible.
}

resizer.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  resizeStart = { y: event.clientY, height: editorPanelHeight() };
  resizer.setPointerCapture(event.pointerId);
  document.body.classList.add('resizing');
  event.preventDefault();
});

resizer.addEventListener('pointermove', (event) => {
  if (!resizeStart) return;
  setEditorHeight(resizeStart.height + event.clientY - resizeStart.y);
});

function finishResize(event) {
  if (!resizeStart) return;
  resizeStart = null;
  if (resizer.hasPointerCapture(event.pointerId)) resizer.releasePointerCapture(event.pointerId);
  document.body.classList.remove('resizing');
}

resizer.addEventListener('pointerup', finishResize);
resizer.addEventListener('pointercancel', finishResize);
resizer.addEventListener('dblclick', () => setEditorHeight(EDITOR_DEFAULT_HEIGHT));
resizer.addEventListener('keydown', (event) => {
  if (event.key === 'Home') {
    event.preventDefault();
    setEditorHeight(EDITOR_DEFAULT_HEIGHT);
    return;
  }
  if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
  event.preventDefault();
  setEditorHeight(editorPanelHeight() + (event.key === 'ArrowDown' ? 24 : -24));
});

window.addEventListener('resize', () => {
  if (window.matchMedia('(max-width: 850px)').matches) return;
  setEditorHeight(editorPanelHeight(), false);
});

highlightSql();
loadTables();
