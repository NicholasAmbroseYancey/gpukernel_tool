const expressionInput = document.getElementById('expression');
const compileBtn = document.getElementById('compileBtn');
const optimizeBtn = document.getElementById('optimizeBtn');
const useLlmEl = document.getElementById('useLlm');
const unitBtn = document.getElementById('unitBtn');
const kernelBtn = document.getElementById('kernelBtn');
const copyBtn = document.getElementById('copyBtn');
const chipsEl = document.getElementById('chips');
const statusEl = document.getElementById('status');
const kernelOutputEl = document.getElementById('kernelOutput');
const testOutputEl = document.getElementById('testOutput');
const testBadgeEl = document.getElementById('testBadge');

const resultsPanel = document.getElementById('resultsPanel');
const correctBadgeEl = document.getElementById('correctBadge');
const comparisonViz = document.getElementById('comparisonViz');
const kernelBar = document.getElementById('kernelBar');
const pytorchBar = document.getElementById('pytorchBar');
const kernelTime = document.getElementById('kernelTime');
const pytorchTime = document.getElementById('pytorchTime');
const performanceSummary = document.getElementById('performanceSummary');
const statTriton = document.getElementById('statTriton');
const statPytorch = document.getElementById('statPytorch');
const statBlock = document.getElementById('statBlock');
const statThroughput = document.getElementById('statThroughput');
const resultsMeta = document.getElementById('resultsMeta');
const planList = document.getElementById('planList');
const sweepBody = document.getElementById('sweepBody');
const referencePanel = document.getElementById('referencePanel');
const referenceExpression = document.getElementById('referenceExpression');
const referenceTableWrap = document.getElementById('referenceTableWrap');
const referenceHead = document.getElementById('referenceHead');
const referenceBody = document.getElementById('referenceBody');

const STORAGE_KEY = 'gpukernel.lastExpression';
// Edit these to match expressions your parser supports.
const EXAMPLES = ['x * y + sin(x)', 'x + y', 'x * y', 'sin(x) * y'];

const allButtons = [compileBtn, optimizeBtn, unitBtn, kernelBtn];
let busy = false;
let lastKernelText = '';
let copyTimer = null;
let compileRequestId = 0;

function loadSavedExpression() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

function saveExpression(expression) {
  try {
    localStorage.setItem(STORAGE_KEY, expression);
  } catch (error) {
    // Storage unavailable; remembering the expression is optional.
  }
}

function setStatus(message, kind = 'neutral') {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`;
}

// Disable every action button while a request is running so a second click
// can't start an overlapping request. The active button gets a spinner.
function setBusy(isBusy, activeBtn = null) {
  busy = isBusy;
  allButtons.forEach((btn) => {
    btn.disabled = isBusy;
    btn.classList.toggle('loading', isBusy && btn === activeBtn);
  });
  expressionInput.disabled = isBusy;
  useLlmEl.disabled = isBusy;
  chipsEl.querySelectorAll('.chip').forEach((chip) => {
    chip.disabled = isBusy;
  });
}

function setTestBadge(kind, text) {
  testBadgeEl.textContent = text;
  testBadgeEl.className = `result-badge ${kind}`;
}

// Show a compiled kernel with line numbers. Copy uses the raw text.
function renderKernel(text) {
  lastKernelText = text;
  kernelOutputEl.classList.remove('kernel-error');
  kernelOutputEl.classList.add('numbered');

  const lines = text.replace(/\s+$/, '').split('\n');
  kernelOutputEl.replaceChildren(
    ...lines.map((line) => {
      const span = document.createElement('span');
      span.className = 'line';
      span.textContent = line;
      return span;
    })
  );
  copyBtn.disabled = false;
}

// Show a plain message (placeholder or error) with no line numbers.
function renderKernelMessage(message, isError = false) {
  lastKernelText = '';
  kernelOutputEl.classList.remove('numbered');
  kernelOutputEl.classList.toggle('kernel-error', isError);
  kernelOutputEl.textContent = message;
  copyBtn.disabled = true;
}

function flashCopyLabel(label) {
  copyBtn.textContent = label;
  clearTimeout(copyTimer);
  copyTimer = setTimeout(() => {
    copyBtn.textContent = 'Copy';
  }, 1500);
}

async function copyKernel() {
  if (!lastKernelText) return;

  try {
    await navigator.clipboard.writeText(lastKernelText);
    flashCopyLabel('Copied!');
  } catch (error) {
    // Fallback for contexts where the Clipboard API is unavailable.
    const temp = document.createElement('textarea');
    temp.value = lastKernelText;
    document.body.appendChild(temp);
    temp.select();
    const ok = document.execCommand('copy');
    temp.remove();
    flashCopyLabel(ok ? 'Copied!' : 'Copy failed');
  }
}

function fmt(value, digits) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '–';
}

// Fill the results card. Everything goes through textContent because plan
// text comes from the LLM.
function renderResults(result) {
  const bench = result.benchmark;
  resultsPanel.hidden = false;

  const validTimes =
    bench &&
    Number.isFinite(bench.triton_ms) &&
    Number.isFinite(bench.pytorch_ms) &&
    bench.triton_ms > 0 &&
    bench.pytorch_ms > 0;
  comparisonViz.hidden = !validTimes;

  if (validTimes) {
    const maxTime = Math.max(bench.triton_ms, bench.pytorch_ms);
    const speedup = bench.pytorch_ms / bench.triton_ms;

    kernelBar.style.width = `${(bench.triton_ms / maxTime) * 100}%`;
    pytorchBar.style.width = `${(bench.pytorch_ms / maxTime) * 100}%`;
    kernelBar.setAttribute('aria-valuenow', String((bench.triton_ms / maxTime) * 100));
    pytorchBar.setAttribute('aria-valuenow', String((bench.pytorch_ms / maxTime) * 100));
    kernelTime.textContent = `${bench.triton_ms.toFixed(4)} ms`;
    pytorchTime.textContent = `${bench.pytorch_ms.toFixed(4)} ms`;

    if (result.correct === false) {
      performanceSummary.textContent =
        'Correctness check failed. Performance results are not considered valid.';
    } else if (result.correct !== true) {
      performanceSummary.textContent =
        'Performance measurements are available, but correctness has not been verified.';
    } else if (speedup > 1.005) {
      performanceSummary.textContent =
        `KernelForge was ${speedup.toFixed(2)}× faster than PyTorch.`;
    } else if (speedup < 0.995) {
      performanceSummary.textContent =
        `PyTorch was ${(1 / speedup).toFixed(2)}× faster than KernelForge.`;
    } else {
      performanceSummary.textContent =
        'Both implementations had similar execution times.';
    }
  } else {
    kernelBar.style.width = '0%';
    pytorchBar.style.width = '0%';
    kernelBar.removeAttribute('aria-valuenow');
    pytorchBar.removeAttribute('aria-valuenow');
    kernelTime.textContent = '';
    pytorchTime.textContent = '';
    performanceSummary.textContent = '';
  }

  if (result.correct === true) {
    correctBadgeEl.textContent = 'Correctness check passed';
    correctBadgeEl.className = 'result-badge passed';
  } else if (result.correct === false) {
    correctBadgeEl.textContent = 'Correctness check FAILED';
    correctBadgeEl.className = 'result-badge failed';
  } else {
    correctBadgeEl.textContent = 'Correctness not verified';
    correctBadgeEl.className = 'result-badge running';
  }

  if (bench) {
    statTriton.textContent = `${fmt(bench.triton_ms, 4)} ms`;
    statPytorch.textContent = `${fmt(bench.pytorch_ms, 4)} ms`;
    statBlock.textContent = String(bench.block_size);
    statThroughput.textContent = `${fmt(bench.throughput_melem_s, 1)} Melem/s`;
  } else {
    [statTriton, statPytorch, statBlock, statThroughput].forEach((el) => {
      el.textContent = '–';
    });
  }

  const meta = [];
  if (result.optimized_source && result.optimized_source !== result.source) {
    meta.push(`Optimized expression: ${result.optimized_source}`);
  }
  if (bench) {
    meta.push(`n = ${Number(bench.n).toLocaleString()}`);
    if (bench.memory) meta.push(`Memory: ${bench.memory}`);
  }
  meta.push(`${result.use_llm ? 'LLM on' : 'LLM off'}, ${fmt(result.elapsed_s, 1)} s`);
  resultsMeta.textContent = meta.join('  ·  ');

  planList.replaceChildren();
  const plans = result.plans || [];
  if (plans.length === 0) {
    const li = document.createElement('li');
    li.className = 'muted-item';
    li.textContent = 'No plans recorded.';
    planList.appendChild(li);
  } else {
    plans.forEach((plan) => {
      const li = document.createElement('li');
      li.textContent = plan.summary;
      planList.appendChild(li);
    });
  }

  sweepBody.replaceChildren();
  const sweep = (bench && bench.sweep) || [];
  if (sweep.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 2;
    td.className = 'muted-item';
    td.textContent = 'No sweep data.';
    tr.appendChild(td);
    sweepBody.appendChild(tr);
  } else {
    const times = sweep.map((item) => item.median_ms).filter((ms) => typeof ms === 'number');
    const best = times.length ? Math.min(...times) : null;
    sweep.forEach((item) => {
      const tr = document.createElement('tr');
      if (best !== null && item.median_ms === best) tr.className = 'best';
      const sizeCell = document.createElement('td');
      sizeCell.textContent = String(item.block_size);
      const msCell = document.createElement('td');
      msCell.textContent = fmt(item.median_ms, 4);
      tr.append(sizeCell, msCell);
      sweepBody.appendChild(tr);
    });
  }
}

function formatReferenceValue(value) {
  if (value === null) return 'Non-finite';
  return Number.isFinite(value) ? String(value) : '—';
}

function renderReferenceResults(expression, result) {
  const inputs = result.inputs;
  const outputs = result.outputs;
  const isNumberArray = (values) =>
    Array.isArray(values) &&
    values.every((value) => typeof value === 'number' && Number.isFinite(value));
  const isOutputArray = (values) =>
    Array.isArray(values) &&
    values.every((value) =>
      value === null || (typeof value === 'number' && Number.isFinite(value))
    );

  if (
    result.mode !== 'pytorch_cpu_reference' ||
    result.kernel_executed !== false ||
    result.performance_measured !== false ||
    result.expression !== expression ||
    !inputs ||
    !isNumberArray(inputs.x) ||
    !isNumberArray(inputs.y) ||
    inputs.x.length === 0 ||
    inputs.y.length === 0 ||
    !outputs ||
    typeof outputs !== 'object' ||
    Array.isArray(outputs)
  ) {
    throw new Error('The reference API returned incomplete or unexpected data.');
  }

  const outputEntries = Object.entries(outputs);
  if (
    outputEntries.length === 0 ||
    outputEntries.some(([, values]) => !isOutputArray(values)) ||
    !outputEntries.some(([, values]) => values.length > 0)
  ) {
    throw new Error('The reference API returned no valid output values.');
  }

  const headers = ['Index', 'Input x', 'Input y', ...outputEntries.map(([name]) => name)];
  const headRow = document.createElement('tr');
  headers.forEach((label) => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = label;
    headRow.appendChild(th);
  });
  referenceHead.replaceChildren(headRow);

  const rowCount = Math.max(
    inputs.x.length,
    inputs.y.length,
    ...outputEntries.map(([, values]) => values.length)
  );
  const rows = [];
  for (let index = 0; index < rowCount; index += 1) {
    const row = document.createElement('tr');
    const cells = [
      String(index + 1),
      index < inputs.x.length ? formatReferenceValue(inputs.x[index]) : '—',
      index < inputs.y.length ? formatReferenceValue(inputs.y[index]) : '—',
      ...outputEntries.map(([, values]) =>
        index < values.length ? formatReferenceValue(values[index]) : '—'
      )
    ];
    cells.forEach((value, cellIndex) => {
      const cell = document.createElement(cellIndex === 0 ? 'th' : 'td');
      if (cellIndex === 0) cell.scope = 'row';
      cell.textContent = value;
      row.appendChild(cell);
    });
    rows.push(row);
  }
  referenceBody.replaceChildren(...rows);
  referenceExpression.textContent = expression;
  referenceTableWrap.hidden = false;
  referencePanel.hidden = false;
}

function clearReferenceResults() {
  referencePanel.hidden = true;
  referenceExpression.textContent = '';
  referenceHead.replaceChildren();
  referenceBody.replaceChildren();
  referenceTableWrap.hidden = true;
}

function markExpressionChanged() {
  compileRequestId += 1;
  clearReferenceResults();
  renderKernelMessage('Expression changed. Compile to generate a matching kernel.');
  setStatus('Expression changed. Compile to refresh the results.', 'neutral');
}

async function compileExpression() {
  if (busy) return;

  const expression = expressionInput.value.trim();
  const requestId = ++compileRequestId;
  clearReferenceResults();
  if (!expression) {
    renderKernelMessage('Enter an expression to compile.');
    setStatus('Please enter an expression first.', 'error');
    return;
  }

  setBusy(true, compileBtn);
  renderKernelMessage('Compiling expression...');
  setStatus('Compiling expression...', 'neutral');

  try {
    const response = await fetch('/api/compile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression })
    });

    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || 'Compile failed.');
    }

    const compiledExpression =
      typeof result.expression === 'string' ? result.expression : expression;
    if (requestId !== compileRequestId) return;

    if (result.kernel) {
      renderKernel(result.kernel);
    } else {
      renderKernelMessage('No kernel generated.');
    }
    saveExpression(compiledExpression);
    setStatus('Compiled. Calculating PyTorch CPU reference...', 'neutral');

    try {
      const referenceResponse = await fetch('/api/reference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: compiledExpression })
      });

      const referenceResult = await referenceResponse.json();
      if (!referenceResponse.ok || !referenceResult || referenceResult.success !== true) {
        throw new Error(
          referenceResult && typeof referenceResult.error === 'string'
            ? referenceResult.error
            : 'CPU reference calculation failed.'
        );
      }

      if (requestId !== compileRequestId) return;
      renderReferenceResults(compiledExpression, referenceResult);
      setStatus('Compiled and reference calculated (PyTorch · CPU).', 'success');
    } catch (error) {
      if (requestId !== compileRequestId) return;
      clearReferenceResults();
      setStatus(
        `Compiled successfully, but CPU reference failed: ${error.message}`,
        'error'
      );
    }
  } catch (error) {
    if (requestId !== compileRequestId) return;
    renderKernelMessage(`Error: ${error.message}`, true);
    setStatus(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

// Optimize can take minutes (LLM calls plus benchmarks). The previous
// kernel stays on screen if a run fails, so a failure doesn't wipe it.
async function optimizeExpression() {
  if (busy) return;

  const expression = expressionInput.value.trim();
  if (!expression) {
    setStatus('Please enter an expression first.', 'error');
    return;
  }

  const useLlm = useLlmEl.checked;
  setBusy(true, optimizeBtn);
  setStatus(
    useLlm
      ? 'Optimizing: LLM rounds plus benchmarks, this can take a few minutes...'
      : 'Benchmarking (LLM off)...',
    'neutral'
  );

  try {
    const response = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression, use_llm: useLlm })
    });

    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || 'Optimize failed.');
    }

    if (result.kernel) {
      renderKernel(result.kernel);
    }
    renderResults(result);
    saveExpression(expression);

    const bench = result.benchmark;
    setStatus(
      bench
        ? `Optimized in ${fmt(result.elapsed_s, 1)} s: ${fmt(bench.speedup, 2)}× vs PyTorch.`
        : `Optimized in ${fmt(result.elapsed_s, 1)} s.`,
      result.correct === false ? 'error' : 'success'
    );
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function runTests(mode, triggerBtn) {
  if (busy) return;

  setBusy(true, triggerBtn);
  setStatus(`Running ${mode} tests...`, 'neutral');
  setTestBadge('running', 'Running...');

  try {
    const response = await fetch(`/api/tests?mode=${mode}`);
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Test run failed.');
    }

    testOutputEl.textContent = result.output || 'No output.';

    const passed = result.returncode === 0;
    setTestBadge(
      passed ? 'passed' : 'failed',
      passed ? `${mode} tests passed` : `${mode} tests failed (exit ${result.returncode})`
    );
    setStatus(
      `${mode} tests exited with ${result.returncode}.`,
      passed ? 'success' : 'error'
    );
  } catch (error) {
    testOutputEl.textContent = `Error: ${error.message}`;
    setTestBadge('failed', `${mode} tests could not run`);
    setStatus(error.message, 'error');
  } finally {
    setBusy(false);
  }
}

function buildChips() {
  EXAMPLES.forEach((example) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = example;
    chip.addEventListener('click', () => {
      if (busy) return;
      expressionInput.value = example;
      compileExpression();
    });
    chipsEl.appendChild(chip);
  });
}

compileBtn.addEventListener('click', compileExpression);
optimizeBtn.addEventListener('click', optimizeExpression);
unitBtn.addEventListener('click', () => runTests('unit', unitBtn));
kernelBtn.addEventListener('click', () => runTests('kernel', kernelBtn));
copyBtn.addEventListener('click', copyKernel);

expressionInput.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    event.preventDefault();
    compileExpression();
  }
});
expressionInput.addEventListener('input', markExpressionChanged);

buildChips();

const saved = loadSavedExpression();
if (saved) {
  expressionInput.value = saved;
}

compileExpression();
