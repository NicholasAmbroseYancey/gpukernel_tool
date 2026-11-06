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

const STORAGE_KEY = 'gpukernel.lastExpression';
// Edit these to match expressions your parser supports.
const EXAMPLES = ['x * y + sin(x)', 'x + y', 'x * y', 'sin(x) * y'];

const allButtons = [compileBtn, optimizeBtn, unitBtn, kernelBtn];
let busy = false;
let lastKernelText = '';
let copyTimer = null;

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

async function compileExpression() {
  if (busy) return;

  const expression = expressionInput.value.trim();
  if (!expression) {
    setStatus('Please enter an expression first.', 'error');
    return;
  }

  setBusy(true, compileBtn);
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

    if (result.kernel) {
      renderKernel(result.kernel);
    } else {
      renderKernelMessage('No kernel generated.');
    }
    saveExpression(expression);
    setStatus(`Compiled: ${result.expression}`, 'success');
  } catch (error) {
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

buildChips();

const saved = loadSavedExpression();
if (saved) {
  expressionInput.value = saved;
}

compileExpression();
