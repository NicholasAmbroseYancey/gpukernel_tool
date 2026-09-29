const expressionInput = document.getElementById('expression');
const compileBtn = document.getElementById('compileBtn');
const unitBtn = document.getElementById('unitBtn');
const kernelBtn = document.getElementById('kernelBtn');
const statusEl = document.getElementById('status');
const kernelOutputEl = document.getElementById('kernelOutput');
const testOutputEl = document.getElementById('testOutput');

function setStatus(message, kind = 'neutral') {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`;
}

async function compileExpression() {
  const expression = expressionInput.value.trim();
  if (!expression) {
    setStatus('Please enter an expression first.', 'error');
    return;
  }

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

    kernelOutputEl.textContent = result.kernel || 'No kernel generated.';
    setStatus(`Compiled: ${result.expression}`, 'success');
  } catch (error) {
    kernelOutputEl.textContent = `Error: ${error.message}`;
    setStatus(error.message, 'error');
  }
}

async function runTests(mode) {
  setStatus(`Running ${mode} tests...`, 'neutral');
  try {
    const response = await fetch(`/api/tests?mode=${mode}`);
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Test run failed.');
    }

    testOutputEl.textContent = result.output || 'No output.';
    setStatus(`${mode} tests exited with ${result.returncode}.`, result.returncode === 0 ? 'success' : 'error');
  } catch (error) {
    testOutputEl.textContent = `Error: ${error.message}`;
    setStatus(error.message, 'error');
  }
}

compileBtn.addEventListener('click', compileExpression);
unitBtn.addEventListener('click', () => runTests('unit'));
kernelBtn.addEventListener('click', () => runTests('kernel'));

compileExpression();
