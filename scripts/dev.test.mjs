import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import { parseWindowsNetstat, paths } from './dev-core.mjs'

const launcher = fileURLToPath(new URL('./dev.mjs', import.meta.url))

async function freePorts(count) {
  const servers = []
  const ports = []
  for (let index = 0; index < count; index += 1) {
    const server = net.createServer()
    await new Promise((resolve, reject) => {
      server.once('error', reject)
      server.listen({ host: '127.0.0.1', port: 0 }, resolve)
    })
    servers.push(server)
    ports.push(server.address().port)
  }
  await Promise.all(servers.map((server) => new Promise((resolve) => server.close(resolve))))
  return ports
}

async function occupyPort() {
  const server = net.createServer()
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0 }, resolve)
  })
  return { server, port: server.address().port }
}

function stopTree(pid) {
  if (!pid) return
  if (process.platform === 'win32') {
    spawnSync('taskkill.exe', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true })
  } else {
    try { process.kill(pid, 'SIGKILL') } catch {}
  }
}

async function runLauncher(args, overrides = {}, timeoutMs = 90000) {
  return await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [launcher, ...args], {
      cwd: paths.root,
      env: { ...process.env, NO_COLOR: '1', ...overrides },
      shell: false,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.setEncoding('utf8').on('data', (chunk) => { stdout += chunk })
    child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk })
    const timer = setTimeout(() => {
      stopTree(child.pid)
      reject(new Error(`Launcher timed out.\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}`))
    }, timeoutMs)
    child.once('error', (error) => { clearTimeout(timer); reject(error) })
    child.once('exit', (code, signal) => {
      clearTimeout(timer)
      resolve({ code, signal, stdout, stderr, output: `${stdout}\n${stderr}` })
    })
  })
}

async function assertPortAvailable(port) {
  const server = net.createServer()
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port }, resolve)
  })
  await new Promise((resolve) => server.close(resolve))
}

test('Windows netstat parsing returns the listening PID', () => {
  const output = '  TCP    127.0.0.1:5173       0.0.0.0:0       LISTENING       4321\r\n'
  assert.equal(parseWindowsNetstat(output, 5173), 4321)
  assert.equal(parseWindowsNetstat(output, 8000), undefined)
})

test('occupied backend and frontend ports fail before starting services', { timeout: 120000 }, async () => {
  const backendOccupant = await occupyPort()
  try {
    const result = await runLauncher(['--backend-only'], { SDP_BACKEND_PORT: String(backendOccupant.port) })
    assert.notEqual(result.code, 0)
    assert.match(result.output, new RegExp(`Port ${backendOccupant.port} is already in use`))
    assert.match(result.output, /Startup cancelled/)
    if (process.platform === 'win32') assert.match(result.output, /PID \d+/)
  } finally {
    await new Promise((resolve) => backendOccupant.server.close(resolve))
  }

  const frontendOccupant = await occupyPort()
  try {
    const result = await runLauncher(['--frontend-only'], { SDP_FRONTEND_PORT: String(frontendOccupant.port) })
    assert.notEqual(result.code, 0)
    assert.match(result.output, new RegExp(`Port ${frontendOccupant.port} is already in use`))
    assert.match(result.output, /Startup cancelled/)
    if (process.platform === 'win32') assert.match(result.output, /PID \d+/)
  } finally {
    await new Promise((resolve) => frontendOccupant.server.close(resolve))
  }
})

test('doctor gives an actionable missing-configuration error', { timeout: 60000 }, async () => {
  const [backendPort, frontendPort] = await freePorts(2)
  const missing = path.join(os.tmpdir(), `missing-service-${process.pid}.json`)
  const result = await runLauncher(['--doctor'], {
    SDP_SERVICE_CONFIG: missing,
    SDP_BACKEND_PORT: String(backendPort),
    SDP_FRONTEND_PORT: String(frontendPort),
  })
  assert.notEqual(result.code, 0)
  assert.match(result.output, /Backend configuration: Missing/)
  assert.match(result.output, /Restore config\/service.json or set SDP_SERVICE_CONFIG/)
})

test('missing historical reports are warnings and do not block predictions', { timeout: 120000 }, async () => {
  const temporaryDirectory = await mkdtemp(path.join(os.tmpdir(), 'defect-launcher-'))
  try {
    const originalPath = path.join(paths.root, 'config', 'service.json')
    const config = JSON.parse(await readFile(originalPath, 'utf8'))
    config.model_manifest = path.resolve(path.dirname(originalPath), config.model_manifest)
    config.historical_reports = {
      baseline_summary: 'missing-baseline.json',
      model_comparison_summary: 'missing-model.json',
      source_similarity_summary: 'missing-similarity.json',
    }
    const temporaryConfig = path.join(temporaryDirectory, 'service.json')
    await writeFile(temporaryConfig, JSON.stringify(config), 'utf8')
    const [backendPort, frontendPort] = await freePorts(2)
    const result = await runLauncher(['--doctor'], {
      SDP_SERVICE_CONFIG: temporaryConfig,
      SDP_BACKEND_PORT: String(backendPort),
      SDP_FRONTEND_PORT: String(frontendPort),
    })
    assert.equal(result.code, 0)
    assert.match(result.output, /Predictions can run; this experiment section will be unavailable/)
    assert.match(result.output, /passed with 3 warning/)
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true })
  }
})

test('backend startup failures surface the Uvicorn error and free the port', { timeout: 120000 }, async () => {
  const [backendPort] = await freePorts(1)
  const result = await runLauncher(['--backend-only'], {
    SDP_BACKEND_PORT: String(backendPort),
    SDP_BACKEND_MODULE: 'defect_prediction.module_that_does_not_exist:app',
    SDP_READINESS_TIMEOUT_MS: '5000',
  })
  assert.notEqual(result.code, 0)
  assert.match(result.output, /Could not import module|No module named|Error loading ASGI app/)
  assert.match(result.output, /exited before startup completed/)
  await assertPortAvailable(backendPort)
})

test('combined startup shuts down cleanly and can restart on the same ports', { timeout: 180000 }, async () => {
  const [backendPort, frontendPort] = await freePorts(2)
  const environment = {
    SDP_BACKEND_PORT: String(backendPort),
    SDP_FRONTEND_PORT: String(frontendPort),
    SDP_TEST_AUTO_SHUTDOWN_MS: '500',
  }
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const result = await runLauncher([], environment, 90000)
    assert.equal(result.code, 0, result.output)
    assert.match(result.output, /Backend ready/)
    assert.match(result.output, /Frontend ready/)
    assert.match(result.output, /Managed services stopped/)
    await assertPortAvailable(backendPort)
    await assertPortAvailable(frontendPort)
  }
})
