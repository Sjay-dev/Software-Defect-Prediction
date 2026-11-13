import { spawn, spawnSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import net from 'node:net'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url))

export const paths = Object.freeze({
  root: path.resolve(scriptDirectory, '..'),
  frontend: path.resolve(scriptDirectory, '..', 'frontend'),
  python: path.resolve(
    scriptDirectory,
    '..',
    '.venv',
    process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
  ),
  vite: path.resolve(scriptDirectory, '..', 'frontend', 'node_modules', 'vite', 'bin', 'vite.js'),
})

export function resolveRepositoryPath(value) {
  return path.isAbsolute(value) ? path.normalize(value) : path.resolve(paths.root, value)
}

export function resolveConfiguredPath(configPath, value) {
  return path.isAbsolute(value) ? path.normalize(value) : path.resolve(path.dirname(configPath), value)
}

function readJson(filePath, label) {
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'))
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`)
  }
}

function item(level, label, detail, fix) {
  return { level, label, detail, fix }
}

function pythonSetupCommand() {
  const executable = process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'
  return `${executable} -m pip install -r requirements-lock.txt; ${executable} -m pip install -e . --no-deps`
}

function checkBackendPackages(pythonPath, configPath) {
  const importCheck = spawnSync(
    pythonPath,
    ['-c', 'import fastapi, uvicorn, joblib, pandas, sklearn; print("imports ok")'],
    { cwd: paths.root, encoding: 'utf8', windowsHide: true },
  )
  if (importCheck.status !== 0) {
    return item(
      'error',
      'Backend packages',
      (importCheck.stderr || importCheck.stdout || 'Required Python packages could not be imported.').trim(),
      pythonSetupCommand(),
    )
  }

  const validationCode = [
    'from pathlib import Path',
    'from defect_prediction.prediction import load_service_settings, load_predictor',
    `settings = load_service_settings(Path(${JSON.stringify(configPath)}))`,
    'predictor = load_predictor(settings)',
    'print(predictor.model_version)',
  ].join('; ')
  const validation = spawnSync(pythonPath, ['-c', validationCode], {
    cwd: paths.root,
    encoding: 'utf8',
    windowsHide: true,
    env: { ...process.env, DEFECT_SERVICE_CONFIG: configPath },
  })
  if (validation.status !== 0) {
    return item(
      'error',
      'Model validation',
      (validation.stderr || validation.stdout || 'The configured model could not be loaded.').trim(),
      `${process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'} -m defect_prediction.deployment`,
    )
  }
  return item('ok', 'Backend packages and model', `Loaded ${validation.stdout.trim()} with checksum and schema validation.`)
}

export function inspectPrerequisites({ includeBackend = true, includeFrontend = true, configOverride } = {}) {
  const results = []
  let configPath = resolveRepositoryPath(configOverride || process.env.SDP_SERVICE_CONFIG || 'config/service.json')

  if (includeBackend) {
    if (!existsSync(paths.python)) {
      results.push(item('error', 'Python virtual environment', `Missing ${paths.python}.`, 'py -3.12 -m venv .venv'))
    } else {
      results.push(item('ok', 'Python virtual environment', paths.python))
    }

    if (!existsSync(configPath)) {
      results.push(item('error', 'Backend configuration', `Missing ${configPath}.`, 'Restore config/service.json or set SDP_SERVICE_CONFIG to an existing config file.'))
    } else {
      results.push(item('ok', 'Backend configuration', configPath))
      try {
        const config = readJson(configPath, 'Backend configuration')
        const manifestPath = resolveConfiguredPath(configPath, config.model_manifest)
        if (!existsSync(manifestPath)) {
          results.push(item('error', 'Deployment manifest', `Missing ${manifestPath}.`, `${process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'} -m defect_prediction.deployment`))
        } else {
          results.push(item('ok', 'Deployment manifest', manifestPath))
          const manifest = readJson(manifestPath, 'Deployment manifest')
          const modelPath = resolveConfiguredPath(manifestPath, manifest.model_file)
          const schemaPath = resolveConfiguredPath(manifestPath, manifest.feature_schema_file)
          results.push(
            existsSync(modelPath)
              ? item('ok', 'Deployment model', modelPath)
              : item('error', 'Deployment model', `Missing ${modelPath}.`, `${process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'} -m defect_prediction.deployment`),
          )
          results.push(
            existsSync(schemaPath)
              ? item('ok', 'Deployment schema', schemaPath)
              : item('error', 'Deployment schema', `Missing ${schemaPath}.`, `${process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'} -m defect_prediction.deployment`),
          )
        }

        const reports = config.historical_reports || {}
        const reportChecks = [
          ['LOPO report', reports.lopo_results, `${process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'} -m defect_prediction.lopo`],
          ['Source-validation report', reports.source_validation_results, `${process.platform === 'win32' ? '.\\.venv\\Scripts\\python.exe' : './.venv/bin/python'} -m defect_prediction.source_validation`],
        ]
        for (const [label, configured, fix] of reportChecks) {
          if (!configured) {
            results.push(item('warn', label, 'Not configured. Predictions can run, but the experiment page will be unavailable.', fix))
            continue
          }
          const reportPath = resolveConfiguredPath(configPath, configured)
          results.push(
            existsSync(reportPath)
              ? item('ok', label, reportPath)
              : item('warn', label, `Missing ${reportPath}. Predictions can run, but the experiment page will be unavailable.`, fix),
          )
        }
      } catch (error) {
        results.push(item('error', 'Backend configuration contents', error.message, 'Restore a valid config/service.json.'))
      }
    }

    if (existsSync(paths.python) && existsSync(configPath)) {
      results.push(checkBackendPackages(paths.python, configPath))
    }
  }

  if (includeFrontend) {
    const packagePath = path.join(paths.frontend, 'package.json')
    if (!existsSync(packagePath)) {
      results.push(item('error', 'Frontend package', `Missing ${packagePath}.`, 'Restore frontend/package.json.'))
    } else {
      results.push(item('ok', 'Frontend package', packagePath))
      const frontendPackage = readJson(packagePath, 'Frontend package')
      const requiredPackages = Object.keys({
        ...frontendPackage.dependencies,
        ...frontendPackage.devDependencies,
      })
      const missingPackages = requiredPackages.filter((packageName) =>
        !existsSync(path.join(paths.frontend, 'node_modules', ...packageName.split('/'), 'package.json')),
      )
      results.push(
        missingPackages.length === 0 && existsSync(paths.vite)
          ? item('ok', 'Frontend dependencies', `${requiredPackages.length} declared packages are installed.`)
          : item('error', 'Frontend dependencies', `Missing installed packages: ${missingPackages.join(', ') || 'vite'}.`, 'npm --prefix frontend install'),
      )
    }
  }

  return { results, configPath }
}

export async function inspectPort(port, host = '127.0.0.1') {
  const available = await new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.once('error', (error) => {
      if (error.code === 'EADDRINUSE' || error.code === 'EACCES') resolve(false)
      else reject(error)
    })
    server.listen({ host, port, exclusive: true }, () => server.close(() => resolve(true)))
  })
  if (available) return { available: true, port }
  return { available: false, port, ...(await findPortOwner(port)) }
}

export function parseWindowsNetstat(output, port) {
  for (const line of output.split(/\r?\n/)) {
    const fields = line.trim().split(/\s+/)
    if (fields.length < 5 || fields[0].toUpperCase() !== 'TCP' || fields[3].toUpperCase() !== 'LISTENING') continue
    const local = fields[1]
    const localPort = Number(local.slice(local.lastIndexOf(':') + 1))
    if (localPort === port) return Number(fields[4])
  }
  return undefined
}

async function findPortOwner(port) {
  if (process.platform === 'win32') {
    const netstat = spawnSync('netstat.exe', ['-ano', '-p', 'tcp'], { encoding: 'utf8', windowsHide: true })
    const pid = parseWindowsNetstat(netstat.stdout || '', port)
    if (!pid) return {}
    const task = spawnSync('tasklist.exe', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true })
    const match = /^"([^"]+)"/.exec((task.stdout || '').trim())
    return { pid, processName: match?.[1] }
  }
  const lsof = spawnSync('lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-t'], { encoding: 'utf8' })
  const pid = Number((lsof.stdout || '').trim().split(/\s+/)[0])
  return Number.isInteger(pid) && pid > 0 ? { pid } : {}
}

export function describePortConflict(info) {
  const identity = info.pid
    ? `PID ${info.pid}${info.processName ? ` (${info.processName})` : ''}`
    : 'an unidentified process'
  const lines = [`Port ${info.port} is already in use by ${identity}.`]
  if (info.pid && process.platform === 'win32') {
    lines.push(`Inspect it: Get-CimInstance Win32_Process -Filter "ProcessId = ${info.pid}" | Select-Object ProcessId, Name, CommandLine`)
    lines.push(`If it is your stale server, stop only it: Stop-Process -Id ${info.pid}`)
  } else if (info.pid) {
    lines.push(`Inspect it: ps -p ${info.pid} -o pid,command`)
    lines.push(`If it is your stale server, stop only it: kill ${info.pid}`)
  }
  return lines.join('\n')
}

export function printChecks(results) {
  for (const result of results) {
    const marker = result.level === 'ok' ? '[OK]' : result.level === 'warn' ? '[WARN]' : '[ERROR]'
    console.log(`${marker} ${result.label}: ${result.detail}`)
    if (result.fix) console.log(`       Fix: ${result.fix}`)
  }
}

export function hasErrors(results) {
  return results.some((result) => result.level === 'error')
}

export function startManagedProcess({ command, args, cwd, env, label }) {
  const child = spawn(command, args, {
    cwd,
    env,
    shell: false,
    windowsHide: false,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  const state = { child, label, exit: null, spawnError: null }
  child.once('error', (error) => { state.spawnError = error })
  child.once('exit', (code, signal) => { state.exit = { code, signal } })
  prefixOutput(child.stdout, label)
  prefixOutput(child.stderr, label)
  return state
}

function prefixOutput(stream, label) {
  let pending = ''
  stream.setEncoding('utf8')
  stream.on('data', (chunk) => {
    pending += chunk
    const lines = pending.split(/\r?\n/)
    pending = lines.pop() || ''
    for (const line of lines) if (line) console.log(`[${label}] ${line}`)
  })
  stream.on('end', () => { if (pending) console.log(`[${label}] ${pending}`) })
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds))

export async function waitForReady(url, state, timeoutMs, peers = []) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    for (const monitored of [state, ...peers]) {
      if (monitored.spawnError) throw new Error(`${monitored.label} could not start: ${monitored.spawnError.message}`)
      if (monitored.exit) throw new Error(`${monitored.label} exited before startup completed (code ${monitored.exit.code ?? 'none'}, signal ${monitored.exit.signal ?? 'none'}).`)
    }
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1000) })
      if (response.ok) {
        for (const monitored of [state, ...peers]) {
          if (monitored.exit) throw new Error(`${monitored.label} exited before startup completed (code ${monitored.exit.code ?? 'none'}, signal ${monitored.exit.signal ?? 'none'}).`)
        }
        return response
      }
    } catch {
      // The server may still be importing or binding its port.
    }
    await delay(200)
  }
  throw new Error(`${state.label} did not become ready at ${url} within ${timeoutMs} ms.`)
}

export async function stopManagedProcess(state) {
  const child = state?.child
  if (!child?.pid) return
  if (process.platform === 'win32') {
    spawnSync('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true })
  } else if (child.exitCode === null && child.signalCode === null) {
    child.kill('SIGTERM')
    await Promise.race([new Promise((resolve) => child.once('exit', resolve)), delay(3000)])
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL')
  }
}
