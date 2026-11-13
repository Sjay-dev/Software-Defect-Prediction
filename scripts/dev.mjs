#!/usr/bin/env node

import process from 'node:process'
import {
  describePortConflict,
  hasErrors,
  inspectPort,
  inspectPrerequisites,
  paths,
  printChecks,
  startManagedProcess,
  stopManagedProcess,
  waitForReady,
} from './dev-core.mjs'

const args = new Set(process.argv.slice(2))
const doctorOnly = args.has('--doctor')
const backendOnly = args.has('--backend-only')
const frontendOnly = args.has('--frontend-only')

if (backendOnly && frontendOnly) {
  console.error('[launcher] Choose either --backend-only or --frontend-only, not both.')
  process.exit(1)
}

const includeBackend = !frontendOnly
const includeFrontend = !backendOnly
const backendPort = Number(process.env.SDP_BACKEND_PORT || 8000)
const frontendPort = Number(process.env.SDP_FRONTEND_PORT || 5173)
const readinessTimeoutMs = Number(process.env.SDP_READINESS_TIMEOUT_MS || 20000)
const backendModule = process.env.SDP_BACKEND_MODULE || 'defect_prediction.api:app'
const frontendApiUrl = `http://127.0.0.1:${backendPort}`
const service = inspectPrerequisites({ includeBackend: doctorOnly || includeBackend, includeFrontend: doctorOnly || includeFrontend })

console.log(`[launcher] Repository: ${paths.root}`)
printChecks(service.results)

const requestedPorts = []
if (doctorOnly || includeBackend) requestedPorts.push({ name: 'Backend', port: backendPort })
if (doctorOnly || includeFrontend) requestedPorts.push({ name: 'Frontend', port: frontendPort })
const portResults = []
for (const requested of requestedPorts) {
  const info = await inspectPort(requested.port)
  portResults.push({ ...info, name: requested.name })
  if (info.available) console.log(`[OK] ${requested.name} port ${requested.port}: available`)
  else console.log(`[${doctorOnly ? 'WARN' : 'ERROR'}] ${requested.name} port conflict:\n${describePortConflict(info)}`)
}

if (doctorOnly) {
  const warningCount = service.results.filter((result) => result.level === 'warn').length + portResults.filter((result) => !result.available).length
  if (hasErrors(service.results)) {
    console.error('[doctor] Blocking prerequisite errors were found.')
    process.exit(1)
  }
  console.log(`[doctor] Core prerequisites passed${warningCount ? ` with ${warningCount} warning(s)` : ''}. No services were started or stopped.`)
  process.exit(0)
}

if (hasErrors(service.results) || portResults.some((result) => !result.available)) {
  console.error('[launcher] Startup cancelled. Fix the errors above and run the command again.')
  process.exit(1)
}

const managed = []
let shuttingDown = false

async function shutdown(exitCode, reason) {
  if (shuttingDown) return
  shuttingDown = true
  if (reason) console.log(`[launcher] ${reason}`)
  for (const state of managed.slice().reverse()) await stopManagedProcess(state)
  console.log('[launcher] Managed services stopped.')
  process.exit(exitCode)
}

process.once('SIGINT', () => { void shutdown(0, 'Ctrl+C received; stopping both services…') })
process.once('SIGTERM', () => { void shutdown(0, 'Termination requested; stopping both services…') })

try {
  if (includeBackend) {
    console.log(`[launcher] Starting backend on ${frontendApiUrl}`)
    const backend = startManagedProcess({
      command: paths.python,
      args: ['-m', 'uvicorn', backendModule, '--host', '127.0.0.1', '--port', String(backendPort)],
      cwd: paths.root,
      env: {
        ...process.env,
        DEFECT_SERVICE_CONFIG: service.configPath,
        PYTHONUNBUFFERED: '1',
      },
      label: 'backend',
    })
    managed.push(backend)
    await waitForReady(`${frontendApiUrl}/health`, backend, readinessTimeoutMs)
    console.log(`[launcher] Backend ready: ${frontendApiUrl}`)
    console.log(`[launcher] API documentation: ${frontendApiUrl}/docs`)
  }

  if (includeFrontend) {
    console.log(`[launcher] Starting frontend on http://127.0.0.1:${frontendPort}`)
    const frontend = startManagedProcess({
      command: process.execPath,
      args: [paths.vite, '--host', '127.0.0.1', '--port', String(frontendPort), '--strictPort'],
      cwd: paths.frontend,
      env: {
        ...process.env,
        VITE_API_BASE_URL: process.env.VITE_API_BASE_URL || frontendApiUrl,
      },
      label: 'frontend',
    })
    managed.push(frontend)
    await waitForReady(
      `http://127.0.0.1:${frontendPort}`,
      frontend,
      readinessTimeoutMs,
      managed.filter((state) => state !== frontend),
    )
    console.log(`[launcher] Frontend ready: http://127.0.0.1:${frontendPort}`)
  }

  console.log('[launcher] Development services are ready. Press Ctrl+C to stop them.')

  const alreadyExited = managed.find((state) => state.exit || state.spawnError)
  if (alreadyExited) {
    throw new Error(`${alreadyExited.label} exited while startup was completing.`)
  }

  for (const state of managed) {
    state.child.once('exit', (code, signal) => {
      if (!shuttingDown) void shutdown(1, `${state.label} exited unexpectedly (code ${code ?? 'none'}, signal ${signal ?? 'none'}); stopping the other service.`)
    })
  }

  const automaticShutdown = Number(process.env.SDP_TEST_AUTO_SHUTDOWN_MS || 0)
  if (automaticShutdown > 0) {
    setTimeout(() => { void shutdown(0, 'Test shutdown timer elapsed.') }, automaticShutdown).unref()
  }
} catch (error) {
  console.error(`[launcher] Startup failed: ${error.message}`)
  await shutdown(1)
}
