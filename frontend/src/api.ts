import type {
  ExperimentResults,
  Health,
  MetricRecord,
  ModelInfo,
  PredictionResponse,
} from './types'

const configuredBase = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
export const API_BASE_URL = configuredBase.replace(/\/$/, '')

function validationMessage(body: unknown): string | null {
  if (!body || typeof body !== 'object' || !('detail' in body)) return null
  const detail = (body as { detail: unknown }).detail
  if (typeof detail === 'string') return detail
  if (!Array.isArray(detail)) return null
  return detail
    .map((item) => {
      if (!item || typeof item !== 'object') return String(item)
      const error = item as { loc?: Array<string | number>; msg?: string }
      const location = error.loc?.slice(1).join(' → ')
      return `${location ? `${location}: ` : ''}${error.msg || 'Invalid input'}`
    })
    .join('; ')
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new Error(`Cannot reach the API at ${API_BASE_URL}. Start the FastAPI service and try again.`)
  }
  const body = (await response.json().catch(() => null)) as unknown
  if (!response.ok) {
    throw new Error(validationMessage(body) || `API request failed with status ${response.status}.`)
  }
  return body as T
}

export const api = {
  health: () => requestJson<Health>('/health'),
  modelInfo: () => requestJson<ModelInfo>('/model-info'),
  experimentResults: () => requestJson<ExperimentResults>('/experiment-results'),
  predict: (records: MetricRecord[]) =>
    requestJson<PredictionResponse>('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records }),
    }),
}
