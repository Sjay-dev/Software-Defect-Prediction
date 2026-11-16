import type { MetricRecord, Prediction, PredictionResponse } from './types'

export type BatchProgress = {
  status: 'running' | 'retrying' | 'complete' | 'failed' | 'cancelled'
  currentBatch: number
  completedBatches: number
  totalBatches: number
  completedRecords: number
  totalRecords: number
  attempt: number
}

type PredictBatch = (records: MetricRecord[], signal?: AbortSignal) => Promise<PredictionResponse>

export type BatchOptions = {
  signal?: AbortSignal
  maxRetries?: number
  retryDelayMs?: number
}

export type BatchedPredictionResult = {
  scoreDescription: string
  predictions: Prediction[]
  modelVersion: string
  batchCount: number
}

export function assignGlobalRanks(predictions: Prediction[]): Prediction[] {
  return predictions
    .slice()
    .sort((left, right) => right.model_score - left.model_score || left.record_id.localeCompare(right.record_id))
    .map((prediction, index) => ({ ...prediction, rank: index + 1 }))
}

export async function predictInBatches(
  records: MetricRecord[],
  batchSize: number,
  expectedModelVersion: string,
  predictBatch: PredictBatch,
  onProgress?: (progress: BatchProgress) => void,
  options: BatchOptions = {},
): Promise<BatchedPredictionResult> {
  if (records.length === 0) throw new Error('There are no records to score.')
  if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error('The API batch size must be a positive integer.')

  const totalBatches = Math.ceil(records.length / batchSize)
  const allPredictions: Prediction[] = []
  let scoreDescription = ''
  const maxRetries = options.maxRetries ?? 0
  const retryDelayMs = options.retryDelayMs ?? 250

  const progress = (status: BatchProgress['status'], batchIndex: number, completedRecords: number, attempt: number) => {
    onProgress?.({
      status,
      currentBatch: batchIndex + 1,
      completedBatches: batchIndex,
      totalBatches,
      completedRecords,
      totalRecords: records.length,
      attempt,
    })
  }

  const abortError = (batchIndex: number, completedRecords: number): Error => {
    progress('cancelled', batchIndex, completedRecords, 1)
    return new Error(
      `Scoring was cancelled after ${completedRecords} of ${records.length} records. ` +
      'No partial results were displayed; run prediction again to score the complete file.',
    )
  }

  for (let batchIndex = 0; batchIndex < totalBatches; batchIndex += 1) {
    const start = batchIndex * batchSize
    const batch = records.slice(start, start + batchSize)
    if (options.signal?.aborted) throw abortError(batchIndex, start)
    progress('running', batchIndex, start, 1)

    let response: PredictionResponse | null = null
    let lastError: unknown = null
    for (let attempt = 1; attempt <= maxRetries + 1; attempt += 1) {
      if (options.signal?.aborted) throw abortError(batchIndex, start)
      if (attempt > 1) progress('retrying', batchIndex, start, attempt)
      try {
        response = await predictBatch(batch, options.signal)
        break
      } catch (error) {
        lastError = error
        if (options.signal?.aborted || (error instanceof DOMException && error.name === 'AbortError')) {
          throw abortError(batchIndex, start)
        }
        if (attempt <= maxRetries) {
          await new Promise<void>((resolve, reject) => {
            const onAbort = () => {
              clearTimeout(timeout)
              reject(abortError(batchIndex, start))
            }
            const timeout = setTimeout(() => {
              options.signal?.removeEventListener('abort', onAbort)
              resolve()
            }, retryDelayMs)
            options.signal?.addEventListener('abort', onAbort, { once: true })
          })
        }
      }
    }
    if (response === null) {
      progress('failed', batchIndex, start, maxRetries + 1)
      const reason = lastError instanceof Error ? lastError.message : 'Unknown API error.'
      throw new Error(
        `Batch ${batchIndex + 1} of ${totalBatches} failed after ${start} of ${records.length} records were scored ` +
        `(${maxRetries + 1} ${maxRetries === 0 ? 'attempt' : 'attempts'}). ` +
        `${reason} Results are incomplete and were not displayed; retry to score the complete file.`,
      )
    }

    const invalidResponse = (message: string): never => {
      progress('failed', batchIndex, start, 1)
      throw new Error(`${message} Results are incomplete and were not displayed.`)
    }
    if (response.predictions.length !== batch.length) invalidResponse(`Batch ${batchIndex + 1} returned ${response.predictions.length} predictions for ${batch.length} records.`)
    const expectedIds = new Set(batch.map((record) => record.record_id))
    const returnedIds = new Set<string>()
    for (const prediction of response.predictions) {
      if (!expectedIds.has(prediction.record_id) || returnedIds.has(prediction.record_id)) {
        invalidResponse(`Batch ${batchIndex + 1} returned an unexpected or duplicate record_id '${prediction.record_id}'.`)
      }
      if (!Number.isFinite(prediction.model_score)) {
        invalidResponse(`Batch ${batchIndex + 1} returned a non-finite model score for '${prediction.record_id}'.`)
      }
      if (prediction.model_version !== expectedModelVersion) {
        invalidResponse(
          `Batch ${batchIndex + 1} used model version '${prediction.model_version}', but the upload started with ` +
          `'${expectedModelVersion}'. Reload the dashboard before retrying so every batch uses one model version.`,
        )
      }
      returnedIds.add(prediction.record_id)
    }
    if (returnedIds.size !== expectedIds.size) {
      invalidResponse(`Batch ${batchIndex + 1} did not return every submitted record_id.`)
    }

    if (batchIndex === 0) scoreDescription = response.score_description
    allPredictions.push(...response.predictions)
    onProgress?.({
      status: batchIndex + 1 === totalBatches ? 'complete' : 'running',
      currentBatch: batchIndex + 1,
      completedBatches: batchIndex + 1,
      totalBatches,
      completedRecords: Math.min(start + batch.length, records.length),
      totalRecords: records.length,
      attempt: 1,
    })
  }

  return {
    scoreDescription,
    predictions: assignGlobalRanks(allPredictions),
    modelVersion: expectedModelVersion,
    batchCount: totalBatches,
  }
}
