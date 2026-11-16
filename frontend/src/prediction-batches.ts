import type { MetricRecord, Prediction, PredictionResponse } from './types'

export type BatchProgress = {
  status: 'running' | 'complete' | 'failed'
  currentBatch: number
  completedBatches: number
  totalBatches: number
  completedRecords: number
  totalRecords: number
}

type PredictBatch = (records: MetricRecord[]) => Promise<PredictionResponse>

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
): Promise<BatchedPredictionResult> {
  if (records.length === 0) throw new Error('There are no records to score.')
  if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error('The API batch size must be a positive integer.')

  const totalBatches = Math.ceil(records.length / batchSize)
  const allPredictions: Prediction[] = []
  let scoreDescription = ''

  for (let batchIndex = 0; batchIndex < totalBatches; batchIndex += 1) {
    const start = batchIndex * batchSize
    const batch = records.slice(start, start + batchSize)
    onProgress?.({
      status: 'running',
      currentBatch: batchIndex + 1,
      completedBatches: batchIndex,
      totalBatches,
      completedRecords: start,
      totalRecords: records.length,
    })

    let response: PredictionResponse
    try {
      response = await predictBatch(batch)
    } catch (error) {
      onProgress?.({
        status: 'failed',
        currentBatch: batchIndex + 1,
        completedBatches: batchIndex,
        totalBatches,
        completedRecords: start,
        totalRecords: records.length,
      })
      const reason = error instanceof Error ? error.message : 'Unknown API error.'
      throw new Error(
        `Batch ${batchIndex + 1} of ${totalBatches} failed after ${start} of ${records.length} records were scored. ` +
        `${reason} No partial results were displayed; retry the upload after resolving the error.`,
      )
    }

    if (response.predictions.length !== batch.length) {
      throw new Error(`Batch ${batchIndex + 1} returned ${response.predictions.length} predictions for ${batch.length} records.`)
    }
    const expectedIds = new Set(batch.map((record) => record.record_id))
    const returnedIds = new Set<string>()
    for (const prediction of response.predictions) {
      if (!expectedIds.has(prediction.record_id) || returnedIds.has(prediction.record_id)) {
        throw new Error(`Batch ${batchIndex + 1} returned an unexpected or duplicate record_id '${prediction.record_id}'.`)
      }
      if (!Number.isFinite(prediction.model_score)) {
        throw new Error(`Batch ${batchIndex + 1} returned a non-finite model score for '${prediction.record_id}'.`)
      }
      if (prediction.model_version !== expectedModelVersion) {
        throw new Error(
          `Batch ${batchIndex + 1} used model version '${prediction.model_version}', but the upload started with ` +
          `'${expectedModelVersion}'. Reload the dashboard before retrying so every batch uses one model version.`,
        )
      }
      returnedIds.add(prediction.record_id)
    }
    if (returnedIds.size !== expectedIds.size) {
      throw new Error(`Batch ${batchIndex + 1} did not return every submitted record_id.`)
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
    })
  }

  return {
    scoreDescription,
    predictions: assignGlobalRanks(allPredictions),
    modelVersion: expectedModelVersion,
    batchCount: totalBatches,
  }
}
