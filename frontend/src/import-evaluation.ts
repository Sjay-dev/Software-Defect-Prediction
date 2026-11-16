import type { ImportRowMetadata } from './file-import'
import type { Prediction } from './types'

export type ImportedEvaluation = {
  labelledRecords: number
  positiveRecords: number
  negativeRecords: number
  averagePrecision: number | null
  rocAuc: number | null
  precisionAtHalf: number | null
  recallAtHalf: number | null
  f1AtHalf: number | null
  confusion: { tn: number; fp: number; fn: number; tp: number }
  explanation: string | null
}

function averagePrecision(rows: Array<{ score: number; label: 0 | 1 }>): number {
  const sorted = rows.slice().sort((left, right) => right.score - left.score)
  let positivesSeen = 0
  let sumPrecision = 0
  sorted.forEach((row, index) => {
    if (row.label === 1) {
      positivesSeen += 1
      sumPrecision += positivesSeen / (index + 1)
    }
  })
  return sumPrecision / positivesSeen
}

function rocAuc(rows: Array<{ score: number; label: 0 | 1 }>): number {
  const positives = rows.filter((row) => row.label === 1)
  const negatives = rows.filter((row) => row.label === 0)
  let wins = 0
  for (const positive of positives) {
    for (const negative of negatives) {
      if (positive.score > negative.score) wins += 1
      else if (positive.score === negative.score) wins += 0.5
    }
  }
  return wins / (positives.length * negatives.length)
}

export function evaluateImportedPredictions(
  predictions: Prediction[],
  metadata: ImportRowMetadata[],
): ImportedEvaluation | null {
  const byId = new Map(metadata.map((row) => [row.recordId, row]))
  const rows = predictions.flatMap((prediction) => {
    const labels = byId.get(prediction.record_id)?.labels ?? []
    const established = labels.find((label) => label.binaryValue !== null)
    return established ? [{ score: prediction.model_score, label: established.binaryValue as 0 | 1 }] : []
  })
  if (rows.length === 0) return null
  const positiveRecords = rows.filter((row) => row.label === 1).length
  const negativeRecords = rows.length - positiveRecords
  const confusion = { tn: 0, fp: 0, fn: 0, tp: 0 }
  for (const row of rows) {
    const predicted = row.score >= 0.5 ? 1 : 0
    if (row.label === 1 && predicted === 1) confusion.tp += 1
    else if (row.label === 1) confusion.fn += 1
    else if (predicted === 1) confusion.fp += 1
    else confusion.tn += 1
  }
  const precision = confusion.tp + confusion.fp > 0 ? confusion.tp / (confusion.tp + confusion.fp) : null
  const recall = confusion.tp + confusion.fn > 0 ? confusion.tp / (confusion.tp + confusion.fn) : null
  const f1 = precision !== null && recall !== null && precision + recall > 0 ? 2 * precision * recall / (precision + recall) : null
  const bothClasses = positiveRecords > 0 && negativeRecords > 0
  return {
    labelledRecords: rows.length,
    positiveRecords,
    negativeRecords,
    averagePrecision: bothClasses ? averagePrecision(rows) : null,
    rocAuc: bothClasses ? rocAuc(rows) : null,
    precisionAtHalf: precision,
    recallAtHalf: recall,
    f1AtHalf: f1,
    confusion,
    explanation: bothClasses ? null : 'Average precision and ROC-AUC are undefined because the imported outcomes contain only one class.',
  }
}
