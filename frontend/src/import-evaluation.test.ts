import { describe, expect, it } from 'vitest'
import { evaluateImportedPredictions } from './import-evaluation'
import type { ImportRowMetadata } from './file-import'

function metadata(labels: Array<0 | 1>): ImportRowMetadata[] {
  return labels.map((label, index) => ({
    sourceRow: index + 2,
    recordId: `r${index}`,
    project: null,
    release: null,
    className: null,
    labels: [{ column: 'defective', meaning: 'defective', rawValue: label, binaryValue: label }],
  }))
}

describe('imported outcome evaluation', () => {
  it('keeps one-class confusion metrics and explains undefined ranking metrics', () => {
    const result = evaluateImportedPredictions([
      { record_id: 'r0', model_score: 0.9, rank: 1, model_version: 'v1' },
      { record_id: 'r1', model_score: 0.2, rank: 2, model_version: 'v1' },
    ], metadata([1, 1]))
    expect(result).toMatchObject({
      averagePrecision: null,
      rocAuc: null,
      positiveRecords: 2,
      negativeRecords: 0,
      confusion: { tn: 0, fp: 0, fn: 1, tp: 1 },
    })
    expect(result?.explanation).toContain('only one class')
  })

  it('computes ranking and threshold metrics only from established labels', () => {
    const result = evaluateImportedPredictions([
      { record_id: 'r0', model_score: 0.9, rank: 1, model_version: 'v1' },
      { record_id: 'r1', model_score: 0.1, rank: 2, model_version: 'v1' },
    ], metadata([1, 0]))
    expect(result).toMatchObject({ averagePrecision: 1, rocAuc: 1, precisionAtHalf: 1, recallAtHalf: 1, f1AtHalf: 1 })
  })
})
