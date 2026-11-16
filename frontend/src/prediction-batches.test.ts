import { describe, expect, it, vi } from 'vitest'
import { assignGlobalRanks, predictInBatches } from './prediction-batches'
import type { MetricRecord, PredictionResponse } from './types'

function records(count: number): MetricRecord[] {
  return Array.from({ length: count }, (_, index) => ({ record_id: `record-${String(index + 1).padStart(3, '0')}`, wmc: index }))
}

describe('batched predictions', () => {
  it('submits bounded batches and reports progress', async () => {
    const calls: number[] = []
    const progress = vi.fn()
    const predict = vi.fn(async (batch: MetricRecord[]): Promise<PredictionResponse> => {
      calls.push(batch.length)
      return {
        score_description: 'Model score',
        predictions: batch.map((record, index) => ({
          record_id: record.record_id,
          model_score: index / 100,
          rank: index + 1,
          model_version: 'v1',
        })),
      }
    })

    const result = await predictInBatches(records(245), 100, 'v1', predict, progress)
    expect(calls).toEqual([100, 100, 45])
    expect(result.predictions).toHaveLength(245)
    expect(result.batchCount).toBe(3)
    expect(progress).toHaveBeenLastCalledWith(expect.objectContaining({
      status: 'complete', completedBatches: 3, completedRecords: 245, totalRecords: 245,
    }))
  })

  it('recalculates ranks globally rather than keeping per-batch ranks', async () => {
    const scores = new Map([
      ['record-001', 0.2],
      ['record-002', 0.1],
      ['record-003', 0.9],
      ['record-004', 0.8],
    ])
    const result = await predictInBatches(records(4), 2, 'v1', async (batch) => ({
      score_description: 'Model score',
      predictions: batch.map((record, index) => ({
        record_id: record.record_id,
        model_score: scores.get(record.record_id) ?? 0,
        rank: index + 1,
        model_version: 'v1',
      })),
    }))

    expect(result.predictions.map(({ record_id, rank }) => ({ record_id, rank }))).toEqual([
      { record_id: 'record-003', rank: 1 },
      { record_id: 'record-004', rank: 2 },
      { record_id: 'record-001', rank: 3 },
      { record_id: 'record-002', rank: 4 },
    ])
  })

  it('uses record identifiers to break equal-score ties deterministically', () => {
    const ranked = assignGlobalRanks([
      { record_id: 'z', model_score: 0.5, rank: 1, model_version: 'v1' },
      { record_id: 'a', model_score: 0.5, rank: 1, model_version: 'v1' },
    ])
    expect(ranked.map((item) => item.record_id)).toEqual(['a', 'z'])
  })

  it('rejects a model-version change between batches', async () => {
    let call = 0
    await expect(predictInBatches(records(3), 2, 'v1', async (batch) => {
      call += 1
      return {
        score_description: 'Model score',
        predictions: batch.map((record, index) => ({
          record_id: record.record_id,
          model_score: index,
          rank: index + 1,
          model_version: call === 1 ? 'v1' : 'v2',
        })),
      }
    })).rejects.toThrow("used model version 'v2'")
  })

  it('identifies the failed batch and does not return partial results', async () => {
    let call = 0
    await expect(predictInBatches(records(5), 2, 'v1', async (batch) => {
      call += 1
      if (call === 2) throw new Error('API unavailable')
      return {
        score_description: 'Model score',
        predictions: batch.map((record, index) => ({
          record_id: record.record_id,
          model_score: index,
          rank: index + 1,
          model_version: 'v1',
        })),
      }
    })).rejects.toThrow('Batch 2 of 3 failed after 2 of 5 records were scored')
  })

  it('retries a failed batch without changing the model version or global ranking', async () => {
    let attempts = 0
    const result = await predictInBatches(records(2), 2, 'v1', async (batch) => {
      attempts += 1
      if (attempts === 1) throw new Error('temporary outage')
      return {
        score_description: 'score',
        predictions: batch.map((record, index) => ({ record_id: record.record_id, model_score: 1 - index, rank: index + 1, model_version: 'v1' })),
      }
    }, undefined, { maxRetries: 1, retryDelayMs: 0 })
    expect(attempts).toBe(2)
    expect(result.predictions.map((prediction) => prediction.rank)).toEqual([1, 2])
  })

  it('cancels between bounded batches and returns no partial result', async () => {
    const controller = new AbortController()
    let calls = 0
    await expect(predictInBatches(records(3), 1, 'v1', async (batch) => {
      calls += 1
      if (calls === 1) controller.abort()
      return {
        score_description: 'score',
        predictions: batch.map((record) => ({ record_id: record.record_id, model_score: 0.5, rank: 1, model_version: 'v1' })),
      }
    }, undefined, { signal: controller.signal })).rejects.toThrow('cancelled after 1 of 3 records')
    expect(calls).toBe(1)
  })
})
