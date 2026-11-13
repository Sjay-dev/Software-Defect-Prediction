import { describe, expect, it } from 'vitest'
import { parseMetricCsv, predictionsCsv, templateCsv } from './csv'

const features = ['wmc', 'loc']

describe('metric CSV handling', () => {
  it('converts valid rows to the backend request shape', () => {
    const parsed = parseMetricCsv('record_id,wmc,loc\nclass-a,4,26\nclass-b,,10\n', features, 10)
    expect(parsed.records).toEqual([
      { record_id: 'class-a', wmc: 4, loc: 26 },
      { record_id: 'class-b', wmc: null, loc: 10 },
    ])
  })

  it('rejects missing columns, non-numeric values, duplicates, and oversized batches', () => {
    expect(() => parseMetricCsv('record_id,wmc\na,1\n', features, 10)).toThrow('Missing required columns: loc')
    expect(() => parseMetricCsv('record_id,wmc,loc\na,nope,1\n', features, 10)).toThrow("Row 2, wmc: 'nope' is not a finite numeric value")
    expect(() => parseMetricCsv('record_id,wmc,loc\na,1,2\na,2,3\n', features, 10)).toThrow("duplicate record_id 'a'")
    expect(() => parseMetricCsv('record_id,wmc,loc\na,1,2\nb,2,3\n', features, 1)).toThrow('configured batch limit is 1')
  })

  it('generates schema and result downloads with stable columns', () => {
    expect(templateCsv(features)).toBe('record_id,wmc,loc\r\n')
    expect(predictionsCsv([{ record_id: 'a,b', model_score: 0.7, rank: 1, model_version: 'v1' }])).toContain('"a,b",0.7,1,v1')
  })
})
