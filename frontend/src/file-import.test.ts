import ExcelJS from 'exceljs'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  analyzeTabularSource,
  parseArffSource,
  parseDelimitedSource,
  parseJsonSource,
  parseXlsxSource,
} from './file-import'
import { predictInBatches } from './prediction-batches'
import type { MetricRecord } from './types'

const features = [
  'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3',
  'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc',
]
const values = [3, 1, 0, 10, 18, 3, 1, 9, 1, 1.1, 106, 0, 0, 0, 0.444444444, 0, 0, 32.66666667, 1, 0.6667]

function canonicalRecord(record: MetricRecord): MetricRecord {
  return Object.fromEntries(['record_id', ...features].map((key) => [key, record[key]])) as MetricRecord
}

function xlsxArrayBuffer(buffer: ExcelJS.Buffer): ArrayBuffer {
  const bytes = new Uint8Array(buffer)
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

describe('multi-format metric imports', () => {
  it('normalizes BOM, whitespace, and header case and detects comma, semicolon, and tab delimiters', () => {
    for (const delimiter of [',', ';', '\t'] as const) {
      const headers = [' RECORD_ID ', ...features.map((feature) => ` ${feature.toUpperCase()} `)]
      const source = parseDelimitedSource(`\uFEFF${headers.join(delimiter)}\nclass-a${delimiter}${values.join(delimiter)}\n`, 'metrics.csv')
      const parsed = analyzeTabularSource(source, features)
      expect(parsed.canPredict).toBe(true)
      expect(parsed.records[0]).toMatchObject({ record_id: 'class-a', wmc: 3, avg_cc: 0.6667 })
      expect(source.delimiter).toBe(delimiter)
    }
    const tsv = parseDelimitedSource(`${['record_id', ...features].join('\t')}\nclass-a\t${values.join('\t')}\n`, 'metrics.tsv', 'tsv')
    expect(tsv.format).toBe('tsv')
    expect(tsv.delimiter).toBe('\t')
  })

  it('imports an available original Jureczko/PROMISE dataset example', () => {
    const text = readFileSync('../data/raw/camel-1.0.csv', 'utf8')
    const parsed = analyzeTabularSource(parseDelimitedSource(text, 'camel-1.0.csv'), features)
    expect(parsed.layout).toBe('jureczko-promise')
    expect(parsed.totalRows).toBeGreaterThan(300)
    expect(parsed.records[0].record_id).toBe('camel:1:org.apache.camel.component.quartz.QuartzExchange')
    expect(parsed.rowMetadata[0].labels[0]).toMatchObject({ column: 'bug', meaning: 'bug_count' })
  })

  it('supports a single JSON record and an array without adding labels to model inputs', () => {
    const base = Object.fromEntries([['record_id', 'json-a'], ...features.map((feature, index) => [feature, values[index]])])
    const single = analyzeTabularSource(parseJsonSource(JSON.stringify(base), 'single.json'), features)
    const array = analyzeTabularSource(parseJsonSource(JSON.stringify([{ ...base, bug_count: 2 }]), 'array.json'), features)
    expect(single.records).toHaveLength(1)
    expect(array.records[0]).not.toHaveProperty('bug_count')
    expect(array.rowMetadata[0].labels[0]).toMatchObject({ meaning: 'bug_count', rawValue: 2, binaryValue: 1 })
    expect(() => parseJsonSource('{"record_id":"a","record_id":"b"}', 'duplicate.json')).toThrow("Duplicate JSON property 'record_id'")
  })

  it('rejects surplus delimited cells rather than silently discarding them', () => {
    expect(() => parseDelimitedSource('record_id,wmc\na,1,unexpected\n', 'wide.csv')).toThrow(
      'row 2 has 3 values but the header has 2 columns',
    )
  })

  it('parses dense ARFF and rejects sparse and unsupported structures with explanations', () => {
    const arff = [
      '@relation defects',
      '@attribute record_id string',
      ...features.map((feature) => `@attribute ${feature} numeric`),
      '@data',
      `class-a,${values.join(',')}`,
    ].join('\n')
    const parsed = analyzeTabularSource(parseArffSource(arff, 'metrics.arff'), features)
    expect(parsed.records[0]).toMatchObject({ record_id: 'class-a', loc: 106 })
    expect(() => parseArffSource(arff.replace(`class-a,${values.join(',')}`, '{0 class-a, 1 3}'), 'sparse.arff')).toThrow('sparse ARFF rows are not supported')
    expect(() => parseArffSource(arff.replace('@attribute wmc numeric', '@attribute wmc date yyyy-MM-dd'), 'date.arff')).toThrow("unsupported type 'date yyyy-MM-dd'")
  })

  it('allows worksheet selection and rejects spreadsheet formulas instead of evaluating cached results', async () => {
    const workbook = new ExcelJS.Workbook()
    const readme = workbook.addWorksheet('Read me')
    readme.addRow(['instructions'])
    readme.addRow(['select Metrics'])
    const metrics = workbook.addWorksheet('Metrics')
    metrics.addRow(['record_id', ...features])
    metrics.addRow(['xlsx-a', ...values])
    const buffer = xlsxArrayBuffer(await workbook.xlsx.writeBuffer())
    const source = await parseXlsxSource(buffer, 'metrics.xlsx', 'Metrics')
    const parsed = analyzeTabularSource(source, features)
    expect(source.availableSheets).toEqual(['Read me', 'Metrics'])
    expect(parsed.records[0]).toMatchObject({ record_id: 'xlsx-a', rfc: 18 })

    const formulaWorkbook = new ExcelJS.Workbook()
    const formulaSheet = formulaWorkbook.addWorksheet('Metrics')
    formulaSheet.addRow(['record_id', ...features])
    formulaSheet.addRow(['formula-a', ...values])
    formulaSheet.getCell('B2').value = { formula: '1+2', result: 3 }
    const formulaSource = await parseXlsxSource(xlsxArrayBuffer(await formulaWorkbook.xlsx.writeBuffer()), 'formula.xlsx')
    const formulaParsed = analyzeTabularSource(formulaSource, features)
    expect(formulaParsed.invalidRows).toEqual([expect.objectContaining({ sourceRow: 2, columns: ['wmc'] })])
    expect(formulaParsed.issues).toEqual(expect.arrayContaining([expect.objectContaining({ code: 'formula-value' })]))
  })

  it('detects duplicate headers before row-object conversion while recognizing the two-name Ant layout', () => {
    const duplicate = parseDelimitedSource(`${['record_id', ...features, ' WMC '].join(',')}\na,${values.join(',')},4\n`, 'duplicate.csv')
    const rejected = analyzeTabularSource(duplicate, features)
    expect(rejected.canPredict).toBe(false)
    expect(rejected.issues).toEqual(expect.arrayContaining([expect.objectContaining({ code: 'duplicate-header', severity: 'error' })]))

    const ant = parseDelimitedSource(`name,version,name,${features.join(',')},bug\nant,1.7,org.example.A,${values.join(',')},2\n`, 'ant.csv')
    const accepted = analyzeTabularSource(ant, features)
    expect(accepted.layout).toBe('jureczko-promise')
    expect(accepted.rowMetadata[0]).toMatchObject({ project: 'ant', release: '1.7', className: 'org.example.A' })
    expect(accepted.records[0]).not.toHaveProperty('bug')
  })

  it('offers explicit mappings for unfamiliar layouts and never guesses similarly named metrics', () => {
    const renamedHeaders = features.map((feature) => `source_${feature}`)
    const source = parseDelimitedSource(`${['id', ...renamedHeaders].join(',')}\nrow-a,${values.join(',')}\n`, 'unfamiliar.csv')
    const preview = analyzeTabularSource(source, features)
    expect(preview.canPredict).toBe(false)
    expect(preview.missingMetrics).toEqual(features)
    const metricMappings = Object.fromEntries(features.map((feature, index) => [feature, source.columns[index + 1].key]))
    const mapped = analyzeTabularSource(source, features, { metricMappings })
    expect(mapped.canPredict).toBe(true)
    expect(mapped.records[0]).toMatchObject({ wmc: 3, avg_cc: 0.6667 })
  })

  it('requires an explicit meaning for an ambiguous generic label column', () => {
    const source = parseDelimitedSource(`${['record_id', ...features, 'bug'].join(',')}\na,${values.join(',')},yes\n`, 'generic.csv')
    const preview = analyzeTabularSource(source, features)
    expect(preview.ambiguousLabelColumns.map((column) => column.header)).toEqual(['bug'])
    expect(preview.records[0]).not.toHaveProperty('bug')
    const ignored = analyzeTabularSource(source, features, { labelMappings: { [source.columns.at(-1)!.key]: 'ignore' } })
    expect(ignored.ambiguousLabelColumns).toEqual([])
    expect(ignored.rowMetadata[0].labels).toEqual([])
  })

  it('reports invalid rows and columns, retains missing values as null, and requires no fabricated values', () => {
    const valid = values.map(String)
    const invalid = values.map(String)
    valid[0] = ''
    invalid[10] = 'Infinity'
    const source = parseDelimitedSource([
      ['record_id', ...features].join(','),
      ['missing-a', ...valid].join(','),
      ['invalid-b', ...invalid].join(','),
    ].join('\n'), 'quality.csv')
    const parsed = analyzeTabularSource(source, features)
    expect(parsed.records).toHaveLength(1)
    expect(parsed.records[0].wmc).toBeNull()
    expect(parsed.invalidRows).toEqual([expect.objectContaining({ sourceRow: 3, columns: ['loc'] })])
    expect(parsed.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'missing-value', sourceRow: 2, column: 'wmc' }),
      expect.objectContaining({ code: 'invalid-value', sourceRow: 3, column: 'loc' }),
    ]))
  })

  it('preserves only unique valid source IDs and deterministically replaces collisions', () => {
    const source = parseDelimitedSource([
      ['record_id', 'project', 'release', 'class_name', ...features].join(','),
      ['same', 'ant', '1.7', 'A', ...values].join(','),
      ['same', 'ant', '1.7', 'B', ...values].join(','),
      ['unique', 'ant', '1.7', 'C', ...values].join(','),
    ].join('\n'), 'identifiers.csv')
    const first = analyzeTabularSource(source, features)
    const second = analyzeTabularSource(source, features)
    expect(first.records.map((record) => record.record_id)).toEqual(['ant:1.7:A', 'ant:1.7:B', 'unique'])
    expect(second.records.map((record) => record.record_id)).toEqual(first.records.map((record) => record.record_id))
  })

  it('produces identical model inputs and scores for identical records in all supported text layouts', async () => {
    const csv = analyzeTabularSource(parseDelimitedSource(`${['record_id', ...features].join(',')}\nsame,${values.join(',')}\n`, 'same.csv'), features)
    const jsonObject = Object.fromEntries([['record_id', 'same'], ...features.map((feature, index) => [feature, values[index]])])
    const json = analyzeTabularSource(parseJsonSource(JSON.stringify(jsonObject), 'same.json'), features)
    const arff = analyzeTabularSource(parseArffSource([
      '@relation same', '@attribute record_id string', ...features.map((feature) => `@attribute ${feature} numeric`), '@data', `same,${values.join(',')}`,
    ].join('\n'), 'same.arff'), features)
    const workbook = new ExcelJS.Workbook()
    const sheet = workbook.addWorksheet('Metrics')
    sheet.addRow(['record_id', ...features]); sheet.addRow(['same', ...values])
    const xlsx = analyzeTabularSource(await parseXlsxSource(xlsxArrayBuffer(await workbook.xlsx.writeBuffer()), 'same.xlsx'), features)
    const records = [csv, json, arff, xlsx].map((parsed) => canonicalRecord(parsed.records[0]))
    expect(records.slice(1)).toEqual(records.slice(1).map(() => records[0]))

    const scores = await Promise.all(records.map(async (record) => (await predictInBatches([record], 1, 'v1', async (batch) => ({
      score_description: 'test',
      predictions: batch.map((item) => ({ record_id: item.record_id, model_score: Number(item.loc) / 1000, rank: 1, model_version: 'v1' })),
    }))).predictions[0].model_score))
    expect(scores).toEqual([0.106, 0.106, 0.106, 0.106])
  })
})
