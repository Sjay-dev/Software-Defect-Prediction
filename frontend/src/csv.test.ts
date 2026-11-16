import { readFileSync } from 'node:fs'
import ExcelJS from 'exceljs'
import { describe, expect, it } from 'vitest'
import {
  MAX_CSV_ROWS,
  parseMetricCsv,
  predictionsCsv,
  templateArff,
  templateCsv,
  templateJson,
  templateXlsx,
  validateCsvFileSize,
} from './csv'

const features = [
  'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3',
  'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc',
]
const metricValues = ['3', '1', '0', '10', '18', '3', '1', '9', '1', '1.1', '106', '0', '0', '0', '0.444444444', '0', '0', '32.66666667', '1', '0.6667']
const antSample = [
  `name,version,name,${features.join(',')},bug`,
  `ant,1.7,org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory,${metricValues.join(',')},0`,
  `ant,1.7,org.apache.tools.ant.taskdefs.optional.junit.OutErrSummaryJUnitResultFormatter,${metricValues.join(',')},1`,
].join('\n')

function templateRow(recordId: string): string {
  return [recordId, ...metricValues].join(',')
}

describe('metric CSV handling', () => {
  it('keeps supporting the record_id plus 20-metric template', () => {
    const parsed = parseMetricCsv(
      `${['record_id', ...features].join(',')}\n${templateRow('class-a')}\n`,
      features,
    )
    expect(parsed.records).toHaveLength(1)
    expect(parsed.records[0]).toMatchObject({ record_id: 'class-a', wmc: 3, avg_cc: 0.6667 })
    expect(parsed.identifierSource).toBe('record_id')
    expect(parsed.identifierDescription).toContain('preserved exactly')
  })

  it('imports the exact original Ant header layout and excludes its label', () => {
    const sourcePath = process.env.ANT_17_CSV_PATH
    const source = sourcePath ? readFileSync(sourcePath, 'utf8') : antSample
    const parsed = parseMetricCsv(source, features)

    expect(parsed.records[0].record_id).toBe('ant:1.7:org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory')
    expect(parsed.records[0]).not.toHaveProperty('bug')
    expect(parsed.records[0]).not.toHaveProperty('name')
    expect(parsed.rowMetadata[0]).toMatchObject({
      project: 'ant',
      release: '1.7',
      className: 'org.apache.tools.ant.taskdefs.rmic.RmicAdapterFactory',
    })
    expect(parsed.identifierDescription).toContain('project:version:class-name')
    expect(parsed.excludedColumns.filter((column) => column.column === 'name')).toHaveLength(2)
    expect(parsed.excludedColumns).toContainEqual({ column: 'bug', columnNumber: 24, role: 'label' })
    expect(parsed.records).toHaveLength(sourcePath ? 745 : 2)
  })

  it('preserves explicit identifiers and original class names while reporting excluded columns', () => {
    const csv = [
      ['record_id', 'project', 'release', 'class_name', ...features, 'bug_count', 'notes'].join(','),
      ['explicit-1', 'ant', '1.7', 'org.example.RealClass', ...metricValues, '4', 'review'].join(','),
    ].join('\n')
    const parsed = parseMetricCsv(csv, features)

    expect(parsed.records[0].record_id).toBe('explicit-1')
    expect(parsed.rowMetadata[0].className).toBe('org.example.RealClass')
    expect(parsed.records[0]).not.toHaveProperty('bug_count')
    expect(parsed.excludedColumns).toEqual(expect.arrayContaining([
      expect.objectContaining({ column: 'record_id', role: 'identifier' }),
      expect.objectContaining({ column: 'bug_count', role: 'label' }),
      expect.objectContaining({ column: 'notes', role: 'extra' }),
    ]))
  })

  it('generates deterministic unique row identifiers when metadata is unavailable', () => {
    const csv = [features.join(','), metricValues.join(','), metricValues.join(',')].join('\n')
    const first = parseMetricCsv(csv, features)
    const second = parseMetricCsv(csv, features)
    expect(first.records.map((record) => record.record_id)).toEqual(['row-000001', 'row-000002'])
    expect(second.records.map((record) => record.record_id)).toEqual(first.records.map((record) => record.record_id))
    expect(first.identifierSource).toBe('row')
  })

  it('makes duplicate metadata identifiers unique with deterministic row suffixes', () => {
    const csv = [
      ['project', 'release', 'class_name', ...features].join(','),
      ['ant', '1.7', 'SameClass', ...metricValues].join(','),
      ['ant', '1.7', 'SameClass', ...metricValues].join(','),
    ].join('\n')
    const parsed = parseMetricCsv(csv, features)
    expect(parsed.records.map((record) => record.record_id)).toEqual([
      'ant:1.7:SameClass',
      'ant:1.7:SameClass:row-000002',
    ])
  })

  it('rejects missing, ambiguous, and incompatible metrics with actionable errors', () => {
    const missing = features.filter((feature) => feature !== 'avg_cc')
    expect(() => parseMetricCsv(
      `${['record_id', ...missing].join(',')}\na,${metricValues.slice(0, -1).join(',')}\n`,
      features,
    )).toThrow('Missing required metric columns: avg_cc')

    const duplicate = ['record_id', ...features, 'wmc'].join(',')
    expect(() => parseMetricCsv(`${duplicate}\na,${metricValues.join(',')},3\n`, features)).toThrow(
      "Required metric 'wmc' appears more than once",
    )

    const badValues = [...metricValues]
    badValues[0] = 'not-a-number'
    expect(() => parseMetricCsv(
      `${['record_id', ...features].join(',')}\na,${badValues.join(',')}\n`,
      features,
    )).toThrow("Row 2, wmc: 'not-a-number' is not a finite numeric value")
  })

  it('enforces total file and row limits independently from API batch size', () => {
    validateCsvFileSize(10 * 1024 * 1024)
    expect(() => validateCsvFileSize(10 * 1024 * 1024 + 1)).toThrow('10 MB upload limit')
    expect(() => parseMetricCsv(antSample, features, 1)).toThrow('upload limit is 1 rows')
    expect(MAX_CSV_ROWS).toBe(10_000)
  })

  it('generates schema and result downloads with stable columns and optional class names', () => {
    expect(templateCsv(['wmc', 'loc'])).toBe('record_id,wmc,loc\r\n')
    expect(predictionsCsv([{ record_id: 'a,b', model_score: 0.7, rank: 1, model_version: 'v1' }])).toContain('"a,b",0.7,1,v1')
    expect(predictionsCsv([{ record_id: 'a', class_name: 'org.example.A', model_score: 0.7, rank: 1, model_version: 'v1' }]))
      .toContain('record_id,class_name,model_score,rank,model_version')
    expect(predictionsCsv([{ record_id: 'a', source_row: 7, labels: { bug: 2 }, model_score: 0.7, rank: 1, model_version: 'v1' }]))
      .toContain('record_id,source_row,model_score,rank,model_version,outcome_bug')
  })

  it('generates templates for every supported format', async () => {
    expect(templateCsv(['wmc'], '\t')).toBe('record_id\twmc\r\n')
    expect(JSON.parse(templateJson(['wmc']))).toEqual({ record_id: 'example-class', wmc: null })
    expect(templateArff(['wmc'])).toContain('@attribute wmc numeric')
    const workbook = new ExcelJS.Workbook()
    await workbook.xlsx.load(await templateXlsx(['wmc', 'loc']).then((blob) => blob.arrayBuffer()))
    expect(workbook.getWorksheet('Metrics')?.getRow(1).values).toEqual([undefined, 'record_id', 'wmc', 'loc'])
  })
})
