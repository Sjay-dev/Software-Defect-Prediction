import Papa from 'papaparse'
import type { MetricRecord, Prediction } from './types'

export type ParsedMetricCsv = {
  records: MetricRecord[]
  fileColumns: string[]
}

function escapeCell(value: string | number): string {
  const text = String(value)
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

export function parseMetricCsv(
  text: string,
  features: string[],
  maxBatchSize: number,
): ParsedMetricCsv {
  const parsed = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: 'greedy',
    transformHeader: (header) => header.replace(/^\uFEFF/, '').trim(),
  })
  if (parsed.errors.length > 0) {
    const error = parsed.errors[0]
    throw new Error(`CSV parsing error${error.row === undefined ? '' : ` on row ${error.row + 2}`}: ${error.message}`)
  }
  const columns = parsed.meta.fields || []
  const expected = ['record_id', ...features]
  const missing = expected.filter((column) => !columns.includes(column))
  const unknown = columns.filter((column) => !expected.includes(column))
  if (missing.length > 0) throw new Error(`Missing required columns: ${missing.join(', ')}.`)
  if (unknown.length > 0) throw new Error(`Unknown columns are not accepted: ${unknown.join(', ')}.`)
  if (parsed.data.length === 0) throw new Error('The CSV contains no data rows.')
  if (parsed.data.length > maxBatchSize) {
    throw new Error(`The CSV has ${parsed.data.length} rows; the configured batch limit is ${maxBatchSize}.`)
  }

  const identifiers = new Set<string>()
  const records = parsed.data.map((row, index) => {
    const csvRow = index + 2
    const recordId = (row.record_id || '').trim()
    if (!recordId) throw new Error(`Row ${csvRow}: record_id must not be blank.`)
    if (identifiers.has(recordId)) throw new Error(`Row ${csvRow}: duplicate record_id '${recordId}'.`)
    identifiers.add(recordId)
    const record: MetricRecord = { record_id: recordId }
    for (const feature of features) {
      const raw = row[feature]?.trim() ?? ''
      if (raw === '') {
        record[feature] = null
        continue
      }
      const value = Number(raw)
      if (!Number.isFinite(value)) {
        throw new Error(`Row ${csvRow}, ${feature}: '${raw}' is not a finite numeric value.`)
      }
      record[feature] = value
    }
    return record
  })
  return { records, fileColumns: columns }
}

export function templateCsv(features: string[]): string {
  return `${['record_id', ...features].join(',')}\r\n`
}

export function predictionsCsv(predictions: Prediction[]): string {
  const rows = predictions
    .slice()
    .sort((a, b) => a.rank - b.rank)
    .map((item) => [item.record_id, item.model_score, item.rank, item.model_version].map(escapeCell).join(','))
  return ['record_id,model_score,rank,model_version', ...rows].join('\r\n') + '\r\n'
}

export function downloadText(filename: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
