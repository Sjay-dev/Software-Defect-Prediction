import { analyzeTabularSource, MAX_IMPORT_FILE_BYTES, MAX_IMPORT_ROWS, parseDelimitedSource, validateImportFileSize } from './file-import'
import type { ImportRowMetadata, ParsedMetricFile } from './file-import'
import type { MetricRecord, Prediction } from './types'

export const MAX_CSV_FILE_BYTES = MAX_IMPORT_FILE_BYTES
export const MAX_CSV_ROWS = MAX_IMPORT_ROWS
export { MAX_RECORD_ID_LENGTH } from './file-import'

export type ExcludedCsvColumn = Omit<ParsedMetricFile['excludedColumns'][number], 'key'>
export type CsvRowMetadata = ImportRowMetadata & { csvRow: number }
export type ParsedMetricCsv = Omit<ParsedMetricFile, 'rowMetadata' | 'excludedColumns'> & {
  rowMetadata: CsvRowMetadata[]
  excludedColumns: ExcludedCsvColumn[]
}

function escapeCell(value: string | number | boolean | null): string {
  const text = value === null ? '' : String(value)
  return /[",\r\n\t;]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

export function validateCsvFileSize(size: number): void {
  if (size > MAX_CSV_FILE_BYTES) throw new Error(`The CSV is larger than the ${MAX_CSV_FILE_BYTES / 1024 / 1024} MB upload limit.`)
}

export function parseMetricCsv(text: string, features: string[], maxRows = MAX_CSV_ROWS): ParsedMetricCsv {
  const source = parseDelimitedSource(text, '', 'csv', 'auto')
  if (source.rows.length > maxRows) {
    throw new Error(`The CSV has ${source.rows.length} rows; the upload limit is ${maxRows.toLocaleString()} rows.`)
  }
  const parsed = analyzeTabularSource(source, features)
  const firstBlocking = parsed.issues.find((issue) => issue.severity === 'error')
  if (firstBlocking?.code === 'missing-metrics') {
    throw new Error(
      `Missing required metric columns: ${parsed.missingMetrics.join(', ')}. ` +
      'Metric names must match the documented schema or be mapped explicitly; values are never fabricated or inferred.',
    )
  }
  if (firstBlocking?.code === 'duplicate-header') {
    const duplicatedFeature = features.find((feature) => firstBlocking.column?.toLocaleLowerCase('en-US').startsWith(feature.toLocaleLowerCase('en-US')))
    if (duplicatedFeature) throw new Error(`Required metric '${duplicatedFeature}' appears more than once. Metric mappings must be unambiguous.`)
    throw new Error(firstBlocking.message)
  }
  if (firstBlocking) {
    if (firstBlocking.code === 'invalid-value' || firstBlocking.code === 'formula-value') {
      throw new Error(`Row ${firstBlocking.sourceRow}, ${firstBlocking.column}: '${firstBlocking.value}' is not a finite numeric value.`)
    }
    throw new Error(firstBlocking.message)
  }
  return {
    ...parsed,
    excludedColumns: parsed.excludedColumns.map(({ key: _key, ...column }) => column),
    rowMetadata: parsed.rowMetadata.map((row) => ({ ...row, csvRow: row.sourceRow })),
  }
}

export function templateCsv(features: string[], delimiter: ',' | '\t' | ';' = ','): string {
  return `${['record_id', ...features].join(delimiter)}\r\n`
}

export function templateJson(features: string[]): string {
  return `${JSON.stringify(Object.fromEntries(['record_id', ...features].map((column) => [column, column === 'record_id' ? 'example-class' : null])), null, 2)}\n`
}

export function templateArff(features: string[]): string {
  return [
    '@relation software_defect_prediction',
    '',
    '@attribute record_id string',
    ...features.map((feature) => `@attribute ${feature} numeric`),
    '',
    '@data',
    `% Add dense rows with ${features.length + 1} comma-separated values.`,
    '',
  ].join('\n')
}

export function predictionsCsv(predictions: Prediction[]): string {
  const includeSourceRow = predictions.some((item) => item.source_row !== undefined)
  const includeClassName = predictions.some((item) => Boolean(item.class_name))
  const labelColumns = [...new Set(predictions.flatMap((item) => Object.keys(item.labels ?? {})))].sort()
  const header = [
    'record_id',
    ...(includeSourceRow ? ['source_row'] : []),
    ...(includeClassName ? ['class_name'] : []),
    'model_score', 'rank', 'model_version',
    ...labelColumns.map((column) => `outcome_${column}`),
  ]
  const rows = predictions
    .slice()
    .sort((left, right) => left.rank - right.rank)
    .map((item) => [
      item.record_id,
      ...(includeSourceRow ? [item.source_row ?? ''] : []),
      ...(includeClassName ? [item.class_name ?? ''] : []),
      item.model_score,
      item.rank,
      item.model_version,
      ...labelColumns.map((column) => item.labels?.[column] ?? ''),
    ].map((value) => escapeCell(value as string | number | boolean | null)).join(','))
  return [header.join(','), ...rows].join('\r\n') + '\r\n'
}

export function downloadText(filename: string, content: string, type = 'text/plain;charset=utf-8'): void {
  downloadBlob(filename, new Blob([content], { type }))
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export async function templateXlsx(features: string[]): Promise<Blob> {
  const ExcelJS = (await import('exceljs')).default
  const workbook = new ExcelJS.Workbook()
  const worksheet = workbook.addWorksheet('Metrics')
  worksheet.addRow(['record_id', ...features])
  worksheet.getRow(1).font = { bold: true }
  const buffer = await workbook.xlsx.writeBuffer()
  return new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
}

export function labelsForPrediction(metadata: ImportRowMetadata): Record<string, string | number | boolean | null> {
  return Object.fromEntries(metadata.labels.map((label) => [label.column, label.rawValue]))
}

export function invalidRowsCsv(rows: ParsedMetricFile['invalidRows']): string {
  const body = rows.map((row) => [
    row.sourceRow,
    row.columns.join('; '),
    row.reasons.join('; '),
  ].map((value) => escapeCell(value)).join(','))
  return ['source_row,invalid_columns,reasons', ...body].join('\r\n') + '\r\n'
}

export function recordsOnly(parsed: ParsedMetricFile): MetricRecord[] {
  return parsed.records
}
