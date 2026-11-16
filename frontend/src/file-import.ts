import type ExcelJS from 'exceljs'
import Papa from 'papaparse'
import type { MetricRecord } from './types'

export const MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024
export const MAX_IMPORT_ROWS = 10_000
export const MAX_IMPORT_COLUMNS = 256
export const MAX_RECORD_ID_LENGTH = 200

export type ImportFormat = 'csv' | 'tsv' | 'json' | 'xlsx' | 'arff'
export type DelimiterChoice = 'auto' | ',' | ';' | '\t'
export type LabelMeaning = 'bug_count' | 'defective'
export type ImportIssueSeverity = 'error' | 'warning'

type FormulaCell = { kind: 'formula'; formula: string }
export type SourceCell = string | number | boolean | null | FormulaCell

export type SourceColumn = {
  key: string
  header: string
  normalizedHeader: string
  columnNumber: number
  declaredType?: string
}

export type SourceRow = {
  sourceRow: number
  values: SourceCell[]
}

export type TabularSource = {
  format: ImportFormat
  fileName: string
  formatDescription: string
  columns: SourceColumn[]
  rows: SourceRow[]
  delimiter?: ',' | ';' | '\t'
  sheetName?: string
  availableSheets?: string[]
}

export type ImportIssue = {
  severity: ImportIssueSeverity
  code: string
  message: string
  sourceRow?: number
  column?: string
  value?: string
}

export type ExcludedImportColumn = {
  key: string
  column: string
  columnNumber: number
  role: 'identifier' | 'metadata' | 'label' | 'extra'
}

export type ImportedLabel = {
  column: string
  meaning: LabelMeaning
  rawValue: string | number | boolean | null
  binaryValue: 0 | 1 | null
}

export type ImportRowMetadata = {
  sourceRow: number
  recordId: string
  project: string | null
  release: string | null
  className: string | null
  labels: ImportedLabel[]
}

export type ImportMapping = {
  metrics: Record<string, string | null>
  recordId: string | null
  project: string | null
  release: string | null
  className: string | null
  labels: Record<string, LabelMeaning | 'ignore'>
}

export type InvalidImportRow = {
  sourceRow: number
  columns: string[]
  reasons: string[]
}

export type ParsedMetricFile = {
  source: TabularSource
  records: MetricRecord[]
  fileColumns: string[]
  excludedColumns: ExcludedImportColumn[]
  identifierDescription: string
  identifierSource: 'record_id' | 'metadata' | 'row' | 'mixed'
  rowMetadata: ImportRowMetadata[]
  layout: 'dashboard-schema' | 'jureczko-promise' | 'explicit-mapping'
  mapping: ImportMapping
  recognizedMetrics: string[]
  missingMetrics: string[]
  ambiguousLabelColumns: SourceColumn[]
  issues: ImportIssue[]
  invalidRows: InvalidImportRow[]
  totalRows: number
  validRows: number
  canPredict: boolean
}

export type ParseImportOptions = {
  delimiter?: DelimiterChoice
  sheetName?: string
  metricMappings?: Record<string, string | null>
  labelMappings?: Record<string, LabelMeaning | 'ignore'>
}

const DECIMAL_NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/
const AMBIGUOUS_LABEL_NAMES = new Set(['bug', 'label', 'target', 'class', 'is_buggy', 'is_defective'])

function normalizedHeader(value: string): string {
  return value.replace(/^\uFEFF/, '').trim().toLocaleLowerCase('en-US')
}

function displayHeader(value: SourceCell, index: number): string {
  if (isFormulaCell(value)) return `[formula column ${index + 1}]`
  const text = value === null ? '' : String(value)
  return index === 0 ? text.replace(/^\uFEFF/, '').trim() : text.trim()
}

function createColumns(headers: SourceCell[], declaredTypes?: string[]): SourceColumn[] {
  if (headers.length === 0 || headers.every((header) => displayHeader(header, 0) === '')) {
    throw new Error('The file header is empty.')
  }
  if (headers.length > MAX_IMPORT_COLUMNS) {
    throw new Error(`The file has ${headers.length} columns; the parsing limit is ${MAX_IMPORT_COLUMNS}.`)
  }
  return headers.map((header, index) => {
    if (isFormulaCell(header)) throw new Error(`Header column ${index + 1} contains a spreadsheet formula. Formula cells are not imported.`)
    const text = displayHeader(header, index)
    if (!text) throw new Error(`Header column ${index + 1} is blank. Give every source column a name.`)
    return {
      key: `column-${index + 1}`,
      header: text,
      normalizedHeader: normalizedHeader(text),
      columnNumber: index + 1,
      declaredType: declaredTypes?.[index],
    }
  })
}

function isFormulaCell(value: SourceCell): value is FormulaCell {
  return typeof value === 'object' && value !== null && 'kind' in value && value.kind === 'formula'
}

function delimiterName(delimiter: ',' | ';' | '\t'): string {
  if (delimiter === '\t') return 'tab'
  if (delimiter === ';') return 'semicolon'
  return 'comma'
}

function normalizePapaErrors(errors: Papa.ParseError[]): void {
  const serious = errors.find((error) => error.code !== 'UndetectableDelimiter')
  if (serious) {
    const row = serious.row === undefined ? '' : ` on row ${serious.row + 1}`
    throw new Error(`Delimited-text parsing error${row}: ${serious.message}`)
  }
}

function detectDelimiter(text: string, choice: DelimiterChoice): ',' | ';' | '\t' {
  if (choice !== 'auto') return choice
  const probe = Papa.parse<string[]>(text.slice(0, 128 * 1024), {
    header: false,
    delimitersToGuess: [',', ';', '\t'],
    preview: 5,
  })
  const delimiter = probe.meta.delimiter
  if (delimiter === ',' || delimiter === ';' || delimiter === '\t') return delimiter
  throw new Error('Could not detect a comma, semicolon, or tab delimiter. Select a delimiter explicitly.')
}

export function parseDelimitedSource(
  text: string,
  fileName: string,
  format: 'csv' | 'tsv' = 'csv',
  delimiterChoice: DelimiterChoice = format === 'tsv' ? '\t' : 'auto',
): TabularSource {
  validateImportFileSize(new TextEncoder().encode(text).byteLength)
  const cleanText = text.replace(/^\uFEFF/, '')
  const delimiter = detectDelimiter(cleanText, delimiterChoice)
  const parsed = Papa.parse<string[]>(cleanText, {
    delimiter,
    header: false,
    skipEmptyLines: 'greedy',
  })
  normalizePapaErrors(parsed.errors)
  if (parsed.data.length === 0) throw new Error('The file is empty.')
  const columns = createColumns(parsed.data[0])
  const rawRows = parsed.data.slice(1)
  if (rawRows.length === 0) throw new Error('The file contains no data rows.')
  if (rawRows.length > MAX_IMPORT_ROWS) {
    throw new Error(`The file has ${rawRows.length.toLocaleString()} rows; the parsing limit is ${MAX_IMPORT_ROWS.toLocaleString()}.`)
  }
  const rows = rawRows.map((values, index) => ({
    sourceRow: index + 2,
    values: values.map((value) => value),
  }))
  const oversizedRow = rows.find((row) => row.values.length > columns.length)
  if (oversizedRow) {
    throw new Error(
      `Delimited-text row ${oversizedRow.sourceRow} has ${oversizedRow.values.length} values but the header has ${columns.length} columns.`,
    )
  }
  return {
    format,
    fileName,
    formatDescription: `${format.toUpperCase()} (${delimiterName(delimiter)}-delimited)`,
    columns,
    rows,
    delimiter,
  }
}

function jsonScalar(value: unknown, row: number, column: string): SourceCell {
  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value
  throw new Error(`JSON record ${row}, property '${column}' contains an object or array. Only scalar cell values are supported.`)
}

export function parseJsonSource(text: string, fileName: string): TabularSource {
  validateImportFileSize(new TextEncoder().encode(text).byteLength)
  let decoded: unknown
  try {
    const cleanText = text.replace(/^\uFEFF/, '')
    assertNoDuplicateJsonKeys(cleanText)
    decoded = JSON.parse(cleanText)
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'invalid JSON'
    throw new Error(`JSON parsing error: ${reason}`)
  }
  const objects = Array.isArray(decoded) ? decoded : [decoded]
  if (objects.length === 0) throw new Error('The JSON array contains no records.')
  if (objects.length > MAX_IMPORT_ROWS) {
    throw new Error(`The JSON contains ${objects.length.toLocaleString()} records; the parsing limit is ${MAX_IMPORT_ROWS.toLocaleString()}.`)
  }
  objects.forEach((record, index) => {
    if (!record || typeof record !== 'object' || Array.isArray(record)) {
      throw new Error(`JSON record ${index + 1} must be an object.`)
    }
  })
  const headers: string[] = []
  const seen = new Set<string>()
  for (const record of objects as Array<Record<string, unknown>>) {
    for (const key of Object.keys(record)) {
      if (!seen.has(key)) {
        seen.add(key)
        headers.push(key)
      }
    }
  }
  const columns = createColumns(headers)
  const rows = (objects as Array<Record<string, unknown>>).map((record, index) => ({
    sourceRow: index + 1,
    values: headers.map((header) => jsonScalar(record[header] ?? null, index + 1, header)),
  }))
  return { format: 'json', fileName, formatDescription: 'JSON records', columns, rows }
}

function assertNoDuplicateJsonKeys(text: string): void {
  let position = 0
  const whitespace = () => { while (/\s/.test(text[position] ?? '')) position += 1 }
  const stringToken = (): string => {
    const start = position
    position += 1
    while (position < text.length) {
      if (text[position] === '\\') {
        position += 2
        continue
      }
      if (text[position] === '"') {
        position += 1
        return JSON.parse(text.slice(start, position)) as string
      }
      position += 1
    }
    throw new Error('Unterminated JSON string.')
  }
  const value = (): void => {
    whitespace()
    if (text[position] === '{') {
      object()
      return
    }
    if (text[position] === '[') {
      position += 1
      whitespace()
      if (text[position] === ']') { position += 1; return }
      while (position < text.length) {
        value(); whitespace()
        if (text[position] === ']') { position += 1; return }
        if (text[position] !== ',') throw new Error('Malformed JSON array.')
        position += 1
      }
      throw new Error('Unterminated JSON array.')
    }
    if (text[position] === '"') {
      stringToken()
      return
    }
    const start = position
    while (position < text.length && !/[\s,}\]]/.test(text[position])) position += 1
    if (position === start) throw new Error('Malformed JSON value.')
  }
  const object = (): void => {
    position += 1
    const keys = new Set<string>()
    whitespace()
    if (text[position] === '}') { position += 1; return }
    while (position < text.length) {
      whitespace()
      if (text[position] !== '"') throw new Error('JSON object keys must be strings.')
      const key = stringToken()
      if (keys.has(key)) throw new Error(`Duplicate JSON property '${key}' is not supported; record keys must be unique.`)
      keys.add(key)
      whitespace()
      if (text[position] !== ':') throw new Error('Malformed JSON object.')
      position += 1
      value()
      whitespace()
      if (text[position] === '}') { position += 1; return }
      if (text[position] !== ',') throw new Error('Malformed JSON object.')
      position += 1
    }
    throw new Error('Unterminated JSON object.')
  }
  value()
}

function parseArffAttribute(line: string, lineNumber: number): { name: string; type: string } {
  const rest = line.replace(/^@attribute\s+/i, '').trim()
  let name = ''
  let type = ''
  if (rest.startsWith("'") || rest.startsWith('"')) {
    const quote = rest[0]
    let end = 1
    let escaped = false
    for (; end < rest.length; end += 1) {
      const character = rest[end]
      if (!escaped && character === quote) break
      escaped = !escaped && character === '\\'
      if (character !== '\\') escaped = false
    }
    if (end >= rest.length) throw new Error(`ARFF line ${lineNumber}: unterminated quoted attribute name.`)
    name = rest.slice(1, end).replace(/\\(['"\\])/g, '$1')
    type = rest.slice(end + 1).trim()
  } else {
    const separator = rest.search(/\s/)
    if (separator < 1) throw new Error(`ARFF line ${lineNumber}: malformed @attribute declaration.`)
    name = rest.slice(0, separator)
    type = rest.slice(separator).trim()
  }
  if (!type) throw new Error(`ARFF line ${lineNumber}: attribute '${name}' has no type.`)
  const normalizedType = type.toLocaleLowerCase('en-US')
  const supported = ['numeric', 'real', 'integer', 'string'].includes(normalizedType) || /^\{.*\}$/.test(type)
  if (!supported) {
    throw new Error(
      `ARFF line ${lineNumber}: attribute '${name}' uses unsupported type '${type}'. ` +
      'Only numeric, real, integer, string, and nominal attributes are supported; relational and date structures are rejected.',
    )
  }
  return { name, type }
}

export function parseArffSource(text: string, fileName: string): TabularSource {
  validateImportFileSize(new TextEncoder().encode(text).byteLength)
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/)
  const attributes: Array<{ name: string; type: string }> = []
  const rows: SourceRow[] = []
  let inData = false
  let relationSeen = false
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const lineNumber = lineIndex + 1
    const trimmed = lines[lineIndex].trim()
    if (!trimmed || trimmed.startsWith('%')) continue
    if (!inData) {
      if (/^@relation(?:\s|$)/i.test(trimmed)) {
        relationSeen = true
        continue
      }
      if (/^@attribute\s+/i.test(trimmed)) {
        attributes.push(parseArffAttribute(trimmed, lineNumber))
        if (attributes.length > MAX_IMPORT_COLUMNS) throw new Error(`The ARFF has more than ${MAX_IMPORT_COLUMNS} attributes.`)
        continue
      }
      if (/^@data(?:\s|$)/i.test(trimmed)) {
        if (attributes.length === 0) throw new Error(`ARFF line ${lineNumber}: @data appears before any attributes.`)
        inData = true
        continue
      }
      throw new Error(`ARFF line ${lineNumber}: unsupported directive or content before @data.`)
    }
    if (trimmed.startsWith('{')) {
      throw new Error(`ARFF line ${lineNumber}: sparse ARFF rows are not supported. Convert the dataset to dense rows first.`)
    }
    if (trimmed.startsWith('@')) throw new Error(`ARFF line ${lineNumber}: directives are not allowed after @data.`)
    const parsed = Papa.parse<string[]>(trimmed, { delimiter: ',', header: false })
    normalizePapaErrors(parsed.errors)
    if (parsed.data.length !== 1) throw new Error(`ARFF line ${lineNumber}: multiline values are not supported.`)
    const values = parsed.data[0]
    if (values.length !== attributes.length) {
      throw new Error(`ARFF line ${lineNumber} has ${values.length} values but ${attributes.length} attributes were declared.`)
    }
    rows.push({ sourceRow: lineNumber, values: values.map((value) => value.trim() === '?' ? null : value.trim()) })
    if (rows.length > MAX_IMPORT_ROWS) throw new Error(`The ARFF has more than ${MAX_IMPORT_ROWS.toLocaleString()} data rows.`)
  }
  if (!relationSeen) throw new Error('ARFF is missing an @relation declaration.')
  if (!inData) throw new Error('ARFF is missing an @data section.')
  if (rows.length === 0) throw new Error('The ARFF @data section contains no records.')
  const columns = createColumns(attributes.map((attribute) => attribute.name), attributes.map((attribute) => attribute.type))
  return { format: 'arff', fileName, formatDescription: 'ARFF (dense)', columns, rows }
}

function excelCellValue(cell: ExcelJS.Cell, formulaValueType: number): SourceCell {
  if (cell.type === formulaValueType) {
    const value = cell.value as { formula?: string; sharedFormula?: string }
    return { kind: 'formula', formula: value.formula || value.sharedFormula || '(shared formula)' }
  }
  const value = cell.value
  if (value === null || value === undefined) return null
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value
  if (value instanceof Date) return value.toISOString()
  if (typeof value === 'object' && 'richText' in value) return value.richText.map((part) => part.text).join('')
  if (typeof value === 'object' && 'text' in value && typeof value.text === 'string') return value.text
  if (typeof value === 'object' && 'error' in value) return String(value.error)
  return String(value)
}

export async function parseXlsxSource(buffer: ArrayBuffer, fileName: string, selectedSheet?: string): Promise<TabularSource> {
  validateImportFileSize(buffer.byteLength)
  const ExcelRuntime = (await import('exceljs')).default
  const workbook = new ExcelRuntime.Workbook()
  try {
    await workbook.xlsx.load(buffer)
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'invalid workbook'
    throw new Error(`XLSX parsing error: ${reason}`)
  }
  const availableSheets = workbook.worksheets.map((worksheet) => worksheet.name)
  if (availableSheets.length === 0) throw new Error('The XLSX workbook contains no worksheets.')
  const sheetName = selectedSheet ?? availableSheets[0]
  const worksheet = workbook.getWorksheet(sheetName)
  if (!worksheet) throw new Error(`Worksheet '${sheetName}' was not found in the XLSX workbook.`)
  if (worksheet.actualColumnCount > MAX_IMPORT_COLUMNS) {
    throw new Error(`Worksheet '${sheetName}' has ${worksheet.actualColumnCount} columns; the parsing limit is ${MAX_IMPORT_COLUMNS}.`)
  }
  if (worksheet.actualRowCount < 2) throw new Error(`Worksheet '${sheetName}' must contain a header and at least one data row.`)
  if (worksheet.actualRowCount - 1 > MAX_IMPORT_ROWS) {
    throw new Error(`Worksheet '${sheetName}' has more than ${MAX_IMPORT_ROWS.toLocaleString()} data rows.`)
  }
  const columnCount = worksheet.actualColumnCount
  const headerRow = worksheet.getRow(1)
  const columns = createColumns(Array.from({ length: columnCount }, (_, index) => excelCellValue(headerRow.getCell(index + 1), ExcelRuntime.ValueType.Formula)))
  const rows: SourceRow[] = []
  for (let rowNumber = 2; rowNumber <= worksheet.actualRowCount; rowNumber += 1) {
    const row = worksheet.getRow(rowNumber)
    const values = Array.from({ length: columnCount }, (_, index) => excelCellValue(row.getCell(index + 1), ExcelRuntime.ValueType.Formula))
    if (values.every((value) => value === null || value === '')) continue
    rows.push({ sourceRow: rowNumber, values })
  }
  if (rows.length === 0) throw new Error(`Worksheet '${sheetName}' contains no data records.`)
  return {
    format: 'xlsx',
    fileName,
    formatDescription: `XLSX worksheet '${sheetName}'`,
    columns,
    rows,
    sheetName,
    availableSheets,
  }
}

function indexesByNormalizedHeader(columns: SourceColumn[]): Map<string, SourceColumn[]> {
  const result = new Map<string, SourceColumn[]>()
  for (const column of columns) {
    const matches = result.get(column.normalizedHeader) ?? []
    matches.push(column)
    result.set(column.normalizedHeader, matches)
  }
  return result
}

function firstUnique(byName: Map<string, SourceColumn[]>, ...names: string[]): string | null {
  for (const name of names) {
    const matches = byName.get(name) ?? []
    if (matches.length === 1) return matches[0].key
  }
  return null
}

function sourceValue(row: SourceRow, key: string | null, columns: SourceColumn[]): SourceCell {
  if (key === null) return null
  const column = columns.find((candidate) => candidate.key === key)
  return column ? (row.values[column.columnNumber - 1] ?? null) : null
}

function textValue(value: SourceCell): string {
  if (value === null || isFormulaCell(value)) return ''
  return String(value).trim()
}

function strictNumber(value: SourceCell): { kind: 'missing' } | { kind: 'invalid'; raw: string } | { kind: 'valid'; value: number } {
  if (value === null || (typeof value === 'string' && (value.trim() === '' || value.trim() === '?'))) return { kind: 'missing' }
  if (isFormulaCell(value)) return { kind: 'invalid', raw: `formula: ${value.formula}` }
  if (typeof value === 'boolean') return { kind: 'invalid', raw: String(value) }
  if (typeof value === 'number') return Number.isFinite(value) ? { kind: 'valid', value } : { kind: 'invalid', raw: String(value) }
  const raw = value.trim()
  if (!DECIMAL_NUMBER.test(raw)) return { kind: 'invalid', raw }
  const number = Number(raw)
  return Number.isFinite(number) ? { kind: 'valid', value: number } : { kind: 'invalid', raw }
}

function stableHash(value: string): string {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

function limitIdentifier(value: string): string {
  if (value.length <= MAX_RECORD_ID_LENGTH) return value
  const suffix = `~${stableHash(value)}`
  return `${value.slice(0, MAX_RECORD_ID_LENGTH - suffix.length)}${suffix}`
}

function safeFileStem(fileName: string): string {
  const withoutExtension = fileName.replace(/\.[^.]+$/, '')
  return withoutExtension.trim().replace(/[^\p{L}\p{N}._-]+/gu, '-') || ''
}

function generatedIdentifierBase(source: TabularSource, idRowNumber: number, project: string, release: string, className: string): { base: string; metadata: boolean } {
  const parts = [project, release, className].filter(Boolean)
  if (parts.length > 0) return { base: parts.join(':'), metadata: true }
  const rowPart = `row-${String(idRowNumber).padStart(6, '0')}`
  const stem = safeFileStem(source.fileName)
  return { base: stem ? `${stem}:${rowPart}` : rowPart, metadata: false }
}

function binaryLabel(meaning: LabelMeaning, raw: SourceCell): 0 | 1 | null {
  if (raw === null || textValue(raw) === '') return null
  if (meaning === 'bug_count') {
    const parsed = strictNumber(raw)
    if (parsed.kind !== 'valid' || parsed.value < 0) return null
    return parsed.value > 0 ? 1 : 0
  }
  if (typeof raw === 'boolean') return raw ? 1 : 0
  const normalized = textValue(raw).toLocaleLowerCase('en-US')
  if (normalized === '1' || normalized === 'true') return 1
  if (normalized === '0' || normalized === 'false') return 0
  return null
}

function rawLabelValue(value: SourceCell): string | number | boolean | null {
  if (isFormulaCell(value)) return `formula: ${value.formula}`
  return value
}

export function analyzeTabularSource(
  source: TabularSource,
  features: string[],
  options: Pick<ParseImportOptions, 'metricMappings' | 'labelMappings'> = {},
): ParsedMetricFile {
  const issues: ImportIssue[] = []
  const byName = indexesByNormalizedHeader(source.columns)
  const nameColumns = byName.get('name') ?? []
  const versionColumns = byName.get('version') ?? []
  const allCanonicalMetricsPresent = features.every((feature) => (byName.get(normalizedHeader(feature)) ?? []).length === 1)
  const hasDocumentedJureczkoPrefix = source.columns.slice(0, 3).map((column) => column.normalizedHeader).join(',') === 'name,version,name'
  const isJureczko = hasDocumentedJureczkoPrefix && nameColumns.length === 2 && versionColumns.length === 1 && allCanonicalMetricsPresent

  for (const [name, columns] of byName) {
    if (columns.length <= 1) continue
    if (name === 'name' && isJureczko) continue
    issues.push({
      severity: 'error',
      code: 'duplicate-header',
      column: columns.map((column) => `${column.header} (#${column.columnNumber})`).join(', '),
      message: `Header '${columns[0].header}' appears ${columns.length} times after trimming and case normalization. Rename duplicates before importing.`,
    })
  }

  const metricMapping: Record<string, string | null> = {}
  for (const feature of features) {
    const manual = options.metricMappings?.[feature]
    if (manual !== undefined) {
      metricMapping[feature] = manual
      continue
    }
    const exact = byName.get(normalizedHeader(feature)) ?? []
    metricMapping[feature] = exact.length === 1 ? exact[0].key : null
  }
  const mappedMetricKeys = Object.values(metricMapping).filter((key): key is string => key !== null)
  const duplicateMetricSources = mappedMetricKeys.filter((key, index) => mappedMetricKeys.indexOf(key) !== index)
  if (duplicateMetricSources.length > 0) {
    issues.push({ severity: 'error', code: 'duplicate-metric-mapping', message: 'One source column cannot be mapped to more than one required metric.' })
  }

  const mapping: ImportMapping = {
    metrics: metricMapping,
    recordId: firstUnique(byName, 'record_id'),
    project: isJureczko ? nameColumns[0].key : firstUnique(byName, 'project'),
    release: isJureczko ? versionColumns[0].key : firstUnique(byName, 'release', 'version'),
    className: isJureczko ? nameColumns[1].key : firstUnique(byName, 'class_name'),
    labels: {},
  }

  const labelColumns = new Map<string, LabelMeaning>()
  const addRecognizedLabel = (name: string, meaning: LabelMeaning) => {
    const matches = byName.get(name) ?? []
    if (matches.length === 1) labelColumns.set(matches[0].key, meaning)
  }
  addRecognizedLabel('bug_count', 'bug_count')
  addRecognizedLabel('defective', 'defective')
  if (isJureczko) addRecognizedLabel('bug', 'bug_count')
  for (const [key, meaning] of labelColumns) mapping.labels[key] = meaning
  for (const [key, meaning] of Object.entries(options.labelMappings ?? {})) mapping.labels[key] = meaning

  const ambiguousLabelColumns = source.columns.filter((column) =>
    AMBIGUOUS_LABEL_NAMES.has(column.normalizedHeader) && !(column.key in mapping.labels),
  )

  const missingMetrics = features.filter((feature) => mapping.metrics[feature] === null)
  const recognizedMetrics = features.filter((feature) => mapping.metrics[feature] !== null)
  if (missingMetrics.length > 0) {
    issues.push({
      severity: 'error',
      code: 'missing-metrics',
      message: `Missing mappings for required metrics: ${missingMetrics.join(', ')}. Select the matching source columns; unrelated columns are never substituted automatically.`,
    })
  }

  const metadataKeys = new Set([mapping.recordId, mapping.project, mapping.release, mapping.className].filter((key): key is string => key !== null))
  const metricKeys = new Set(mappedMetricKeys)
  const activeLabelKeys = new Set(Object.entries(mapping.labels).filter(([, meaning]) => meaning !== 'ignore').map(([key]) => key))
  const excludedColumns = source.columns.flatMap((column): ExcludedImportColumn[] => {
    if (metricKeys.has(column.key)) return []
    let role: ExcludedImportColumn['role'] = 'extra'
    if (column.key === mapping.recordId) role = 'identifier'
    else if (metadataKeys.has(column.key)) role = 'metadata'
    else if (activeLabelKeys.has(column.key)) role = 'label'
    return [{ key: column.key, column: column.header, columnNumber: column.columnNumber, role }]
  })

  const explicitIdValues = source.rows.map((row) => textValue(sourceValue(row, mapping.recordId, source.columns)))
  const validExplicitIdCounts = new Map<string, number>()
  for (const value of explicitIdValues) {
    if (value && value.length <= MAX_RECORD_ID_LENGTH) validExplicitIdCounts.set(value, (validExplicitIdCounts.get(value) ?? 0) + 1)
  }
  const reservedExplicitIds = new Set(
    [...validExplicitIdCounts.entries()].filter(([, count]) => count === 1).map(([identifier]) => identifier),
  )

  const usedIdentifiers = new Set<string>()
  const records: MetricRecord[] = []
  const rowMetadata: ImportRowMetadata[] = []
  const invalidRows: InvalidImportRow[] = []
  let explicitIdentifiers = 0
  let metadataIdentifiers = 0
  let rowIdentifiers = 0

  source.rows.forEach((row, rowIndex) => {
    const idRowNumber = source.fileName ? row.sourceRow : rowIndex + 1
    const project = textValue(sourceValue(row, mapping.project, source.columns))
    const release = textValue(sourceValue(row, mapping.release, source.columns))
    const className = textValue(sourceValue(row, mapping.className, source.columns))
    const explicitId = explicitIdValues[rowIndex]
    let recordId = ''
    if (explicitId && explicitId.length <= MAX_RECORD_ID_LENGTH && validExplicitIdCounts.get(explicitId) === 1 && !usedIdentifiers.has(explicitId)) {
      recordId = explicitId
      explicitIdentifiers += 1
    } else {
      const generated = generatedIdentifierBase(source, idRowNumber, project, release, className)
      recordId = limitIdentifier(generated.base)
      if (usedIdentifiers.has(recordId) || reservedExplicitIds.has(recordId)) recordId = limitIdentifier(`${generated.base}:row-${String(idRowNumber).padStart(6, '0')}`)
      let duplicate = 2
      while (usedIdentifiers.has(recordId) || reservedExplicitIds.has(recordId)) {
        recordId = limitIdentifier(`${generated.base}:row-${String(idRowNumber).padStart(6, '0')}-${duplicate}`)
        duplicate += 1
      }
      if (generated.metadata) metadataIdentifiers += 1
      else rowIdentifiers += 1
      if (mapping.recordId) {
        issues.push({
          severity: 'warning',
          code: 'replaced-record-id',
          sourceRow: row.sourceRow,
          column: source.columns.find((column) => column.key === mapping.recordId)?.header,
          value: explicitId || '(blank)',
          message: !explicitId
            ? `Source row ${row.sourceRow}: record_id is blank, so a stable generated identifier is used.`
            : explicitId.length > MAX_RECORD_ID_LENGTH
            ? `Source row ${row.sourceRow}: record_id exceeds ${MAX_RECORD_ID_LENGTH} characters, so a stable generated identifier is used.`
            : `Source row ${row.sourceRow}: record_id '${explicitId}' is duplicated, so a stable generated identifier is used.`,
        })
      }
    }
    usedIdentifiers.add(recordId)

    const record: MetricRecord = { record_id: recordId }
    const reasons: string[] = []
    const badColumns: string[] = []
    const formulaColumnKeys = new Set<string>()
    source.columns.forEach((column) => {
      const raw = row.values[column.columnNumber - 1] ?? null
      if (!isFormulaCell(raw)) return
      formulaColumnKeys.add(column.key)
      reasons.push(`'${column.header}' contains a spreadsheet formula`)
      badColumns.push(column.header)
      issues.push({
        severity: 'error', code: 'formula-value', sourceRow: row.sourceRow, column: column.header,
        value: raw.formula,
        message: `Source row ${row.sourceRow}, column '${column.header}' contains a formula. Formulas and cached formula results are not imported.`,
      })
    })
    for (const feature of features) {
      const key = mapping.metrics[feature]
      if (key === null) continue
      const column = source.columns.find((candidate) => candidate.key === key)
      if (!column) {
        reasons.push(`mapping for ${feature} refers to an unknown column`)
        badColumns.push(feature)
        continue
      }
      const raw = sourceValue(row, key, source.columns)
      if (formulaColumnKeys.has(key)) continue
      const parsed = strictNumber(raw)
      if (parsed.kind === 'missing') {
        record[feature] = null
        issues.push({
          severity: 'warning', code: 'missing-value', sourceRow: row.sourceRow, column: column.header,
          message: `Source row ${row.sourceRow}, column '${column.header}': metric is missing. It remains null for the saved inference preprocessor.`,
        })
      } else if (parsed.kind === 'invalid') {
        reasons.push(`'${column.header}' is not a finite decimal number`)
        badColumns.push(column.header)
        issues.push({
          severity: 'error', code: 'invalid-value', sourceRow: row.sourceRow,
          column: column.header, value: parsed.raw,
          message: `Source row ${row.sourceRow}, column '${column.header}': '${parsed.raw}' is not a finite decimal number.`,
        })
      } else {
        record[feature] = parsed.value
      }
    }

    const labels: ImportedLabel[] = []
    for (const [key, meaning] of Object.entries(mapping.labels)) {
      if (meaning === 'ignore') continue
      const column = source.columns.find((candidate) => candidate.key === key)
      if (!column) continue
      const raw = sourceValue(row, key, source.columns)
      const binaryValue = binaryLabel(meaning, raw)
      labels.push({ column: column.header, meaning, rawValue: rawLabelValue(raw), binaryValue })
      if (raw !== null && textValue(raw) !== '' && binaryValue === null) {
        issues.push({
          severity: 'warning', code: 'invalid-label', sourceRow: row.sourceRow, column: column.header,
          message: `Source row ${row.sourceRow}, label '${column.header}' cannot be interpreted as ${meaning}. It is preserved for export but excluded from evaluation and model inputs.`,
        })
      }
    }
    const metadata = {
      sourceRow: row.sourceRow,
      recordId,
      project: project || null,
      release: release || null,
      className: className || null,
      labels,
    }
    if (reasons.length > 0) {
      invalidRows.push({ sourceRow: row.sourceRow, columns: [...new Set(badColumns)], reasons })
    } else if (missingMetrics.length === 0) {
      records.push(record)
      rowMetadata.push(metadata)
    }
  })

  let identifierSource: ParsedMetricFile['identifierSource'] = 'row'
  let identifierDescription = ''
  if (explicitIdentifiers === source.rows.length) {
    identifierSource = 'record_id'
    identifierDescription = 'Unique, valid source record_id values are preserved exactly.'
  } else if (metadataIdentifiers > 0 && rowIdentifiers === 0 && explicitIdentifiers === 0) {
    identifierSource = 'metadata'
    identifierDescription = isJureczko
      ? 'Stable IDs are built as project:version:class-name from the documented Jureczko/PROMISE fields.'
      : 'Stable IDs are built from available project, release/version, and class-name metadata.'
  } else if (rowIdentifiers > 0 && metadataIdentifiers === 0 && explicitIdentifiers === 0) {
    identifierSource = 'row'
    identifierDescription = source.fileName
      ? 'No class metadata is available, so stable IDs are built from the file name and source row.'
      : 'No class metadata is available, so stable row-number IDs are generated.'
  } else {
    identifierSource = 'mixed'
    identifierDescription = 'Unique source IDs are preserved; invalid or repeated IDs use deterministic metadata or file-and-row identifiers.'
  }

  const blockingSchemaIssue = issues.some((issue) => issue.severity === 'error' && ['duplicate-header', 'duplicate-metric-mapping', 'missing-metrics'].includes(issue.code))
  return {
    source,
    records,
    fileColumns: source.columns.map((column) => column.header),
    excludedColumns,
    identifierDescription,
    identifierSource,
    rowMetadata,
    layout: isJureczko ? 'jureczko-promise' : allCanonicalMetricsPresent ? 'dashboard-schema' : 'explicit-mapping',
    mapping,
    recognizedMetrics,
    missingMetrics,
    ambiguousLabelColumns,
    issues,
    invalidRows,
    totalRows: source.rows.length,
    validRows: records.length,
    canPredict: !blockingSchemaIssue && records.length > 0,
  }
}

export function validateImportFileSize(size: number): void {
  if (size > MAX_IMPORT_FILE_BYTES) {
    throw new Error(`The file is larger than the ${MAX_IMPORT_FILE_BYTES / 1024 / 1024} MB parsing limit.`)
  }
}

export async function parseMetricFile(file: File, features: string[], options: ParseImportOptions = {}): Promise<ParsedMetricFile> {
  validateImportFileSize(file.size)
  const extension = file.name.split('.').pop()?.toLocaleLowerCase('en-US')
  let source: TabularSource
  if (extension === 'csv' || extension === 'tsv') {
    source = parseDelimitedSource(await file.text(), file.name, extension, options.delimiter ?? (extension === 'tsv' ? '\t' : 'auto'))
  } else if (extension === 'json') {
    source = parseJsonSource(await file.text(), file.name)
  } else if (extension === 'arff') {
    source = parseArffSource(await file.text(), file.name)
  } else if (extension === 'xlsx') {
    source = await parseXlsxSource(await file.arrayBuffer(), file.name, options.sheetName)
  } else {
    throw new Error('Unsupported file type. Choose CSV, TSV, JSON, XLSX, or ARFF. Raw source-code files are not metric datasets.')
  }
  return analyzeTabularSource(source, features, options)
}

export function formatDelimiter(delimiter: ',' | ';' | '\t' | undefined): string {
  return delimiter === undefined ? 'not applicable' : delimiterName(delimiter)
}
