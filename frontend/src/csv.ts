import Papa from 'papaparse'
import type { MetricRecord, Prediction } from './types'

export const MAX_CSV_FILE_BYTES = 10 * 1024 * 1024
export const MAX_CSV_ROWS = 10_000
export const MAX_RECORD_ID_LENGTH = 200

export type ExcludedCsvColumn = {
  column: string
  columnNumber: number
  role: 'identifier' | 'metadata' | 'label' | 'extra'
}

export type CsvRowMetadata = {
  csvRow: number
  recordId: string
  project: string | null
  release: string | null
  className: string | null
}

export type ParsedMetricCsv = {
  records: MetricRecord[]
  fileColumns: string[]
  excludedColumns: ExcludedCsvColumn[]
  identifierDescription: string
  identifierSource: 'record_id' | 'metadata' | 'row' | 'mixed'
  rowMetadata: CsvRowMetadata[]
}

type ParsedRows = Papa.ParseResult<string[]>

function escapeCell(value: string | number): string {
  const text = String(value)
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

function normalizeHeader(header: string, index: number): string {
  const trimmed = header.trim()
  return index === 0 ? trimmed.replace(/^\uFEFF/, '') : trimmed
}

function indexesOf(columns: string[], name: string): number[] {
  const indexes: number[] = []
  columns.forEach((column, index) => {
    if (column === name) indexes.push(index)
  })
  return indexes
}

function requireUnambiguousColumn(columns: string[], name: string): number | null {
  const indexes = indexesOf(columns, name)
  if (indexes.length > 1) {
    throw new Error(`Column '${name}' appears more than once. Remove or rename the duplicate before uploading.`)
  }
  return indexes[0] ?? null
}

function stableHash(value: string): string {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

function limitGeneratedIdentifier(value: string): string {
  if (value.length <= MAX_RECORD_ID_LENGTH) return value
  const suffix = `~${stableHash(value)}`
  return `${value.slice(0, MAX_RECORD_ID_LENGTH - suffix.length)}${suffix}`
}

function valueAt(row: string[], index: number | null): string {
  return index === null ? '' : (row[index] ?? '').trim()
}

function excludedRole(
  index: number,
  recordIdIndex: number | null,
  metadataIndexes: Set<number>,
  labelIndexes: Set<number>,
): ExcludedCsvColumn['role'] {
  if (index === recordIdIndex) return 'identifier'
  if (metadataIndexes.has(index)) return 'metadata'
  if (labelIndexes.has(index)) return 'label'
  return 'extra'
}

export function validateCsvFileSize(size: number): void {
  if (size > MAX_CSV_FILE_BYTES) {
    throw new Error(`The CSV is larger than the ${MAX_CSV_FILE_BYTES / 1024 / 1024} MB upload limit.`)
  }
}

export function parseMetricCsv(
  text: string,
  features: string[],
  maxRows = MAX_CSV_ROWS,
): ParsedMetricCsv {
  validateCsvFileSize(new TextEncoder().encode(text).byteLength)

  const parsed = Papa.parse<string[]>(text, {
    header: false,
    skipEmptyLines: 'greedy',
  }) as ParsedRows
  if (parsed.errors.length > 0) {
    const error = parsed.errors[0]
    throw new Error(`CSV parsing error${error.row === undefined ? '' : ` on row ${error.row + 1}`}: ${error.message}`)
  }
  if (parsed.data.length === 0) throw new Error('The CSV is empty.')

  const columns = parsed.data[0].map(normalizeHeader)
  if (columns.length === 0 || columns.every((column) => column === '')) {
    throw new Error('The CSV header is empty.')
  }

  const rows = parsed.data.slice(1)
  if (rows.length === 0) throw new Error('The CSV contains no data rows.')
  if (rows.length > maxRows) {
    throw new Error(`The CSV has ${rows.length} rows; the upload limit is ${maxRows.toLocaleString()} rows.`)
  }

  const featureIndexes = new Map<string, number>()
  const missingFeatures: string[] = []
  for (const feature of features) {
    const indexes = indexesOf(columns, feature)
    if (indexes.length === 0) {
      missingFeatures.push(feature)
    } else if (indexes.length > 1) {
      throw new Error(`Required metric '${feature}' appears more than once. Metric mappings must be unambiguous.`)
    } else {
      featureIndexes.set(feature, indexes[0])
    }
  }
  if (missingFeatures.length > 0) {
    throw new Error(
      `Missing required metric columns: ${missingFeatures.join(', ')}. ` +
      'Metric names must exactly match the downloadable 20-metric schema; values are never fabricated or inferred.',
    )
  }

  const recordIdIndex = requireUnambiguousColumn(columns, 'record_id')
  const projectIndex = requireUnambiguousColumn(columns, 'project')
  const releaseIndex = requireUnambiguousColumn(columns, 'release')
  const versionIndex = requireUnambiguousColumn(columns, 'version')
  const classNameIndex = requireUnambiguousColumn(columns, 'class_name')
  const nameIndexes = indexesOf(columns, 'name')

  // The published Jureczko/PROMISE files deliberately repeat `name`: the first
  // column is the project and the second is the fully qualified class name.
  const isJureczkoLayout = nameIndexes.length === 2 && versionIndex !== null
  if (nameIndexes.length > 1 && !isJureczkoLayout) {
    throw new Error(
      "The repeated 'name' columns do not match the documented Jureczko layout (name, version, name). " +
      'Rename metadata columns explicitly rather than relying on an ambiguous mapping.',
    )
  }

  const mappedProjectIndex = projectIndex ?? (isJureczkoLayout ? nameIndexes[0] : null)
  const mappedReleaseIndex = releaseIndex ?? versionIndex
  const mappedClassNameIndex = classNameIndex ?? (isJureczkoLayout ? nameIndexes[1] : null)
  const metadataIndexes = new Set(
    [mappedProjectIndex, mappedReleaseIndex, mappedClassNameIndex]
      .filter((index): index is number => index !== null),
  )
  const labelIndexes = new Set<number>()
  for (const label of ['bug', 'bug_count', 'defective']) {
    for (const index of indexesOf(columns, label)) labelIndexes.add(index)
  }
  const usedFeatureIndexes = new Set(featureIndexes.values())
  const excludedColumns = columns.flatMap((column, index) => usedFeatureIndexes.has(index)
    ? []
    : [{
        column,
        columnNumber: index + 1,
        role: excludedRole(index, recordIdIndex, metadataIndexes, labelIndexes),
      } satisfies ExcludedCsvColumn])

  const identifiers = new Set<string>()
  const records: MetricRecord[] = []
  const rowMetadata: CsvRowMetadata[] = []
  let metadataIdentifiers = 0
  let rowIdentifiers = 0

  rows.forEach((row, rowIndex) => {
    const csvRow = rowIndex + 2
    if (row.length > columns.length) {
      throw new Error(`Row ${csvRow} has ${row.length} values but the header has ${columns.length} columns.`)
    }

    const project = valueAt(row, mappedProjectIndex) || null
    const release = valueAt(row, mappedReleaseIndex) || null
    const className = valueAt(row, mappedClassNameIndex) || null
    let recordId: string

    if (recordIdIndex !== null) {
      recordId = valueAt(row, recordIdIndex)
      if (!recordId) throw new Error(`Row ${csvRow}: record_id must not be blank.`)
      if (recordId.length > MAX_RECORD_ID_LENGTH) {
        throw new Error(`Row ${csvRow}: record_id exceeds the API limit of ${MAX_RECORD_ID_LENGTH} characters.`)
      }
      if (identifiers.has(recordId)) throw new Error(`Row ${csvRow}: duplicate record_id '${recordId}'.`)
    } else {
      const metadataParts = [project, release, className].filter((part): part is string => Boolean(part))
      const hasMetadata = metadataParts.length > 0
      const base = hasMetadata ? metadataParts.join(':') : `row-${String(rowIndex + 1).padStart(6, '0')}`
      recordId = limitGeneratedIdentifier(base)
      if (identifiers.has(recordId)) {
        recordId = limitGeneratedIdentifier(`${base}:row-${String(rowIndex + 1).padStart(6, '0')}`)
      }
      let collision = 2
      while (identifiers.has(recordId)) {
        recordId = limitGeneratedIdentifier(`${base}:row-${String(rowIndex + 1).padStart(6, '0')}-${collision}`)
        collision += 1
      }
      if (hasMetadata) metadataIdentifiers += 1
      else rowIdentifiers += 1
    }
    identifiers.add(recordId)

    const record: MetricRecord = { record_id: recordId }
    for (const feature of features) {
      const featureIndex = featureIndexes.get(feature)
      if (featureIndex === undefined) throw new Error(`Internal mapping error for metric '${feature}'.`)
      const raw = (row[featureIndex] ?? '').trim()
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
    records.push(record)
    rowMetadata.push({ csvRow, recordId, project, release, className })
  })

  let identifierSource: ParsedMetricCsv['identifierSource']
  let identifierDescription: string
  if (recordIdIndex !== null) {
    identifierSource = 'record_id'
    identifierDescription = 'Existing record_id values were preserved exactly.'
  } else if (metadataIdentifiers > 0 && rowIdentifiers === 0) {
    identifierSource = 'metadata'
    identifierDescription = isJureczkoLayout
      ? 'record_id was absent, so each ID was built as project:version:class-name from the documented Jureczko name,version,name metadata.'
      : 'record_id was absent, so each ID was built from the available project, release/version, and class-name metadata.'
  } else if (metadataIdentifiers === 0) {
    identifierSource = 'row'
    identifierDescription = 'No usable identifier metadata was present, so deterministic row-000001 style IDs were generated.'
  } else {
    identifierSource = 'mixed'
    identifierDescription = 'IDs were built from available metadata; rows without usable metadata received deterministic row IDs.'
  }

  return {
    records,
    fileColumns: columns,
    excludedColumns,
    identifierDescription,
    identifierSource,
    rowMetadata,
  }
}

export function templateCsv(features: string[]): string {
  return `${['record_id', ...features].join(',')}\r\n`
}

export function predictionsCsv(predictions: Prediction[]): string {
  const includeClassName = predictions.some((item) => Boolean(item.class_name))
  const header = includeClassName
    ? 'record_id,class_name,model_score,rank,model_version'
    : 'record_id,model_score,rank,model_version'
  const rows = predictions
    .slice()
    .sort((a, b) => a.rank - b.rank)
    .map((item) => {
      const values = includeClassName
        ? [item.record_id, item.class_name ?? '', item.model_score, item.rank, item.model_version]
        : [item.record_id, item.model_score, item.rank, item.model_version]
      return values.map(escapeCell).join(',')
    })
  return [header, ...rows].join('\r\n') + '\r\n'
}

export function downloadText(filename: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
