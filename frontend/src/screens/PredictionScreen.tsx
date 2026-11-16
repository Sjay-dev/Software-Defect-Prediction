import {
  ArrowDownUp,
  CheckCircle2,
  ChevronDown,
  Download,
  FileSpreadsheet,
  LoaderCircle,
  Play,
  ShieldAlert,
  UploadCloud,
  X,
} from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { api } from '../api'
import {
  downloadBlob,
  downloadText,
  labelsForPrediction,
  invalidRowsCsv,
  predictionsCsv,
  templateArff,
  templateCsv,
  templateJson,
  templateXlsx,
} from '../csv'
import {
  analyzeTabularSource,
  formatDelimiter,
  MAX_IMPORT_FILE_BYTES,
  MAX_IMPORT_ROWS,
  parseDelimitedSource,
  parseMetricFile,
} from '../file-import'
import type { DelimiterChoice, LabelMeaning, ParsedMetricFile } from '../file-import'
import { evaluateImportedPredictions } from '../import-evaluation'
import { predictInBatches } from '../prediction-batches'
import type { BatchProgress } from '../prediction-batches'
import { SAMPLE_RECORD } from '../sample'
import type { ModelInfo, Prediction } from '../types'

type SortKey = 'rank' | 'record_id' | 'class_name' | 'model_score'
type UploadState = ParsedMetricFile & { fileSize: number }

const megabytes = MAX_IMPORT_FILE_BYTES / 1024 / 1024
const acceptedExtensions = ['csv', 'tsv', 'json', 'xlsx', 'arff']

function numberText(value: number | null): string {
  return value === null ? 'undefined' : value.toFixed(4)
}

export function PredictionScreen({ model }: { model: ModelInfo }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [upload, setUpload] = useState<UploadState | null>(null)
  const [metricMappings, setMetricMappings] = useState<Record<string, string | null>>({})
  const [labelMappings, setLabelMappings] = useState<Record<string, LabelMeaning | 'ignore'>>({})
  const [delimiterChoice, setDelimiterChoice] = useState<DelimiterChoice>('auto')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [isParsing, setIsParsing] = useState(false)
  const [acceptValidRowsOnly, setAcceptValidRowsOnly] = useState(false)
  const [includeOutcomes, setIncludeOutcomes] = useState(true)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [scoreDescription, setScoreDescription] = useState(model.score_description)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [sort, setSort] = useState<{ key: SortKey; direction: 'asc' | 'desc' }>({ key: 'rank', direction: 'asc' })

  const evaluation = useMemo(
    () => upload && predictions.length > 0 ? evaluateImportedPredictions(predictions, upload.rowMetadata) : null,
    [predictions, upload],
  )
  const sortedPredictions = useMemo(() => predictions.slice().sort((leftItem, rightItem) => {
    const left = leftItem[sort.key] ?? ''
    const right = rightItem[sort.key] ?? ''
    const result = typeof left === 'string' ? left.localeCompare(String(right)) : Number(left) - Number(right)
    return sort.direction === 'asc' ? result : -result
  }), [predictions, sort])

  function chooseSort(key: SortKey) {
    setSort((current) => ({ key, direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc' }))
  }

  function resetResults() {
    setRequestError(null)
    setPredictions([])
    setBatchProgress(null)
  }

  async function parseFile(
    file: File,
    options: { delimiter?: DelimiterChoice; sheetName?: string } = {},
    mappings: { metrics?: Record<string, string | null>; labels?: Record<string, LabelMeaning | 'ignore'> } = {},
  ) {
    setIsParsing(true)
    setValidationError(null)
    setAcceptValidRowsOnly(false)
    resetResults()
    try {
      const parsed = await parseMetricFile(file, model.features, {
        ...options,
        metricMappings: mappings.metrics,
        labelMappings: mappings.labels,
      })
      setUpload({ ...parsed, fileSize: file.size })
    } catch (error) {
      setUpload(null)
      setValidationError(error instanceof Error ? error.message : 'The file could not be parsed safely.')
    } finally {
      setIsParsing(false)
    }
  }

  async function loadFile(file: File) {
    const extension = file.name.split('.').pop()?.toLocaleLowerCase('en-US') ?? ''
    if (!acceptedExtensions.includes(extension)) {
      setSourceFile(null)
      setUpload(null)
      setValidationError('Choose a CSV, TSV, JSON, XLSX, or ARFF metric dataset. Raw source code and arbitrary business files are not compatible.')
      return
    }
    setSourceFile(file)
    setMetricMappings({})
    setLabelMappings({})
    const initialDelimiter: DelimiterChoice = extension === 'tsv' ? '\t' : 'auto'
    setDelimiterChoice(initialDelimiter)
    await parseFile(file, { delimiter: initialDelimiter })
  }

  function remapMetric(feature: string, key: string) {
    if (!upload) return
    const nextMappings = { ...metricMappings, [feature]: key || null }
    setMetricMappings(nextMappings)
    setUpload({
      ...analyzeTabularSource(upload.source, model.features, { metricMappings: nextMappings, labelMappings }),
      fileSize: upload.fileSize,
    })
    setAcceptValidRowsOnly(false)
    resetResults()
  }

  function mapAmbiguousLabel(key: string, meaning: LabelMeaning | 'ignore') {
    if (!upload) return
    const nextMappings = { ...labelMappings, [key]: meaning }
    setLabelMappings(nextMappings)
    setUpload({
      ...analyzeTabularSource(upload.source, model.features, { metricMappings, labelMappings: nextMappings }),
      fileSize: upload.fileSize,
    })
    resetResults()
  }

  function loadSample() {
    const values = [SAMPLE_RECORD.record_id, ...model.features.map((feature) => SAMPLE_RECORD[feature])]
    const text = `${['record_id', ...model.features].join(',')}\n${values.join(',')}\n`
    const source = parseDelimitedSource(text, 'schema-sample.csv')
    setUpload({ ...analyzeTabularSource(source, model.features), fileSize: 0 })
    setSourceFile(null)
    setMetricMappings({})
    setLabelMappings({})
    setValidationError(null)
    resetResults()
  }

  async function runPrediction() {
    if (!upload) return
    const controller = new AbortController()
    abortRef.current = controller
    setIsPredicting(true)
    setRequestError(null)
    setPredictions([])
    try {
      const response = await predictInBatches(
        upload.records,
        model.max_batch_size,
        model.model_version,
        (records, signal) => api.predict(records, signal),
        setBatchProgress,
        { signal: controller.signal, maxRetries: 1, retryDelayMs: 400 },
      )
      const metadataById = new Map(upload.rowMetadata.map((row) => [row.recordId, row]))
      setPredictions(response.predictions.map((prediction) => {
        const metadata = metadataById.get(prediction.record_id)
        return {
          ...prediction,
          class_name: metadata?.className ?? undefined,
          source_row: metadata?.sourceRow,
          labels: metadata ? labelsForPrediction(metadata) : undefined,
        }
      }))
      setScoreDescription(response.scoreDescription)
      setSort({ key: 'rank', direction: 'asc' })
    } catch (error) {
      setPredictions([])
      setRequestError(error instanceof Error ? error.message : 'Prediction failed.')
    } finally {
      abortRef.current = null
      setIsPredicting(false)
    }
  }

  function clearUpload() {
    if (isPredicting) return
    setSourceFile(null)
    setUpload(null)
    setMetricMappings({})
    setLabelMappings({})
    setValidationError(null)
    resetResults()
    if (inputRef.current) inputRef.current.value = ''
  }

  function downloadResults() {
    const rows = includeOutcomes ? predictions : predictions.map(({ labels: _labels, ...prediction }) => prediction)
    downloadText('defect-prediction-results.csv', predictionsCsv(rows), 'text/csv;charset=utf-8')
  }

  const batchCount = upload ? Math.ceil(upload.records.length / model.max_batch_size) : 0
  const progressPercent = batchProgress ? Math.round((batchProgress.completedRecords / batchProgress.totalRecords) * 100) : 0
  const unresolvedAmbiguousLabels = upload?.ambiguousLabelColumns.length ?? 0
  const partialNeedsConsent = Boolean(upload && upload.invalidRows.length > 0 && !acceptValidRowsOnly)
  const readyToPredict = Boolean(upload?.canPredict && !partialNeedsConsent && unresolvedAmbiguousLabels === 0)
  const issuePreview = upload?.issues.slice(0, 20) ?? []

  return (
    <div className="space-y-7">
      <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="eyebrow">Prediction workspace</p>
          <h1 className="page-title">Rank software classes for review</h1>
          <p className="page-intro">Import compatible class-level metrics, review the detected mapping locally, then score every valid record with one frozen model version.</p>
        </div>
        <details className="relative">
          <summary className="secondary-button cursor-pointer list-none"><Download size={16} /> Download templates</summary>
          <div className="absolute right-0 z-10 mt-2 grid w-52 gap-1 rounded-xl border border-slate-200 bg-white p-2 shadow-lg">
            <button type="button" onClick={() => downloadText('software-metrics-template.csv', templateCsv(model.features), 'text/csv;charset=utf-8')} className="rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50">CSV template</button>
            <button type="button" onClick={() => downloadText('software-metrics-template.tsv', templateCsv(model.features, '\t'), 'text/tab-separated-values;charset=utf-8')} className="rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50">TSV template</button>
            <button type="button" onClick={() => downloadText('software-metrics-template.json', templateJson(model.features), 'application/json;charset=utf-8')} className="rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50">JSON template</button>
            <button type="button" onClick={() => downloadText('software-metrics-template.arff', templateArff(model.features), 'text/plain;charset=utf-8')} className="rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50">ARFF template</button>
            <button type="button" onClick={() => void templateXlsx(model.features).then((blob) => downloadBlob('software-metrics-template.xlsx', blob)).catch((error) => setValidationError(error instanceof Error ? error.message : 'Could not create XLSX template.'))} className="rounded-lg px-3 py-2 text-left text-sm hover:bg-slate-50">XLSX template</button>
          </div>
        </details>
      </section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(300px,0.55fr)]">
        <section className="card p-5 sm:p-7">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="section-title">Upload metrics</h2>
              <p className="mt-1 text-sm text-slate-500">CSV, TSV, JSON, XLSX, or ARFF · {MAX_IMPORT_ROWS.toLocaleString()} rows · {megabytes} MB · batches of {model.max_batch_size}</p>
            </div>
            <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">{model.features.length} metrics</span>
          </div>

          <label
            onDragEnter={(event) => { event.preventDefault(); if (!isPredicting) setIsDragging(true) }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault(); setIsDragging(false)
              if (!isPredicting && event.dataTransfer.files[0]) void loadFile(event.dataTransfer.files[0])
            }}
            className={`mt-6 grid min-h-48 place-items-center rounded-2xl border-2 border-dashed px-5 text-center transition ${
              isPredicting ? 'cursor-not-allowed border-slate-200 bg-slate-100 opacity-70' :
              isDragging ? 'cursor-pointer border-teal-500 bg-teal-50' : 'cursor-pointer border-slate-300 bg-slate-50/70 hover:border-teal-400 hover:bg-teal-50/50'
            }`}
          >
            <input ref={inputRef} type="file" accept=".csv,.tsv,.json,.xlsx,.arff,text/csv,text/tab-separated-values,application/json" disabled={isPredicting} className="sr-only" onChange={(event) => { const file = event.target.files?.[0]; if (file) void loadFile(file) }} />
            <div>
              {isParsing ? <LoaderCircle className="mx-auto animate-spin text-teal-700" size={30} /> : <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-white text-teal-700 shadow-sm ring-1 ring-slate-200"><UploadCloud size={25} /></span>}
              <p className="mt-4 font-semibold text-slate-900">{isParsing ? 'Inspecting file safely…' : 'Drop a supported metric file here, or browse'}</p>
              <p className="mt-1 text-sm text-slate-500">Only explicitly mapped required metrics are sent to the model.</p>
            </div>
          </label>

          {validationError && <Alert title="Import failed" message={validationError} />}

          {upload && (
            <div className={`mt-4 space-y-4 rounded-xl border p-4 ${upload.canPredict ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  {upload.canPredict ? <CheckCircle2 className="shrink-0 text-emerald-600" size={20} /> : <ShieldAlert className="shrink-0 text-amber-600" size={20} />}
                  <div>
                    <p className="font-semibold text-slate-950">{upload.canPredict ? 'Import preview ready' : 'Mapping needs attention'}</p>
                    <p className="text-sm text-slate-700">{upload.source.fileName} · {upload.totalRows.toLocaleString()} source rows · {upload.validRows.toLocaleString()} valid · {batchCount} {batchCount === 1 ? 'batch' : 'batches'}</p>
                  </div>
                </div>
                <button type="button" disabled={isPredicting} onClick={clearUpload} className="inline-flex items-center gap-1.5 self-start text-xs font-semibold text-slate-700 hover:text-slate-950 disabled:opacity-50"><X size={14} /> Clear</button>
              </div>

              <div className="grid gap-2 rounded-lg border border-slate-200 bg-white/80 p-3 text-xs text-slate-700 sm:grid-cols-2">
                <p><span className="font-semibold text-slate-950">Detected:</span> {upload.source.formatDescription}</p>
                <p><span className="font-semibold text-slate-950">Layout:</span> {upload.layout}</p>
                <p><span className="font-semibold text-slate-950">Metrics:</span> {upload.recognizedMetrics.length}/{model.features.length} mapped</p>
                <p><span className="font-semibold text-slate-950">Identifiers:</span> {upload.identifierSource}</p>
                {upload.source.delimiter && <p><span className="font-semibold text-slate-950">Delimiter:</span> {formatDelimiter(upload.source.delimiter)}</p>}
                {upload.source.sheetName && <p><span className="font-semibold text-slate-950">Worksheet:</span> {upload.source.sheetName}</p>}
              </div>

              {sourceFile && upload.source.delimiter && (
                <label className="block text-xs font-semibold text-slate-700">Delimited-text separator
                  <select value={delimiterChoice} disabled={isParsing || isPredicting} onChange={(event) => { const value = event.target.value as DelimiterChoice; setDelimiterChoice(value); setMetricMappings({}); setLabelMappings({}); void parseFile(sourceFile, { delimiter: value }) }} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal">
                    <option value="auto">Auto-detect</option><option value=",">Comma</option><option value=";">Semicolon</option><option value={'\t'}>Tab</option>
                  </select>
                </label>
              )}
              {sourceFile && upload.source.availableSheets && upload.source.availableSheets.length > 1 && (
                <label className="block text-xs font-semibold text-slate-700">Worksheet
                  <select value={upload.source.sheetName} disabled={isParsing || isPredicting} onChange={(event) => { setMetricMappings({}); setLabelMappings({}); void parseFile(sourceFile, { sheetName: event.target.value }) }} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal">
                    {upload.source.availableSheets.map((sheet) => <option key={sheet} value={sheet}>{sheet}</option>)}
                  </select>
                </label>
              )}

              {upload.missingMetrics.length > 0 && (
                <div className="rounded-lg border border-amber-200 bg-white p-3">
                  <p className="text-sm font-semibold text-amber-950">Explicit column mapping required</p>
                  <p className="mt-1 text-xs leading-5 text-amber-800">Only choose a column when its metric definition matches. Similar names are not assumed equivalent.</p>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2">{upload.missingMetrics.map((feature) => (
                    <label key={feature} className="text-xs font-semibold text-slate-700">{feature}
                      <select value={metricMappings[feature] ?? ''} onChange={(event) => remapMetric(feature, event.target.value)} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-2 py-2 font-normal">
                        <option value="">Select a source column…</option>
                        {upload.source.columns.map((column) => <option key={column.key} value={column.key}>{column.header} (column {column.columnNumber})</option>)}
                      </select>
                    </label>
                  ))}</div>
                </div>
              )}

              {upload.ambiguousLabelColumns.length > 0 && (
                <div className="rounded-lg border border-amber-200 bg-white p-3">
                  <p className="text-sm font-semibold text-amber-950">Confirm ambiguous outcome columns</p>
                  <p className="mt-1 text-xs leading-5 text-amber-800">Outcome meaning must be explicit. These columns never enter model inputs.</p>
                  {upload.ambiguousLabelColumns.map((column) => (
                    <label key={column.key} className="mt-2 block text-xs font-semibold text-slate-700">{column.header}
                      <select defaultValue="" onChange={(event) => mapAmbiguousLabel(column.key, event.target.value as LabelMeaning | 'ignore')} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-2 py-2 font-normal">
                        <option value="" disabled>Choose meaning…</option><option value="ignore">Not an outcome / exclude</option><option value="bug_count">Documented bug count (defective when &gt; 0)</option><option value="defective">Documented binary defect indicator</option>
                      </select>
                    </label>
                  ))}
                </div>
              )}

              <details className="rounded-lg border border-slate-200 bg-white/80 p-3 text-xs text-slate-700" open={upload.invalidRows.length > 0}>
                <summary className="cursor-pointer font-semibold text-slate-950">Validation details ({upload.issues.length} findings)</summary>
                <p className="mt-2"><span className="font-semibold">Identifier mapping:</span> {upload.identifierDescription}</p>
                {issuePreview.length > 0 && <div className="mt-2 max-h-48 overflow-auto rounded-md bg-slate-50 p-2"><ul className="space-y-1">{issuePreview.map((issue, index) => <li key={`${issue.code}-${issue.sourceRow ?? 0}-${issue.column ?? ''}-${index}`} className={issue.severity === 'error' ? 'text-rose-700' : 'text-amber-700'}>{issue.message}</li>)}</ul>{upload.issues.length > issuePreview.length && <p className="mt-2 font-medium">Showing the first {issuePreview.length} of {upload.issues.length} findings.</p>}</div>}
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full min-w-[560px] text-left"><thead className="text-[10px] uppercase tracking-wider text-slate-400"><tr><th className="pb-1 pr-3">Source row</th><th className="pb-1 pr-3">Original class</th><th className="pb-1">Stable record_id</th></tr></thead>
                    <tbody>{upload.rowMetadata.slice(0, 5).map((row) => <tr key={row.recordId} className="border-t border-slate-100"><td className="py-1.5 pr-3">{row.sourceRow}</td><td className="max-w-56 truncate py-1.5 pr-3 font-mono" title={row.className ?? undefined}>{row.className ?? '—'}</td><td className="max-w-72 truncate py-1.5 font-mono" title={row.recordId}>{row.recordId}</td></tr>)}</tbody>
                  </table>
                </div>
                <p className="mt-3 font-semibold text-slate-950">Excluded from model inputs</p>
                <div className="mt-1 flex flex-wrap gap-1.5">{upload.excludedColumns.length === 0 ? <span>None</span> : upload.excludedColumns.map((column) => <span key={column.key} className="rounded-md bg-slate-100 px-2 py-0.5 text-slate-600" title={`Source column ${column.columnNumber}`}>{column.column} <span className="text-slate-400">· {column.role}</span></span>)}</div>
              </details>

              {upload.invalidRows.length > 0 && (
                <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-900">
                  <label className="flex items-start gap-2"><input type="checkbox" checked={acceptValidRowsOnly} onChange={(event) => setAcceptValidRowsOnly(event.target.checked)} className="mt-1" /><span><strong>Use only the {upload.validRows.toLocaleString()} valid rows.</strong> Exclude {upload.invalidRows.length.toLocaleString()} invalid rows: {upload.invalidRows.slice(0, 8).map((row) => `${row.sourceRow} (${row.columns.join(', ')})`).join('; ')}{upload.invalidRows.length > 8 ? '; …' : ''}. Results will explicitly cover valid rows only.</span></label>
                  <button type="button" onClick={() => downloadText('excluded-import-rows.csv', invalidRowsCsv(upload.invalidRows), 'text/csv;charset=utf-8')} className="mt-2 font-semibold underline underline-offset-2">Download the complete excluded-row report</button>
                </div>
              )}
              <p className="text-xs leading-5 text-slate-600">Missing metric cells remain null for the saved inference preprocessor. Values are never imputed in the importer, and labels/metadata are never sent as predictors.</p>
            </div>
          )}

          <div className="mt-5 flex gap-2">
            <button type="button" disabled={!readyToPredict || isPredicting || isParsing} onClick={() => void runPrediction()} className="primary-button flex-1 justify-center disabled:cursor-not-allowed disabled:opacity-45">
              {isPredicting ? <LoaderCircle className="animate-spin" size={17} /> : <Play size={17} fill="currentColor" />}
              {isPredicting && batchProgress ? `${batchProgress.status === 'retrying' ? 'Retrying' : 'Scoring'} batch ${batchProgress.currentBatch} of ${batchProgress.totalBatches}…` : 'Run prediction'}
            </button>
            {isPredicting && <button type="button" onClick={() => abortRef.current?.abort()} className="secondary-button text-rose-700"><X size={16} /> Cancel</button>}
          </div>
          {batchProgress && <div className="mt-3" aria-live="polite"><div className="mb-1.5 flex justify-between text-xs font-medium text-slate-600"><span>{batchProgress.status === 'complete' ? 'Scoring complete' : batchProgress.status === 'failed' ? 'Batch failed' : batchProgress.status === 'cancelled' ? 'Cancelled' : batchProgress.status === 'retrying' ? `Retrying batch ${batchProgress.currentBatch} (attempt ${batchProgress.attempt})` : `Batch ${batchProgress.currentBatch} of ${batchProgress.totalBatches}`}</span><span>{batchProgress.completedRecords.toLocaleString()} / {batchProgress.totalRecords.toLocaleString()} records</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className={`h-full rounded-full transition-all ${['failed', 'cancelled'].includes(batchProgress.status) ? 'bg-rose-500' : 'bg-teal-500'}`} style={{ width: `${progressPercent}%` }} /></div></div>}
          {requestError && <Alert title="Incomplete prediction" message={requestError} />}
        </section>

        <aside className="space-y-5">
          <section className="card p-5">
            <div className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-50 text-indigo-700"><FileSpreadsheet size={19} /></span><div><p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Unlabelled example</p><h2 className="font-semibold">Schema-valid sample</h2></div></div>
            <p className="mt-4 text-sm leading-6 text-slate-600">A stable identifier and all 20 required metrics. It intentionally contains no outcome label.</p>
            <dl className="mt-4 grid grid-cols-2 gap-2 rounded-xl bg-slate-50 p-3 text-xs"><div className="col-span-2"><dt className="text-slate-400">record_id</dt><dd className="truncate font-mono text-slate-800">{SAMPLE_RECORD.record_id}</dd></div><div><dt className="text-slate-400">wmc</dt><dd className="font-semibold">4</dd></div><div><dt className="text-slate-400">loc</dt><dd className="font-semibold">26</dd></div></dl>
            <button type="button" disabled={isPredicting} onClick={loadSample} className="secondary-button mt-4 w-full justify-center disabled:opacity-50">Use sample input</button>
          </section>
          <section className="rounded-2xl bg-slate-950 p-5 text-white shadow-sm"><p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-300">Compatibility boundary</p><p className="mt-3 text-sm leading-6 text-slate-300">This model requires the documented 20 class-level static metrics. Arbitrary business tables and raw Java files need a separate metric extractor; different feature sets require a separately trained model.</p><p className="mt-3 text-xs leading-5 text-slate-400">Uploads are never used to retrain, tune, or calibrate the saved model.</p></section>
        </aside>
      </div>

      {evaluation && <section className="card p-5 sm:p-7"><div className="flex flex-col justify-between gap-2 sm:flex-row"><div><h2 className="section-title">Optional recorded-outcome summary</h2><p className="mt-1 text-sm text-slate-500">Computed only from explicitly interpreted labels; labels never entered model inputs.</p></div><p className="text-xs text-slate-500">{evaluation.labelledRecords} labelled · {evaluation.positiveRecords} positive · {evaluation.negativeRecords} negative</p></div><div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">{[['Average precision', numberText(evaluation.averagePrecision)], ['ROC-AUC', numberText(evaluation.rocAuc)], ['Precision @ 0.5', numberText(evaluation.precisionAtHalf)], ['Recall @ 0.5', numberText(evaluation.recallAtHalf)], ['F1 @ 0.5', numberText(evaluation.f1AtHalf)], ['TN / FP / FN / TP', `${evaluation.confusion.tn} / ${evaluation.confusion.fp} / ${evaluation.confusion.fn} / ${evaluation.confusion.tp}`]].map(([label, value]) => <div key={label} className="rounded-xl bg-slate-50 p-3"><p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p><p className="mt-1 font-mono text-sm font-semibold text-slate-800">{value}</p></div>)}</div>{evaluation.explanation && <p className="mt-3 text-sm text-amber-700">{evaluation.explanation}</p>}</section>}

      <section className="card overflow-hidden">
        <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7"><div><h2 className="section-title">Prediction results</h2><p className="mt-1 text-sm text-slate-500">Global ranks are assigned only after every batch succeeds. No partial result set is presented as a complete evaluation.</p></div>{predictions.length > 0 && <div className="flex flex-wrap items-center gap-3">{predictions.some((prediction) => Object.keys(prediction.labels ?? {}).length > 0) && <label className="text-xs text-slate-600"><input type="checkbox" checked={includeOutcomes} onChange={(event) => setIncludeOutcomes(event.target.checked)} className="mr-1.5" />Join recorded outcomes</label>}<button type="button" onClick={downloadResults} className="secondary-button"><Download size={15} /> Download results CSV</button></div>}</div>
        {predictions.length === 0 ? <div className="grid min-h-64 place-items-center px-5 text-center"><div><ArrowDownUp className="mx-auto text-slate-300" size={30} /><p className="mt-3 font-medium text-slate-700">No complete prediction results yet</p><p className="mt-1 text-sm text-slate-500">Import and confirm a compatible dataset, then run prediction.</p></div></div> : <><div className="overflow-x-auto"><table className="data-table min-w-[980px]"><thead><tr><th><SortButton label="Global rank" active={sort.key === 'rank'} direction={sort.direction} onClick={() => chooseSort('rank')} /></th><th><SortButton label="Original class name" active={sort.key === 'class_name'} direction={sort.direction} onClick={() => chooseSort('class_name')} /></th><th><SortButton label="Record identifier" active={sort.key === 'record_id'} direction={sort.direction} onClick={() => chooseSort('record_id')} /></th><th><SortButton label="Model risk score" active={sort.key === 'model_score'} direction={sort.direction} onClick={() => chooseSort('model_score')} /></th><th>Model version</th></tr></thead><tbody>{sortedPredictions.map((item) => <tr key={item.record_id}><td><span className="inline-grid h-8 w-8 place-items-center rounded-lg bg-slate-950 text-xs font-bold text-white">{item.rank}</span></td><td className="max-w-72 truncate font-mono text-xs text-slate-600" title={item.class_name}>{item.class_name ?? '—'}</td><td className="max-w-80 truncate font-mono text-xs font-medium text-slate-800" title={item.record_id}>{item.record_id}</td><td><div className="flex min-w-48 items-center gap-3"><div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-teal-500" style={{ width: `${Math.max(2, item.model_score * 100)}%` }} /></div><span className="w-16 text-right font-mono text-xs font-semibold">{item.model_score.toFixed(4)}</span></div></td><td className="text-xs text-slate-500">{item.model_version}</td></tr>)}</tbody></table></div><div className="border-t border-slate-200 bg-slate-50 px-5 py-3 text-xs text-slate-500 sm:px-7">{scoreDescription}</div></>}
      </section>
    </div>
  )
}

function Alert({ title, message }: { title: string; message: string }) {
  return <div role="alert" className="mt-4 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900"><ShieldAlert className="mt-0.5 shrink-0 text-rose-600" size={18} /><div><p className="font-semibold">{title}</p><p className="mt-0.5">{message}</p></div></div>
}

function SortButton({ label, active, direction, onClick }: { label: string; active: boolean; direction: 'asc' | 'desc'; onClick: () => void }) {
  return <button type="button" onClick={onClick} className="inline-flex items-center gap-1.5 font-semibold text-slate-600 hover:text-slate-950">{label}<ChevronDown size={13} className={`transition ${active ? 'text-teal-600' : 'text-slate-300'} ${active && direction === 'desc' ? 'rotate-180' : ''}`} /></button>
}
