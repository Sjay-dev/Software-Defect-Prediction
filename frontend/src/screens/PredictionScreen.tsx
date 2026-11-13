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
import { downloadText, parseMetricCsv, predictionsCsv, templateCsv } from '../csv'
import { SAMPLE_RECORD } from '../sample'
import type { MetricRecord, ModelInfo, Prediction } from '../types'

type SortKey = 'rank' | 'record_id' | 'model_score'
type UploadState = { fileName: string; records: MetricRecord[] } | null

export function PredictionScreen({ model }: { model: ModelInfo }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [upload, setUpload] = useState<UploadState>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [scoreDescription, setScoreDescription] = useState(model.score_description)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [sort, setSort] = useState<{ key: SortKey; direction: 'asc' | 'desc' }>({ key: 'rank', direction: 'asc' })

  const sortedPredictions = useMemo(() => {
    return predictions.slice().sort((a, b) => {
      const left = a[sort.key]
      const right = b[sort.key]
      const result = typeof left === 'string' ? left.localeCompare(String(right)) : Number(left) - Number(right)
      return sort.direction === 'asc' ? result : -result
    })
  }, [predictions, sort])

  function chooseSort(key: SortKey) {
    setSort((current) => ({
      key,
      direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc',
    }))
  }

  async function loadFile(file: File) {
    setValidationError(null)
    setRequestError(null)
    setPredictions([])
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUpload(null)
      setValidationError('Choose a .csv file using the provided schema template.')
      return
    }
    try {
      const parsed = parseMetricCsv(await file.text(), model.features, model.max_batch_size)
      setUpload({ fileName: file.name, records: parsed.records })
    } catch (error) {
      setUpload(null)
      setValidationError(error instanceof Error ? error.message : 'The CSV could not be validated.')
    }
  }

  function loadSample() {
    setUpload({ fileName: 'labelled-schema-sample.csv', records: [{ ...SAMPLE_RECORD }] })
    setValidationError(null)
    setRequestError(null)
    setPredictions([])
  }

  async function runPrediction() {
    if (!upload) return
    setIsPredicting(true)
    setRequestError(null)
    try {
      const response = await api.predict(upload.records)
      setPredictions(response.predictions.slice().sort((a, b) => a.rank - b.rank))
      setScoreDescription(response.score_description)
      setSort({ key: 'rank', direction: 'asc' })
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'Prediction failed.')
    } finally {
      setIsPredicting(false)
    }
  }

  function clearUpload() {
    setUpload(null)
    setPredictions([])
    setValidationError(null)
    setRequestError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="space-y-7">
      <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="eyebrow">Prediction workspace</p>
          <h1 className="page-title">Rank a review batch</h1>
          <p className="page-intro">Upload static code metrics, validate them locally, then score the batch with the saved pipeline.</p>
        </div>
        <button
          type="button"
          onClick={() => downloadText('software-metrics-template.csv', templateCsv(model.features))}
          className="secondary-button"
        >
          <Download size={16} /> Download CSV template
        </button>
      </section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(300px,0.55fr)]">
        <section className="card p-5 sm:p-7">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="section-title">Upload metrics</h2>
              <p className="mt-1 text-sm text-slate-500">1–{model.max_batch_size} records · CSV only · blanks become missing metric values</p>
            </div>
            <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">{model.features.length} metrics</span>
          </div>

          <label
            onDragEnter={(event) => { event.preventDefault(); setIsDragging(true) }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setIsDragging(false)
              const file = event.dataTransfer.files[0]
              if (file) void loadFile(file)
            }}
            className={`mt-6 grid min-h-56 cursor-pointer place-items-center rounded-2xl border-2 border-dashed px-5 text-center transition ${
              isDragging ? 'border-teal-500 bg-teal-50' : 'border-slate-300 bg-slate-50/70 hover:border-teal-400 hover:bg-teal-50/50'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void loadFile(file)
              }}
            />
            <div>
              <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-white text-teal-700 shadow-sm ring-1 ring-slate-200">
                <UploadCloud size={25} />
              </span>
              <p className="mt-4 font-semibold text-slate-900">Drop a CSV here, or browse</p>
              <p className="mt-1 text-sm text-slate-500">Columns are checked before anything is sent.</p>
            </div>
          </label>

          {validationError && (
            <div role="alert" className="mt-4 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
              <ShieldAlert className="mt-0.5 shrink-0 text-rose-600" size={18} />
              <div><p className="font-semibold">CSV validation failed</p><p className="mt-0.5">{validationError}</p></div>
            </div>
          )}

          {upload && (
            <div className="mt-4 flex flex-col gap-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="shrink-0 text-emerald-600" size={20} />
                <div>
                  <p className="font-semibold text-emerald-950">Ready to score</p>
                  <p className="text-sm text-emerald-800">{upload.fileName} · {upload.records.length} {upload.records.length === 1 ? 'record' : 'records'}</p>
                </div>
              </div>
              <button type="button" onClick={clearUpload} className="inline-flex items-center gap-1.5 self-start text-xs font-semibold text-emerald-900 hover:text-emerald-700 sm:self-auto"><X size={14} /> Clear</button>
            </div>
          )}

          <button type="button" disabled={!upload || isPredicting} onClick={() => void runPrediction()} className="primary-button mt-5 w-full justify-center disabled:cursor-not-allowed disabled:opacity-45">
            {isPredicting ? <LoaderCircle className="animate-spin" size={17} /> : <Play size={17} fill="currentColor" />}
            {isPredicting ? 'Scoring with saved pipeline…' : 'Run prediction'}
          </button>
          {requestError && <p role="alert" className="mt-3 text-sm font-medium text-rose-700">{requestError}</p>}
        </section>

        <aside className="space-y-5">
          <section className="card p-5">
            <div className="flex items-center gap-3">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-indigo-50 text-indigo-700"><FileSpreadsheet size={19} /></span>
              <div><p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Labelled example</p><h2 className="font-semibold">Schema-valid sample</h2></div>
            </div>
            <p className="mt-4 text-sm leading-6 text-slate-600">A sample identifier and all required metrics. It intentionally contains no defect or ground-truth label.</p>
            <dl className="mt-4 grid grid-cols-2 gap-2 rounded-xl bg-slate-50 p-3 text-xs">
              <div className="col-span-2"><dt className="text-slate-400">record_id</dt><dd className="truncate font-mono text-slate-800">{SAMPLE_RECORD.record_id}</dd></div>
              <div><dt className="text-slate-400">wmc</dt><dd className="font-semibold">4</dd></div>
              <div><dt className="text-slate-400">loc</dt><dd className="font-semibold">26</dd></div>
              <div><dt className="text-slate-400">cbo</dt><dd className="font-semibold">6</dd></div>
              <div><dt className="text-slate-400">avg_cc</dt><dd className="font-semibold">0.75</dd></div>
            </dl>
            <details className="mt-3 text-xs text-slate-500">
              <summary className="cursor-pointer font-medium text-slate-700">View all sample fields</summary>
              <pre className="mt-2 max-h-52 overflow-auto rounded-lg bg-slate-950 p-3 text-[10px] leading-5 text-slate-200">{JSON.stringify(SAMPLE_RECORD, null, 2)}</pre>
            </details>
            <button type="button" onClick={loadSample} className="secondary-button mt-4 w-full justify-center">Use sample input</button>
          </section>

          <section className="rounded-2xl bg-slate-950 p-5 text-white shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-300">Score meaning</p>
            <p className="mt-3 text-sm leading-6 text-slate-300">{model.score_description}</p>
            <p className="mt-3 text-xs leading-5 text-slate-400">Use scores to prioritize review inside this batch. They are not verified defect probabilities.</p>
          </section>
        </aside>
      </div>

      <section className="card overflow-hidden">
        <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
          <div>
            <h2 className="section-title">Prediction results</h2>
            <p className="mt-1 text-sm text-slate-500">Highest-risk records appear first by default.</p>
          </div>
          {predictions.length > 0 && (
            <button type="button" onClick={() => downloadText('defect-prediction-results.csv', predictionsCsv(predictions))} className="secondary-button">
              <Download size={15} /> Download results CSV
            </button>
          )}
        </div>

        {predictions.length === 0 ? (
          <div className="grid min-h-64 place-items-center px-5 text-center">
            <div><ArrowDownUp className="mx-auto text-slate-300" size={30} /><p className="mt-3 font-medium text-slate-700">No prediction results yet</p><p className="mt-1 text-sm text-slate-500">Upload a valid CSV or use the labelled sample, then run prediction.</p></div>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="data-table min-w-[720px]">
                <thead><tr>
                  <th><SortButton label="Rank" active={sort.key === 'rank'} direction={sort.direction} onClick={() => chooseSort('rank')} /></th>
                  <th><SortButton label="Record identifier" active={sort.key === 'record_id'} direction={sort.direction} onClick={() => chooseSort('record_id')} /></th>
                  <th><SortButton label="Model risk score" active={sort.key === 'model_score'} direction={sort.direction} onClick={() => chooseSort('model_score')} /></th>
                  <th>Model version</th>
                </tr></thead>
                <tbody>{sortedPredictions.map((item) => (
                  <tr key={item.record_id}>
                    <td><span className="inline-grid h-8 w-8 place-items-center rounded-lg bg-slate-950 text-xs font-bold text-white">{item.rank}</span></td>
                    <td className="font-mono text-xs font-medium text-slate-800">{item.record_id}</td>
                    <td><div className="flex min-w-48 items-center gap-3"><div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-teal-500" style={{ width: `${Math.max(2, item.model_score * 100)}%` }} /></div><span className="w-16 text-right font-mono text-xs font-semibold">{item.model_score.toFixed(4)}</span></div></td>
                    <td className="text-xs text-slate-500">{item.model_version}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div className="border-t border-slate-200 bg-slate-50 px-5 py-3 text-xs text-slate-500 sm:px-7">{scoreDescription}</div>
          </>
        )}
      </section>
    </div>
  )
}

function SortButton({ label, active, direction, onClick }: { label: string; active: boolean; direction: 'asc' | 'desc'; onClick: () => void }) {
  return <button type="button" onClick={onClick} className="inline-flex items-center gap-1.5 font-semibold text-slate-600 hover:text-slate-950">{label}<ChevronDown size={13} className={`transition ${active ? 'text-teal-600' : 'text-slate-300'} ${active && direction === 'desc' ? 'rotate-180' : ''}`} /></button>
}
