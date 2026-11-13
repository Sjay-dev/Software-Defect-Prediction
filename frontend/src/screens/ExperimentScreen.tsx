import { AlertTriangle, Archive, ArrowUpRight, BarChart3, Database, Target } from 'lucide-react'
import type { ExperimentResults } from '../types'

const score = (value: number) => value.toFixed(3)
const percent = (value: number) => `${(value * 100).toFixed(1)}%`

export function ExperimentScreen({ results }: { results: ExperimentResults }) {
  const nearlyAll = new Set(results.source_selected_thresholds_flagging_all_or_nearly_all)
  return (
    <div className="space-y-7">
      <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="eyebrow">Frozen evaluation</p>
          <h1 className="page-title">Seven-project results</h1>
          <p className="page-intro">Historical leave-one-project-out results. These metrics are not generated from new dashboard predictions.</p>
        </div>
        <div className="inline-flex items-center gap-2 self-start rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-semibold text-indigo-800">
          <Archive size={14} /> Historical report
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryCard icon={Database} label="Target projects" value={String(results.project_count)} note="Every project held out once" tone="slate" />
        <SummaryCard icon={BarChart3} label="Dummy mean AP" value={score(results.summary.dummy_prior.average_precision.mean)} note="Tracks mean defect prevalence" tone="amber" />
        <SummaryCard icon={ArrowUpRight} label="Logistic mean AP" value={score(results.summary.logistic_regression.average_precision.mean)} note="Macro average across projects" tone="teal" />
        <SummaryCard icon={Target} label="Mean top-20% recall" value={percent(results.summary.top_20_percent_recall.mean)} note="Ranking result, fixed review budget" tone="indigo" />
      </section>

      <section className="card p-5 sm:p-7">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div><h2 className="section-title">Average precision comparison</h2><p className="mt-1 text-sm text-slate-500">Dummy prior versus logistic regression for each held-out project.</p></div>
          <div className="flex gap-4 text-xs text-slate-500"><Legend color="bg-amber-400" label="Dummy" /><Legend color="bg-teal-500" label="Logistic" /></div>
        </div>
        <div className="mt-7 grid gap-x-8 gap-y-5 lg:grid-cols-2">
          {results.projects.map((project) => (
            <div key={project.project} className="grid grid-cols-[70px_1fr_42px] items-center gap-3">
              <span className="text-sm font-semibold capitalize text-slate-700">{project.project}</span>
              <div className="space-y-1.5">
                <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-amber-400" style={{ width: `${project.dummy_prior.average_precision * 100}%` }} /></div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-teal-500" style={{ width: `${project.logistic_regression.average_precision * 100}%` }} /></div>
              </div>
              <span className="text-right font-mono text-xs font-semibold text-teal-800">{score(project.logistic_regression.average_precision)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card overflow-hidden">
        <div className="border-b border-slate-200 px-5 py-5 sm:px-7">
          <h2 className="section-title">Per-project metrics</h2>
          <p className="mt-1 text-sm text-slate-500">Classification metrics use the original fixed threshold of {results.decision_threshold}. AP and top-20% recall evaluate ranking.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table min-w-[940px]">
            <thead><tr><th>Project</th><th>Records</th><th>Dummy AP</th><th>Logistic AP</th><th>Precision</th><th>Recall</th><th>F1</th><th>Top-20% recall</th></tr></thead>
            <tbody>{results.projects.map((project) => (
              <tr key={project.project}>
                <td><span className="font-semibold capitalize text-slate-900">{project.project}</span></td>
                <td className="font-mono text-xs">{project.records.toLocaleString()}</td>
                <td className="font-mono text-xs">{score(project.dummy_prior.average_precision)}</td>
                <td><span className="rounded-md bg-teal-50 px-2 py-1 font-mono text-xs font-semibold text-teal-800">{score(project.logistic_regression.average_precision)}</span></td>
                <td className="font-mono text-xs">{score(project.logistic_regression.precision)}</td>
                <td className="font-mono text-xs">{score(project.logistic_regression.recall)}</td>
                <td className="font-mono text-xs">{score(project.logistic_regression.f1)}</td>
                <td className="font-mono text-xs font-semibold">{percent(project.top_20_percent_recall)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <div className="border-t border-slate-200 bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-500 sm:px-7">{results.ranking_note}</div>
      </section>

      <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-100 text-amber-800"><AlertTriangle size={20} /></span>
          <div>
            <h2 className="font-semibold text-amber-950">Selected-threshold caution</h2>
            <p className="mt-1 max-w-4xl text-sm leading-6 text-amber-900">Source-selected thresholds flag all or nearly all target records for the projects below. High recall in these cases comes with an impractically broad review set. {results.nearly_all_definition}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {results.projects.filter((project) => nearlyAll.has(project.project)).map((project) => (
                <span key={project.project} className="rounded-full border border-amber-300 bg-white/70 px-3 py-1.5 text-xs font-semibold capitalize text-amber-950">{project.project} · {project.source_selected_flagged_percentage.toFixed(1)}% flagged</span>
              ))}
            </div>
          </div>
        </div>
      </section>

      <p className="text-xs leading-5 text-slate-500">{results.macro_average_note} Outer labels were used for {results.target_label_usage}; tuning remains “{results.tuning}”.</p>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return <span className="inline-flex items-center gap-1.5"><span className={`h-2.5 w-2.5 rounded-full ${color}`} />{label}</span>
}

function SummaryCard({ icon: Icon, label, value, note, tone }: { icon: typeof Database; label: string; value: string; note: string; tone: 'slate' | 'amber' | 'teal' | 'indigo' }) {
  const styles = { slate: 'bg-slate-100 text-slate-700', amber: 'bg-amber-100 text-amber-800', teal: 'bg-teal-100 text-teal-800', indigo: 'bg-indigo-100 text-indigo-800' }
  return <article className="card p-5"><div className={`grid h-10 w-10 place-items-center rounded-xl ${styles[tone]}`}><Icon size={19} /></div><p className="mt-5 text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</p><p className="mt-1 text-3xl font-bold tracking-tight text-slate-950">{value}</p><p className="mt-2 text-xs text-slate-500">{note}</p></article>
}
