import { AlertTriangle, Archive, ArrowUpRight, BarChart3, Database, GitCompareArrows, Target } from 'lucide-react'
import type {
  BaselineResults,
  ExperimentResults,
  FixedModelName,
  ModelComparisonResults,
  SimilarityResults,
} from '../types'

const score = (value: number) => value.toFixed(3)
const signed = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
const percent = (value: number) => `${(value * 100).toFixed(1)}%`
const modelLabel: Record<FixedModelName, string> = {
  dummy_prior: 'Dummy prior',
  logistic_regression: 'Logistic regression',
  random_forest: 'Random forest',
  hist_gradient_boosting: 'Histogram gradient boosting',
}

export function ExperimentScreen({ results }: { results: ExperimentResults }) {
  const hasSections = results.baseline || results.model_comparison || results.source_similarity
  return (
    <div className="space-y-8">
      <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="eyebrow">Frozen evaluation</p>
          <h1 className="page-title">Seven-project research results</h1>
          <p className="page-intro">Validated, read-only summaries from completed experiments. Opening this page never trains a model or changes the deployed pipeline.</p>
        </div>
        <div className="inline-flex items-center gap-2 self-start rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-semibold text-indigo-800">
          <Archive size={14} /> Historical reports
        </div>
      </section>

      {results.unavailable_sections.length > 0 && <MissingReports items={results.unavailable_sections} />}
      {!hasSections && (
        <section className="card p-6 text-sm text-slate-600">
          Research summaries are unavailable. Prediction remains separate and can continue when the configured model is healthy.
        </section>
      )}
      {results.baseline && <BaselineSection results={results.baseline} />}
      {results.model_comparison && <ModelComparisonSection results={results.model_comparison} />}
      {results.source_similarity && <SimilaritySection results={results.source_similarity} />}
    </div>
  )
}

function MissingReports({ items }: { items: Array<{ section: string; reason: string }> }) {
  return (
    <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 shrink-0 text-amber-700" size={20} />
        <div><h2 className="font-semibold text-amber-950">Some research sections are unavailable</h2><ul className="mt-2 space-y-1 text-sm text-amber-900">{items.map((item) => <li key={`${item.section}-${item.reason}`}><span className="font-semibold">{item.section}:</span> {item.reason}</li>)}</ul></div>
      </div>
    </section>
  )
}

function BaselineSection({ results }: { results: BaselineResults }) {
  const nearlyAll = new Set(results.source_selected_thresholds_flagging_all_or_nearly_all)
  return (
    <section className="space-y-4" aria-labelledby="baseline-heading">
      <SectionHeading eyebrow="Baseline" title="Original logistic evaluation" description="All six non-target projects train the fixed logistic pipeline. Ranking metrics and thresholded classification are reported separately." id="baseline-heading" />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryCard icon={Database} label="Target projects" value={String(results.project_count)} note="Every project held out once" tone="slate" />
        <SummaryCard icon={BarChart3} label="Dummy macro AP" value={score(results.summary.dummy_prior.average_precision.mean)} note="Equals mean target prevalence" tone="amber" />
        <SummaryCard icon={ArrowUpRight} label="Logistic macro AP" value={score(results.summary.logistic_regression.average_precision.mean)} note="Primary ranking metric" tone="teal" />
        <SummaryCard icon={Target} label="Top-20% recall" value={percent(results.summary.top_20_percent_recall.mean)} note="Exact review budget" tone="indigo" />
      </div>
      <section className="card overflow-hidden">
        <TableHeader title="Baseline per-project results" note={`Precision, recall, and F1 use threshold ${results.decision_threshold}; AP and top-20% recall evaluate ranking.`} />
        <div className="overflow-x-auto"><table className="data-table min-w-[940px]"><thead><tr><th>Project</th><th>Records</th><th>Dummy AP</th><th>Logistic AP</th><th>Top-20% recall</th><th>Precision @ 0.5</th><th>Recall @ 0.5</th><th>F1 @ 0.5</th></tr></thead><tbody>{results.projects.map((project) => <tr key={project.project}><ProjectCell name={project.project} /><NumberCell value={project.records} integer /><NumberCell value={project.dummy_prior.average_precision} /><HighlightCell value={project.logistic_regression.average_precision} /><NumberCell value={project.top_20_percent_recall} /><NumberCell value={project.logistic_regression.precision} /><NumberCell value={project.logistic_regression.recall} /><NumberCell value={project.logistic_regression.f1} /></tr>)}</tbody></table></div>
        <div className="border-t border-slate-200 bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-500 sm:px-7">{results.ranking_note}</div>
      </section>
      <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 sm:p-6">
        <div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 shrink-0 text-amber-700" size={20} /><div><h3 className="font-semibold text-amber-950">Source-selected threshold caution</h3><p className="mt-1 max-w-4xl text-sm leading-6 text-amber-900">Source-only threshold selection raised recall and F1 by flagging very broad sets. It did not change ranking scores or AP. {results.nearly_all_definition}</p><div className="mt-3 flex flex-wrap gap-2">{results.projects.filter((project) => nearlyAll.has(project.project)).map((project) => <span key={project.project} className="rounded-full border border-amber-300 bg-white/70 px-3 py-1.5 text-xs font-semibold capitalize text-amber-950">{project.project} · {project.source_selected_flagged_percentage.toFixed(1)}% flagged</span>)}</div></div></div>
      </section>
    </section>
  )
}

function ModelComparisonSection({ results }: { results: ModelComparisonResults }) {
  return (
    <section className="space-y-4" aria-labelledby="comparison-heading">
      <SectionHeading eyebrow="Fixed model comparison" title="Same folds, four fixed configurations" description={results.finding} id="comparison-heading" />
      <section className="card overflow-hidden">
        <TableHeader title="Equal-weight macro summary" note={`Classification columns use threshold ${results.decision_threshold}. Values are means across seven projects.`} />
        <div className="overflow-x-auto"><table className="data-table min-w-[780px]"><thead><tr><th>Model</th><th>AP</th><th>Top-20% recall</th><th>Precision @ 0.5</th><th>Recall @ 0.5</th><th>F1 @ 0.5</th></tr></thead><tbody>{results.models.map((model) => { const values = results.summary.models[model]; return <tr key={model}><td className="font-semibold text-slate-900">{modelLabel[model]}</td><NumberCell value={values.average_precision.mean} /><NumberCell value={values.top_20_percent_recall.mean} /><NumberCell value={values.precision.mean} /><NumberCell value={values.recall.mean} /><NumberCell value={values.f1.mean} /></tr> })}</tbody></table></div>
      </section>
      <section className="card overflow-hidden">
        <TableHeader title="Every held-out project and model" note="Random forest has the highest top-20% mean by a small margin; logistic regression has the highest macro AP." />
        <div className="overflow-x-auto"><table className="data-table min-w-[940px]"><thead><tr><th>Project</th><th>Model</th><th>AP</th><th>Top-20% recall</th><th>Precision @ 0.5</th><th>Recall @ 0.5</th><th>F1 @ 0.5</th></tr></thead><tbody>{results.projects.flatMap((project) => results.models.map((model) => { const metrics = project.models[model]; return <tr key={`${project.project}-${model}`}><ProjectCell name={project.project} /><td className="font-medium text-slate-800">{modelLabel[model]}</td><NumberCell value={metrics.average_precision} /><NumberCell value={metrics.top_20_percent_recall} /><NumberCell value={metrics.precision} /><NumberCell value={metrics.recall} /><NumberCell value={metrics.f1} /></tr> }))}</tbody></table></div>
        <div className="border-t border-slate-200 bg-slate-50 px-5 py-3 text-xs leading-5 text-slate-500 sm:px-7">Higher F1 at threshold 0.5 does not establish improved ranking. {results.ranking_note}</div>
      </section>
    </section>
  )
}

function SimilaritySection({ results }: { results: SimilarityResults }) {
  const diff = results.summary.differences_from_all_sources.similar_minus_all
  const randomDiff = results.summary.differences_from_all_sources.similar_minus_random_mean
  return (
    <section className="space-y-4" aria-labelledby="similarity-heading">
      <SectionHeading eyebrow="Source-project selection" title="Unlabelled-target similarity experiment" description={results.finding} id="similarity-heading" />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <SummaryCard icon={GitCompareArrows} label="Similar vs all AP" value={signed(diff.average_precision.mean)} note="Equal-weight project difference" tone="amber" />
        <SummaryCard icon={Target} label="Similar vs all top-20" value={signed(diff.top_20_percent_recall.mean)} note="Primary ranking difference" tone="amber" />
        <SummaryCard icon={Database} label="Subsets per target" value={String(results.random_subset_count)} note="Every three-project combination" tone="slate" />
        <SummaryCard icon={BarChart3} label="Similar vs subset AP" value={signed(randomDiff.average_precision.mean)} note="Against exhaustive subset mean" tone="indigo" />
      </div>
      <section className="card overflow-hidden">
        <TableHeader title="Equal-weight strategy summary" note={`The exhaustive reference is the mean of all ${results.random_subset_count} subsets within each target, never the best subset. Classification uses threshold ${results.decision_threshold}.`} />
        <div className="overflow-x-auto"><table className="data-table min-w-[820px]"><thead><tr><th>Strategy</th><th>AP</th><th>Top-20% recall</th><th>Precision @ 0.5</th><th>Recall @ 0.5</th><th>F1 @ 0.5</th></tr></thead><tbody>{(['all_sources', 'similar_sources', 'random_subset_mean'] as const).map((strategy) => { const values = results.summary.strategies[strategy]; return <tr key={strategy}><td className="font-semibold text-slate-900">{strategy.replaceAll('_', ' ')}</td><NumberCell value={values.average_precision.mean} /><NumberCell value={values.top_20_percent_recall.mean} /><NumberCell value={values.precision.mean} /><NumberCell value={values.recall.mean} /><NumberCell value={values.f1.mean} /></tr> })}</tbody></table></div>
      </section>
      <section className="card overflow-hidden">
        <TableHeader title="Per-project ranking and exhaustive reference" note="Random values show mean ± sample SD and the full range across all 20 subsets." />
        <div className="overflow-x-auto"><table className="data-table min-w-[1180px]"><thead><tr><th>Target</th><th>All AP</th><th>Similar AP</th><th>Random AP mean ± SD [range]</th><th>All top-20</th><th>Similar top-20</th><th>Random top-20 mean ± SD [range]</th></tr></thead><tbody>{results.projects.map((project) => { const randomAp = project.random_subset_reference.distributions.average_precision; const randomTop = project.random_subset_reference.distributions.top_20_percent_recall; return <tr key={project.project}><ProjectCell name={project.project} /><NumberCell value={project.all_sources.metrics.average_precision} /><NumberCell value={project.similar_sources.metrics.average_precision} /><td className="font-mono text-xs">{distributionLabel(randomAp)}</td><NumberCell value={project.all_sources.metrics.top_20_percent_recall} /><NumberCell value={project.similar_sources.metrics.top_20_percent_recall} /><td className="font-mono text-xs">{distributionLabel(randomTop)}</td></tr> })}</tbody></table></div>
      </section>
      <section className="card p-5 sm:p-7">
        <h3 className="section-title">Selected sources and Euclidean distances</h3><p className="mt-1 text-sm leading-6 text-slate-500">Distances use source-fitted preprocessing and 40-value median/IQR summaries. Target metrics are accessed without target labels.</p>
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">{results.projects.map((project) => <article key={project.project} className="rounded-xl border border-slate-200 bg-slate-50 p-4"><h4 className="font-semibold capitalize text-slate-950">{project.project}</h4><div className="mt-3 space-y-2">{project.selected_sources.map((source) => <div key={source.source_project} className={`flex items-center justify-between rounded-lg px-3 py-2 text-xs ${source.selected ? 'bg-teal-100 font-semibold text-teal-950' : 'bg-white text-slate-500'}`}><span className="capitalize">#{source.rank} {source.source_project}</span><span className="font-mono">{source.distance.toFixed(3)}</span></div>)}</div></article>)}</div>
        <p className="mt-5 text-xs leading-5 text-slate-500">{results.unlabelled_target_access} The three-project reference matches project count, not training-row count. {results.ranking_note}</p>
      </section>
    </section>
  )
}

function distributionLabel(value: { mean: number; sample_standard_deviation: number; minimum: number; maximum: number }) {
  return `${score(value.mean)} ± ${score(value.sample_standard_deviation)} [${score(value.minimum)}, ${score(value.maximum)}]`
}

function SectionHeading({ eyebrow, title, description, id }: { eyebrow: string; title: string; description: string; id: string }) {
  return <div className="pt-2"><p className="eyebrow">{eyebrow}</p><h2 id={id} className="mt-1 text-2xl font-bold tracking-tight text-slate-950">{title}</h2><p className="mt-2 max-w-4xl text-sm leading-6 text-slate-500">{description}</p></div>
}

function TableHeader({ title, note }: { title: string; note: string }) {
  return <div className="border-b border-slate-200 px-5 py-5 sm:px-7"><h3 className="section-title">{title}</h3><p className="mt-1 text-sm leading-6 text-slate-500">{note}</p></div>
}

function ProjectCell({ name }: { name: string }) { return <td><span className="font-semibold capitalize text-slate-900">{name}</span></td> }
function NumberCell({ value, integer = false }: { value: number; integer?: boolean }) { return <td className="font-mono text-xs">{integer ? value.toLocaleString() : score(value)}</td> }
function HighlightCell({ value }: { value: number }) { return <td><span className="rounded-md bg-teal-50 px-2 py-1 font-mono text-xs font-semibold text-teal-800">{score(value)}</span></td> }

function SummaryCard({ icon: Icon, label, value, note, tone }: { icon: typeof Database; label: string; value: string; note: string; tone: 'slate' | 'amber' | 'teal' | 'indigo' }) {
  const styles = { slate: 'bg-slate-100 text-slate-700', amber: 'bg-amber-100 text-amber-800', teal: 'bg-teal-100 text-teal-800', indigo: 'bg-indigo-100 text-indigo-800' }
  return <article className="card p-5"><div className={`grid h-10 w-10 place-items-center rounded-xl ${styles[tone]}`}><Icon size={19} /></div><p className="mt-5 text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</p><p className="mt-1 text-3xl font-bold tracking-tight text-slate-950">{value}</p><p className="mt-2 text-xs text-slate-500">{note}</p></article>
}
