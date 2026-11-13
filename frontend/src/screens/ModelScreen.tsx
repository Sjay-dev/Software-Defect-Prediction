import { Boxes, Check, Cpu, Database, Info, Layers3, ShieldCheck } from 'lucide-react'
import type { ModelInfo } from '../types'

const limitations = [
  'Scores are relative model risk scores, not calibrated or verified defect probabilities.',
  'The model was evaluated across seven historical Java projects; transfer to a new codebase is not guaranteed.',
  'Ranks compare records only within the submitted batch and change when the batch changes.',
  'Only the listed static code metrics are used; identifiers and historical defect labels are excluded.',
]

export function ModelScreen({ model }: { model: ModelInfo }) {
  return (
    <div className="space-y-7">
      <section>
        <p className="eyebrow">Deployment metadata</p>
        <h1 className="page-title">Model information</h1>
        <p className="page-intro">The exact schema and provenance returned by the running FastAPI service.</p>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="card overflow-hidden">
          <div className="bg-slate-950 p-6 text-white sm:p-8">
            <div className="flex items-start justify-between gap-4">
              <span className="grid h-12 w-12 place-items-center rounded-2xl bg-teal-400 text-slate-950"><Cpu size={23} /></span>
              <span className="rounded-full border border-white/15 px-3 py-1 text-xs text-slate-300">Loaded at API startup</span>
            </div>
            <p className="mt-7 text-xs font-semibold uppercase tracking-[0.18em] text-teal-300">Model version</p>
            <h2 className="mt-2 break-words text-2xl font-bold tracking-tight">{model.model_version}</h2>
            <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-300">{model.model_type}</p>
          </div>
          <dl className="grid gap-px bg-slate-200 sm:grid-cols-2">
            <div className="bg-white p-5"><dt className="text-xs font-semibold uppercase tracking-wider text-slate-400">Feature count</dt><dd className="mt-2 text-xl font-bold">{model.features.length}</dd></div>
            <div className="bg-white p-5"><dt className="text-xs font-semibold uppercase tracking-wider text-slate-400">Maximum batch</dt><dd className="mt-2 text-xl font-bold">{model.max_batch_size} records</dd></div>
          </dl>
        </div>

        <div className="card p-6">
          <div className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-emerald-50 text-emerald-700"><ShieldCheck size={20} /></span><div><p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Artifact handling</p><h2 className="font-semibold">Frozen and verified</h2></div></div>
          <p className="mt-5 text-sm leading-6 text-slate-600">{model.packaging_method}</p>
          <div className="mt-5 rounded-xl border border-teal-200 bg-teal-50 p-4 text-sm leading-6 text-teal-950">{model.score_description}</div>
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        <div className="card p-6">
          <div className="flex items-center gap-3"><Database className="text-teal-700" size={20} /><h2 className="section-title">Training projects</h2></div>
          <p className="mt-2 text-sm text-slate-500">These projects contributed rows to the served pipeline.</p>
          <div className="mt-5 flex flex-wrap gap-2">{model.training_projects.map((project) => <span key={project} className="inline-flex items-center gap-1.5 rounded-full bg-slate-950 px-3 py-1.5 text-xs font-semibold capitalize text-white"><Check size={12} className="text-teal-300" />{project}</span>)}</div>
        </div>
        <div className="card p-6">
          <div className="flex items-center gap-3"><Layers3 className="text-indigo-700" size={20} /><h2 className="section-title">Evaluation-only projects</h2></div>
          <p className="mt-2 text-sm text-slate-500">Held out from this model’s training and used for evaluation.</p>
          <div className="mt-5 flex flex-wrap gap-2">{model.evaluation_only_projects.map((project) => <span key={project} className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-xs font-semibold capitalize text-indigo-900">{project}</span>)}</div>
        </div>
      </section>

      <section className="card p-6 sm:p-7">
        <div className="flex items-center gap-3"><Boxes className="text-teal-700" size={20} /><div><h2 className="section-title">Required feature schema</h2><p className="mt-1 text-sm text-slate-500">CSV column names must match these backend feature names exactly.</p></div></div>
        <div className="mt-6 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-5">{model.features.map((feature, index) => <div key={feature} className="rounded-xl border border-slate-200 bg-slate-50 p-3"><span className="text-[10px] font-semibold text-slate-400">{String(index + 1).padStart(2, '0')}</span><p className="mt-1 font-mono text-sm font-semibold text-slate-800">{feature}</p></div>)}</div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-7">
        <div className="flex items-center gap-3"><Info className="text-amber-600" size={20} /><h2 className="section-title">Limitations</h2></div>
        <ul className="mt-5 grid gap-3 lg:grid-cols-2">{limitations.map((limitation) => <li key={limitation} className="flex items-start gap-3 rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-700"><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" />{limitation}</li>)}</ul>
      </section>
    </div>
  )
}
