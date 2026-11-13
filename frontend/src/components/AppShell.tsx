import {
  Activity,
  BarChart3,
  BrainCircuit,
  PanelLeft,
  UploadCloud,
} from 'lucide-react'
import type { ReactNode } from 'react'
import type { Health, Loadable } from '../types'

export type Screen = 'prediction' | 'experiments' | 'model'

const navigation = [
  { id: 'prediction' as const, label: 'Prediction', icon: UploadCloud },
  { id: 'experiments' as const, label: 'Experiments', icon: BarChart3 },
  { id: 'model' as const, label: 'Model information', icon: BrainCircuit },
]

export function AppShell({
  active,
  onNavigate,
  health,
  children,
}: {
  active: Screen
  onNavigate: (screen: Screen) => void
  health: Loadable<Health>
  children: ReactNode
}) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-950 lg:grid lg:grid-cols-[264px_1fr]">
      <aside className="relative overflow-hidden bg-slate-950 px-5 pb-4 pt-6 text-white lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col lg:px-6 lg:py-8">
        <div className="pointer-events-none absolute -right-20 -top-28 h-64 w-64 rounded-full bg-teal-400/10 blur-3xl" />
        <div className="relative flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-2xl bg-teal-400 text-slate-950 shadow-lg shadow-teal-950/30">
            <Activity size={22} strokeWidth={2.4} />
          </div>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-teal-300">Defect review</p>
            <p className="mt-0.5 font-semibold tracking-tight">Software Risk Lab</p>
          </div>
        </div>

        <nav aria-label="Main navigation" className="relative mt-6 flex gap-2 overflow-x-auto pb-1 lg:mt-12 lg:flex-col lg:overflow-visible">
          {navigation.map(({ id, label, icon: Icon }) => {
            const selected = active === id
            return (
              <button
                key={id}
                type="button"
                onClick={() => onNavigate(id)}
                aria-current={selected ? 'page' : undefined}
                className={`flex min-w-fit items-center gap-3 rounded-xl px-4 py-3 text-left text-sm font-medium transition ${
                  selected
                    ? 'bg-white text-slate-950 shadow-sm'
                    : 'text-slate-300 hover:bg-white/8 hover:text-white'
                }`}
              >
                <Icon size={18} />
                {label}
              </button>
            )
          })}
        </nav>

        <div className="relative mt-5 border-t border-white/10 pt-4 lg:mt-auto">
          <div className="flex items-center gap-2.5 text-xs text-slate-300">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                health.state === 'success'
                  ? 'bg-emerald-400 shadow-[0_0_0_4px_rgba(52,211,153,0.12)]'
                  : health.state === 'error'
                    ? 'bg-rose-400'
                    : 'animate-pulse bg-amber-300'
              }`}
            />
            {health.state === 'success'
              ? `API connected · ${health.data.model_version}`
              : health.state === 'error'
                ? 'API unavailable'
                : 'Checking API…'}
          </div>
        </div>
      </aside>

      <main className="min-w-0">
        <header className="flex h-16 items-center justify-between border-b border-slate-200/80 bg-white/80 px-5 backdrop-blur lg:px-9">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <PanelLeft size={16} />
            <span>Cross-project defect prediction</span>
          </div>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">Local research dashboard</span>
        </header>
        <div className="mx-auto max-w-[1480px] p-5 sm:p-7 lg:p-9">{children}</div>
      </main>
    </div>
  )
}
