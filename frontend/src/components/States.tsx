import { AlertCircle, LoaderCircle, RefreshCw } from 'lucide-react'

export function LoadingPanel({ label }: { label: string }) {
  return (
    <div className="grid min-h-72 place-items-center rounded-2xl border border-slate-200 bg-white">
      <div className="text-center text-sm text-slate-500">
        <LoaderCircle className="mx-auto mb-3 animate-spin text-teal-600" size={26} />
        {label}
      </div>
    </div>
  )
}

export function ErrorPanel({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-rose-950">
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 shrink-0 text-rose-600" size={20} />
        <div>
          <p className="font-semibold">Unable to load this view</p>
          <p className="mt-1 text-sm text-rose-800">{message}</p>
          {onRetry && (
            <button type="button" onClick={onRetry} className="mt-4 inline-flex items-center gap-2 rounded-lg bg-rose-900 px-3 py-2 text-xs font-semibold text-white hover:bg-rose-800">
              <RefreshCw size={14} /> Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
