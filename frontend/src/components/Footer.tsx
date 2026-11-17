const links = [
  {
    label: 'GitHub',
    href: 'https://github.com/Sjay-dev',
    newTab: true,
  },
  {
    label: 'Methodology',
    href: 'https://github.com/Sjay-dev/Software-Defect-Prediction',
    newTab: false,
  },
  {
    label: 'Dataset sources',
    href: 'https://github.com/Sjay-dev/Software-Defect-Prediction/blob/main/docs/DATASET.md',
    newTab: false,
  },
] as const

export function Footer() {
  return (
    <footer className="border-t border-slate-200/80 bg-white/50 px-5 py-6 sm:px-7 lg:px-9">
      <div className="mx-auto flex max-w-[1480px] flex-col items-center text-center text-xs leading-5 text-slate-500">
        <p>© 2026 Joseph Sanusi. All rights reserved.</p>
        <nav aria-label="Footer links" className="mt-2 flex flex-wrap items-center justify-center gap-x-2 gap-y-1">
          {links.map((link, index) => (
            <span key={link.label} className="inline-flex items-center gap-x-2">
              {index > 0 && <span aria-hidden="true" className="text-slate-300">·</span>}
              <a
                href={link.href}
                {...(link.newTab ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
                className="rounded-sm font-medium text-slate-600 underline-offset-4 transition hover:text-teal-700 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-600"
              >
                {link.label}
              </a>
            </span>
          ))}
        </nav>
      </div>
    </footer>
  )
}
