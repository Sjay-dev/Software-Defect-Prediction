import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Footer } from './Footer'

describe('Footer', () => {
  it('renders the shared attribution and verified destinations', () => {
    const markup = renderToStaticMarkup(<Footer />)

    expect(markup).toContain('© 2026 Joseph Sanusi. All rights reserved.')
    expect(markup).toContain('href="https://github.com/Sjay-dev"')
    expect(markup).toContain('target="_blank"')
    expect(markup).toContain('rel="noopener noreferrer"')
    expect(markup).toContain(
      'href="https://github.com/Sjay-dev/Software-Defect-Prediction"',
    )
    expect(markup).toContain(
      'href="https://github.com/Sjay-dev/Software-Defect-Prediction/blob/main/docs/DATASET.md"',
    )
  })
})
