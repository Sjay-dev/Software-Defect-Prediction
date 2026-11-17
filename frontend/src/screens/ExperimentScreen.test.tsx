import { readFileSync } from 'node:fs'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { BaselineResults, ExperimentResults, ModelComparisonResults, SimilarityResults } from '../types'
import { ExperimentScreen } from './ExperimentScreen'

function report<T>(relativePath: string): T {
  return JSON.parse(
    readFileSync(new URL(`../../../reports/${relativePath}`, import.meta.url), 'utf8'),
  ) as T
}

const complete: ExperimentResults = {
  report_type: 'consolidated_historical_research',
  historical: true,
  available_sections: ['baseline', 'model_comparison', 'source_similarity'],
  unavailable_sections: [],
  baseline: report<BaselineResults>('baseline/summary.json'),
  model_comparison: report<ModelComparisonResults>('model_comparison/summary.json'),
  source_similarity: report<SimilarityResults>('source_similarity/summary.json'),
}

describe('ExperimentScreen', () => {
  it('renders the values and selected sources from the tracked reports', () => {
    const markup = renderToStaticMarkup(<ExperimentScreen results={complete} />)

    expect(markup).toContain('Original logistic evaluation')
    expect(markup).toContain('0.554')
    expect(markup).toContain('Same folds, four fixed configurations')
    expect(markup).toContain('Unlabelled-target similarity experiment')
    expect(markup).toContain('Random values show mean ± sample SD')
    expect(markup).toContain('#1 lucene')
    expect(markup).toContain('Opening this page never trains a model')
  })

  it('keeps available results visible when another report is missing', () => {
    const partial: ExperimentResults = {
      ...complete,
      available_sections: ['baseline'],
      unavailable_sections: [{ section: 'model_comparison', reason: 'missing' }],
      model_comparison: null,
      source_similarity: null,
    }
    const markup = renderToStaticMarkup(<ExperimentScreen results={partial} />)

    expect(markup).toContain('Some research sections are unavailable')
    expect(markup).toContain('Original logistic evaluation')
    expect(markup).not.toContain('Same folds, four fixed configurations')
  })
})
