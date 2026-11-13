export type Health = {
  status: string
  model_loaded: boolean
  model_version: string
}

export type ModelInfo = {
  model_version: string
  model_type: string
  score_description: string
  training_projects: string[]
  evaluation_only_projects: string[]
  features: string[]
  max_batch_size: number
  packaging_method: string
}

export type MetricRecord = {
  record_id: string
  [feature: string]: string | number | null
}

export type Prediction = {
  record_id: string
  model_score: number
  rank: number
  model_version: string
}

export type PredictionResponse = {
  score_description: string
  predictions: Prediction[]
}

export type MetricSummary = {
  mean: number
  minimum: number
  maximum: number
  sample_standard_deviation: number
}

export type EvaluationMetrics = {
  average_precision: number
  precision: number
  recall: number
  f1: number
}

export type ExperimentProject = {
  project: string
  records: number
  defect_prevalence: number
  dummy_prior: EvaluationMetrics
  logistic_regression: EvaluationMetrics
  top_20_percent_recall: number
  top_20_percent_reviewed_records: number
  source_selected_threshold: number
  source_selected_flagged_records: number
  source_selected_flagged_percentage: number
}

export type ExperimentResults = {
  report_type: string
  historical: true
  project_count: number
  decision_threshold: number
  tuning: string
  macro_average_note: string
  target_label_usage: string
  ranking_note: string
  nearly_all_definition: string
  source_selected_thresholds_flagging_all_or_nearly_all: string[]
  summary: {
    dummy_prior: { average_precision: MetricSummary }
    logistic_regression: {
      average_precision: MetricSummary
      precision: MetricSummary
      recall: MetricSummary
      f1: MetricSummary
    }
    top_20_percent_recall: MetricSummary
  }
  projects: ExperimentProject[]
}

export type Loadable<T> =
  | { state: 'loading' }
  | { state: 'success'; data: T }
  | { state: 'error'; message: string }
