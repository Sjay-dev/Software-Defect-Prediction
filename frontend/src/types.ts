export type Health = {
  status: string
  model_loaded: boolean
  model_version: string
}

export type ModelInfo = {
  model_version: string
  model_sha256: string
  model_type: string
  score_description: string
  training_projects: string[]
  evaluation_only_projects: string[]
  features: string[]
  max_batch_size: number
  packaging_method: string
  decision_threshold?: number | null
  calibration_status?: string | null
}

export type MetricRecord = {
  record_id: string
  [feature: string]: string | number | null
}

export type Prediction = {
  record_id: string
  class_name?: string
  source_row?: number
  labels?: Record<string, string | number | boolean | null>
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
  top_20_percent_recall?: number
  precision: number
  recall: number
  f1: number
  confusion_matrix?: {
    tn: number
    fp: number
    fn: number
    tp: number
  }
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
  source_selected_metrics: EvaluationMetrics
  source_selected_flagged_records: number
  source_selected_flagged_percentage: number
}

export type BaselineResults = {
  report_type: string
  schema_version: number
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
    source_selected_threshold: {
      precision: MetricSummary
      recall: MetricSummary
      f1: MetricSummary
    }
  }
  projects: ExperimentProject[]
}

export type FixedModelName = 'dummy_prior' | 'logistic_regression' | 'random_forest' | 'hist_gradient_boosting'

export type RankingEvaluationMetrics = Required<EvaluationMetrics>

export type ModelComparisonResults = {
  report_type: 'fixed_model_comparison'
  schema_version: 1
  historical: true
  project_count: number
  decision_threshold: number
  top_fraction: number
  models: FixedModelName[]
  ranking_note: string
  finding: string
  summary: {
    models: Record<FixedModelName, Record<'average_precision' | 'top_20_percent_recall' | 'precision' | 'recall' | 'f1', MetricSummary>>
  }
  projects: Array<{
    project: string
    records: number
    defect_prevalence: number
    models: Record<FixedModelName, RankingEvaluationMetrics>
  }>
}

export type Distribution = MetricSummary

export type SimilarityResults = {
  report_type: 'source_project_similarity_selection'
  schema_version: 1
  historical: true
  project_count: number
  decision_threshold: number
  top_fraction: number
  selected_source_count: number
  random_subset_count: number
  unlabelled_target_access: string
  ranking_note: string
  finding: string
  summary: {
    strategies: Record<'all_sources' | 'similar_sources' | 'random_subset_mean', Record<'average_precision' | 'top_20_percent_recall' | 'precision' | 'recall' | 'f1', MetricSummary>>
    differences_from_all_sources: Record<'similar_minus_all' | 'random_mean_minus_all' | 'similar_minus_random_mean', Record<'average_precision' | 'top_20_percent_recall' | 'precision' | 'recall' | 'f1', MetricSummary>>
  }
  projects: Array<{
    project: string
    records: number
    defect_prevalence: number
    selected_sources: Array<{
      source_project: string
      distance: number
      rank: number
      selected: boolean
    }>
    all_sources: {
      source_projects: string[]
      training_rows: number
      training_defect_prevalence: number
      metrics: RankingEvaluationMetrics
    }
    similar_sources: {
      source_projects: string[]
      training_rows: number
      training_defect_prevalence: number
      metrics: RankingEvaluationMetrics
    }
    random_subset_reference: {
      combination_count: number
      aggregation: string
      distributions: Record<'average_precision' | 'top_20_percent_recall' | 'precision' | 'recall' | 'f1' | 'tn' | 'fp' | 'fn' | 'tp' | 'training_rows' | 'training_defect_prevalence' | 'training_seconds' | 'prediction_seconds', Distribution>
    }
  }>
}

export type ExperimentResults = {
  report_type: 'consolidated_historical_research'
  historical: true
  available_sections: Array<'baseline' | 'model_comparison' | 'source_similarity'>
  unavailable_sections: Array<{ section: string; reason: string }>
  baseline: BaselineResults | null
  model_comparison: ModelComparisonResults | null
  source_similarity: SimilarityResults | null
}

export type Loadable<T> =
  | { state: 'loading' }
  | { state: 'success'; data: T }
  | { state: 'error'; message: string }
