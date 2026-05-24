/**
 * Model Insights – Types aligned to flow_inference + model_versions
 */

export interface FlowInferenceRow {
  id: string;
  flow_id: string;
  model_version_id: string;
  engine: string;
  predicted_label: string | null;
  confidence: number | null;
  anomaly_score: number | null;
  is_anomaly: boolean | null;
  created_at: string;
}

export interface ModelComparisonRow {
  id: string;
  version: string;
  engine: string;
  accuracy: number | null;
  f1_score: number | null;
  precision: number | null;
  recall: number | null;
  latency_ms: number | null;
  false_positive_rate: number | null;
  created_at: string;
}
