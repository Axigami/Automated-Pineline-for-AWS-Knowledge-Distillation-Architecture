/**
 * MLOps – Types aligned to retrain_jobs_all, model_versions, deployments_all
 */

export interface ModelVersionRow {
  id: string;
  model_id: string;
  version: string;
  status: string;
  accuracy: number | null;
  f1_score: number | null;
  precision: number | null;
  recall: number | null;
  latency_ms: number | null;
  memory_mb: number | null;
  false_positive_rate: number | null;
  throughput_per_s: number | null;
  artifact_uri: string | null;
  created_at: string;
  author: string | null;
}

export interface RetrainJobRow {
  job_id: string;
  job_status: string | null;
  job_progress_percent: number | null;
  job_epochs: number | null;
  job_started_at: string | null;
  job_finished_at: string | null;
  job_created_at: string | null;
  event_step: string | null;
  event_message: string | null;
  event_progress_percent: number | null;
  job_knowledge_distillation: boolean | null;
  job_data_range: string | null;
}

export interface RetrainConfig {
  homeId: string;
  dataDays: number;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
}

export type RetrainJobStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface OtaDeployRequest {
  modelVersionId: string;
  targetNodeIds: string[];
}
