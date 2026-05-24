/**
 * MLOps Service
 * Handles all database operations for MLOps module
 * - Retrain job management and tracking
 * - Model version management
 * - Pipeline progress monitoring
 * - Deployment tracking
 */

import { supabase } from '../../core/lib/supabaseClient';

// ============================================================================
// TYPES
// ============================================================================

export interface RetrainJob {
  job_id: string;
  job_requested_by: string | null;
  job_home_id: string | null;
  job_status: string | null;
  job_data_range: string | null;
  job_epochs: number | null;
  job_knowledge_distillation: boolean | null;
  job_teacher_from_version_id: string | null;
  job_teacher_to_version_id: string | null;
  job_student_from_version_id: string | null;
  job_student_to_version_id: string | null;
  job_progress_percent: number | null;
  job_started_at: string | null;
  job_finished_at: string | null;
  job_created_at: string | null;
  job_pipeline_steps_json: string | null;
  job_training_batch_size: number | null;
  job_training_learning_rate: number | null;
}

export interface JobEvent {
  event_id: string;
  event_job_id: string;
  event_ts: string;
  event_step: string;
  event_message: string;
  event_progress_percent: number | null;
}

export interface ModelVersion {
  id: string;
  model_id: string;
  version: string;
  status: string;
  artifact_uri: string | null;
  metrics_json: string | null;
  created_at: string;
  author: string | null;
  accuracy: number | null;
  f1_score: number | null;
  precision: number | null;
  recall: number | null;
  latency_ms: number | null;
  memory_mb: number | null;
  false_positive_rate: number | null;
  throughput_per_s: number | null;
}

export interface Model {
  id: string;
  name: string;
  kind: 'cloud' | 'edge';
  description: string | null;
  created_at: string;
}

export interface PipelineStep {
  step: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  metrics?: Record<string, any>;
}

export interface TriggerRetrainParams {
  home_id: string;
  user_id: string;
  data_range?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  knowledge_distillation?: boolean;
}

// ============================================================================
// RETRAIN JOBS - Training pipeline management
// ============================================================================

/**
 * Get all retrain jobs ordered by creation date
 */
export async function getRetrainJobs(limit = 20): Promise<RetrainJob[]> {
  const { data, error } = await supabase
    .from('retrain_jobs_all')
    .select('*')
    .order('job_created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch retrain jobs: ${error.message}`);
  return (data ?? []) as RetrainJob[];
}

/**
 * Get active (running/queued) retrain jobs
 */
export async function getActiveRetrainJobs(): Promise<RetrainJob[]> {
  const { data, error } = await supabase
    .from('retrain_jobs_all')
    .select('*')
    .in('job_status', ['queued', 'running'])
    .order('job_created_at', { ascending: false });

  if (error) throw new Error(`Failed to fetch active jobs: ${error.message}`);
  return (data ?? []) as RetrainJob[];
}

/**
 * Get single retrain job by ID
 */
export async function getRetrainJob(jobId: string): Promise<RetrainJob | null> {
  const { data, error } = await supabase
    .from('retrain_jobs_all')
    .select('*')
    .eq('job_id', jobId)
    .single();

  if (error) {
    if (error.code === 'PGRST116') return null; // Not found
    throw new Error(`Failed to fetch job: ${error.message}`);
  }
  return data as RetrainJob;
}

/**
 * Get job events (progress updates)
 */
export async function getJobEvents(jobId: string): Promise<JobEvent[]> {
  const { data, error } = await supabase
    .from('retrain_jobs_all')
    .select('event_id, event_job_id, event_ts, event_step, event_message, event_progress_percent')
    .eq('event_job_id', jobId)
    .not('event_step', 'is', null)
    .order('event_ts', { ascending: true });

  if (error) throw new Error(`Failed to fetch job events: ${error.message}`);
  return (data ?? []) as JobEvent[];
}

/**
 * Trigger new retrain job
 * This creates a job record in Supabase and DynamoDB.
 * The DynamoDB Stream triggers the RetrainJobHandler Lambda which starts the training pipeline.
 */
export async function triggerRetrain(params: TriggerRetrainParams): Promise<string> {
  const jobId = crypto.randomUUID();
  
  const jobData = {
    job_id: jobId,
    job_requested_by: params.user_id,
    job_home_id: params.home_id,
    job_status: 'queued',
    job_data_range: params.data_range ?? '30d',
    job_epochs: params.epochs ?? 10,
    job_knowledge_distillation: params.knowledge_distillation ?? true,
    job_training_batch_size: params.batch_size ?? 128,
    job_training_learning_rate: params.learning_rate ?? 0.001,
    job_progress_percent: 0,
    job_created_at: new Date().toISOString(),
    job_pipeline_steps_json: JSON.stringify([
      { step: 'fine_tune', status: 'pending' },
      { step: 'distillation', status: 'pending' },
      { step: 'export_onnx', status: 'pending' },
    ] as PipelineStep[]),
  };

  const { error } = await supabase
    .from('retrain_jobs_all')
    .insert(jobData);

  if (error) throw new Error(`Failed to trigger retrain: ${error.message}`);
  
  // Note: The actual Lambda trigger happens via DynamoDB Stream
  // When this record is written to DynamoDB (via dynamodbClient in useMLOps controller),
  // the RetrainJobHandler Lambda is automatically invoked by the Stream event
  
  return jobId;
}

/**
 * Cancel retrain job
 */
export async function cancelRetrainJob(jobId: string): Promise<void> {
  const { error } = await supabase
    .from('retrain_jobs_all')
    .update({
      job_status: 'cancelled',
      job_finished_at: new Date().toISOString(),
    })
    .eq('job_id', jobId);

  if (error) throw new Error(`Failed to cancel job: ${error.message}`);
}

// ============================================================================
// MODEL VERSIONS - Model lifecycle management
// ============================================================================

/**
 * Get all model versions
 */
export async function getModelVersions(limit = 50): Promise<ModelVersion[]> {
  const { data, error } = await supabase
    .from('model_versions')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch model versions: ${error.message}`);
  return (data ?? []) as ModelVersion[];
}

/**
 * Get model versions for specific model
 */
export async function getModelVersionsByModel(modelId: string, limit = 20): Promise<ModelVersion[]> {
  const { data, error } = await supabase
    .from('model_versions')
    .select('*')
    .eq('model_id', modelId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch model versions: ${error.message}`);
  return (data ?? []) as ModelVersion[];
}

/**
 * Get active/production model versions
 */
export async function getActiveModelVersions(): Promise<ModelVersion[]> {
  const { data, error } = await supabase
    .from('model_versions')
    .select('*')
    .in('status', ['active', 'production'])
    .order('created_at', { ascending: false });

  if (error) throw new Error(`Failed to fetch active versions: ${error.message}`);
  return (data ?? []) as ModelVersion[];
}

/**
 * Get latest model version for each model
 */
export async function getLatestModelVersions(): Promise<ModelVersion[]> {
  // Get all models
  const { data: models, error: modelsError } = await supabase
    .from('models')
    .select('id');

  if (modelsError) throw new Error(`Failed to fetch models: ${modelsError.message}`);

  const latestVersions: ModelVersion[] = [];

  // Get latest version for each model
  for (const model of (models ?? [])) {
    const { data, error } = await supabase
      .from('model_versions')
      .select('*')
      .eq('model_id', model.id)
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (!error && data) {
      latestVersions.push(data as ModelVersion);
    }
  }

  return latestVersions;
}

/**
 * Compare two model versions
 */
export async function compareModelVersions(
  versionId1: string,
  versionId2: string
): Promise<{ version1: ModelVersion; version2: ModelVersion }> {
  const [v1Result, v2Result] = await Promise.all([
    supabase.from('model_versions').select('*').eq('id', versionId1).single(),
    supabase.from('model_versions').select('*').eq('id', versionId2).single(),
  ]);

  if (v1Result.error) throw new Error(`Failed to fetch version 1: ${v1Result.error.message}`);
  if (v2Result.error) throw new Error(`Failed to fetch version 2: ${v2Result.error.message}`);

  return {
    version1: v1Result.data as ModelVersion,
    version2: v2Result.data as ModelVersion,
  };
}

// ============================================================================
// MODELS - Model registry
// ============================================================================

/**
 * Get all models
 */
export async function getModels(): Promise<Model[]> {
  const { data, error } = await supabase
    .from('models')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) throw new Error(`Failed to fetch models: ${error.message}`);
  return (data ?? []) as Model[];
}

/**
 * Get models by kind (cloud/edge)
 */
export async function getModelsByKind(kind: 'cloud' | 'edge'): Promise<Model[]> {
  const { data, error } = await supabase
    .from('models')
    .select('*')
    .eq('kind', kind)
    .order('created_at', { ascending: false });

  if (error) throw new Error(`Failed to fetch models: ${error.message}`);
  return (data ?? []) as Model[];
}

// ============================================================================
// PIPELINE MONITORING - Real-time progress tracking
// ============================================================================

/**
 * Get pipeline steps for a job
 */
export function getPipelineSteps(job: RetrainJob): PipelineStep[] {
  if (!job.job_pipeline_steps_json) {
    return [
      { step: 'fine_tune', status: 'pending' },
      { step: 'distillation', status: 'pending' },
      { step: 'export_onnx', status: 'pending' },
    ];
  }

  try {
    return JSON.parse(job.job_pipeline_steps_json) as PipelineStep[];
  } catch {
    return [];
  }
}

/**
 * Get current pipeline step
 */
export function getCurrentPipelineStep(job: RetrainJob): PipelineStep | null {
  const steps = getPipelineSteps(job);
  return steps.find(s => s.status === 'running') ?? null;
}

/**
 * Calculate overall pipeline progress
 */
export function calculatePipelineProgress(job: RetrainJob): number {
  if (job.job_progress_percent != null) {
    return job.job_progress_percent;
  }

  const steps = getPipelineSteps(job);
  if (steps.length === 0) return 0;

  const completedSteps = steps.filter(s => s.status === 'completed').length;
  return Math.round((completedSteps / steps.length) * 100);
}

// ============================================================================
// STATISTICS & ANALYTICS
// ============================================================================

/**
 * Get job statistics
 */
export async function getJobStatistics(): Promise<{
  total: number;
  completed: number;
  failed: number;
  running: number;
  avg_duration_minutes: number;
}> {
  const { data, error } = await supabase
    .from('retrain_jobs_all')
    .select('job_status, job_started_at, job_finished_at');

  if (error) throw new Error(`Failed to fetch job stats: ${error.message}`);

  const jobs = data ?? [];
  const completed = jobs.filter(j => j.job_status === 'completed');
  const failed = jobs.filter(j => j.job_status === 'failed');
  const running = jobs.filter(j => j.job_status === 'running' || j.job_status === 'queued');

  // Calculate average duration
  const durations = completed
    .filter(j => j.job_started_at && j.job_finished_at)
    .map(j => {
      const start = new Date(j.job_started_at!).getTime();
      const end = new Date(j.job_finished_at!).getTime();
      return (end - start) / 1000 / 60; // minutes
    });

  const avg_duration_minutes = durations.length > 0
    ? durations.reduce((a, b) => a + b, 0) / durations.length
    : 0;

  return {
    total: jobs.length,
    completed: completed.length,
    failed: failed.length,
    running: running.length,
    avg_duration_minutes: Math.round(avg_duration_minutes),
  };
}

/**
 * Get model performance trends
 */
export async function getModelPerformanceTrends(
  modelId: string,
  limit = 10
): Promise<Array<{ version: string; accuracy: number; f1_score: number; created_at: string }>> {
  const { data, error } = await supabase
    .from('model_versions')
    .select('version, accuracy, f1_score, created_at')
    .eq('model_id', modelId)
    .order('created_at', { ascending: true })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch trends: ${error.message}`);
  return (data ?? []) as Array<{ version: string; accuracy: number; f1_score: number; created_at: string }>;
}

// ============================================================================
// REAL-TIME SUBSCRIPTIONS
// ============================================================================

/**
 * Subscribe to retrain job updates
 */
export function subscribeToJobUpdates(
  jobId: string,
  callback: (job: RetrainJob) => void
): () => void {
  const subscription = supabase
    .channel(`job_${jobId}`)
    .on(
      'postgres_changes',
      {
        event: '*',
        schema: 'public',
        table: 'retrain_jobs_all',
        filter: `job_id=eq.${jobId}`,
      },
      (payload) => {
        callback(payload.new as RetrainJob);
      }
    )
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Subscribe to all job updates
 */
export function subscribeToAllJobUpdates(
  callback: (job: RetrainJob) => void
): () => void {
  const subscription = supabase
    .channel('all_jobs')
    .on(
      'postgres_changes',
      {
        event: '*',
        schema: 'public',
        table: 'retrain_jobs_all',
      },
      (payload) => {
        callback(payload.new as RetrainJob);
      }
    )
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Subscribe to new model versions
 */
export function subscribeToModelVersions(
  callback: (version: ModelVersion) => void
): () => void {
  const subscription = supabase
    .channel('model_versions')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'model_versions',
      },
      (payload) => {
        callback(payload.new as ModelVersion);
      }
    )
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}
