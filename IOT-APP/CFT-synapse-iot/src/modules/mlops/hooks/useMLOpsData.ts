import { useEffect, useState, useCallback } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';

const POLL_FAST_MS = 3000;

/** Trạng thái “đang phục vụ” production — đồng bộ với pipeline/backend thường dùng. */
const ACTIVE_STATUSES = new Set(['production', 'deployed', 'active']);

export interface VersionLog {
  id: string;
  modelId: string;
  version: string;
  timestamp: string;
  accuracy: number;
  status: string;
  author: string;
  f1_score: number;
  precision: number;
  recall: number;
  latency_ms: number;
  memory_mb: number;
  false_positive_rate: number;
  throughput_per_s: number;
  metricsJson: string | null;
}

export interface RadarPoint {
  subject: string;
  v_old: number;
  v_new: number;
  fullMark: number;
}

export interface BenchmarkRow {
  name: string;
  accuracy: number;
  latency: number;
  memory: number;
}

export interface PipelineStep {
  id: number;
  name: string;
  status: 'completed' | 'active' | 'pending';
}

export interface ActiveJobInfo {
  jobId: string | null;
  jobStatus: string | null;
  progressPercent: number;
  eventStep: string | null;
  eventMessage: string | null;
}

export interface MLOpsData {
  versionHistory: VersionLog[];
  radarData: RadarPoint[];
  benchmarkData: BenchmarkRow[];
  pipelineSteps: PipelineStep[];
  activeVersion: string;
  /** 0–100: từ metrics_json (distillation) hoặc proxy F1 */
  fidelityScore: number;
  fidelityIsProxy: boolean;
  activeJob: ActiveJobInfo;
  triggerRetrain: (params: { batchSize: number; learningRate: number }) => Promise<void>;
  refreshData: () => void;
  promoteVersionToProduction: (versionId: string, modelId: string) => Promise<{ ok: boolean; error?: string }>;
  clearHistory: () => Promise<{ ok: boolean; error?: string }>;
  loading: boolean;
  error: string | null;
}

function toPercent(val: number | null, max: number): number {
  if (val == null) return 0;
  return Math.round((val / max) * 100);
}

function pickActiveVersionLabel(rows: VersionLog[]): string {
  const prod = rows.find((v) => ACTIVE_STATUSES.has((v.status || '').toLowerCase()));
  if (prod) return prod.version;
  return rows[0]?.version ?? '—';
}

/** Đọc điểm fidelity/distillation từ metrics_json nếu có. */
function extractFidelityFromMetrics(metricsJson: string | null, f1Fallback: number | null): { score: number; isProxy: boolean } {
  if (!metricsJson || !metricsJson.trim()) {
    const f1 = f1Fallback ?? 0;
    return { score: Math.round(f1 * 100), isProxy: true };
  }
  try {
    const m = JSON.parse(metricsJson) as Record<string, unknown>;
    const keys = ['knowledge_fidelity', 'distillation_fidelity', 'fidelity', 'teacher_student_agreement'];
    for (const k of keys) {
      const v = m[k];
      if (typeof v === 'number' && !Number.isNaN(v)) {
        let pct = v;
        if (pct <= 1 && pct >= 0) pct = v * 100;
        return { score: Math.round(Math.min(100, Math.max(0, pct))), isProxy: false };
      }
    }
  } catch {
    /* ignore */
  }
  const f1 = f1Fallback ?? 0;
  return { score: Math.round(f1 * 100), isProxy: true };
}

interface DbPipelineStep {
  step?: string;
  name?: string;
  status?: string;
}

function mapPipelineStatus(s: string | undefined): 'completed' | 'active' | 'pending' {
  const u = (s || '').toLowerCase();
  if (u === 'completed' || u === 'success' || u === 'done') return 'completed';
  if (u === 'running' || u === 'active') return 'active';
  if (u === 'failed') return 'pending';
  return 'pending';
}

function buildPipelineStepsFromJob(
  jobPipelineJson: string | null,
  jobStatus: string | null,
  eventStep: string | null,
  progressPercent: number
): PipelineStep[] {
  let raw: DbPipelineStep[] = [];
  if (jobPipelineJson) {
    try {
      const parsed = JSON.parse(jobPipelineJson);
      if (Array.isArray(parsed)) raw = parsed;
    } catch {
      raw = [];
    }
  }

  if (raw.length === 0) {
    return [
      { id: 1, name: 'Data Preparation', status: 'pending' },
      { id: 2, name: 'Teacher Training', status: 'pending' },
      { id: 3, name: 'Distillation', status: 'pending' },
      { id: 4, name: 'ONNX Quantization', status: 'pending' },
      { id: 5, name: 'Edge Push', status: 'pending' },
    ];
  }

  const js = (jobStatus || '').toLowerCase();
  const steps: PipelineStep[] = raw.map((item, i) => {
    const name = (item.name || item.step || `step_${i + 1}`) as string;
    let st = mapPipelineStatus(item.status);
    return { id: i + 1, name, status: st };
  });

  if (js === 'completed') {
    return steps.map((s) => ({ ...s, status: 'completed' as const }));
  }
  if (js === 'failed' || js === 'cancelled') {
    return steps.map((s) => ({ ...s, status: 'pending' as const }));
  }

  if (eventStep) {
    const norm = eventStep.toLowerCase();
    let seenActive = false;
    return steps.map((s) => {
      const sn = s.name.toLowerCase();
      if (sn.includes(norm) || norm.includes(sn)) {
        seenActive = true;
        return { ...s, status: 'active' };
      }
      if (!seenActive) return { ...s, status: 'completed' };
      return { ...s, status: 'pending' };
    });
  }

  if (js === 'running' || js === 'queued') {
    const idx = Math.min(
      steps.length - 1,
      Math.max(0, Math.floor((progressPercent / 100) * steps.length))
    );
    return steps.map((s, i) => ({
      ...s,
      status: i < idx ? 'completed' : i === idx ? 'active' : 'pending',
    }));
  }

  return steps;
}

export function useMLOpsData(): MLOpsData {
  const [data, setData] = useState<MLOpsData>({
    versionHistory: [],
    radarData: [],
    benchmarkData: [],
    pipelineSteps: [],
    activeVersion: '—',
    fidelityScore: 0,
    fidelityIsProxy: true,
    activeJob: {
      jobId: null,
      jobStatus: null,
      progressPercent: 0,
      eventStep: null,
      eventMessage: null,
    },
    triggerRetrain: async () => {},
    refreshData: () => {},
    promoteVersionToProduction: async () => ({ ok: false }),
    clearHistory: async () => ({ ok: false }),
    loading: true,
    error: null,
  });

  const [refreshKey, setRefreshKey] = useState(0);
  const refreshData = useCallback(() => setRefreshKey((k) => k + 1), []);

  const promoteVersionToProduction = useCallback(async (versionId: string, modelId: string) => {
    try {
      const { error: e1 } = await supabase
        .from('model_versions')
        .update({ status: 'archived' as any })
        .eq('model_id', modelId)
        .neq('id', versionId);
      if (e1) return { ok: false, error: e1.message };

      const { error: e2 } = await supabase
        .from('model_versions')
        .update({ status: 'production' as any })
        .eq('id', versionId);
      if (e2) return { ok: false, error: e2.message };

      refreshData();
      return { ok: true };
    } catch (e: unknown) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  }, [refreshData]);

  const clearHistory = useCallback(async () => {
    try {
      // Strategy: Keep only the latest 5 model versions, delete the rest
      const KEEP_LATEST = 5;
      
      // Get all model versions ordered by created_at
      const { data: allVersions, error: fetchError } = await supabase
        .from('model_versions')
        .select('id, version, created_at, status')
        .order('created_at', { ascending: false });
      
      if (fetchError) return { ok: false, error: fetchError.message };
      
      if (!allVersions || allVersions.length <= KEEP_LATEST) {
        return { ok: false, error: `Only ${allVersions?.length ?? 0} version(s) found. Keeping all (minimum ${KEEP_LATEST} kept).` };
      }
      
      // Get IDs of versions to delete (all except the latest KEEP_LATEST)
      const versionsToDelete = allVersions.slice(KEEP_LATEST);
      const idsToDelete = versionsToDelete.map(v => v.id);
      
      if (idsToDelete.length === 0) {
        return { ok: false, error: 'No old versions to delete' };
      }
      
      // Delete old versions
      const { error: deleteError } = await supabase
        .from('model_versions')
        .delete()
        .in('id', idsToDelete);
      
      if (deleteError) return { ok: false, error: deleteError.message };

      refreshData();
      return { ok: true, deletedCount: idsToDelete.length };
    } catch (e: unknown) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  }, [refreshData]);

  const triggerRetrain = useCallback(
    async (params: { batchSize: number; learningRate: number }) => {
      const { lambdaClient } = await import('../../../core/lib/lambdaClient');
      await lambdaClient.triggerFineTuning({
        triggered_by: 'mlops_ui',
        hyperparameters: {
          batch_size: params.batchSize,
          learning_rate: params.learningRate,
        },
      });
      refreshData();
    },
    [refreshData]
  );

  useEffect(() => {
    let cancelled = false;

    async function fetchAll() {
      try {
        console.log('[MLOps] Fetching model versions...');
        const { data: versions, error: vErr } = await supabase
          .from('model_versions')
          .select(
            'id, model_id, version, created_at, accuracy, status, author, f1_score, precision, recall, latency_ms, memory_mb, false_positive_rate, throughput_per_s, metrics_json'
          )
          .order('created_at', { ascending: false })
          .limit(10);

        if (vErr) {
          console.error('[MLOps] Error fetching model versions:', vErr);
          throw new Error(vErr.message);
        }
        if (cancelled) return;
        
        console.log('[MLOps] Fetched versions:', versions?.length ?? 0);

        const versionHistory: VersionLog[] = (versions ?? []).map((v: any) => ({
          id: v.id,
          modelId: v.model_id,
          version: v.version ?? '—',
          timestamp: v.created_at ? new Date(v.created_at).toLocaleString('sv-SE').slice(0, 16) : '—',
          accuracy: v.accuracy ?? 0,
          status: v.status ?? 'archived',
          author: v.author ?? 'system',
          f1_score: v.f1_score ?? 0,
          precision: v.precision ?? 0,
          recall: v.recall ?? 0,
          latency_ms: v.latency_ms ?? 0,
          memory_mb: v.memory_mb ?? 0,
          false_positive_rate: v.false_positive_rate ?? 0,
          throughput_per_s: v.throughput_per_s ?? 0,
          metricsJson: v.metrics_json ?? null,
        }));

        const activeVersion = pickActiveVersionLabel(versionHistory);

        const newest = versionHistory[0];
        const fid = extractFidelityFromMetrics(newest?.metricsJson ?? null, newest?.f1_score ?? null);

        const v_new = versionHistory[0];
        const v_old = versionHistory[1];

        const radarData: RadarPoint[] = v_new
          ? [
              { subject: 'F1-Score', v_old: toPercent(v_old?.f1_score ?? null, 1), v_new: toPercent(v_new.f1_score, 1), fullMark: 100 },
              { subject: 'Precision', v_old: toPercent(v_old?.precision ?? null, 1), v_new: toPercent(v_new.precision, 1), fullMark: 100 },
              { subject: 'Recall', v_old: toPercent(v_old?.recall ?? null, 1), v_new: toPercent(v_new.recall, 1), fullMark: 100 },
              {
                subject: 'Latency',
                v_old: v_old ? Math.max(0, 100 - (v_old.latency_ms ?? 0)) : 0,
                v_new: Math.max(0, 100 - (v_new.latency_ms ?? 0)),
                fullMark: 100,
              },
              {
                subject: 'Memory',
                v_old: v_old ? Math.max(0, 100 - Math.round((v_old.memory_mb ?? 0) / 10)) : 0,
                v_new: Math.max(0, 100 - Math.round((v_new.memory_mb ?? 0) / 10)),
                fullMark: 100,
              },
              {
                subject: 'FPR',
                v_old: v_old ? toPercent(1 - (v_old.false_positive_rate ?? 0), 1) : 0,
                v_new: toPercent(1 - (v_new.false_positive_rate ?? 0), 1),
                fullMark: 100,
              },
            ]
          : [];

        const { data: models, error: mErr } = await supabase.from('models').select('id, name, kind');
        if (mErr) {
          console.error('[MLOps] Error fetching models:', mErr);
          throw new Error(mErr.message);
        }
        
        console.log('[MLOps] Fetched models:', models?.length ?? 0);

        const benchmarkData: BenchmarkRow[] = [];
        for (const model of models ?? []) {
          const { data: mv } = await supabase
            .from('model_versions')
            .select('accuracy, latency_ms, memory_mb')
            .eq('model_id', model.id)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();
          if (mv) {
            benchmarkData.push({
              name: (model as { name?: string; kind?: string }).name ?? (model as any).kind,
              accuracy: Math.round(((mv as any).accuracy ?? 0) * 100 * 10) / 10,
              latency: (mv as any).latency_ms ?? 0,
              memory: (mv as any).memory_mb ?? 0,
            });
          }
        }

        const { data: job } = await supabase
          .from('retrain_jobs_all')
          .select(
            'job_id, job_pipeline_steps_json, job_status, job_progress_percent, event_step, event_message'
          )
          .order('job_created_at', { ascending: false })
          .limit(1)
          .maybeSingle();

        const jStatus = (job as any)?.job_status ?? null;
        const progressPercent = Math.min(100, Math.max(0, (job as any)?.job_progress_percent ?? 0));
        const pipelineSteps = buildPipelineStepsFromJob(
          (job as any)?.job_pipeline_steps_json ?? null,
          jStatus,
          (job as any)?.event_step ?? null,
          progressPercent
        );

        const activeJob: ActiveJobInfo = {
          jobId: (job as any)?.job_id ?? null,
          jobStatus: jStatus,
          progressPercent,
          eventStep: (job as any)?.event_step ?? null,
          eventMessage: (job as any)?.event_message ?? null,
        };

        if (!cancelled) {
          setData({
            versionHistory,
            radarData,
            benchmarkData,
            pipelineSteps,
            activeVersion,
            fidelityScore: fid.score,
            fidelityIsProxy: fid.isProxy,
            activeJob,
            triggerRetrain,
            refreshData,
            promoteVersionToProduction,
            clearHistory,
            loading: false,
            error: null,
          });
        }
      } catch (e: unknown) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          console.error('[MLOps] Fatal error:', msg, e);
          setData((prev) => ({
            ...prev,
            loading: false,
            error: msg,
            triggerRetrain,
            refreshData,
            promoteVersionToProduction,
            clearHistory,
          }));
        }
      }
    }

    fetchAll();
    const interval = window.setInterval(fetchAll, POLL_FAST_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [refreshKey, triggerRetrain, refreshData, promoteVersionToProduction, clearHistory]);

  return data;
}
