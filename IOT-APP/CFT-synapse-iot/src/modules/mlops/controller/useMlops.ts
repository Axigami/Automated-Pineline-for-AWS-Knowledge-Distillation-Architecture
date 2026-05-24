import { useEffect, useCallback, useRef } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';
import { dynamodbClient } from '../../../core/lib/dynamodbClient';
import { useAuthContext } from '../../../core/auth/AuthProvider';
import { useMlopsStore } from '../model/store';
import {
  getMlopsCacheForUser,
  isCacheValid,
  writeCache,
  invalidateCache,
} from '../model/cache';
import type { ModelVersionRow, RetrainJobRow, RetrainConfig, OtaDeployRequest } from '../model/types';

const POLL_INTERVAL = 5000; // 5 seconds (increased from 3s for better performance)

/**
 * useMlops – Controller with caching.
 * model_versions: latest metrics.
 * retrain_jobs_all: trigger + progress polling.
 * deployments_all: OTA deploy.
 */
export function useMlops() {
  const { user } = useAuthContext();
  const userId = user?.id ?? null;
  const userIdRef = useRef(userId);
  userIdRef.current = userId;

  const store = useMlopsStore();
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const syncFromCache = useCallback(
    (uid: string) => {
      const c = getMlopsCacheForUser(uid);
      store.setMetrics(c.metrics);
      store.setActiveJob(c.activeJob);
      if (c.activeJob) {
        store.setActiveTaskId((c.activeJob as any).job_id);
      }
    },
    [store],
  );

  const fetchMetrics = useCallback(async () => {
    const { data } = await supabase
      .from('model_versions')
      .select('id, model_id, version, status, accuracy, f1_score, precision, recall, latency_ms, memory_mb, false_positive_rate, throughput_per_s, artifact_uri, created_at, author')
      .eq('status', 'production')
      .order('created_at', { ascending: false })
      .limit(10);
    if (data) {
      store.setMetrics(data as ModelVersionRow[]);
      // Update cache
      const uid = userIdRef.current;
      if (uid) {
        writeCache(uid, { metrics: data as ModelVersionRow[] });
      }
    }
  }, [store]);

  const fetchActiveJob = useCallback(async () => {
    const res = await supabase
      .from('retrain_jobs_all' as any)
      .select('job_id, job_status, job_progress_percent, job_epochs, job_started_at, job_finished_at, job_created_at, event_step, event_message, event_progress_percent, job_knowledge_distillation, job_data_range')
      .in('job_status', ['queued', 'running'])
      .order('job_created_at', { ascending: false })
      .limit(1)
      .maybeSingle();
    const data = res.data as RetrainJobRow | null;
    if (data) {
      store.setActiveJob(data);
      store.setActiveTaskId((data as any).job_id);
    } else {
      store.setActiveJob(null);
      store.setActiveTaskId(null);
    }
    // Update cache
    const uid = userIdRef.current;
    if (uid) {
      writeCache(uid, { activeJob: data });
    }
  }, [store]);

  const triggerRetrain = useCallback(async (config: RetrainConfig) => {
    const jobId = crypto.randomUUID();
    const now = new Date().toISOString();
    
    // Security: Get user info for audit trail
    const currentUserId = userIdRef.current;
    if (!currentUserId) {
      store.setError('Authentication required to trigger retraining');
      return;
    }
    
    const jobData = {
      job_id: jobId,
      job_home_id: config.homeId,
      job_status: 'queued' as const,
      job_data_range: `${config.dataDays}d`,
      job_epochs: config.epochs ?? 10,
      job_created_at: now,
    };
    
    // 1. Write to Supabase (for frontend to read)
    const res = await supabase
      .from('retrain_jobs_all' as any)
      .insert(jobData as any)
      .select('job_id')
      .single();
    
    const error = (res as any).error;
    const data = (res as any).data as { job_id: string } | null;
    
    if (error) {
      store.setError('Failed to start Retraining: ' + error.message);
      return;
    }
    
    // 2. Write to DynamoDB (for Lambda to process via Stream)
    // Security: Set auth token and include user info
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.access_token) {
        dynamodbClient.setAuthToken(session.access_token);
      }
      
      await dynamodbClient.writeRetrainJob({
        job_id: jobId,
        job_requested_by: currentUserId,
        job_home_id: config.homeId,
        job_status: 'queued',
        job_data_range: `${config.dataDays}d`,
        job_epochs: config.epochs ?? 10,
        job_created_at: now,
        job_progress_percent: 0,
        job_training_batch_size: config.batchSize,
        job_training_learning_rate: config.learningRate,
      });
    } catch (err) {
      console.warn('Failed to write to DynamoDB (non-critical):', err);
      // Don't show error to user - Supabase write succeeded
    }
    
    if (data) {
      store.setActiveTaskId(data.job_id);
      // Invalidate cache to force refresh
      const uid = userIdRef.current;
      if (uid) invalidateCache(uid);
    }
  }, [store]);

  const deployModel = useCallback(async (req: OtaDeployRequest) => {
    const now = new Date().toISOString();
    
    // Security: Get user info for audit trail
    const currentUserId = userIdRef.current;
    if (!currentUserId) {
      store.setError('Authentication required to deploy model');
      return;
    }
    
    // Security: Get user email for audit trail
    const { data: { user: currentUser } } = await supabase.auth.getUser();
    const userEmail = currentUser?.email || 'unknown';
    
    const rows = req.targetNodeIds.map((nodeId) => ({
      deployment_id: crypto.randomUUID(),
      deployment_model_version_id: req.modelVersionId,
      deployment_status: 'pending',
      target_node_id: nodeId,
      deployment_created_at: now,
      deployment_requested_by: currentUserId, // Audit trail
    }));
    
    // 1. Write to Supabase (for frontend to read)
    const { error } = await supabase.from('deployments_all' as any).insert(rows as any);
    if (error) {
      store.setError('OTA Deploy failed: ' + error.message);
      return;
    }
    
    // 2. Write to DynamoDB (for Lambda to process via Stream)
    // Security: Set auth token and include user info
    // NOTE: This is optional - deployment works without DynamoDB if Lambda is not set up
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.access_token) {
        dynamodbClient.setAuthToken(session.access_token);
      }
      
      await dynamodbClient.writeDeployments(
        rows.map(row => ({
          deployment_id: row.deployment_id,
          deployment_requested_by: currentUserId,
          deployment_model_version_id: row.deployment_model_version_id,
          deployment_status: row.deployment_status as 'pending',
          target_node_id: row.target_node_id,
          deployment_created_at: row.deployment_created_at,
        }))
      );
      
      console.log(`[MLOps Security] User ${userEmail} (${currentUserId}) deployed model ${req.modelVersionId} to ${req.targetNodeIds.length} nodes`);
      console.log('[MLOps] DynamoDB write successful - Lambda will process deployment');
    } catch (err) {
      console.warn('[MLOps] Failed to write to DynamoDB (non-critical):', err);
      console.warn('[MLOps] Deployment recorded in Supabase but Lambda will not be triggered automatically');
      console.warn('[MLOps] You may need to manually trigger deployment or fix API Gateway CORS settings');
      // Don't show error to user - Supabase write succeeded and deployment is recorded
    }
  }, [store]);

  const clearDeploymentHistory = useCallback(async () => {
    try {
      // Security: Get user info for audit trail
      const currentUserId = userIdRef.current;
      if (!currentUserId) {
        store.setError('Authentication required to clear deployment history');
        return { ok: false, error: 'Authentication required' };
      }
      
      // Security: Get user email for audit trail
      const { data: { user: currentUser } } = await supabase.auth.getUser();
      const userEmail = currentUser?.email || 'unknown';
      
      // Count deployments before deleting
      const { count: deploymentCount } = await supabase
        .from('deployments_all' as any)
        .select('*', { count: 'exact', head: true })
        .in('deployment_status', ['completed', 'failed', 'cancelled']);
      
      if (deploymentCount === 0) {
        store.setError('No completed/failed/cancelled deployments to delete');
        return { ok: false, error: 'No deployments to delete' };
      }
      
      // Delete all deployments with status 'completed', 'failed', or 'cancelled'
      const { error } = await supabase
        .from('deployments_all' as any)
        .delete()
        .in('deployment_status', ['completed', 'failed', 'cancelled']);
      
      if (error) {
        store.setError('Failed to clear deployment history: ' + error.message);
        return { ok: false, error: error.message };
      }
      
      console.log(`[MLOps Security] User ${userEmail} (${currentUserId}) cleared ${deploymentCount} deployment record(s)`);
      return { ok: true, deletedCount: deploymentCount };
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      store.setError('Failed to clear deployment history: ' + msg);
      return { ok: false, error: msg };
    }
  }, [store]);

  // Polling active job progress (FR 4.2)
  useEffect(() => {
    if (!store.activeTaskId) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    const taskId = store.activeTaskId;
    const poll = async () => {
      const res = await supabase
        .from('retrain_jobs_all' as any)
        .select('job_id, job_status, job_progress_percent, event_step, event_message, event_progress_percent, job_finished_at')
        .eq('job_id', taskId)
        .maybeSingle();
      const data = res.data as RetrainJobRow | null;
      if (data) {
        store.setActiveJob(data);
        const status = (data as any).job_status as string | null;
        if (status === 'completed' || status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current);
          store.setActiveTaskId(null);
          fetchMetrics();
        }
      }
    };
    pollRef.current = setInterval(poll, POLL_INTERVAL);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [store.activeTaskId, store, fetchMetrics]);

  useEffect(() => {
    if (!userId) {
      store.setIsLoading(false);
      return;
    }

    // Check cache first
    if (isCacheValid(userId)) {
      syncFromCache(userId);
      store.setIsLoading(false);
    } else {
      store.setIsLoading(true);
      Promise.allSettled([fetchMetrics(), fetchActiveJob()])
        .finally(() => store.setIsLoading(false));
    }
  }, [userId, fetchMetrics, fetchActiveJob, store, syncFromCache]);

  return { ...store, triggerRetrain, deployModel, clearDeploymentHistory, refreshMetrics: fetchMetrics };
}
