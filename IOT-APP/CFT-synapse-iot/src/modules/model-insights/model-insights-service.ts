/**
 * Model Insights Service
 * Handles all database operations for model-insights module
 * - Forensic flow analysis with inference logic
 * - Historical attack intelligence
 * - Human-in-the-loop feedback management
 */

import { supabase } from '../../core/lib/supabaseClient';

// ============================================================================
// TYPES
// ============================================================================

export interface ForensicFlow {
  flow_id: string;
  flow_home_id: string;
  flow_node_id: string | null;
  flow_ts: string;
  flow_protocol: string | null;
  flow_src_ip: string | null;
  flow_dst_ip: string | null;
  flow_src_port: number | null;
  flow_dst_port: number | null;
  flow_total_bytes: number | null;
  predicted_label: string | null;
  confidence: number | null;
  anomaly_score: number | null;
  is_anomaly: boolean | null;
  inference_logic: string | null;
  feedback_true_label: string | null;
  feedback_action: string | null;
  feedback_note: string | null;
  feedback_created_at: string | null;
}

export interface HistoricalAttack {
  alert_id: string;
  alert_home_id: string;
  alert_first_seen_at: string;
  alert_last_seen_at: string | null;
  alert_threat_type: string;
  alert_confidence: number | null;
  alert_severity: string;
  alert_source_ip: string | null;
  alert_target_ip: string | null;
  alert_description: string | null;
}

export interface FeedbackSubmission {
  flow_id: string;
  user_id: string;
  true_label: string;
  action: 'analyst_correction' | 'false_positive' | 'confirmed_attack';
  note?: string;
}

export interface InferenceResult {
  flow_id: string;
  flow_home_id: string;
  flow_node_id?: string;
  flow_ts: string;
  flow_protocol?: string;
  flow_src_ip?: string;
  flow_dst_ip?: string;
  flow_src_port?: number;
  flow_dst_port?: number;
  flow_total_bytes?: number;
  predicted_label: string;
  confidence: number;
  anomaly_score: number;
  is_anomaly: boolean;
  inference_logic: string;
}

// ============================================================================
// FORENSIC FLOWS - Real-time inference results with AI reasoning
// ============================================================================

/**
 * Get recent forensic flows with inference details
 * Used for live analysis view showing AI decision-making process
 */
export async function getForensicFlows(limit = 200): Promise<ForensicFlow[]> {
  const { data, error } = await supabase
    .from('network_flows_feedback_all')
    .select('*')
    .order('flow_ts', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch forensic flows: ${error.message}`);
  return (data ?? []) as ForensicFlow[];
}

/**
 * Get forensic flows filtered by anomaly status
 */
export async function getForensicFlowsByType(
  type: 'all' | 'anomaly' | 'benign',
  limit = 200
): Promise<ForensicFlow[]> {
  let query = supabase
    .from('network_flows_feedback_all')
    .select('*')
    .order('flow_ts', { ascending: false })
    .limit(limit);

  if (type === 'anomaly') {
    query = query.eq('is_anomaly', true);
  } else if (type === 'benign') {
    query = query.eq('is_anomaly', false);
  }

  const { data, error } = await query;
  if (error) throw new Error(`Failed to fetch flows by type: ${error.message}`);
  return (data ?? []) as ForensicFlow[];
}

/**
 * Get single flow details for forensic analysis
 */
export async function getFlowDetails(flowId: string): Promise<ForensicFlow | null> {
  const { data, error } = await supabase
    .from('network_flows_feedback_all')
    .select('*')
    .eq('flow_id', flowId)
    .single();

  if (error) {
    if (error.code === 'PGRST116') return null; // Not found
    throw new Error(`Failed to fetch flow details: ${error.message}`);
  }
  return data as ForensicFlow;
}

/**
 * Record inference result from edge/cloud model
 * Called by Lambda after model inference
 */
export async function recordInferenceResult(result: InferenceResult): Promise<void> {
  const { error } = await supabase
    .from('network_flows_feedback_all')
    .upsert({
      flow_id: result.flow_id,
      flow_home_id: result.flow_home_id,
      flow_node_id: result.flow_node_id ?? null,
      flow_ts: result.flow_ts,
      flow_protocol: result.flow_protocol ?? null,
      flow_src_ip: result.flow_src_ip ?? null,
      flow_dst_ip: result.flow_dst_ip ?? null,
      flow_src_port: result.flow_src_port ?? null,
      flow_dst_port: result.flow_dst_port ?? null,
      flow_total_bytes: result.flow_total_bytes ?? null,
      flow_created_at: new Date().toISOString(),
      predicted_label: result.predicted_label,
      confidence: result.confidence,
      anomaly_score: result.anomaly_score,
      is_anomaly: result.is_anomaly,
      inference_logic: result.inference_logic,
    }, {
      onConflict: 'flow_id'
    });

  if (error) throw new Error(`Failed to record inference: ${error.message}`);
}

// ============================================================================
// HISTORICAL ATTACKS - Attack intelligence from alerts
// ============================================================================

/**
 * Get historical attack records
 * Used for historical view showing past attack patterns
 */
export async function getHistoricalAttacks(limit = 20): Promise<HistoricalAttack[]> {
  const { data, error } = await supabase
    .from('alerts_all')
    .select('alert_id, alert_home_id, alert_first_seen_at, alert_last_seen_at, alert_threat_type, alert_confidence, alert_severity, alert_source_ip, alert_target_ip, alert_description')
    .order('alert_first_seen_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch historical attacks: ${error.message}`);
  return (data ?? []) as HistoricalAttack[];
}

/**
 * Get attacks filtered by severity
 */
export async function getAttacksBySeverity(
  severity: 'critical' | 'high' | 'medium' | 'low',
  limit = 20
): Promise<HistoricalAttack[]> {
  const { data, error } = await supabase
    .from('alerts_all')
    .select('*')
    .eq('alert_severity', severity)
    .order('alert_first_seen_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch attacks by severity: ${error.message}`);
  return (data ?? []) as HistoricalAttack[];
}

/**
 * Get attacks by threat type
 */
export async function getAttacksByType(
  threatType: string,
  limit = 20
): Promise<HistoricalAttack[]> {
  const { data, error } = await supabase
    .from('alerts_all')
    .select('*')
    .eq('alert_threat_type', threatType)
    .order('alert_first_seen_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch attacks by type: ${error.message}`);
  return (data ?? []) as HistoricalAttack[];
}

// ============================================================================
// HUMAN-IN-THE-LOOP FEEDBACK - Analyst corrections for model improvement
// ============================================================================

/**
 * Get all feedback submissions (analyst corrections)
 * Used for training queue and pending knowledge updates
 */
export async function getFeedbackSubmissions(limit = 50): Promise<ForensicFlow[]> {
  const { data, error } = await supabase
    .from('network_flows_feedback_all')
    .select('*')
    .not('feedback_true_label', 'is', null)
    .order('feedback_created_at', { ascending: false })
    .limit(limit);

  if (error) throw new Error(`Failed to fetch feedback: ${error.message}`);
  return (data ?? []) as ForensicFlow[];
}

/**
 * Submit analyst feedback/correction for a flow
 * Used when analyst corrects model prediction
 */
export async function submitFeedback(feedback: FeedbackSubmission): Promise<void> {
  const { error } = await supabase
    .from('network_flows_feedback_all')
    .update({
      feedback_user_id: feedback.user_id,
      feedback_true_label: feedback.true_label,
      feedback_action: feedback.action,
      feedback_note: feedback.note ?? null,
      feedback_created_at: new Date().toISOString(),
    })
    .eq('flow_id', feedback.flow_id);

  if (error) throw new Error(`Failed to submit feedback: ${error.message}`);
}

/**
 * Batch submit multiple feedback corrections
 * Used when analyst reviews multiple flows at once
 */
export async function submitBatchFeedback(feedbacks: FeedbackSubmission[]): Promise<void> {
  const updates = feedbacks.map(f => ({
    flow_id: f.flow_id,
    feedback_user_id: f.user_id,
    feedback_true_label: f.true_label,
    feedback_action: f.action,
    feedback_note: f.note ?? null,
    feedback_created_at: new Date().toISOString(),
  }));

  const { error } = await supabase
    .from('network_flows_feedback_all')
    .upsert(updates, { onConflict: 'flow_id' });

  if (error) throw new Error(`Failed to submit batch feedback: ${error.message}`);
}

/**
 * Get feedback statistics (for training queue display)
 */
export async function getFeedbackStats(): Promise<{
  total: number;
  by_label: Record<string, number>;
  by_action: Record<string, number>;
}> {
  const { data, error } = await supabase
    .from('network_flows_feedback_all')
    .select('feedback_true_label, feedback_action')
    .not('feedback_true_label', 'is', null);

  if (error) throw new Error(`Failed to fetch feedback stats: ${error.message}`);

  const by_label: Record<string, number> = {};
  const by_action: Record<string, number> = {};

  (data ?? []).forEach(row => {
    if (row.feedback_true_label) {
      by_label[row.feedback_true_label] = (by_label[row.feedback_true_label] || 0) + 1;
    }
    if (row.feedback_action) {
      by_action[row.feedback_action] = (by_action[row.feedback_action] || 0) + 1;
    }
  });

  return {
    total: data?.length ?? 0,
    by_label,
    by_action,
  };
}

// ============================================================================
// REAL-TIME SUBSCRIPTIONS
// ============================================================================

/**
 * Subscribe to new inference results in real-time
 * Used for live dashboard updates
 */
export function subscribeToInferenceResults(
  callback: (flow: ForensicFlow) => void
): () => void {
  const subscription = supabase
    .channel('inference_results')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'network_flows_feedback_all',
      },
      (payload) => {
        callback(payload.new as ForensicFlow);
      }
    )
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Subscribe to new attacks in real-time
 */
export function subscribeToNewAttacks(
  callback: (attack: HistoricalAttack) => void
): () => void {
  const subscription = supabase
    .channel('new_attacks')
    .on(
      'postgres_changes',
      {
        event: 'INSERT',
        schema: 'public',
        table: 'alerts_all',
      },
      (payload) => {
        callback(payload.new as HistoricalAttack);
      }
    )
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}

/**
 * Subscribe to feedback updates
 */
export function subscribeToFeedbackUpdates(
  callback: (flow: ForensicFlow) => void
): () => void {
  const subscription = supabase
    .channel('feedback_updates')
    .on(
      'postgres_changes',
      {
        event: 'UPDATE',
        schema: 'public',
        table: 'network_flows_feedback_all',
        filter: 'feedback_true_label=not.is.null',
      },
      (payload) => {
        callback(payload.new as ForensicFlow);
      }
    )
    .subscribe();

  return () => {
    subscription.unsubscribe();
  };
}
