import { useEffect, useState } from 'react';
import { supabase } from '../../../core/lib/supabaseClient';

export interface ForensicFlow {
  flow_id: string;
  flow_ts: string;
  flow_protocol: string | null;
  flow_src_ip: string | null;
  flow_dst_ip: string | null;
  flow_src_port: number | null;
  flow_dst_port: number | null;
  flow_total_bytes: number | null;
  flow_in_bytes: number | null;      // dst2src bytes
  flow_out_bytes: number | null;     // src2dst bytes
  flow_duration_s: number | null;    // duration in seconds
  anomaly_score: number | null;
  inference_logic: string | null;
  predicted_label: string | null;
  feedback_true_label: string | null;
  is_anomaly: boolean | null;
}

export interface HistoricalAttack {
  alert_id: string;
  alert_first_seen_at: string;
  alert_threat_type: string;
  alert_confidence: number | null;
  alert_severity: string;
  alert_source_ip: string | null;
  alert_target_ip: string | null;
}

export interface FeedbackRow {
  flow_id: string;
  predicted_label: string | null;
  feedback_true_label: string | null;
  feedback_action: string | null;
  feedback_note: string | null;
  feedback_created_at: string | null;
}

export interface TrainingQueueStatus {
  pending_count: number;
  relabeled_count: number;
  used_count: number;
  total_count: number;
  by_label: Record<string, number>;
  ready_for_training: boolean;
}

export interface ModelInsightsData {
  forensicFlows: ForensicFlow[];
  historicalAttacks: HistoricalAttack[];
  feedbackRows: FeedbackRow[];
  trainingQueueStatus: TrainingQueueStatus | null;
  loading: boolean;
  error: string | null;
}

export function useModelInsightsData(): ModelInsightsData {
  const [data, setData] = useState<ModelInsightsData>({
    forensicFlows: [],
    historicalAttacks: [],
    feedbackRows: [],
    trainingQueueStatus: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    async function fetch() {
      try {
        // Fetch Supabase data with optimized limits
        // PERFORMANCE: Reduced from 200 to 100 flows for faster load
        const [flowsRes, attacksRes, feedbackRes] = await Promise.all([
          supabase
            .from('network_flows_feedback_all')
            .select('flow_id, flow_ts, flow_protocol, flow_src_ip, flow_dst_ip, flow_src_port, flow_dst_port, flow_total_bytes, flow_in_bytes, flow_out_bytes, flow_duration_s, anomaly_score, inference_logic, predicted_label, feedback_true_label, is_anomaly')
            .order('flow_ts', { ascending: false })
            .limit(100),  // Reduced from 200 for faster load

          supabase
            .from('alerts_all')
            .select('alert_id, alert_first_seen_at, alert_threat_type, alert_confidence, alert_severity, alert_source_ip, alert_target_ip')
            .order('alert_first_seen_at', { ascending: false })
            .limit(20),

          supabase
            .from('network_flows_feedback_all')
            .select('flow_id, predicted_label, feedback_true_label, feedback_action, feedback_note, feedback_created_at')
            .not('feedback_true_label', 'is', null)
            .order('feedback_created_at', { ascending: false })
            .limit(30),
        ]);

        if (flowsRes.error) throw new Error(flowsRes.error.message);
        if (attacksRes.error) throw new Error(attacksRes.error.message);
        if (feedbackRes.error) throw new Error(feedbackRes.error.message);
        if (cancelled) return;

        // Fetch training queue status from DynamoDB via Lambda
        let trainingQueueStatus: TrainingQueueStatus | null = null;
        try {
          const { lambdaClient } = await import('../../../core/lib/lambdaClient');
          trainingQueueStatus = await lambdaClient.getTrainingQueueStatus();
        } catch (queueError) {
          console.warn('Failed to fetch training queue status:', queueError);
          // Don't fail the entire load if queue status fails
        }

        setData({
          forensicFlows: (flowsRes.data ?? []) as ForensicFlow[],
          historicalAttacks: (attacksRes.data ?? []) as HistoricalAttack[],
          feedbackRows: (feedbackRes.data ?? []) as FeedbackRow[],
          trainingQueueStatus,
          loading: false,
          error: null,
        });
      } catch (e: unknown) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setData((prev) => ({ ...prev, loading: false, error: msg }));
        }
      }
    }
    fetch();
    return () => { cancelled = true; };
  }, []);

  return data;
}
