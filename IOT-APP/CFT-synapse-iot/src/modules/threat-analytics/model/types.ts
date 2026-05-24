/**
 * Threat Analytics – Types aligned to network_flows_feedback_all (full schema)
 */

// ─── DB Row Type ──────────────────────────────────────────────────────────────
export interface FlowRow {
  flow_id: string;
  flow_home_id: string;
  flow_node_id: string | null;
  flow_ts: string;
  flow_protocol: string | null;
  flow_src_ip: string | null;
  flow_dst_ip: string | null;
  flow_src_port: number | null;
  flow_dst_port: number | null;
  flow_duration_s: number | null;
  flow_in_bytes: number | null;
  flow_out_bytes: number | null;
  flow_tcp_flags: string | null;
  flow_total_bytes: number | null;
  predicted_label: string | null;
  confidence: number | null;
  anomaly_score: number | null;
  is_anomaly: boolean | null;
  inference_logic: string | null;
  feedback_action: string | null;
  feedback_true_label: string | null;
  feedback_note: string | null;
  feedback_user_id: string | null;
  feedback_created_at: string | null;
}

// ─── UI Model Type ────────────────────────────────────────────────────────────
export interface FlowUIModel {
  id: string;
  homeId: string;    // flow_home_id – used for @ autocomplete
  time: string;
  rawTs: string;
  srcIp: string;
  dstIp: string;
  srcPort: string;
  dstPort: string;
  protocol: string;
  duration: string;
  inBytes: string;
  outBytes: string;
  tcpFlags: string;
  predictedLabel: string;
  trueLabel: string | null;
  confidencePct: string;
  isAnomaly: boolean;
  hasFeedback: boolean;
  inferenceLogic: string;
}

// ─── Aggregations / Charts ────────────────────────────────────────────────────
export interface LabelAggregation {
  label: string;
  count: number;
  color: string;
}

export interface TopAttacker {
  ip: string;
  count: number;
  label: string;
}

export interface TimelinePoint {
  time: string;
  [label: string]: number | string;
}

// ─── Action Types ─────────────────────────────────────────────────────────────
export interface LabelFeedbackRequest {
  flowId: string;
  trueLabel: string;
  note?: string;
}

export interface FlowQueryParams {
  query: string;
  from: string;
  to: string;
}
