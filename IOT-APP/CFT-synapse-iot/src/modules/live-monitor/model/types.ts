/**
 * Live Monitor – Types aligned to DB schema
 *
 * Nguồn dữ liệu:
 *  - Flow (Raw Data Table): bảng network_flows_feedback_all (Raspberry Pi → Edge → Supabase)
 *  - Alert (Live Alert Stack): bảng alerts_all (Cloud model → Supabase)
 *  - EdgeSyncStatus: bảng edge_nodes (trạng thái node Pi)
 *  - RadarPoint: anomaly score tổng hợp từ flows theo thời gian
 */

// ─────────────── Legacy type (giữ backward-compat) ───────────────
export type WsMessageType = 'telemetry_log' | 'alert_log';

export interface LogEntry {
  id: string;
  time: string;
  nodeId: string;
  srcIp: string;
  type: WsMessageType;
  label: string;
  confidencePct?: string;
  severity?: string;
}

// ─────────────── Flow – Raw network flow từ edge (Pi) ───────────────
/** Row trả về từ network_flows_feedback_all */
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
  flow_total_bytes: number | null;
  flow_in_bytes: number | null;
  flow_out_bytes: number | null;
  is_anomaly: boolean | null;
  predicted_label: string | null;
  confidence: number | null;
  anomaly_score: number | null;
  inference_logic: string | null;
}

/** UI Model cho Dual-Stream Raw Data Table */
export interface FlowUIModel {
  id: string;
  timestamp: string; // Full ISO timestamp for sorting/filtering
  timestampDisplay?: string; // Formatted time for display (HH:MM:SS)
  protocol: string;
  srcIp: string;
  dstIp: string;
  srcPort: number;
  dstPort: number;
  duration: number;
  bytes: number;
  isAnomaly: boolean;
  predictedLabel: string | null;
  confidence: number | null;
  anomalyScore: number | null;
  nodeId: string | null;
  homeId: string;
}

// ─────────────── Alert – Cảnh báo từ Cloud model ───────────────
/** Row trả về từ alerts_all */
export interface AlertRow {
  alert_id: string;
  alert_home_id: string;
  alert_first_seen_at: string;
  alert_threat_type: string;
  alert_severity: string;
  alert_status: string;
  alert_confidence: number | null;
  alert_source_ip: string | null;
  alert_target_ip: string | null;
  alert_predicted_label: string | null;
  alert_verdict_text: string | null;
  alert_sequence_values_json: string | null;
  alert_node_id: string | null;
  alert_class_id: number | null;
}

/** Severity giới hạn để type-check */
export type AlertSeverity = 'critical' | 'warning';

/** Status của alert */
export type AlertStatus = 'pending' | 'verifying' | 'confirmed' | 'false-positive';

/** UI Model cho Live Alert Stack */
export interface AlertUIModel {
  id: string;
  time: string; // Full ISO timestamp for sorting/filtering
  timeDisplay?: string; // Formatted time for display (HH:MM:SS)
  threat: string;
  severity: AlertSeverity;
  sequence: number[];
  status: AlertStatus;
  verdict?: string;
  confidence: number | null;
  srcIp: string | null;
  nodeId: string | null;
  homeId: string;
  isAggregated?: boolean;
  aggregatedCount?: number;
  aggregatedAlerts?: AlertUIModel[];
  alert_sequence_values_json?: string | null;
}

// ─────────────── Edge Sync Status ───────────────
/** Row từ edge_nodes */
export interface EdgeNodeRow {
  id: string;
  home_id: string;
  node_code: string;
  status: string;
  current_cpu_percent: number | null;
  current_ram_percent: number | null;
  current_temperature_c: number | null;
  current_latency_ms: number | null;
  last_seen_at: string | null;
}

/** UI Model cho Edge-to-Cloud Sync panel */
export interface EdgeSyncStatus {
  id: string;
  homeId: string;
  nodeCode: string;
  isOnline: boolean;
  cpu: number | null;
  ram: number | null;
  temperature: number | null;
  latencyMs: number | null;
  lastSeenAt: string | null;
}

// ─────────────── Radar Point ───────────────
export interface RadarPoint {
  time: number;
  score: number;
}
