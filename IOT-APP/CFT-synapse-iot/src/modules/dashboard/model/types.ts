/**
 * Dashboard Module – Types aligned to real DB schema.
 * Bảng: alerts_all, edge_nodes, model_versions, homes
 */

// ---- Raw DB row types (subset dùng cho UI) ----
export interface AlertRow {
  alert_id: string;
  alert_home_id: string;
  alert_first_seen_at: string;
  alert_source_ip: string | null;
  alert_target_ip: string | null;
  alert_predicted_label: string | null;
  alert_confidence: number | null;
  alert_severity: string;
  alert_status: string;
  alert_threat_type: string;
  alert_node_id: string | null;
  alert_source_text: string | null;
  alert_sequence_values_json: string | null;
  alert_sequence_steps_json: string | null;
  // Joined fields for user-friendly display
  home_name?: string;
  home_code?: string;
  node_code?: string;
  node_location?: string;
}

// ---- Home (location) ----
export interface HomeInfo {
  id: string;
  code: string;
  name: string;
}

export interface EdgeNodeRow {
  id: string;
  node_code: string;
  status: string;
  location_text: string | null;
  ip_address: string | null;
  last_seen_at: string | null;
  current_cpu_percent: number | null;
  current_ram_percent: number | null;
  current_temperature_c: number | null;
  current_latency_ms: number | null;
  framework: string | null;
  model_version_text: string | null;
}

export interface ModelStats {
  version: string;
  status: string;
  accuracy: number | null;
  f1_score: number | null;
  precision: number | null;
  recall: number | null;
  latency_ms: number | null;
  throughput_per_s: number | null;
  false_positive_rate: number | null;
}

// ---- Summary card ----
export interface TelemetrySummary {
  totalDevices: number;
  onlineDevices: number;
  alertsToday: number;
  criticalAlerts: number;
}

// ---- Attack distribution for bar chart ----
export interface AttackDistPoint {
  name: string;
  value: number;
  color: string;
}

// ---- UI Model ----
export interface AlertUIModel {
  id: string;
  time: string;
  srcIp: string;
  targetIp: string;
  label: string;
  confidencePct: string;
  confidenceVal: number;  // 0–1 raw
  status: string;
  severity: 'high' | 'medium' | 'low';
  source: string;
  seqValues: number[];
  seqSteps: string[];
  // User-friendly display fields (no raw IPs for regular users)
  homeName: string;
  homeCode: string;
  deviceName: string;    // e.g. "Thiết bị tại Smart Home A" or node_code
  locationName: string;  // e.g. "Smart Home A" or location_text
  alert_node_id: string | null;
  alert_home_id: string;
}

/**
 * Traffic series point — mỗi giờ, mỗi home là một key riêng.
 * Ví dụ: { hour: "10:00", "Home 101": 2, "EDGE102": 0, "EDGE103": 1 }
 * Dùng với recharts Line, mỗi home là một <Line dataKey={home.code} />
 */
export interface TrafficSeriesPoint {
  hour: string;
  [homeCode: string]: number | string; // string vì `hour` vẫn là string
}
