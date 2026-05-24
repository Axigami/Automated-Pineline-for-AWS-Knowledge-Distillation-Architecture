/**
 * Live Monitor – Adapter layer
 * Chuyển đổi raw DB rows → UI Models (clean data cho View)
 */
import type {
  FlowRow, FlowUIModel,
  AlertRow, AlertUIModel, AlertSeverity, AlertStatus,
  EdgeNodeRow, EdgeSyncStatus,
} from './types';
import { normalizeEdgePresence } from '../../../utils/edgeNodeStatus';

// ─────────────── Flow Adapter ───────────────

export function adaptFlowRow(row: FlowRow): FlowUIModel {
  // Store full ISO timestamp for sorting/filtering, display formatted time
  const fullTimestamp = new Date(row.flow_ts);
  
  return {
    id: row.flow_id,
    timestamp: fullTimestamp.toISOString(), // Store full ISO timestamp
    timestampDisplay: fullTimestamp.toLocaleTimeString('en-GB', { hour12: false }), // Display only time
    protocol: row.flow_protocol ?? '—',
    srcIp: row.flow_src_ip ?? '—',
    dstIp: row.flow_dst_ip ?? '—',
    srcPort: row.flow_src_port ?? 0,
    dstPort: row.flow_dst_port ?? 0,
    duration: row.flow_duration_s ?? 0,
    bytes: row.flow_total_bytes ?? (row.flow_in_bytes ?? 0) + (row.flow_out_bytes ?? 0),
    isAnomaly: row.is_anomaly ?? false,
    predictedLabel: row.predicted_label ?? null,
    confidence: row.confidence ?? null,
    anomalyScore: row.anomaly_score ?? null,
    nodeId: row.flow_node_id ?? null,
    homeId: row.flow_home_id,
  };
}

// ─────────────── Alert Adapter ───────────────

const SEVERITY_MAP: Record<string, AlertSeverity> = {
  critical: 'critical',
  high: 'critical',
  warning: 'warning',
  medium: 'warning',
  low: 'warning',
};

const STATUS_MAP: Record<string, AlertStatus> = {
  pending: 'pending',
  open: 'pending',
  verifying: 'verifying',
  verified: 'confirmed',
  confirmed: 'confirmed',
  resolved: 'confirmed',
  acknowledged: 'confirmed',
  'false-positive': 'false-positive',
  false_positive: 'false-positive',
  closed: 'false-positive',
};

/**
 * Parse JSON sequence values từ DB column.
 * Trả về mảng số từ 0-1 (10 phần tử).
 */
function parseSequenceValues(json: string | null): number[] {
  if (!json) return Array.from({ length: 10 }, () => 0.05);
  try {
    const raw = JSON.parse(json);
    if (Array.isArray(raw)) {
      return raw.slice(0, 10).map((v) => typeof v === 'number' ? Math.min(Math.max(v, 0), 1) : 0);
    }
    return Array.from({ length: 10 }, () => 0);
  } catch {
    return Array.from({ length: 10 }, () => 0);
  }
}

export function adaptAlertRow(row: AlertRow): AlertUIModel {
  const severityRaw = (row.alert_severity ?? 'warning').toLowerCase();
  const statusRaw = (row.alert_status ?? 'pending').toLowerCase().replace(/ /g, '-');
  const fullTimestamp = new Date(row.alert_first_seen_at);

  return {
    id: row.alert_id,
    time: fullTimestamp.toISOString(), // Store full ISO timestamp
    timeDisplay: fullTimestamp.toLocaleTimeString('en-GB', { hour12: false }), // Display only time
    threat: row.alert_predicted_label ?? row.alert_threat_type,
    severity: SEVERITY_MAP[severityRaw] ?? 'warning',
    sequence: parseSequenceValues(row.alert_sequence_values_json),
    status: STATUS_MAP[statusRaw] ?? 'pending',
    verdict: row.alert_verdict_text ?? undefined,
    confidence: row.alert_confidence,
    srcIp: row.alert_source_ip ?? null,
    nodeId: row.alert_node_id ?? null,
    homeId: row.alert_home_id,
  };
}

// ─────────────── Edge Node Adapter ───────────────

export function adaptEdgeNodeRow(row: EdgeNodeRow): EdgeSyncStatus {
  const lastSeen = row.last_seen_at ? new Date(row.last_seen_at) : null;
  const thresholdMs = 5 * 60 * 1000; // 5 phút
  const byStatus = normalizeEdgePresence(row.status) === 'online';
  const byHeartbeat = lastSeen != null && Date.now() - lastSeen.getTime() < thresholdMs;
  const isOnline = byStatus || byHeartbeat;

  return {
    id: row.id,
    homeId: row.home_id,
    nodeCode: row.node_code,
    isOnline,
    cpu: row.current_cpu_percent ?? null,
    ram: row.current_ram_percent ?? null,
    temperature: row.current_temperature_c ?? null,
    latencyMs: row.current_latency_ms ?? null,
    lastSeenAt: lastSeen
      ? lastSeen.toLocaleTimeString('en-GB', { hour12: false })
      : null,
  };
}
