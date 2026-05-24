import type { AlertRow, AlertUIModel, EdgeNodeRow } from './types';

const ATTACK_COLORS: Record<string, string> = {
  ddos: '#f43f5e',
  dos: '#fb923c',
  portscan: '#facc15',
  botnet: '#a78bfa',
  default: '#64748b',
};

function mapSeverity(severity: string): AlertUIModel['severity'] {
  const s = severity.toLowerCase();
  if (s === 'critical' || s === 'high') return 'high';
  if (s === 'medium') return 'medium';
  return 'low';
}

function parseJsonArray(json: string | null): any[] {
  if (!json) return [];
  try { return JSON.parse(json); } catch { return []; }
}

export function adaptAlert(row: AlertRow): AlertUIModel {
  const seqValues = parseJsonArray(row.alert_sequence_values_json) as number[];
  const seqSteps  = parseJsonArray(row.alert_sequence_steps_json)  as string[];

  // Build user-friendly display names
  const homeName = row.home_name ?? '';
  const homeCode = row.home_code ?? '';
  const nodeCode = row.node_code ?? '';
  const nodeLocation = row.node_location ?? '';
  
  // deviceName: prefer node_code, fallback to source_text
  const deviceName = nodeCode || row.alert_source_text || '—';
  // locationName: prefer home_name, fallback to node_location
  const locationName = homeName || nodeLocation || '—';

  return {
    id:           row.alert_id,
    time:         new Date(row.alert_first_seen_at).toLocaleTimeString('vi-VN', { hour12: false }),
    srcIp:        row.alert_source_ip ?? '—',
    targetIp:     row.alert_target_ip ?? row.alert_source_ip ?? '—',
    label:        row.alert_predicted_label ?? row.alert_threat_type,
    confidencePct: row.alert_confidence != null
      ? `${Math.round(row.alert_confidence * 100)}%`
      : '—',
    confidenceVal: row.alert_confidence ?? 0,
    status:       row.alert_status,
    severity:     mapSeverity(row.alert_severity),
    source:       row.alert_source_text ?? row.alert_node_id ?? '—',
    seqValues,
    seqSteps,
    homeName,
    homeCode,
    deviceName,
    locationName,
    alert_node_id: row.alert_node_id ?? null,
    alert_home_id: row.alert_home_id,
  };
}

export function adaptAlerts(rows: AlertRow[]): AlertUIModel[] {
  return rows.map(adaptAlert);
}

// Màu badge theo loại tấn công
export function getAttackColor(name: string): string {
  return ATTACK_COLORS[name.toLowerCase()] ?? ATTACK_COLORS.default;
}
